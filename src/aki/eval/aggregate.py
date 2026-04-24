"""Aggregate per-model tables into a single results file.

Reads every ``reports/tables/per_model/<tag>/bootstrap_ci.csv`` and
produces::

    reports/tables/final_results.csv   :one row per (task, family, model)
    reports/tables/final_results.md    :Markdown table for the thesis
    reports/tables/best_per_task.csv   :best model per task w/ CIs

Also records the *input economy gate* result (minimal vs combined)
when both families were evaluated.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from aki.utils.config import Config
from aki.utils.paths import paths
from aki.utils.subset import artifact_triple_from_path, matches_selector, output_path

_KEY_METRICS = ("auroc", "auprc", "brier", "calibration_slope",
                "calibration_intercept", "ece",
                "sensitivity_at_spec_90", "specificity_at_sens_90")


def build_final_results(
    cfg: Config,
    *,
    tasks: list[str] | None = None,
    families: list[str] | None = None,
    models: list[str] | None = None,
    output_tag: str | None = None,
) -> pd.DataFrame:
    """Collect CI-annotated metrics and emit summary tables."""
    rows: list[dict] = []
    for tag_dir in sorted((paths.tables / "per_model").glob("*")):
        if not tag_dir.is_dir():
            continue
        try:
            task, family, model = artifact_triple_from_path(tag_dir)
        except ValueError:
            continue
        if not matches_selector(
            task=task, family=family, model=model,
            tasks=tasks, families=families, models=models,
        ):
            continue
        ci_path = tag_dir / "bootstrap_ci.csv"
        if not ci_path.exists():
            continue

        ci = pd.read_csv(ci_path).set_index("metric")

        row = {"task": task, "family": family, "model": model}
        for m in _KEY_METRICS:
            if m not in ci.index:
                continue
            row[m] = ci.at[m, "point"]
            row[f"{m}_lo"] = ci.at[m, "ci_lower"]
            row[f"{m}_hi"] = ci.at[m, "ci_upper"]
        rows.append(row)

    if not rows:
        logger.warning("No per-model bootstrap tables found — run `aki evaluate` first.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(["task", "family", "model"], ignore_index=True)
    csv_path = output_path(paths.tables / "final_results.csv", output_tag)
    df.to_csv(csv_path, index=False)

    _write_markdown(df, output_path(paths.tables / "final_results.md", output_tag))
    _write_best_per_task(df, output_path(paths.tables / "best_per_task.csv", output_tag))
    _write_input_economy_table(
        df,
        output_path(paths.tables / "input_economy.csv", output_tag),
        cfg,
    )
    logger.info(f"final results -> {csv_path}")
    return df


def _write_markdown(df: pd.DataFrame, out: Path) -> None:
    lines = ["# Final results", ""]
    for task, grp in df.groupby("task"):
        lines += [f"## {task}", "",
                  "| Family | Model | AUROC (95% CI) | AUPRC (95% CI) | Brier | Slope | ECE |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in grp.iterrows():
            lines.append(
                f"| {r['family']} | {r['model']} "
                f"| {r['auroc']:.3f} ({r['auroc_lo']:.3f}–{r['auroc_hi']:.3f}) "
                f"| {r['auprc']:.3f} ({r['auprc_lo']:.3f}–{r['auprc_hi']:.3f}) "
                f"| {r['brier']:.3f} "
                f"| {r.get('calibration_slope', float('nan')):.2f} "
                f"| {r.get('ece', float('nan')):.3f} |"
            )
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def _write_best_per_task(df: pd.DataFrame, out: Path) -> None:
    best = (
        df.sort_values("auroc", ascending=False)
        .groupby("task", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best.to_csv(out, index=False)


def _write_input_economy_table(df: pd.DataFrame, out: Path, cfg: Config) -> None:
    if "minimal" not in df["family"].unique() or "combined" not in df["family"].unique():
        return

    gate_cfg = cfg.eval["input_economy"]
    auroc_tol = float(gate_cfg["auroc_tolerance"])
    slope_tol = float(gate_cfg.get("calibration_tolerance", 0.05))
    ece_tol = float(gate_cfg.get("ece_tolerance", 0.02))
    nb_tol = float(gate_cfg.get("net_benefit_tolerance", 0.0))

    metrics = ["auroc", "calibration_slope", "ece"]
    combined = df[df["family"] == "combined"][["task", "model", *metrics]]
    minimal = df[df["family"] == "minimal"][["task", "model", *metrics]]
    merged = combined.merge(
        minimal, on=["task", "model"], suffixes=("_combined", "_minimal")
    )
    if merged.empty:
        return

    merged["delta_auroc"] = merged["auroc_combined"] - merged["auroc_minimal"]
    merged["delta_ece"] = merged["ece_minimal"] - merged["ece_combined"]
    merged["minimal_slope_error"] = (merged["calibration_slope_minimal"] - 1.0).abs()
    merged["passes_auroc"] = merged["delta_auroc"] <= auroc_tol
    merged["passes_calibration"] = (
        (merged["minimal_slope_error"] <= slope_tol)
        & (merged["delta_ece"] <= ece_tol)
    )

    thresholds = [float(t) for t in cfg.eval["decision_curve"]["clinical_thresholds"]]
    for threshold in thresholds:
        col = f"delta_net_benefit_at_{threshold:g}"
        merged[col] = merged.apply(
            lambda r, threshold=threshold: _net_benefit_delta(
                r["task"], r["model"], threshold,
            ),
            axis=1,
        )
    nb_cols = [c for c in merged.columns if c.startswith("delta_net_benefit_at_")]
    if nb_cols:
        merged["passes_decision_curve"] = merged[nb_cols].ge(-nb_tol).all(axis=1)
    else:
        merged["passes_decision_curve"] = True

    merged["passes_gate"] = (
        merged["passes_auroc"]
        & merged["passes_calibration"]
        & merged["passes_decision_curve"]
    )
    merged.to_csv(out, index=False)


def _net_benefit_delta(task: str, model: str, threshold: float) -> float:
    """Return minimal - combined net benefit at the nearest saved threshold."""
    combined = _net_benefit_at(task, "combined", model, threshold)
    minimal = _net_benefit_at(task, "minimal", model, threshold)
    if pd.isna(combined) or pd.isna(minimal):
        return float("nan")
    return float(minimal - combined)


def _net_benefit_at(task: str, family: str, model: str, threshold: float) -> float:
    dc_path = (
        paths.tables / "per_model" / f"{task}__{family}__{model}" / "decision_curve.csv"
    )
    if not dc_path.exists():
        return float("nan")
    dc = pd.read_csv(dc_path)
    idx = (dc["threshold"] - threshold).abs().idxmin()
    return float(dc.loc[idx, "nb_model"])
