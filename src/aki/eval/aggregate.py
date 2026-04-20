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

_KEY_METRICS = ("auroc", "auprc", "brier", "calibration_slope",
                "calibration_intercept", "ece",
                "sensitivity_at_spec_90", "specificity_at_sens_90")


def build_final_results(cfg: Config) -> pd.DataFrame:
    """Collect CI-annotated metrics and emit summary tables."""
    rows: list[dict] = []
    for tag_dir in sorted((paths.tables / "per_model").glob("*")):
        if not tag_dir.is_dir():
            continue
        ci_path = tag_dir / "bootstrap_ci.csv"
        if not ci_path.exists():
            continue

        task, family, model = tag_dir.name.split("__")
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
    csv_path = paths.tables / "final_results.csv"
    df.to_csv(csv_path, index=False)

    _write_markdown(df, paths.tables / "final_results.md")
    _write_best_per_task(df, paths.tables / "best_per_task.csv")
    _write_input_economy_table(df, paths.tables / "input_economy.csv", cfg)
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
    tol = float(cfg.eval["input_economy"]["auroc_tolerance"])
    pivot = df.pivot_table(
        index=["task", "model"], columns="family", values="auroc"
    ).reset_index()
    if "minimal" not in pivot.columns or "combined" not in pivot.columns:
        return
    pivot["delta"] = pivot["combined"] - pivot["minimal"]
    pivot["passes_gate"] = pivot["delta"] <= tol
    pivot.to_csv(out, index=False)
