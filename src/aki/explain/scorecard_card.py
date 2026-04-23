"""Printable scorecard artifact.

Converts a fitted sparse-logistic ``ModelArtifact`` into:
- ``scorecard.csv``: feature-level summary in raw and standardized units,
- ``scorecard_points.csv``: simple raw-value anchor rules with points,
- ``scorecard.md``: thesis-friendly markdown with both tables.

The points table is intentionally approximate: it is derived from the
fitted linear predictor and training quantiles stored on the artifact.
That makes it useful for interpretation and draft score-sheet design
without pretending these are clinician-validated bedside cutoffs yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aki.models.base import ModelArtifact


def build_scorecard_artifact(
    art: ModelArtifact,
    out_dir: Path,
) -> dict[str, Path]:
    """Write scorecard tables into ``out_dir``."""
    if art.name != "scorecard":
        raise ValueError("build_scorecard_artifact requires a scorecard artifact")

    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _feature_summary_table(art)
    summary_path = out_dir / "scorecard.csv"
    summary.to_csv(summary_path, index=False)

    points = _points_table(art, summary)
    points_path = out_dir / "scorecard_points.csv"
    points.to_csv(points_path, index=False)

    md_path = out_dir / "scorecard.md"
    md_path.write_text(_markdown_report(art, summary, points), encoding="utf-8")

    return {"csv": summary_path, "points_csv": points_path, "md": md_path}


def _feature_summary_table(art: ModelArtifact) -> pd.DataFrame:
    clf = art.estimator.named_steps["clf"]
    coef = clf.coef_.ravel()
    profiles = art.extra.get("feature_profiles", {})

    rows: list[dict[str, Any]] = []
    for feature, coefficient in zip(art.feature_names, coef, strict=True):
        profile = profiles.get(feature, {})
        scale = float(profile.get("scale", 1.0) or 1.0)
        quantiles = profile.get("quantiles", {})

        rows.append(
            {
                "feature": feature,
                "kind": profile.get("kind", "continuous"),
                "coefficient_per_sd": float(coefficient),
                "odds_ratio_per_sd": float(np.exp(coefficient)),
                "logodds_per_unit": float(coefficient / scale),
                "odds_ratio_per_unit": float(np.exp(coefficient / scale)),
                "impute_median": float(profile.get("impute_median", np.nan)),
                "train_mean": float(profile.get("mean", np.nan)),
                "train_sd": scale,
                "q10": float(quantiles.get("0.10", np.nan)),
                "q25": float(quantiles.get("0.25", np.nan)),
                "median": float(quantiles.get("0.50", np.nan)),
                "q75": float(quantiles.get("0.75", np.nan)),
                "q90": float(quantiles.get("0.90", np.nan)),
                "selected_C": float(art.extra.get("selected_C", np.nan)),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    nonzero = table.loc[table["coefficient_per_sd"] != 0, "coefficient_per_sd"].abs()
    point_scale = 1.0 if nonzero.empty else 1.0 / float(nonzero.min())
    table["points_per_sd"] = np.round(table["coefficient_per_sd"] * point_scale).astype(int)
    return table.sort_values(
        "coefficient_per_sd",
        key=np.abs,
        ascending=False,
        ignore_index=True,
    )


def _points_table(art: ModelArtifact, summary: pd.DataFrame) -> pd.DataFrame:
    profiles = art.extra.get("feature_profiles", {})
    rows: list[dict[str, Any]] = []

    for row in summary.itertuples(index=False):
        profile = profiles.get(row.feature, {})
        baseline = float(profile.get("quantiles", {}).get("0.50", np.nan))
        baseline = baseline if np.isfinite(baseline) else float(profile.get("impute_median", 0.0))
        scale = float(profile.get("scale", 1.0) or 1.0)
        mean = float(profile.get("mean", 0.0))
        coef = float(row.coefficient_per_sd)
        baseline_z = (baseline - mean) / scale

        anchors = _anchor_values(profile)
        for label, value in anchors:
            value = float(value)
            z_value = (value - mean) / scale
            delta_logodds = coef * (z_value - baseline_z)
            rows.append(
                {
                    "feature": row.feature,
                    "kind": row.kind,
                    "anchor": label,
                    "representative_value": value,
                    "baseline_value": baseline,
                    "delta_logodds_vs_median": float(delta_logodds),
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    nonzero = table.loc[
        table["delta_logodds_vs_median"] != 0,
        "delta_logodds_vs_median",
    ].abs()
    point_scale = 1.0 if nonzero.empty else 1.0 / float(nonzero.min())
    table["points_vs_median"] = np.round(
        table["delta_logodds_vs_median"] * point_scale
    ).astype(int)
    return table.sort_values(
        ["feature", "representative_value"],
        ignore_index=True,
    )


def _anchor_values(profile: dict[str, Any]) -> list[tuple[str, float]]:
    kind = profile.get("kind", "continuous")
    if kind == "binary":
        unique_values = [float(v) for v in profile.get("unique_values", [])]
        if not unique_values:
            unique_values = [0.0, 1.0]
        labels = {0.0: "0 / absent", 1.0: "1 / present"}
        return [(labels.get(v, str(v)), v) for v in unique_values]

    quantiles = profile.get("quantiles", {})
    anchors = [
        ("Q25", quantiles.get("0.25")),
        ("Median", quantiles.get("0.50")),
        ("Q75", quantiles.get("0.75")),
        ("Q90", quantiles.get("0.90")),
    ]
    deduped: list[tuple[str, float]] = []
    seen: set[float] = set()
    for label, value in anchors:
        if value is None:
            continue
        v = float(value)
        if not np.isfinite(v):
            continue
        rounded = round(v, 8)
        if rounded in seen:
            continue
        seen.add(rounded)
        deduped.append((label, v))
    return deduped


def _markdown_report(
    art: ModelArtifact,
    summary: pd.DataFrame,
    points: pd.DataFrame,
) -> str:
    intercept = float(art.estimator.named_steps["clf"].intercept_.ravel()[0])
    base_rate = art.extra.get("base_rate")
    selected_c = art.extra.get("selected_C")
    lines = [
        f"# Scorecard - {art.task} ({art.family})",
        "",
        f"- Intercept: **{intercept:+.3f}** log-odds",
        f"- Selected L1 C: **{selected_c:.4g}**" if selected_c is not None else "- Selected L1 C: n/a",
        f"- Training prevalence: **{float(base_rate):.3f}**" if base_rate is not None else "- Training prevalence: n/a",
        f"- Selected features: **{len(summary)}**",
        "",
        "## Feature Summary",
        "",
        "| Feature | OR / +1 SD | OR / raw unit | Q25 | Median | Q75 | Points / +1 SD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| `{row.feature}` | {row.odds_ratio_per_sd:.2f} | {row.odds_ratio_per_unit:.2f} "
            f"| {_fmt(row.q25)} | {_fmt(row.median)} | {_fmt(row.q75)} | {row.points_per_sd:+d} |"
        )

    lines += [
        "",
        "## Anchor Points",
        "",
        "| Feature | Anchor | Representative Value | Delta Log-Odds vs Median | Points |",
        "|---|---|---:|---:|---:|",
    ]
    for row in points.itertuples(index=False):
        lines.append(
            f"| `{row.feature}` | {row.anchor} | {_fmt(row.representative_value)} "
            f"| {row.delta_logodds_vs_median:+.3f} | {row.points_vs_median:+d} |"
        )

    lines += [
        "",
        "These anchor points are data-driven summaries from the training set, not",
        "prescriptive bedside cutoffs or causal treatment thresholds.",
    ]
    return "\n".join(lines)


def _fmt(value: float) -> str:
    if not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.2f}"
