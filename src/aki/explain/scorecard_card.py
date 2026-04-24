"""Printable scorecard artifact.

For the original linear sparse-logistic scorecard, this module exports:
- ``scorecard.csv``: feature-level summary in raw and standardized units
- ``scorecard_points.csv``: simple anchor-point rules from training quantiles

For the newer bedside-binned scorecard, it exports:
- ``scorecard.csv``: one row per raw feature summarizing the score sheet
- ``scorecard_points.csv``: explicit bedside bins / levels with odds ratios
  and points relative to the reference level
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

    if art.extra.get("scorecard_representation") == "binned":
        points = _binned_points_table(art)
        summary = _binned_feature_summary_table(art, points)
        markdown = _binned_markdown_report(art, summary, points)
    else:
        summary = _linear_feature_summary_table(art)
        points = _linear_points_table(art, summary)
        markdown = _linear_markdown_report(art, summary, points)

    summary_path = out_dir / "scorecard.csv"
    summary.to_csv(summary_path, index=False)

    points_path = out_dir / "scorecard_points.csv"
    points.to_csv(points_path, index=False)

    md_path = out_dir / "scorecard.md"
    md_path.write_text(markdown, encoding="utf-8")

    return {"csv": summary_path, "points_csv": points_path, "md": md_path}


def _linear_feature_summary_table(art: ModelArtifact) -> pd.DataFrame:
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


def _linear_points_table(art: ModelArtifact, summary: pd.DataFrame) -> pd.DataFrame:
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

        anchors = _linear_anchor_values(profile)
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


def _binned_points_table(art: ModelArtifact) -> pd.DataFrame:
    clf = art.estimator.named_steps["clf"]
    coef_map = {
        meta.get("term_name", term): float(coef)
        for meta, term, coef in zip(
            art.extra.get("term_metadata", []),
            art.extra.get("scorecard_terms", []),
            clf.coef_.ravel(),
            strict=True,
        )
    }

    rows: list[dict[str, Any]] = []
    profiles = art.extra.get("feature_profiles", {})
    for feature in art.feature_names:
        profile = profiles.get(feature, {})
        kind = profile.get("kind")
        if kind == "binned_continuous":
            for level in profile.get("bins", []):
                coef = 0.0 if level.get("reference") else coef_map.get(level.get("term_name"), 0.0)
                rows.append(
                    {
                        "feature": feature,
                        "kind": kind,
                        "level_label": level.get("label"),
                        "range_display": level.get("range_display"),
                        "lower_bound": level.get("lower"),
                        "upper_bound": level.get("upper"),
                        "reference": bool(level.get("reference")),
                        "coefficient_logodds_vs_reference": float(coef),
                        "odds_ratio_vs_reference": float(np.exp(coef)),
                    }
                )
            continue

        if kind == "binary":
            rows.extend(
                [
                    {
                        "feature": feature,
                        "kind": kind,
                        "level_label": "0 / absent",
                        "range_display": "0 / absent",
                        "lower_bound": float("nan"),
                        "upper_bound": float("nan"),
                        "reference": True,
                        "coefficient_logodds_vs_reference": 0.0,
                        "odds_ratio_vs_reference": 1.0,
                    },
                    {
                        "feature": feature,
                        "kind": kind,
                        "level_label": "1 / present",
                        "range_display": "1 / present",
                        "lower_bound": float("nan"),
                        "upper_bound": float("nan"),
                        "reference": False,
                        "coefficient_logodds_vs_reference": float(coef_map.get(feature, 0.0)),
                        "odds_ratio_vs_reference": float(np.exp(coef_map.get(feature, 0.0))),
                    },
                ]
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    nonzero = table.loc[
        ~table["reference"] & (table["coefficient_logodds_vs_reference"] != 0),
        "coefficient_logodds_vs_reference",
    ].abs()
    point_scale = 1.0 if nonzero.empty else 1.0 / float(nonzero.min())
    table["points_vs_reference"] = np.round(
        table["coefficient_logodds_vs_reference"] * point_scale
    ).astype(int)
    table.loc[table["reference"], "points_vs_reference"] = 0

    order = (
        table.groupby("feature")["points_vs_reference"]
        .apply(lambda s: float(s.abs().max()))
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    feature_order = {feature: idx for idx, feature in enumerate(order)}
    return table.sort_values(
        ["feature", "reference", "points_vs_reference"],
        key=lambda s: s.map(feature_order) if s.name == "feature" else s,
        ascending=[True, False, False],
        ignore_index=True,
    )


def _binned_feature_summary_table(art: ModelArtifact, points: pd.DataFrame) -> pd.DataFrame:
    profiles = art.extra.get("feature_profiles", {})
    rows: list[dict[str, Any]] = []
    for feature in art.feature_names:
        subset = points[points["feature"] == feature]
        profile = profiles.get(feature, {})
        reference_level = ""
        if profile.get("kind") == "binned_continuous":
            reference_level = str(profile.get("reference_bin_label", "reference"))
        elif profile.get("kind") == "binary":
            reference_level = "0 / absent"
        rows.append(
            {
                "feature": feature,
                "kind": profile.get("kind", "unknown"),
                "n_levels": int(len(subset)),
                "reference_level": reference_level,
                "max_abs_logodds": float(subset["coefficient_logodds_vs_reference"].abs().max()),
                "max_abs_points": int(subset["points_vs_reference"].abs().max()),
                "selected_C": float(art.extra.get("selected_C", np.nan)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "max_abs_points", ascending=False, ignore_index=True
    )


def _linear_anchor_values(profile: dict[str, Any]) -> list[tuple[str, float]]:
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


def _linear_markdown_report(
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


def _binned_markdown_report(
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
        f"- Regularization C: **{selected_c:.4g}**" if selected_c is not None else "- Regularization C: n/a",
        "- Scorecard form: **clinically binned logistic scorecard**",
        f"- Training prevalence: **{float(base_rate):.3f}**" if base_rate is not None else "- Training prevalence: n/a",
        f"- Raw features: **{len(summary)}**",
        f"- Additive terms: **{int(art.extra.get('n_terms', 0))}**",
        "",
        "## Feature Summary",
        "",
        "| Feature | Levels | Reference | Max |",
        "|---|---:|---|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| `{row.feature}` | {int(row.n_levels)} | {row.reference_level} | {row.max_abs_points:+d} |"
        )

    lines += [
        "",
        "## Bedside Points",
        "",
        "| Feature | Level | OR vs Ref | Points |",
        "|---|---|---:|---:|",
    ]
    for row in points.itertuples(index=False):
        label = f"{row.range_display} (ref)" if row.reference else row.range_display
        lines.append(
            f"| `{row.feature}` | {label} | {row.odds_ratio_vs_reference:.2f} | {int(row.points_vs_reference):+d} |"
        )

    lines += [
        "",
        "These bins are the configured bedside scorecard levels used during training.",
        "They are clinically oriented working cutoffs for thesis evaluation, not",
        "causal treatment thresholds or externally validated practice rules.",
    ]
    return "\n".join(lines)


def _fmt(value: float) -> str:
    if not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.2f}"
