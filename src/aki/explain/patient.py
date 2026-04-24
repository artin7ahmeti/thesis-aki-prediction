"""Patient-level additive contributions.

For the two glass-box models (EBM, scorecard), decomposes an individual
prediction into per-feature contributions on the log-odds scale. This is
what a clinician sees on a case card: "feature X pushed the probability
up by Y".

Important: these are associations, not causal effects. The returned
frame should be labeled as such in any figures/text.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aki.models.base import ModelArtifact


def patient_additive_contributions(
    art: ModelArtifact,
    x_row: pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    """Return (feature, value, contribution_logodds) for one landmark row."""
    if isinstance(x_row, pd.DataFrame):
        if len(x_row) != 1:
            raise ValueError("x_row must be a single row")
        x_row = x_row.iloc[0]

    if art.name == "ebm":
        return _ebm_contributions(art, x_row)
    if art.name == "scorecard":
        return _scorecard_contributions(art, x_row)
    raise ValueError(f"patient contributions not supported for {art.name}")


def _ebm_contributions(art: ModelArtifact, x_row: pd.Series) -> pd.DataFrame:
    """Use EBM's local explanation API."""
    X = pd.DataFrame([x_row[art.feature_names].values], columns=art.feature_names)
    local = art.estimator.explain_local(X, y=None)
    d = local.data(0)
    return (
        pd.DataFrame({
            "feature": d["names"],
            "value": d.get("values", [np.nan] * len(d["names"])),
            "contribution_logodds": d["scores"],
        })
        .sort_values("contribution_logodds", key=np.abs, ascending=False, ignore_index=True)
    )


def _scorecard_contributions(art: ModelArtifact, x_row: pd.Series) -> pd.DataFrame:
    """Return per-feature scorecard contributions for one patient row."""
    if art.extra.get("scorecard_representation") == "binned":
        return _binned_scorecard_contributions(art, x_row)
    return _linear_scorecard_contributions(art, x_row)


def _linear_scorecard_contributions(art: ModelArtifact, x_row: pd.Series) -> pd.DataFrame:
    pipeline = art.estimator
    raw = x_row[art.feature_names].values.reshape(1, -1)
    imputed = pipeline.named_steps["impute"].transform(raw)
    scaled = pipeline.named_steps["scale"].transform(imputed)
    coef = pipeline.named_steps["clf"].coef_.ravel()

    contribs = coef * scaled.ravel()
    return (
        pd.DataFrame({
            "feature": art.feature_names,
            "value": raw.ravel(),
            "scaled_value": scaled.ravel(),
            "coefficient": coef,
            "contribution_logodds": contribs,
        })
        .sort_values("contribution_logodds", key=np.abs, ascending=False, ignore_index=True)
    )


def _binned_scorecard_contributions(art: ModelArtifact, x_row: pd.Series) -> pd.DataFrame:
    pipeline = art.estimator
    X = pd.DataFrame([x_row[art.feature_names]], columns=art.feature_names)
    transformed = pipeline.named_steps["design"].transform(X)
    coef = pipeline.named_steps["clf"].coef_.ravel()
    term_meta = art.extra.get("term_metadata", [])
    profiles = art.extra.get("feature_profiles", {})

    coeff_by_term = {
        meta.get("term_name"): float(c)
        for meta, c in zip(term_meta, coef, strict=True)
    }
    value_by_term = {
        meta.get("term_name"): float(v)
        for meta, v in zip(term_meta, transformed.ravel(), strict=True)
    }

    rows: list[dict[str, object]] = []
    for feature in art.feature_names:
        raw_value = float(x_row[feature])
        profile = profiles.get(feature, {})
        if profile.get("kind") == "binary":
            active = bool(round(raw_value))
            coefficient = coeff_by_term.get(feature, 0.0)
            rows.append(
                {
                    "feature": feature,
                    "value": raw_value,
                    "active_level": "1 / present" if active else "0 / absent",
                    "coefficient": coefficient if active else 0.0,
                    "contribution_logodds": coefficient if active else 0.0,
                }
            )
            continue

        active_level = profile.get("reference_bin_label", "reference")
        coefficient = 0.0
        for level in profile.get("bins", []):
            if level.get("reference"):
                continue
            term_name = level.get("term_name")
            if float(value_by_term.get(term_name, 0.0)) >= 0.5:
                active_level = level.get("range_display", level.get("label"))
                coefficient = coeff_by_term.get(term_name, 0.0)
                break
        rows.append(
            {
                "feature": feature,
                "value": raw_value,
                "active_level": active_level,
                "coefficient": coefficient,
                "contribution_logodds": coefficient,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("contribution_logodds", key=np.abs, ascending=False, ignore_index=True)
    )
