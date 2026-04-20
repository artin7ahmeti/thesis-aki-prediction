"""Patient-level additive contributions.

For the two glass-box models (EBM, scorecard), decomposes an individual
prediction into per-feature contributions on the log-odds scale. This is
what a clinician sees on a *case card*: "feature X pushed the probability
up by Y".

Important: these are **associations**, not causal effects. The returned
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
            "feature":                d["names"],
            "value":                  d.get("values", [np.nan] * len(d["names"])),
            "contribution_logodds":   d["scores"],
        })
        .sort_values("contribution_logodds", key=np.abs, ascending=False, ignore_index=True)
    )


def _scorecard_contributions(art: ModelArtifact, x_row: pd.Series) -> pd.DataFrame:
    """Contribution = coefficient × standardized value."""
    pipeline = art.estimator
    raw = x_row[art.feature_names].values.reshape(1, -1)
    imputed = pipeline.named_steps["impute"].transform(raw)
    scaled  = pipeline.named_steps["scale"].transform(imputed)
    coef = pipeline.named_steps["clf"].coef_.ravel()

    contribs = coef * scaled.ravel()
    return (
        pd.DataFrame({
            "feature":                art.feature_names,
            "value":                  raw.ravel(),
            "scaled_value":           scaled.ravel(),
            "coefficient":            coef,
            "contribution_logodds":   contribs,
        })
        .sort_values("contribution_logodds", key=np.abs, ascending=False, ignore_index=True)
    )
