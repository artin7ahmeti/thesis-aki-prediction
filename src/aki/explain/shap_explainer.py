"""SHAP-based explanations for the LightGBM baseline.

Uses :class:`shap.TreeExplainer`, O(N · T · D) exact Shapley values,
fast enough on HPC nodes even for the full test split.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from loguru import logger

from aki.models.base import ModelArtifact


def lightgbm_global_shap(
    art: ModelArtifact,
    X: pd.DataFrame,
    out_dir: Path,
    max_samples: int = 5000,
    random_state: int = 42,
) -> Path:
    """Write a mean-|SHAP| importance table + summary PNG."""
    if art.name != "lightgbm":
        raise ValueError("lightgbm_global_shap requires a LightGBM artifact")
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = X[art.feature_names]
    if len(sample) > max_samples:
        sample = sample.sample(max_samples, random_state=random_state)

    booster = art.estimator.booster_
    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(sample)
    if isinstance(values, list):  # older SHAP: [class0, class1]
        values = values[1]

    mean_abs = np.abs(values).mean(axis=0)
    table = (
        pd.DataFrame({"feature": art.feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    )
    csv_path = out_dir / "shap_global.csv"
    table.to_csv(csv_path, index=False)

    fig = plt.figure(figsize=(7, 0.35 * min(len(art.feature_names), 25)))
    shap.summary_plot(values, sample, show=False, max_display=25)
    png_path = out_dir / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"SHAP summary -> {png_path}")
    return csv_path


def lightgbm_local_shap(
    art: ModelArtifact,
    x_row: pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    """Per-feature SHAP contributions for a single landmark."""
    if art.name != "lightgbm":
        raise ValueError("lightgbm_local_shap requires a LightGBM artifact")
    if isinstance(x_row, pd.Series):
        x_row = x_row.to_frame().T
    x_row = x_row[art.feature_names]

    booster = art.estimator.booster_
    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(x_row)
    if isinstance(values, list):
        values = values[1]

    return (
        pd.DataFrame({
            "feature":              art.feature_names,
            "value":                x_row.iloc[0].values,
            "contribution_logodds": values.ravel(),
        })
        .sort_values("contribution_logodds", key=np.abs, ascending=False, ignore_index=True)
    )
