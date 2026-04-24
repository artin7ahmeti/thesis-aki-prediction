from __future__ import annotations

import numpy as np
import pandas as pd

from aki.explain.scorecard_card import build_scorecard_artifact
from aki.models.scorecard import ScorecardModel


def _toy_binary(n: int = 400, d: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    logit = X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2]
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(int)
    return (
        pd.DataFrame(X, columns=[f"f{i}" for i in range(d)]),
        pd.Series(y, name="y"),
    )


def test_scorecard_artifact_contains_summary_and_anchor_points(tmp_path):
    X, y = _toy_binary(n=500, d=12, seed=3)
    model = ScorecardModel(
        {
            "C_grid": [0.03, 0.1],
            "target_features": 5,
            "random_state": 0,
        }
    ).fit(X, y, groups=np.arange(len(X)) // 5)

    art = model.artifact(task="aki_stage1_24h", family="minimal")
    outputs = build_scorecard_artifact(art, tmp_path)

    summary = pd.read_csv(outputs["csv"])
    points = pd.read_csv(outputs["points_csv"])
    md = outputs["md"].read_text(encoding="utf-8")

    assert {"feature", "odds_ratio_per_sd", "odds_ratio_per_unit", "points_per_sd"} <= set(summary.columns)
    assert {"feature", "anchor", "representative_value", "points_vs_median"} <= set(points.columns)
    assert "## Feature Summary" in md
    assert "## Anchor Points" in md


def test_scorecard_honors_scalar_tuned_c_over_default_grid():
    X, y = _toy_binary(n=500, d=12, seed=7)
    model = ScorecardModel(
        {
            "C": 0.03,
            "C_grid": [0.001, 0.003, 0.01],
            "target_features": 6,
            "random_state": 0,
        }
    ).fit(X, y, groups=np.arange(len(X)) // 5)

    assert abs(model.extra_["selected_C"] - 0.03) < 1e-12
    assert model.extra_["C_grid"] == [0.03]


def test_binned_scorecard_artifact_contains_bedside_points(tmp_path):
    X, y = _toy_binary(n=500, d=5, seed=11)
    X["binary"] = (X["f0"] > 0).astype(int)
    model = ScorecardModel(
        {
            "representation": "binned",
            "selection_mode": "fixed",
            "C": 0.3,
            "bin_edges": {
                "f0": [-0.5, 0.5],
                "f1": [-0.5, 0.5],
                "f2": [-0.5, 0.5],
                "f3": [-0.5, 0.5],
                "f4": [-0.5, 0.5],
            },
            "random_state": 0,
        }
    ).fit(X[["f0", "f1", "f2", "f3", "f4", "binary"]], y, groups=np.arange(len(X)) // 5)

    art = model.artifact(task="aki_stage1_24h", family="scorecard_primary")
    outputs = build_scorecard_artifact(art, tmp_path)

    summary = pd.read_csv(outputs["csv"])
    points = pd.read_csv(outputs["points_csv"])
    md = outputs["md"].read_text(encoding="utf-8")

    assert {"feature", "reference_level", "max_abs_points"} <= set(summary.columns)
    assert {"feature", "range_display", "points_vs_reference"} <= set(points.columns)
    assert "## Bedside Points" in md
