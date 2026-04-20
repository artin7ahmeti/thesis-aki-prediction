"""Smoke-level tests that each model class fits and predicts on toy data.

These don't test performance — just that the interface works end-to-end.
"""

import numpy as np
import pandas as pd
import pytest

from aki.models.ebm import EBMModel
from aki.models.lightgbm_model import LightGBMModel
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


@pytest.mark.parametrize("cls", [LightGBMModel])
def test_tree_model_fits_and_predicts(cls):
    X, y = _toy_binary()
    m = cls({"random_state": 0}).fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (len(y),)
    assert 0.0 <= p.min() and p.max() <= 1.0


def test_scorecard_selects_sparse_set():
    X, y = _toy_binary(n=600, d=20)
    m = ScorecardModel({
        "C_grid": [0.05, 0.1],
        "target_features": 5,
        "random_state": 0,
    }).fit(X, y, groups=np.arange(len(X)) // 3)  # synthetic patient groups
    assert len(m.feature_names_) <= 5
    assert m.estimator_ is not None
    p = m.predict_proba(X)
    assert p.shape == (len(y),)


def test_ebm_produces_global_importance():
    pytest.importorskip("interpret")
    X, y = _toy_binary(n=300, d=5)
    m = EBMModel({"outer_bags": 2, "interactions": 0, "random_state": 0}).fit(X, y)
    imp = m.global_importance()
    assert len(imp) > 0
    assert set(imp.columns) == {"term", "importance"}


def test_calibration_changes_predictions():
    X, y = _toy_binary(n=600)
    m = LightGBMModel({"random_state": 0, "n_estimators": 100}).fit(X, y)
    p_uncal = m.predict_proba(X)
    m.calibrate(X, y, method="isotonic")
    p_cal = m.predict_proba(X)
    assert not np.allclose(p_uncal, p_cal)
