"""Evaluation-metric sanity tests on synthetic data."""

import numpy as np
import pandas as pd

from aki.eval.bootstrap import patient_bootstrap_ci
from aki.eval.calibration import calibration_metrics, reliability_curve
from aki.eval.decision_curve import decision_curve
from aki.eval.metrics import discrimination_metrics, full_report


def _synthetic(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    # well-separated probabilities
    p = np.where(y == 1, rng.beta(5, 2, size=n), rng.beta(2, 5, size=n))
    return y, p


def test_discrimination_on_perfect_ranker():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = discrimination_metrics(y, p)
    assert m["auroc"] == 1.0
    assert m["n"] == 4 and m["n_pos"] == 2


def test_full_report_contains_all_keys():
    y, p = _synthetic()
    rep = full_report(y, p)
    for k in ("auroc", "auprc", "brier",
              "calibration_slope", "calibration_intercept", "ece"):
        assert k in rep


def test_calibration_slope_of_perfect_probs():
    rng = np.random.default_rng(42)
    p = rng.uniform(0.01, 0.99, size=5000)
    y = (rng.uniform(size=5000) < p).astype(int)
    m = calibration_metrics(y, p)
    assert 0.8 < m["calibration_slope"] < 1.2
    assert m["ece"] < 0.05


def test_reliability_curve_bins():
    y, p = _synthetic(200)
    rc = reliability_curve(y, p, n_bins=10)
    assert len(rc) == 10
    assert rc["n"].sum() == len(y)


def test_decision_curve_treat_none_is_zero():
    y, p = _synthetic()
    dc = decision_curve(y, p, threshold_min=0.05, threshold_max=0.4, threshold_step=0.05)
    assert (dc["nb_none"] == 0.0).all()
    # treat-all NB should equal prevalence at threshold 0
    prev = y.mean()
    assert np.isclose(dc["nb_all"].iloc[0], prev - (1 - prev) * (0.05 / 0.95), atol=1e-6)


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=300)
    p = rng.uniform(size=300)
    subj = rng.integers(0, 50, size=300)  # 50 patients
    ci = patient_bootstrap_ci(y, p, subj, n_iterations=50, random_state=0)
    auroc_row = ci[ci["metric"] == "auroc"].iloc[0]
    assert auroc_row["ci_lower"] <= auroc_row["point"] <= auroc_row["ci_upper"]


def test_bootstrap_patient_resampling_uses_subject_unit():
    """Sanity: bootstrap with 1 patient -> all iterations resample same rows."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=100)
    p = rng.uniform(size=100)
    subj = np.zeros(100, dtype=int)
    ci = patient_bootstrap_ci(y, p, subj, n_iterations=10, random_state=0)
    # Only one unique draw -> CI is degenerate at point
    auroc = ci[ci["metric"] == "auroc"].iloc[0]
    assert auroc["ci_lower"] == auroc["ci_upper"] == auroc["point"]


def test_report_to_frame_roundtrip():
    from aki.eval.metrics import report_to_frame
    y, p = _synthetic()
    row = report_to_frame(full_report(y, p), task="aki_stage1_24h", model="ebm")
    assert isinstance(row, pd.DataFrame)
    assert "auroc" in row.columns and "task" in row.columns
