"""Calibration metrics: Brier, slope, intercept, ECE, reliability curve."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

_EPS = 1e-9


def calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Calibration slope + intercept (logit) and expected calibration error."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), _EPS, 1.0 - _EPS)
    logit = np.log(y_prob / (1.0 - y_prob)).reshape(-1, 1)

    # Slope + intercept via logistic regression on logit predictions
    try:
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(logit, y_true)
        slope     = float(lr.coef_.ravel()[0])
        intercept = float(lr.intercept_.ravel()[0])
    except Exception:
        slope, intercept = float("nan"), float("nan")

    return {
        "calibration_slope":     slope,
        "calibration_intercept": intercept,
        "ece":                   expected_calibration_error(y_true, y_prob, n_bins),
    }


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Standard equal-width binned ECE."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    n = len(y_prob)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(y_prob[m].mean() - y_true[m].mean())
    return float(ece)


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return bin -> (predicted mean, observed rate, n) for reliability plots."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({
            "bin":       b,
            "bin_left":  bins[b],
            "bin_right": bins[b + 1],
            "n":         int(m.sum()),
            "pred_mean": float(y_prob[m].mean()) if m.any() else float("nan"),
            "obs_rate":  float(y_true[m].mean()) if m.any() else float("nan"),
        })
    return pd.DataFrame(rows)
