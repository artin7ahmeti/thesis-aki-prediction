"""Decision curve analysis (Vickers & Elkin, 2006).

Net benefit at threshold ``p_t``::

    NB = (TP/n) - (FP/n) * (p_t / (1 - p_t))

Compared against two baselines:
- *treat-all*:  NB = prevalence - (1 - prevalence) * (p_t / (1 - p_t))
- *treat-none*: NB = 0
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def decision_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
    threshold_min: float = 0.01,
    threshold_max: float = 0.50,
    threshold_step: float = 0.01,
) -> pd.DataFrame:
    """Compute net benefit for the model and both reference strategies."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    prevalence = y_true.mean()

    if thresholds is None:
        thresholds = np.arange(threshold_min, threshold_max + 1e-9, threshold_step)

    rows = []
    for pt in thresholds:
        odds = pt / (1.0 - pt) if pt < 1.0 else np.inf
        pred = (y_prob >= pt).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        nb_model = (tp / n) - (fp / n) * odds
        nb_all   = prevalence - (1 - prevalence) * odds
        rows.append({
            "threshold":  float(pt),
            "nb_model":   float(nb_model),
            "nb_all":     float(nb_all),
            "nb_none":    0.0,
            "tp":         tp,
            "fp":         fp,
            "n":          n,
        })
    return pd.DataFrame(rows)


def net_benefit_at(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> float:
    """Scalar net-benefit at a single clinical threshold."""
    return float(
        decision_curve(
            y_true, y_prob, thresholds=np.array([threshold])
        )["nb_model"].iloc[0]
    )
