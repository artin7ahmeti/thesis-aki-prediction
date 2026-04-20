"""Patient-level bootstrap confidence intervals.

Resamples ``subject_id`` (not rows) so CIs reflect patient-level
uncertainty, closer to how the model will be deployed.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger

from aki.eval.metrics import full_report


def patient_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    subject_ids: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], dict[str, float]] | None = None,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return per-metric point estimate + (lower, upper) CI bounds."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    subject_ids = np.asarray(subject_ids)
    metric_fn = metric_fn or full_report

    unique_sids, inverse = np.unique(subject_ids, return_inverse=True)
    sid_to_rows: list[np.ndarray] = [
        np.where(inverse == i)[0] for i in range(len(unique_sids))
    ]

    rng = np.random.default_rng(random_state)
    point = metric_fn(y_true, y_prob)
    names = list(point.keys())

    draws: dict[str, list[float]] = {k: [] for k in names}
    for _ in range(n_iterations):
        sampled = rng.integers(0, len(unique_sids), size=len(unique_sids))
        idx = np.concatenate([sid_to_rows[i] for i in sampled])
        try:
            res = metric_fn(y_true[idx], y_prob[idx])
        except Exception as e:
            logger.debug(f"bootstrap iteration failed: {e}")
            continue
        for k, v in res.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                draws[k].append(float(v))

    alpha = (1.0 - confidence_level) / 2.0
    rows: list[dict] = []
    for k in names:
        vals = np.asarray(draws[k], dtype=float)
        est = point[k]
        if not isinstance(est, (int, float)) or len(vals) == 0:
            rows.append({"metric": k, "point": est, "ci_lower": np.nan, "ci_upper": np.nan})
            continue
        lo = float(np.quantile(vals, alpha))
        hi = float(np.quantile(vals, 1.0 - alpha))
        rows.append({"metric": k, "point": float(est), "ci_lower": lo, "ci_upper": hi})
    return pd.DataFrame(rows)
