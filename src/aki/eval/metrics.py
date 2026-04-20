"""Discrimination metrics + unified report bundle."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

from aki.eval.calibration import calibration_metrics


def discrimination_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """AUROC, AUPRC, and operating-point sensitivities/specificities."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    out = {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "prevalence": float(y_true.mean()),
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
    }
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    spec = 1.0 - fpr
    out["sensitivity_at_spec_90"] = float(
        tpr[spec >= 0.90].max() if np.any(spec >= 0.90) else np.nan
    )
    out["specificity_at_sens_90"] = float(
        spec[tpr >= 0.90].max() if np.any(tpr >= 0.90) else np.nan
    )
    return out


def full_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Discrimination + Brier + calibration slope/intercept/ECE."""
    out = discrimination_metrics(y_true, y_prob)
    out["brier"] = float(brier_score_loss(y_true, y_prob))
    out.update(calibration_metrics(y_true, y_prob, n_bins=n_bins))
    return out


def report_to_frame(report: dict[str, float], **extra: str) -> pd.DataFrame:
    """Convert a metrics dict into a one-row DataFrame for aggregation."""
    row = {**extra, **report}
    return pd.DataFrame([row])
