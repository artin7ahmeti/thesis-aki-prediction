"""Evaluation: metrics, calibration, decision curves, fairness, bootstrap."""

from aki.eval.aggregate import build_final_results
from aki.eval.bootstrap import patient_bootstrap_ci
from aki.eval.calibration import calibration_metrics, reliability_curve
from aki.eval.decision_curve import decision_curve
from aki.eval.drift import compute_drift_report
from aki.eval.evaluate import evaluate_all
from aki.eval.fairness import subgroup_metrics
from aki.eval.metrics import discrimination_metrics, full_report

__all__ = [
    "discrimination_metrics",
    "full_report",
    "calibration_metrics",
    "reliability_curve",
    "decision_curve",
    "subgroup_metrics",
    "patient_bootstrap_ci",
    "evaluate_all",
    "compute_drift_report",
    "build_final_results",
]
