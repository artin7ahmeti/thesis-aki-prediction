"""Evaluation: metrics, calibration, decision curves, fairness, bootstrap."""

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


def __getattr__(name: str):
    if name in {"discrimination_metrics", "full_report"}:
        from aki.eval.metrics import discrimination_metrics, full_report

        return {"discrimination_metrics": discrimination_metrics, "full_report": full_report}[name]
    if name in {"calibration_metrics", "reliability_curve"}:
        from aki.eval.calibration import calibration_metrics, reliability_curve

        return {"calibration_metrics": calibration_metrics, "reliability_curve": reliability_curve}[name]
    if name == "decision_curve":
        from aki.eval.decision_curve import decision_curve

        return decision_curve
    if name == "subgroup_metrics":
        from aki.eval.fairness import subgroup_metrics

        return subgroup_metrics
    if name == "patient_bootstrap_ci":
        from aki.eval.bootstrap import patient_bootstrap_ci

        return patient_bootstrap_ci
    if name == "evaluate_all":
        from aki.eval.evaluate import evaluate_all

        return evaluate_all
    if name == "compute_drift_report":
        from aki.eval.drift import compute_drift_report

        return compute_drift_report
    if name == "build_final_results":
        from aki.eval.aggregate import build_final_results

        return build_final_results
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
