"""Model explanation: global importance, shape plots, scorecards."""

from aki.explain.global_importance import global_importance_table
from aki.explain.patient import patient_additive_contributions
from aki.explain.plots import plot_ebm_shapes, plot_reliability
from aki.explain.report import run_explanations
from aki.explain.scorecard_card import build_scorecard_artifact
from aki.explain.shap_explainer import lightgbm_global_shap, lightgbm_local_shap

__all__ = [
    "global_importance_table",
    "plot_ebm_shapes",
    "plot_reliability",
    "build_scorecard_artifact",
    "patient_additive_contributions",
    "lightgbm_global_shap",
    "lightgbm_local_shap",
    "run_explanations",
]
