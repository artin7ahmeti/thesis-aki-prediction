"""Model explanation utilities.

Keep package imports lightweight so `aki explain --model ebm` does not pull in
optional SHAP/IPython dependencies unless a LightGBM explanation is actually
requested.
"""

from aki.explain.global_importance import global_importance_table
from aki.explain.patient import (
    patient_additive_contributions,
    select_representative_patient_case,
)
from aki.explain.plots import (
    plot_ebm_shapes,
    plot_patient_contributions,
    plot_reliability,
)
from aki.explain.scorecard_card import build_scorecard_artifact


def lightgbm_global_shap(*args, **kwargs):
    from aki.explain.shap_explainer import lightgbm_global_shap as _impl

    return _impl(*args, **kwargs)


def lightgbm_local_shap(*args, **kwargs):
    from aki.explain.shap_explainer import lightgbm_local_shap as _impl

    return _impl(*args, **kwargs)


def run_explanations(*args, **kwargs):
    from aki.explain.report import run_explanations as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "build_scorecard_artifact",
    "global_importance_table",
    "lightgbm_global_shap",
    "lightgbm_local_shap",
    "patient_additive_contributions",
    "plot_ebm_shapes",
    "plot_patient_contributions",
    "plot_reliability",
    "run_explanations",
    "select_representative_patient_case",
]
