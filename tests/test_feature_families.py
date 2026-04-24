from __future__ import annotations

import pandas as pd

from aki.features.engineer import _select_family
from aki.utils.config import load_configs


def test_scorecard_primary_family_is_configured():
    cfg = load_configs()
    family = cfg.features["feature_families"]["scorecard_primary"]
    assert family["include_encoded_demographics"] is False
    assert family["selected_features"] == [
        "age",
        "urine_output_ml_6h",
        "creatinine_max_24h",
        "map_mean_12h",
        "fluid_input_ml_12h",
        "glucose_std_24h",
        "hematocrit_max_24h",
    ]
    assert family["model_overrides"]["scorecard"]["representation"] == "binned"
    assert family["model_overrides"]["scorecard"]["selection_mode"] == "fixed"


def test_scorecard_core_and_augmented_families_are_configured():
    cfg = load_configs()
    core = cfg.features["feature_families"]["scorecard_core"]
    primary = cfg.features["feature_families"]["scorecard_primary"]
    augmented = cfg.features["feature_families"]["scorecard_augmented"]

    assert core["include_encoded_demographics"] is False
    assert core["selected_features"] == [
        "age",
        "urine_output_ml_6h",
        "creatinine_max_24h",
        "map_mean_12h",
        "hematocrit_max_24h",
    ]

    assert augmented["include_encoded_demographics"] is False
    assert augmented["selected_features"] == [
        "age",
        "urine_output_ml_6h",
        "creatinine_max_24h",
        "map_mean_12h",
        "hematocrit_max_24h",
        "loop_diuretic_24h",
        "fluid_input_ml_12h",
        "glucose_std_24h",
    ]
    assert primary["selected_features"] != augmented["selected_features"]
    assert core["model_overrides"]["scorecard"]["representation"] == "binned"
    assert augmented["model_overrides"]["scorecard"]["representation"] == "binned"


def test_select_family_supports_curated_selected_features_without_encoded_demographics():
    cfg = load_configs()
    family_cfg = cfg.features["feature_families"]["scorecard_primary"]
    full = pd.DataFrame(
        {
            "stay_id": [1],
            "subject_id": [11],
            "landmark_time": ["2026-01-01"],
            "anchor_year_group": ["2017 - 2019"],
            "age": [70],
            "age_group": ["65-85"],
            "sex": ["M"],
            "ethnicity": ["WHITE"],
            "hours_since_icu_admit": [24.0],
            "sex_male": [1],
            "eth_White": [1],
            "y_stage1_24h": [0],
            "y_stage1_48h": [0],
            "y_stage2_24h": [0],
            "y_stage2_48h": [0],
            "y_cr_only_stage1_24h": [0],
            "y_cr_only_stage1_48h": [0],
            "urine_output_ml_6h": [800.0],
            "creatinine_max_24h": [1.8],
            "map_mean_12h": [72.0],
            "fluid_input_ml_12h": [1200.0],
            "glucose_std_24h": [18.0],
            "hematocrit_max_24h": [34.0],
            "extraneous_feature": [999.0],
        }
    )

    subset = _select_family(full, family_cfg, cfg)
    assert "extraneous_feature" not in subset.columns
    assert "sex_male" not in subset.columns
    assert "eth_White" not in subset.columns
    for feature in family_cfg["selected_features"]:
        assert feature in subset.columns
