"""Tests for the feature-engineering helpers.

These tests hit the pandas-side helpers only — the SQL layer is exercised
in integration tests (not part of the default suite, need real MIMIC).
"""

import numpy as np
import pandas as pd

from aki.features.engineer import _add_missingness_indicators


def test_missingness_indicator_from_count_column():
    df = pd.DataFrame({
        "stay_id": [1, 2, 3],
        "creatinine_count_6h": [3, np.nan, 0],
        "creatinine_mean_6h":  [1.1, np.nan, np.nan],
    })
    out = _add_missingness_indicators(df)
    assert "creatinine_missing_6h" in out.columns
    assert out["creatinine_missing_6h"].tolist() == [0, 1, 1]


def test_missingness_indicator_skips_non_count_columns():
    df = pd.DataFrame({
        "stay_id": [1, 2],
        "creatinine_latest_6h": [1.0, 2.0],
        "creatinine_count_12h": [1, 0],
    })
    out = _add_missingness_indicators(df)
    missing_cols = [c for c in out.columns if c.endswith("_missing_12h")]
    assert missing_cols == ["creatinine_missing_12h"]
    # latest should not have generated a missing column
    assert not any(c.endswith("_missing_6h") for c in out.columns)


def test_feature_columns_strip_meta_and_labels():
    from aki.models.base import feature_columns

    df = pd.DataFrame(columns=[
        "stay_id", "subject_id", "landmark_time", "anchor_year_group",
        "age", "age_group", "sex", "ethnicity", "hours_since_icu_admit",
        "split",
        "y_stage1_24h", "y_stage1_48h",
        "creatinine_latest_6h", "sbp_mean_12h",
    ])
    feats = feature_columns(df)
    assert "creatinine_latest_6h" in feats
    assert "sbp_mean_12h" in feats
    assert "age" in feats  # age is a feature, not meta
    assert "split" not in feats
    assert "y_stage1_24h" not in feats
    assert "subject_id" not in feats
