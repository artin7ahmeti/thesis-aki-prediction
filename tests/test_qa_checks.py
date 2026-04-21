"""QA invariant checks."""

import pandas as pd
import pytest

from aki.qa.checks import assert_qa_invariants


def _good_tables() -> dict[str, pd.DataFrame]:
    return {
        "qa.cohort_summary": pd.DataFrame(
            [
                {
                    "n_stays": 100,
                    "n_patients": 90,
                    "min_age": 18,
                }
            ]
        ),
        "qa.landmark_summary": pd.DataFrame(
            [
                {
                    "n_landmarks": 300,
                    "n_stays_with_landmarks": 80,
                }
            ]
        ),
        "qa.label_prevalence": pd.DataFrame(
            [
                {
                    "n_landmarks": 300,
                    "prev_stage1_24h": 0.10,
                    "prev_stage1_48h": 0.15,
                    "prev_stage2_24h": 0.04,
                    "prev_stage2_48h": 0.07,
                    "prev_cr_only_24h": 0.05,
                    "prev_cr_only_48h": 0.09,
                }
            ]
        ),
        "qa.leakage_check": pd.DataFrame(
            [
                {
                    "n_rolling_rows": 1000,
                    "n_future_feature_rows": 0,
                    "n_before_window_feature_rows": 0,
                }
            ]
        ),
        "qa.cohort_baseline_coverage": pd.DataFrame(
            [
                {
                    "n_stays": 100,
                    "n_with_baseline": 70,
                }
            ]
        ),
    }


def test_qa_invariants_accept_valid_summary():
    assert_qa_invariants(_good_tables())


def test_qa_invariants_reject_future_feature_rows():
    tables = _good_tables()
    tables["qa.leakage_check"].loc[0, "n_future_feature_rows"] = 1

    with pytest.raises(AssertionError, match="after landmark_time"):
        assert_qa_invariants(tables)


def test_qa_invariants_reject_nonmonotone_label_prevalence():
    tables = _good_tables()
    tables["qa.label_prevalence"].loc[0, "prev_stage1_48h"] = 0.05

    with pytest.raises(AssertionError, match="stage1 48h"):
        assert_qa_invariants(tables)
