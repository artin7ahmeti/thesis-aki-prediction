"""Temporal + patient-level split invariants."""

import pandas as pd
import pytest

from aki.split.splits import assign_splits
from aki.utils.config import load_configs


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame([
        # Two landmarks per patient -> ensure no patient straddles splits
        {"subject_id": 1, "stay_id": 10, "anchor_year_group": "2008 - 2010"},
        {"subject_id": 1, "stay_id": 10, "anchor_year_group": "2008 - 2010"},
        {"subject_id": 2, "stay_id": 20, "anchor_year_group": "2011 - 2013"},
        {"subject_id": 3, "stay_id": 30, "anchor_year_group": "2014 - 2016"},
        {"subject_id": 4, "stay_id": 40, "anchor_year_group": "2017 - 2019"},
        {"subject_id": 5, "stay_id": 50, "anchor_year_group": "2020 - 2022"},
    ])


def test_splits_assign_correct_bucket():
    cfg = load_configs()
    df = assign_splits(_toy_df(), cfg)
    bucket = dict(zip(df["subject_id"], df["split"]))
    assert bucket[1] == "train"
    assert bucket[2] == "train"
    assert bucket[3] == "val"
    assert bucket[4] == "test"
    assert bucket[5] == "test"


def test_no_patient_crosses_splits():
    cfg = load_configs()
    df = assign_splits(_toy_df(), cfg)
    per_subject = df.groupby("subject_id")["split"].nunique()
    assert (per_subject == 1).all()


def test_leaked_patient_raises():
    """If a subject somehow has two anchor_year_groups spanning splits, fail."""
    cfg = load_configs()
    bad = pd.DataFrame([
        {"subject_id": 1, "stay_id": 10, "anchor_year_group": "2008 - 2010"},
        {"subject_id": 1, "stay_id": 11, "anchor_year_group": "2017 - 2019"},
    ])
    with pytest.raises(AssertionError, match="leakage"):
        assign_splits(bad, cfg)


def test_unmapped_group_is_dropped():
    cfg = load_configs()
    df = pd.DataFrame([
        {"subject_id": 1, "stay_id": 10, "anchor_year_group": "2008 - 2010"},
        {"subject_id": 2, "stay_id": 20, "anchor_year_group": "1990 - 1992"},  # unmapped
    ])
    out = assign_splits(df, cfg)
    assert 2 not in out["subject_id"].values
