"""Config loading smoke tests."""

from aki.utils.config import load_configs


def test_configs_load():
    cfg = load_configs()
    assert cfg.project
    assert cfg.data
    assert cfg.cohort
    assert cfg.features
    assert cfg.models
    assert cfg.eval


def test_feature_itemids_are_integers():
    cfg = load_configs()
    for vital in cfg.features["vitals"]:
        for iid in vital.get("itemids", []):
            assert isinstance(iid, int), f"{vital['name']}: {iid} is not int"
    for lab in cfg.features["labs"]:
        for iid in lab.get("itemids", []):
            assert isinstance(iid, int), f"{lab['name']}: {iid} is not int"


def test_eval_tasks_have_horizons():
    cfg = load_configs()
    for task in cfg.eval["tasks"]:
        assert task["horizon_hours"] in (24, 48)
        assert task["outcome"] in ("kdigo_stage1", "kdigo_stage2")


def test_splits_are_temporal_and_disjoint():
    cfg = load_configs()
    s = cfg.eval["splits"]
    assert s["strategy"] == "anchor_year_group"
    combined = s["train_groups"] + s["val_groups"] + s["test_groups"]
    assert len(combined) == len(set(combined)), "anchor_year_group overlap across splits"


def test_config_hash_is_stable():
    cfg = load_configs()
    h1 = cfg.config_hash()
    h2 = cfg.config_hash()
    assert h1 == h2 and len(h1) == 16
