from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("optuna")

from aki.models import tune
from aki.utils.config import load_configs


class RecorderTrial:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, tuple, dict]] = []

    def suggest_int(self, name, low, high, **kwargs):
        self.calls.append(("int", name, (low, high), kwargs))
        return low

    def suggest_float(self, name, low, high, **kwargs):
        self.calls.append(("float", name, (low, high), kwargs))
        return low


def test_suggest_lightgbm_uses_tightened_runtime_safe_ranges():
    cfg = load_configs()
    trial = RecorderTrial()

    params = tune._suggest_lightgbm(trial, cfg)

    assert params["num_leaves"] == 15
    assert params["max_depth"] == 4
    assert params["learning_rate"] == 1e-2
    assert params["min_child_samples"] == 20

    expected = {
        ("int", "num_leaves"): ((15, 63), {"log": True}),
        ("int", "max_depth"): ((4, 8), {}),
        ("float", "learning_rate"): ((1e-2, 8e-2), {"log": True}),
        ("int", "min_child_samples"): ((20, 120), {"log": True}),
        ("float", "subsample"): ((0.7, 1.0), {}),
        ("float", "colsample_bytree"): ((0.7, 1.0), {}),
        ("float", "reg_alpha"): ((1e-6, 5.0), {"log": True}),
        ("float", "reg_lambda"): ((1e-6, 5.0), {"log": True}),
    }

    actual = {(kind, name): (bounds, kwargs) for kind, name, bounds, kwargs in trial.calls}
    assert actual == expected


def test_make_pruner_is_more_aggressive_for_lightgbm():
    lightgbm_pruner = tune._make_pruner("lightgbm")
    ebm_pruner = tune._make_pruner("ebm")

    assert type(lightgbm_pruner).__name__ == "MedianPruner"
    assert type(ebm_pruner).__name__ == "MedianPruner"
    assert getattr(lightgbm_pruner, "_n_startup_trials") == 2
    assert getattr(lightgbm_pruner, "_n_warmup_steps") == 1
    assert getattr(ebm_pruner, "_n_startup_trials") == 5
    assert getattr(ebm_pruner, "_n_warmup_steps") == 2


def test_checkpoint_callback_saves_current_incumbent(monkeypatch):
    saved: dict[str, object] = {}

    def fake_save_best_params(params, tag):
        saved["params"] = params
        saved["tag"] = tag

    monkeypatch.setattr(tune, "save_best_params", fake_save_best_params)

    study = SimpleNamespace(
        best_params={"learning_rate": 0.03, "num_leaves": 31},
        best_value=0.781,
        best_trial=SimpleNamespace(user_attrs={"mean_auprc": 0.331}),
        trials=[object(), object(), object()],
    )

    callback = tune._checkpoint_callback("aki_stage1_48h__combined__lightgbm")
    callback(study, None)

    assert saved["tag"] == "aki_stage1_48h__combined__lightgbm"
    assert saved["params"] == {
        "learning_rate": 0.03,
        "num_leaves": 31,
        "_cv_mean_auroc": 0.781,
        "_cv_mean_auprc": 0.331,
        "_n_trials": 3,
    }
