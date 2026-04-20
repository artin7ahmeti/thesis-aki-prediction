"""Optuna hyperparameter tuning with patient-grouped CV.

All models are tuned on the *train* split using :class:`GroupKFold` with
``subject_id`` as the group — no patient ever crosses CV folds, which
matches the patient-level temporal split used for the held-out test set.

The objective is mean ROC-AUC across folds (tie-broken by AUPRC). The
best-params dict is persisted to ``reports/artifacts/tune/<tag>.json``
so the final training pass can load it directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from aki.models.ebm import EBMModel
from aki.models.lightgbm_model import LightGBMModel
from aki.models.scorecard import ScorecardModel
from aki.utils.config import Config
from aki.utils.paths import paths

# Silence optuna's per-trial INFO chatter — loguru handles progress.
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    cfg: Config,
    n_trials: int = 60,
    n_folds: int = 5,
    seed: int = 42,
    timeout_s: int | None = None,
) -> dict[str, Any]:
    """Run Optuna HPO and return the best-params dict."""
    if model_name not in {"ebm", "scorecard", "lightgbm"}:
        raise ValueError(f"Unknown model {model_name!r}")

    suggester = _SUGGESTERS[model_name]
    fit_eval = _FIT_EVAL[model_name]

    cv = GroupKFold(n_splits=n_folds)
    folds = list(cv.split(X, y, groups=groups))

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    # MedianPruner: after a warmup of 5 full trials, compare a trial's
    # running mean AUROC against the median at the same fold index and
    # kill it early if it's already worse. Saves ~30-50% of HPO wall
    # time without touching the final-trial quality.
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=2,      # don't prune until 2 folds have completed
        interval_steps=1,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def objective(trial: optuna.Trial) -> float:
        params = suggester(trial, cfg)
        auroc_folds: list[float] = []
        auprc_folds: list[float] = []
        for step, (tr_idx, va_idx) in enumerate(folds):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            if y_va.nunique() < 2:
                continue
            try:
                p_va = fit_eval(params, X_tr, y_tr, X_va, y_va, groups[tr_idx], seed)
            except Exception as e:
                logger.debug(f"trial {trial.number} fold failed: {e}")
                raise optuna.TrialPruned() from e
            auroc_folds.append(roc_auc_score(y_va, p_va))
            auprc_folds.append(average_precision_score(y_va, p_va))
            # Report running mean so the pruner can decide after each fold.
            trial.report(float(np.mean(auroc_folds)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if not auroc_folds:
            raise optuna.TrialPruned("no valid folds")
        trial.set_user_attr("mean_auprc", float(np.mean(auprc_folds)))
        return float(np.mean(auroc_folds))

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_s,
        show_progress_bar=False,
    )
    best = dict(study.best_params)
    best["_cv_mean_auroc"] = float(study.best_value)
    best["_cv_mean_auprc"] = float(study.best_trial.user_attrs.get("mean_auprc", np.nan))
    best["_n_trials"] = len(study.trials)
    logger.info(
        f"HPO {model_name}: trials={len(study.trials)} "
        f"AUROC={study.best_value:.4f} AUPRC={best['_cv_mean_auprc']:.4f}"
    )
    return best


def save_best_params(params: dict[str, Any], tag: str) -> Path:
    out_dir = paths.artifacts / "tune"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tag}.json"
    out.write_text(json.dumps(params, indent=2), encoding="utf-8")
    return out


def load_best_params(tag: str) -> dict[str, Any] | None:
    path = paths.artifacts / "tune" / f"{tag}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ #
# Per-model suggestion spaces
# ------------------------------------------------------------------ #
def _suggest_ebm(trial: optuna.Trial, cfg: Config) -> dict[str, Any]:
    base = dict(cfg.models.get("ebm", {}))
    base.update({
        "learning_rate":   trial.suggest_float("learning_rate", 1e-3, 5e-2, log=True),
        "max_bins":        trial.suggest_categorical("max_bins", [128, 256, 512]),
        "interactions":    trial.suggest_int("interactions", 0, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
        "outer_bags":      trial.suggest_int("outer_bags", 4, 8),
    })
    return base


def _suggest_scorecard(trial: optuna.Trial, cfg: Config) -> dict[str, Any]:
    base = dict(cfg.models.get("sparse_logistic", {}))
    base.update({
        "C_grid":          [trial.suggest_float("C", 1e-3, 1e1, log=True)],
        "target_features": trial.suggest_int("target_features", 6, 14),
    })
    return base


def _suggest_lightgbm(trial: optuna.Trial, cfg: Config) -> dict[str, Any]:
    base = dict(cfg.models.get("lightgbm", {}))
    base.update({
        "num_leaves":        trial.suggest_int("num_leaves", 15, 255, log=True),
        "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    })
    return base


# ------------------------------------------------------------------ #
# Per-model fit/predict (returns validation P(y=1))
# ------------------------------------------------------------------ #
def _fit_eval_ebm(
    params, X_tr, y_tr, X_va, _y_va, _groups_tr, seed: int,
) -> np.ndarray:
    params = {**params, "random_state": seed}
    m = EBMModel(params).fit(X_tr, y_tr)
    return m.predict_proba(X_va)


def _fit_eval_scorecard(
    params, X_tr, y_tr, X_va, _y_va, groups_tr, seed: int,
) -> np.ndarray:
    params = {**params, "random_state": seed}
    m = ScorecardModel(params).fit(X_tr, y_tr, groups=groups_tr)
    return m.predict_proba(X_va)


def _fit_eval_lightgbm(
    params, X_tr, y_tr, X_va, y_va, _groups_tr, seed: int,
) -> np.ndarray:
    params = {**params, "random_state": seed}
    m = LightGBMModel(params).fit(X_tr, y_tr, X_val=X_va, y_val=y_va)
    return m.predict_proba(X_va)


_SUGGESTERS: dict[str, Callable[[optuna.Trial, Config], dict[str, Any]]] = {
    "ebm":       _suggest_ebm,
    "scorecard": _suggest_scorecard,
    "lightgbm":  _suggest_lightgbm,
}

_FIT_EVAL: dict[str, Callable[..., np.ndarray]] = {
    "ebm":       _fit_eval_ebm,
    "scorecard": _fit_eval_scorecard,
    "lightgbm":  _fit_eval_lightgbm,
}
