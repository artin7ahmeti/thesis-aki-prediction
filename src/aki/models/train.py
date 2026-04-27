"""Training orchestration.

Walks every (task, family, model) combination, fits on train, calibrates
on val, logs metrics + artifact to MLflow, and saves a joblib artifact
under ``reports/artifacts/models/``.

When a tuned-parameters file exists under ``reports/artifacts/tune/``
for a given (task, family, model) tag, it is loaded automatically and
merged with the config defaults before fitting.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
from loguru import logger

from aki.eval.metrics import full_report
from aki.models.base import BaseModel, split_xy
from aki.models.ebm import EBMModel
from aki.models.lightgbm_model import LightGBMModel
from aki.models.scorecard import ScorecardModel
from aki.models.tune import load_best_params, save_best_params, tune_model
from aki.split.splits import assign_splits, load_split
from aki.utils.config import Config
from aki.utils.mlflow_utils import init_mlflow, run
from aki.utils.paths import paths
from aki.utils.seed import seed_everything

_MODEL_BUILDERS = {
    "ebm":        lambda p: EBMModel(p),
    "scorecard":  lambda p: ScorecardModel(p),
    "lightgbm":   lambda p: LightGBMModel(p),
}


def train_all(
    cfg: Config,
    tune: bool = False,
    n_trials: int = 60,
    families: list[str] | None = None,
    tasks: list[str] | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Fit every (task × family × model) and return a metrics summary table.

    Parameters
    ----------
    tune
        If True, run Optuna HPO on the train split (patient-grouped CV)
        before fitting the final model. Best params are persisted to
        ``reports/artifacts/tune/<tag>.json``.
    families, tasks, models
        Optional subset lists (useful for HPC array jobs).
    """
    seed_everything(cfg.random_seed)
    init_mlflow(cfg, experiment="aki-train")

    # Default training set excludes "minimal", that family is built
    # *after* the first pass, from the EBM importance on "combined".
    families = families or [
        f for f in cfg.eval["feature_families_to_train"] if f != "minimal"
    ]
    model_names = models or list(_MODEL_BUILDERS)

    rows: list[dict] = []
    for family in families:
        family_cfg = cfg.features["feature_families"].get(family, {})
        features_df = _load_features(cfg, family)
        features_df = assign_splits(features_df, cfg)

        for task in cfg.eval["tasks"]:
            if tasks and task["name"] not in tasks:
                continue
            label_col = _label_col(task)
            if label_col not in features_df.columns:
                logger.warning(f"missing label {label_col} for family={family}, skipping")
                continue

            for model_name in model_names:
                default_params = cfg.models.get(
                    "sparse_logistic" if model_name == "scorecard" else model_name,
                    {},
                )
                family_overrides = (
                    family_cfg.get("model_overrides", {}).get(model_name, {})
                )
                metrics = _train_one(
                    cfg=cfg,
                    features_df=features_df,
                    task=task["name"],
                    label_col=label_col,
                    family=family,
                    model_name=model_name,
                    model_params={**dict(default_params), **dict(family_overrides)},
                    tune=tune,
                    n_trials=n_trials,
                )
                rows.append(metrics)

    summary = pd.DataFrame(rows)
    out = paths.tables / "train_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    logger.info(f"train summary -> {out}")
    return summary


def _train_one(
    *,
    cfg: Config,
    features_df: pd.DataFrame,
    task: str,
    label_col: str,
    family: str,
    model_name: str,
    model_params: dict,
    tune: bool = False,
    n_trials: int = 60,
) -> dict:
    tr = load_split(features_df, "train")
    va = load_split(features_df, "val")

    X_tr, y_tr, feats = split_xy(tr, label_col)
    X_va, y_va, _     = split_xy(va, label_col, feature_names=feats)

    tag = f"{task}__{family}__{model_name}"
    run_name = f"{task}.{family}.{model_name}"
    tags = {"task": task, "family": family, "model": model_name}

    model_params.setdefault("random_state", cfg.random_seed)

    # Inject tuned params if available / requested
    if tune:
        groups = tr.loc[X_tr.index, "subject_id"].values
        best = tune_model(
            model_name, X_tr, y_tr, groups=groups, cfg=cfg,
            base_params=model_params,
            n_trials=n_trials, seed=cfg.random_seed, tag=tag,
        )
        # Drop private diagnostic keys before merging
        public_best = {k: v for k, v in best.items() if not k.startswith("_")}
        model_params.update(public_best)
        save_best_params(best, tag)
    else:
        cached = load_best_params(tag)
        if cached:
            public_cached = {k: v for k, v in cached.items() if not k.startswith("_")}
            model_params.update(public_cached)
            logger.info(f"{tag}: loaded cached tuned params")

    with run(cfg, run_name=run_name, tags=tags):
        mlflow.log_params(
            {f"param.{k}": v for k, v in model_params.items() if _loggable(v)}
        )

        model = _fit(model_name, model_params, X_tr, y_tr, X_va, y_va, tr)

        cal_method = cfg.models.get("calibration", {}).get("method", "isotonic")
        calibrate_models = cfg.models.get("calibration", {}).get("apply_to", [])
        should_calibrate = model_name in calibrate_models or (
            model_name == "scorecard" and "sparse_logistic" in calibrate_models
        )
        if should_calibrate:
            model.calibrate(X_va, y_va, method=cal_method)

        p_va = model.predict_proba(X_va)
        metrics = full_report(y_va.values, p_va)
        mlflow.log_metrics({f"val.{k}": float(v) for k, v in metrics.items() if _finite(v)})

        artifact_path = _save_artifact(model, task, family)
        mlflow.log_artifact(str(artifact_path))

    return {"task": task, "family": family, "model": model_name, **metrics}


def _fit(
    model_name: str,
    params: dict,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    train_df: pd.DataFrame,
) -> BaseModel:
    model = _MODEL_BUILDERS[model_name](params)
    if model_name == "lightgbm":
        model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)
    elif model_name == "scorecard":
        model.fit(X_tr, y_tr, groups=train_df.loc[X_tr.index, "subject_id"].values)
    else:
        model.fit(X_tr, y_tr)
    return model


def _save_artifact(model: BaseModel, task: str, family: str) -> Path:
    art = model.artifact(task=task, family=family)
    out = paths.artifacts / "models" / f"{task}__{family}__{model.name}.joblib"
    art.save(out)
    return out


def _load_features(cfg: Config, family: str) -> pd.DataFrame:
    path = cfg.curated_dir / "features" / f"{family}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature file missing: {path}. Run `aki features` first."
        )
    return pd.read_parquet(path)


def _label_col(task: dict) -> str:
    stage = task["outcome"].replace("kdigo_", "")  # stage1 / stage2
    return f"y_{stage}_{task['horizon_hours']}h"


def _finite(v) -> bool:
    try:
        return float(v) == float(v) and abs(float(v)) != float("inf")
    except Exception:
        return False


def _loggable(v) -> bool:
    return isinstance(v, (int, float, str, bool))
