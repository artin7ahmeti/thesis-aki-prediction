"""End-to-end evaluation on the held-out test split.

For every trained artifact under ``reports/artifacts/models/``:

1. Predict on the test split for the matching (task, family).
2. Compute full metric report + patient-level bootstrap CIs.
3. Compute decision curve, subgroup fairness, reliability curve.
4. Save all tables under ``reports/tables/`` (CSV) and log to MLflow.
5. Apply the *input-economy* gate: the ``minimal`` family must come
   within ``auroc_tolerance`` of ``combined`` on every task.
"""

from __future__ import annotations

import mlflow
import pandas as pd
from loguru import logger

from aki.eval.bootstrap import patient_bootstrap_ci
from aki.eval.calibration import reliability_curve
from aki.eval.decision_curve import decision_curve
from aki.eval.fairness import subgroup_metrics
from aki.eval.metrics import full_report
from aki.models.base import ModelArtifact
from aki.split.splits import assign_splits, load_split
from aki.utils.config import Config
from aki.utils.mlflow_utils import init_mlflow, run
from aki.utils.paths import paths


def evaluate_all(cfg: Config) -> pd.DataFrame:
    """Evaluate every model artifact on the test split."""
    init_mlflow(cfg, experiment="aki-evaluate")

    artifacts = sorted((paths.artifacts / "models").glob("*.joblib"))
    if not artifacts:
        raise FileNotFoundError("No trained artifacts found. Run `aki train` first.")

    summary_rows: list[dict] = []
    for art_path in artifacts:
        art = ModelArtifact.load(art_path)
        result = _evaluate_one(cfg, art)
        summary_rows.append(result)

    summary = pd.DataFrame(summary_rows)
    summary_path = paths.tables / "test_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"test summary -> {summary_path}")

    gate = _input_economy_gate(summary, cfg)
    if not gate.empty:
        gate.to_csv(paths.tables / "input_economy_gate.csv", index=False)
    return summary


def _evaluate_one(cfg: Config, art: ModelArtifact) -> dict:
    features_df = pd.read_parquet(
        cfg.curated_dir / "features" / f"{art.family}.parquet"
    )
    features_df = assign_splits(features_df, cfg)
    te = load_split(features_df, "test")

    label_col = _label_col_from_task(art.task)
    te = te.dropna(subset=[label_col]).copy()
    X = te[art.feature_names]
    y = te[label_col].astype(int).values
    subjects = te["subject_id"].values

    if art.calibrator is not None:
        p = art.calibrator.predict_proba(X)[:, 1]
    else:
        p = art.estimator.predict_proba(X)[:, 1]

    metrics = full_report(y, p)
    tag = f"{art.task}__{art.family}__{art.name}"

    with run(cfg, run_name=f"eval.{tag}",
             tags={"task": art.task, "family": art.family, "model": art.name}):
        mlflow.log_metrics({f"test.{k}": float(v)
                            for k, v in metrics.items() if _finite(v)})

        _save_auxiliary_tables(cfg, art, tag, y, p, subjects, te)

    return {"task": art.task, "family": art.family, "model": art.name, **metrics}


def _save_auxiliary_tables(
    cfg: Config,
    art: ModelArtifact,
    tag: str,
    y,
    p,
    subjects,
    te: pd.DataFrame,
) -> None:
    out_dir = paths.tables / "per_model" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bootstrap CIs
    boot_cfg = cfg.eval["bootstrap"]
    ci = patient_bootstrap_ci(
        y, p, subjects,
        n_iterations=int(boot_cfg["n_iterations"]),
        confidence_level=float(boot_cfg["confidence_level"]),
        random_state=cfg.random_seed,
    )
    ci.to_csv(out_dir / "bootstrap_ci.csv", index=False)

    # Decision curve
    dc_cfg = cfg.eval["decision_curve"]
    dc = decision_curve(
        y, p,
        threshold_min=dc_cfg["threshold_min"],
        threshold_max=dc_cfg["threshold_max"],
        threshold_step=dc_cfg["threshold_step"],
    )
    dc.to_csv(out_dir / "decision_curve.csv", index=False)

    # Reliability curve
    rc = reliability_curve(y, p, n_bins=cfg.eval["metrics"]["calibration_bins"])
    rc.to_csv(out_dir / "reliability_curve.csv", index=False)

    # Subgroup fairness
    fair = subgroup_metrics(y, p, te, cfg)
    if not fair.empty:
        fair.to_csv(out_dir / "fairness.csv", index=False)


def _input_economy_gate(summary: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compare ``minimal`` vs. ``combined`` per (task, model).

    Returns an empty frame if the ``minimal`` family hasn't been trained.
    """
    gate_cfg = cfg.eval["input_economy"]
    auroc_tol = float(gate_cfg["auroc_tolerance"])
    slope_tol = float(gate_cfg.get("calibration_tolerance", 0.05))
    ece_tol = float(gate_cfg.get("ece_tolerance", 0.02))

    if "minimal" not in summary["family"].unique():
        return pd.DataFrame()

    metrics = ["task", "model", "auroc", "calibration_slope", "ece"]
    combined = summary[summary["family"] == "combined"][metrics]
    minimal  = summary[summary["family"] == "minimal"][metrics]

    merged = combined.merge(
        minimal, on=["task", "model"], suffixes=("_combined", "_minimal")
    )
    merged["delta_auroc"] = merged["auroc_combined"] - merged["auroc_minimal"]
    merged["delta_ece"] = merged["ece_minimal"] - merged["ece_combined"]
    merged["minimal_slope_error"] = (merged["calibration_slope_minimal"] - 1.0).abs()
    merged["passes_auroc"] = merged["delta_auroc"] <= auroc_tol
    merged["passes_calibration"] = (
        (merged["minimal_slope_error"] <= slope_tol)
        & (merged["delta_ece"] <= ece_tol)
    )
    merged["passes_gate"] = merged["passes_auroc"] & merged["passes_calibration"]
    return merged


def _label_col_from_task(task_name: str) -> str:
    stage, horizon = task_name.replace("aki_", "").rsplit("_", 1)
    return f"y_{stage}_{horizon}"


def _finite(v) -> bool:
    try:
        return float(v) == float(v) and abs(float(v)) != float("inf")
    except Exception:
        return False
