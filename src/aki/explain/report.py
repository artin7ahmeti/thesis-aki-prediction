"""Explanation-pass orchestrator.

Walks every trained artifact and emits:
- ``reports/tables/per_model/<tag>/global_importance.csv``
- ``reports/figures/per_model/<tag>/ebm_shape_*.png`` (EBM only)
- ``reports/tables/per_model/<tag>/patient_contributions.csv`` (EBM/scorecard)
- ``reports/figures/per_model/<tag>/patient_explanation.png`` (EBM/scorecard)
- ``reports/figures/per_model/<tag>/reliability.png``
- ``reports/artifacts/scorecards/<tag>/scorecard.{csv,md}`` (scorecard only)
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

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
from aki.models.base import ModelArtifact
from aki.split.splits import assign_splits, load_split
from aki.utils.config import Config
from aki.utils.paths import paths
from aki.utils.subset import artifact_triple_from_path, matches_selector


def run_explanations(
    cfg: Config,
    *,
    tasks: list[str] | None = None,
    families: list[str] | None = None,
    models: list[str] | None = None,
) -> None:
    """Generate explanation artifacts for every trained model."""
    artifacts_dir = paths.artifacts / "models"
    artifacts = []
    for art_path in sorted(artifacts_dir.glob("*.joblib")):
        task, family, model = artifact_triple_from_path(art_path)
        if matches_selector(
            task=task, family=family, model=model,
            tasks=tasks, families=families, models=models,
        ):
            artifacts.append(art_path)
    if not artifacts:
        raise FileNotFoundError("No trained artifacts found. Run `aki train` first.")

    # Cache per-family test-split features for local explanation artifacts.
    _test_cache: dict[str, pd.DataFrame] = {}

    for art_path in artifacts:
        art = ModelArtifact.load(art_path)
        tag = f"{art.task}__{art.family}__{art.name}"

        # Global importance
        imp_dir = paths.tables / "per_model" / tag
        imp_dir.mkdir(parents=True, exist_ok=True)
        importance = global_importance_table(art)
        importance.to_csv(imp_dir / "global_importance.csv", index=False)

        # Shape plots (EBM only)
        if art.name == "ebm":
            fig_dir = paths.figures / "per_model" / tag
            plot_ebm_shapes(art, out_dir=fig_dir, top_k=12)

        # Patient-level additive explanation (glass-box models only)
        if art.name in {"ebm", "scorecard"}:
            test_df = _test_split_for_family(cfg, art.family, _test_cache)
            _write_patient_level_artifacts(cfg, art, tag, test_df)

        # Reliability plot
        rc_path = imp_dir / "reliability_curve.csv"
        if rc_path.exists():
            rc = pd.read_csv(rc_path)
            plot_reliability(
                rc,
                out_path=paths.figures / "per_model" / tag / "reliability.png",
                title=f"Reliability — {tag}",
            )

        # Scorecard (sparse logistic only)
        if art.name == "scorecard":
            card_dir = paths.artifacts / "scorecards" / tag
            build_scorecard_artifact(art, out_dir=card_dir)

        # SHAP (LightGBM only)
        if art.name == "lightgbm":
            from aki.explain.shap_explainer import lightgbm_global_shap

            if art.family not in _test_cache:
                features_df = pd.read_parquet(
                    cfg.curated_dir / "features" / f"{art.family}.parquet"
                )
                features_df = assign_splits(features_df, cfg)
                _test_cache[art.family] = load_split(features_df, "test")
            test_df = _test_cache[art.family]
            lightgbm_global_shap(
                art, test_df, out_dir=paths.figures / "per_model" / tag,
                random_state=cfg.random_seed,
            )

        logger.info(f"explanations written for {tag}")


def _test_split_for_family(
    cfg: Config,
    family: str,
    cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if family not in cache:
        features_df = pd.read_parquet(
            cfg.curated_dir / "features" / f"{family}.parquet"
        )
        features_df = assign_splits(features_df, cfg)
        cache[family] = load_split(features_df, "test")
    return cache[family]


def _write_patient_level_artifacts(
    cfg: Config,
    art: ModelArtifact,
    tag: str,
    test_df: pd.DataFrame,
) -> None:
    label_col = _label_col_from_task(art.task)
    if label_col not in test_df.columns:
        logger.warning(f"patient explanation skipped for {tag}: missing label {label_col}")
        return

    reference = _bedside_reference_bundle(cfg=cfg, art=art)
    case = select_representative_patient_case(
        art,
        test_df,
        label_col,
        quantile=0.85,
        reference_art=reference["artifact"] if reference else None,
        reference_df=reference["test_df"] if reference else None,
        reference_label_col=label_col,
    )
    contrib = patient_additive_contributions(art, case)

    table_dir = paths.tables / "per_model" / tag
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = paths.figures / "per_model" / tag
    fig_dir.mkdir(parents=True, exist_ok=True)

    contrib.to_csv(table_dir / "patient_contributions.csv", index=False)

    meta_cols = [
        col
        for col in [
            "subject_id",
            "stay_id",
            "hadm_id",
            "landmark_time",
            "hours_since_icu_admit",
            label_col,
            "_pred_proba",
            "_missing_features",
            "_selection_pool",
            "_selection_quantile",
            "_selection_target_prob",
            "_selection_strategy",
            "_agreement_gap",
            "_reference_pred_proba",
            "_reference_target_prob",
            "_reference_model_tag",
        ]
        if col in case.index
    ]
    case.loc[meta_cols].to_frame().T.to_csv(table_dir / "patient_case.csv", index=False)

    if reference:
        ref_case = _align_reference_case(reference["test_df"], case)
        if ref_case is not None:
            ref_contrib = patient_additive_contributions(reference["artifact"], ref_case)
            ref_contrib.to_csv(table_dir / "patient_contributions_reference_scorecard.csv", index=False)

    subtitle = _patient_plot_subtitle(case, label_col)
    plot_patient_contributions(
        contrib,
        out_path=fig_dir / "patient_explanation.png",
        title="Patient-level explanation",
        subtitle=subtitle,
        top_n=10,
    )


def _patient_plot_subtitle(case: pd.Series, label_col: str) -> str:
    parts = [
        "Held-out test landmark",
        f"true label: {int(case[label_col])}",
    ]
    if "_pred_proba" in case.index:
        parts.append(f"predicted risk: {100.0 * float(case['_pred_proba']):.1f}%")
    if "_reference_pred_proba" in case.index and pd.notna(case["_reference_pred_proba"]):
        parts.append(f"bedside scorecard risk: {100.0 * float(case['_reference_pred_proba']):.1f}%")
    if "hours_since_icu_admit" in case.index and pd.notna(case["hours_since_icu_admit"]):
        parts.append(f"{float(case['hours_since_icu_admit']):.0f}h since ICU admission")
    if "_selection_strategy" in case.index:
        if str(case["_selection_strategy"]) == "primary_plus_bedside_agreement":
            parts.append("selected for agreement with bedside scorecard")
        elif "_selection_pool" in case.index:
            pool = "positive cases" if str(case["_selection_pool"]) == "positive" else "all test cases"
            parts.append(f"selected from {pool}")
    return " | ".join(parts)


def _bedside_reference_bundle(
    cfg: Config,
    art: ModelArtifact,
) -> dict[str, object] | None:
    """Return the best bedside scorecard artifact/test split for matching stage-1 tasks."""
    if art.name != "ebm":
        return None
    if art.task not in {"aki_stage1_24h", "aki_stage1_48h"}:
        return None

    reference_family = "scorecard_augmented_v2"
    reference_tag = f"{art.task}__{reference_family}__scorecard"
    art_path = paths.artifacts / "models" / f"{reference_tag}.joblib"
    if not art_path.exists():
        return None

    reference_art = ModelArtifact.load(art_path)
    reference_df = _load_reference_test_df(reference_family, cfg)
    if reference_df is None:
        return None
    return {
        "artifact": reference_art,
        "test_df": reference_df,
        "tag": reference_tag,
    }


def _load_reference_test_df(reference_family: str, cfg: Config) -> pd.DataFrame | None:
    feature_path = cfg.curated_dir / "features" / f"{reference_family}.parquet"
    if not feature_path.exists():
        return None
    features_df = pd.read_parquet(feature_path)
    features_df = assign_splits(features_df, cfg)
    return load_split(features_df, "test")


def _align_reference_case(reference_df: pd.DataFrame, case: pd.Series) -> pd.Series | None:
    key_cols = [c for c in ("stay_id", "subject_id", "landmark_time") if c in reference_df.columns and c in case.index]
    if not key_cols:
        return None
    mask = pd.Series(True, index=reference_df.index)
    for col in key_cols:
        mask &= reference_df[col] == case[col]
    matched = reference_df.loc[mask]
    if matched.empty:
        return None
    return matched.iloc[0].copy()


def _label_col_from_task(task_name: str) -> str:
    stage, horizon = task_name.replace("aki_", "").rsplit("_", 1)
    return f"y_{stage}_{horizon}"
