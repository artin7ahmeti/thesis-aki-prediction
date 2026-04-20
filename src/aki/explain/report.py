"""Explanation-pass orchestrator.

Walks every trained artifact and emits:
- ``reports/tables/per_model/<tag>/global_importance.csv``
- ``reports/figures/per_model/<tag>/ebm_shape_*.png`` (EBM only)
- ``reports/figures/per_model/<tag>/reliability.png``
- ``reports/artifacts/scorecards/<tag>/scorecard.{csv,md}`` (scorecard only)
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from aki.explain.global_importance import global_importance_table
from aki.explain.plots import plot_ebm_shapes, plot_reliability
from aki.explain.scorecard_card import build_scorecard_artifact
from aki.explain.shap_explainer import lightgbm_global_shap
from aki.models.base import ModelArtifact
from aki.split.splits import assign_splits, load_split
from aki.utils.config import Config
from aki.utils.paths import paths


def run_explanations(cfg: Config) -> None:
    """Generate explanation artifacts for every trained model."""
    artifacts_dir = paths.artifacts / "models"
    artifacts = sorted(artifacts_dir.glob("*.joblib"))
    if not artifacts:
        raise FileNotFoundError("No trained artifacts found. Run `aki train` first.")

    # Cache per-family test-split features for SHAP
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
