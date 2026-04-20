"""Derive the ``minimal`` feature family from EBM importance.

Workflow, run after the first training pass on the ``combined`` family:

1. Load the per-task EBM artifact for the ``combined`` family.
2. Rank univariate terms by mean-|score| (interactions excluded).
3. Pick the top-``k`` terms (``k`` = feature family ``max_features``,
   default 10), keeping the *underlying column name* — e.g. an EBM
   term "creatinine_latest_6h" becomes a single feature column.
4. Write a slim parquet ``data/curated/features/minimal.parquet`` that
   the normal ``train_all`` loop can consume.

The resulting file contains: the selected columns + demographics + meta
+ label columns, so split/train/evaluate can proceed unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from aki.models.base import ModelArtifact
from aki.utils.config import Config
from aki.utils.paths import paths


def derive_minimal_family(cfg: Config, source_family: str = "combined") -> Path:
    """Build the ``minimal`` family parquet from the best EBM model.

    Parameters
    ----------
    source_family
        Feature family to mine for top-k features (defaults to ``combined``).
    """
    family_cfg = cfg.features["feature_families"]["minimal"]
    k = int(family_cfg.get("max_features", 10))

    combined_path = cfg.curated_dir / "features" / f"{source_family}.parquet"
    if not combined_path.exists():
        raise FileNotFoundError(
            f"{combined_path} missing. Run `aki features` first."
        )
    df = pd.read_parquet(combined_path)

    # Rank across every EBM we trained on the source family; take the union
    # of top-k features per task so the minimal family remains stable
    # across 24h/48h and stage-1/stage-2 outcomes.
    chosen: list[str] = []
    for art in _ebm_artifacts(source_family):
        ranked = _rank_univariate_terms(art)
        for feat in ranked:
            if feat in chosen:
                continue
            if feat in df.columns:
                chosen.append(feat)
            if len(chosen) >= k:
                break
        if len(chosen) >= k:
            break

    if not chosen:
        raise RuntimeError(
            "No EBM artifacts found. Run `aki train` on the combined family first."
        )

    logger.info(f"minimal family (top-{k}): {chosen}")
    meta_cols = _meta_and_label_cols(df)
    demo_cols = ["age", "sex_male"] + [c for c in df.columns if c.startswith("eth_")]
    keep = list(dict.fromkeys(meta_cols + demo_cols + chosen))
    keep = [c for c in keep if c in df.columns]
    minimal = df[keep].copy()

    out = cfg.curated_dir / "features" / "minimal.parquet"
    minimal.to_parquet(out, index=False, compression="zstd")
    logger.info(f"minimal.parquet: {len(minimal):,} rows × {minimal.shape[1]} cols -> {out}")
    return out


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _ebm_artifacts(family: str) -> list[ModelArtifact]:
    """Load every EBM artifact trained on ``family`` — sorted for determinism."""
    out: list[ModelArtifact] = []
    for p in sorted((paths.artifacts / "models").glob(f"*__{family}__ebm.joblib")):
        out.append(ModelArtifact.load(p))
    return out


def _rank_univariate_terms(art: ModelArtifact) -> list[str]:
    """Return univariate feature names from an EBM artifact, sorted by |importance|."""
    expl = art.estimator.explain_global()
    data = expl.data()
    terms = list(data["names"])
    scores = list(data["scores"])

    # Univariate term names match feature columns; interactions contain "&" or " x "
    return [
        term for term, _ in sorted(
            zip(terms, scores), key=lambda kv: -abs(kv[1])
        )
        if "&" not in term and " x " not in term
    ]


def _meta_and_label_cols(df: pd.DataFrame) -> list[str]:
    meta = [
        "stay_id", "subject_id", "landmark_time",
        "anchor_year_group", "age", "age_group", "sex", "ethnicity",
        "hours_since_icu_admit",
    ]
    labels = [c for c in df.columns if c.startswith("y_")]
    return [c for c in meta + labels if c in df.columns]
