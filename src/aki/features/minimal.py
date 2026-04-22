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

    artifacts = _ebm_artifacts(source_family)
    importance = _aggregate_univariate_importance(artifacts, available=set(df.columns))
    if importance.empty:
        raise RuntimeError(
            "No EBM artifacts found. Run `aki train` on the combined family first."
        )

    chosen = importance["feature"].head(k).tolist()
    logger.info(f"minimal family (top-{k}): {chosen}")
    _write_selection_table(importance, chosen, source_family)

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
    return [feature for feature, _score in _univariate_term_scores(art)]


def _univariate_term_scores(art: ModelArtifact) -> list[tuple[str, float]]:
    """Return univariate EBM terms with absolute global-importance scores."""
    expl = art.estimator.explain_global()
    data = expl.data()
    terms = list(data["names"])
    scores = list(data["scores"])

    # Univariate term names match feature columns; interactions contain "&" or " x "
    return [
        (term, abs(float(score))) for term, score in sorted(
            zip(terms, scores, strict=True),
            key=lambda kv: (-abs(float(kv[1])), kv[0]),
        )
        if "&" not in term and " x " not in term
    ]


def _aggregate_univariate_importance(
    artifacts: list[ModelArtifact],
    available: set[str],
) -> pd.DataFrame:
    """Average EBM univariate importance across all source-family tasks."""
    rows: list[dict] = []
    for art in artifacts:
        for rank, (feature, score) in enumerate(_univariate_term_scores(art), start=1):
            if feature not in available:
                continue
            rows.append({
                "task": art.task,
                "family": art.family,
                "model": art.name,
                "feature": feature,
                "abs_importance": score,
                "rank": rank,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "feature", "mean_abs_importance", "max_abs_importance",
            "n_artifacts", "best_rank",
        ])

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby("feature", as_index=False)
        .agg(
            mean_abs_importance=("abs_importance", "mean"),
            max_abs_importance=("abs_importance", "max"),
            n_artifacts=("task", "nunique"),
            best_rank=("rank", "min"),
        )
        .sort_values(
            ["mean_abs_importance", "n_artifacts", "best_rank", "feature"],
            ascending=[False, False, True, True],
            ignore_index=True,
        )
    )
    return summary


def _write_selection_table(
    importance: pd.DataFrame,
    chosen: list[str],
    source_family: str,
) -> Path:
    out = paths.tables / "minimal_feature_selection.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table = importance.copy()
    chosen_order = {feature: i + 1 for i, feature in enumerate(chosen)}
    table["selected"] = table["feature"].isin(chosen)
    table["selected_rank"] = table["feature"].map(chosen_order)
    table["source_family"] = source_family
    table.to_csv(out, index=False)
    logger.info(f"minimal feature selection table -> {out}")
    return out


def _meta_and_label_cols(df: pd.DataFrame) -> list[str]:
    meta = [
        "stay_id", "subject_id", "landmark_time",
        "anchor_year_group", "age", "age_group", "sex", "ethnicity",
        "hours_since_icu_admit",
    ]
    labels = [c for c in df.columns if c.startswith("y_")]
    return [c for c in meta + labels if c in df.columns]
