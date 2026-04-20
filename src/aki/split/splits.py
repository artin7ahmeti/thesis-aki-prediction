"""Temporal + patient-level train/val/test splits.

Splits by ``anchor_year_group``: a 3-year band that MIMIC-IV assigns at
the *patient* level. Using it guarantees:

1. **Temporal ordering**, earlier bands -> train, later bands -> test.
2. **Patient exclusivity**, no ``subject_id`` ever appears in more than
   one split (since anchor_year_group is constant per subject).

Integrity checks below enforce (2) defensively even if the config is
mis-specified.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from aki.utils.config import Config

_SPLITS = ("train", "val", "test")


def assign_splits(features_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return a copy of ``features_df`` with a ``split`` column added.

    Parameters
    ----------
    features_df
        Must contain ``subject_id`` and ``anchor_year_group`` columns.
    cfg
        Reads ``cfg.eval["splits"]``.
    """
    splits_cfg = cfg.eval["splits"]
    if splits_cfg["strategy"] != "anchor_year_group":
        raise ValueError(
            f"Unsupported split strategy: {splits_cfg['strategy']!r}"
        )

    group_to_split: dict[str, str] = {}
    for name in _SPLITS:
        for g in splits_cfg[f"{name}_groups"]:
            group_to_split[g] = name

    df = features_df.copy()
    df["split"] = df["anchor_year_group"].map(group_to_split)

    unmapped = df["split"].isna().sum()
    if unmapped:
        logger.warning(
            f"{unmapped:,} rows have anchor_year_group outside configured splits — dropping"
        )
        df = df.dropna(subset=["split"]).copy()

    _assert_patient_exclusivity(df)

    for name in _SPLITS:
        n_rows = (df["split"] == name).sum()
        n_pat = df.loc[df["split"] == name, "subject_id"].nunique()
        logger.info(f"  split={name}: {n_rows:,} rows / {n_pat:,} patients")
    return df


def load_split(
    features_df: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    """Select one split from an already-assigned features frame."""
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {_SPLITS}, got {split!r}")
    if "split" not in features_df.columns:
        raise ValueError("features_df has no 'split' column — call assign_splits first")
    return features_df[features_df["split"] == split].copy()


def write_split_manifest(df: pd.DataFrame, out_path: Path) -> None:
    """Write a tiny audit file listing patient counts per split."""
    manifest = (
        df.groupby("split")
        .agg(
            n_rows=("stay_id", "size"),
            n_stays=("stay_id", "nunique"),
            n_patients=("subject_id", "nunique"),
        )
        .reset_index()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)
    logger.info(f"split manifest -> {out_path}")


def _assert_patient_exclusivity(df: pd.DataFrame) -> None:
    """Fail loudly if any subject_id appears in more than one split."""
    per_subject_splits = df.groupby("subject_id")["split"].nunique()
    leaked = per_subject_splits[per_subject_splits > 1]
    if len(leaked) > 0:
        raise AssertionError(
            f"Patient leakage: {len(leaked)} subject_ids span multiple splits. "
            f"First offenders: {leaked.head().to_dict()}"
        )
