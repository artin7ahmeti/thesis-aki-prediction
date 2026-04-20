"""Dataset drift checks across temporal splits.

Produces two tables in ``reports/tables/qa/``:

- ``split_prevalence.csv`` :label prevalence per (split, task).
- ``feature_drift.csv`` :mean + std per feature in each split, plus
  the standardized mean difference (SMD) vs. train.

A |SMD| > 0.1 is a commonly-cited threshold for meaningful drift, rows
above that threshold are flagged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from aki.models.base import feature_columns
from aki.split.splits import assign_splits
from aki.utils.config import Config
from aki.utils.paths import paths


def compute_drift_report(cfg: Config, family: str = "combined") -> dict[str, Path]:
    """Build + write drift tables for the given family."""
    out_dir = paths.tables / "qa"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.curated_dir / "features" / f"{family}.parquet")
    df = assign_splits(df, cfg)

    prev = _prevalence_table(df)
    prev_path = out_dir / f"split_prevalence__{family}.csv"
    prev.to_csv(prev_path, index=False)

    drift = _feature_drift_table(df)
    drift_path = out_dir / f"feature_drift__{family}.csv"
    drift.to_csv(drift_path, index=False)

    flagged = int((drift["abs_smd"] > 0.1).sum())
    logger.info(
        f"drift report ({family}): {flagged}/{len(drift)} features |SMD|>0.1 "
        f"-> {drift_path}"
    )
    return {"prevalence": prev_path, "feature_drift": drift_path}


def _prevalence_table(df: pd.DataFrame) -> pd.DataFrame:
    label_cols = [c for c in df.columns if c.startswith("y_")]
    rows = []
    for split in df["split"].dropna().unique():
        sub = df[df["split"] == split]
        row = {
            "split":      split,
            "n_rows":     len(sub),
            "n_stays":    sub["stay_id"].nunique(),
            "n_patients": sub["subject_id"].nunique(),
        }
        for c in label_cols:
            row[f"prev.{c}"] = float(sub[c].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_drift_table(df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = feature_columns(df)
    tr = df[df["split"] == "train"]
    splits = [s for s in ("val", "test") if s in df["split"].unique()]

    rows = []
    for c in feat_cols:
        tr_vals = pd.to_numeric(tr[c], errors="coerce").dropna()
        if tr_vals.empty:
            continue
        tr_mean = float(tr_vals.mean())
        tr_std = float(tr_vals.std(ddof=1))
        for s in splits:
            sv = pd.to_numeric(df.loc[df["split"] == s, c], errors="coerce").dropna()
            if sv.empty:
                continue
            s_mean = float(sv.mean())
            s_std = float(sv.std(ddof=1))
            pooled = np.sqrt(((tr_std ** 2) + (s_std ** 2)) / 2.0)
            smd = (s_mean - tr_mean) / pooled if pooled > 0 else 0.0
            rows.append({
                "feature":   c,
                "split":     s,
                "n_train":   int(len(tr_vals)),
                "n_split":   int(len(sv)),
                "mean_train": tr_mean,
                "mean_split": s_mean,
                "smd":       smd,
                "abs_smd":   abs(smd),
                "missing_rate_train": float(tr[c].isna().mean()),
                "missing_rate_split": float(df.loc[df["split"] == s, c].isna().mean()),
            })
    return (
        pd.DataFrame(rows)
        .sort_values("abs_smd", ascending=False, ignore_index=True)
    )
