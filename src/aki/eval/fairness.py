"""Subgroup fairness evaluation.

Computes the full metric report per subgroup (sex, age_group, ethnicity)
and flags absolute gaps that exceed ``disparity_threshold``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from aki.eval.metrics import full_report
from aki.utils.config import Config


def subgroup_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    subgroup_df: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Per-subgroup metric table + disparity flag column.

    Parameters
    ----------
    subgroup_df
        DataFrame aligned row-wise with ``y_true``/``y_prob``. Must contain
        ``sex``, ``age``, ``ethnicity_group`` (or ``eth_*`` one-hots), etc.
    """
    fair_cfg = cfg.eval["fairness"]
    min_n = int(fair_cfg.get("min_subgroup_n", 100))
    disparity = float(fair_cfg.get("disparity_threshold", 0.05))

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    rows: list[dict] = []
    for attr, values in fair_cfg["groups"].items():
        for v in values:
            mask = _subgroup_mask(subgroup_df, attr, v)
            if mask is None:
                continue
            n = int(mask.sum())
            if n < min_n:
                logger.debug(f"skip subgroup {attr}={v}: n={n} < min_subgroup_n={min_n}")
                continue
            if y_true[mask].sum() == 0 or y_true[mask].sum() == n:
                # AUROC undefined, skip gracefully
                continue
            rep = full_report(y_true[mask], y_prob[mask])
            rows.append({"attribute": attr, "value": str(v), "n": n, **rep})

    result = pd.DataFrame(rows)
    if not result.empty:
        result = _flag_disparities(result, disparity)
    return result


def _subgroup_mask(df: pd.DataFrame, attr: str, value) -> np.ndarray | None:
    if attr == "age_group" and isinstance(value, (list, tuple)):
        lo, hi = value
        if "age" not in df.columns:
            return None
        return (df["age"] >= lo).values & (df["age"] < hi).values
    if attr == "sex":
        if "sex" in df.columns:
            return (df["sex"] == value).values
        if "sex_male" in df.columns:
            target = 1 if value == "M" else 0
            return (df["sex_male"] == target).values
        return None
    if attr == "ethnicity_group":
        if "ethnicity_group" in df.columns:
            return (df["ethnicity_group"] == value).values
        col = f"eth_{value}"
        if col in df.columns:
            return (df[col] == 1).values
        return None
    if attr in df.columns:
        return (df[attr] == value).values
    return None


def _flag_disparities(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """For each attribute, flag subgroups whose metric gap vs. the max
    exceeds ``threshold``."""
    out = df.copy()
    for metric in ("auroc", "auprc", "brier"):
        if metric not in out.columns:
            continue
        flag_col = f"{metric}_disparity_flag"
        out[flag_col] = False
        for _, grp in out.groupby("attribute"):
            best = grp[metric].max() if metric != "brier" else grp[metric].min()
            gap = (best - grp[metric]) if metric != "brier" else (grp[metric] - best)
            flagged = gap.abs() > threshold
            out.loc[grp.index, flag_col] = flagged.values
    return out
