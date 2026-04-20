"""End-to-end feature engineering.

1. Build the signal_map from configs/features.yaml
2. Run sql/features/01_numeric_events_long.sql -> long-format events
3. Run sql/features/02_rolling_aggregations.sql -> long-format windowed aggs
4. Pivot to wide (one column per {feature}_{agg}_{window}h)
5. Join demographics + treatments
6. Write one parquet per feature family to data/curated/features/
"""

from __future__ import annotations

import duckdb
import pandas as pd
from loguru import logger

from aki.data.db import run_sql_file
from aki.features.signal_map import build_signal_map
from aki.utils.config import Config
from aki.utils.paths import paths

_AGGS = ["latest", "delta", "min", "max", "mean", "std", "count"]


def build_features(conn: duckdb.DuckDBPyConnection, cfg: Config) -> None:
    """Build feature matrices for every family listed in configs/features.yaml."""
    build_signal_map(conn, cfg)

    # Long-format numeric events + rolling aggregations in DuckDB
    run_sql_file(conn, paths.sql / "features" / "01_numeric_events_long.sql")
    run_sql_file(conn, paths.sql / "features" / "02_rolling_aggregations.sql")
    run_sql_file(conn, paths.sql / "features" / "03_treatments.sql")

    # Pivot to wide in pandas (small enough after landmarking)
    wide = _pivot_rolling_to_wide(conn)

    # Attach demographics, treatments, cohort metadata
    demographics = _build_demographic_columns(conn, cfg)
    treatments = conn.execute("SELECT * FROM features.treatments").df()
    landmarks = conn.execute(
        "SELECT stay_id, subject_id, landmark_time, anchor_year_group, "
        "       age, age_group, sex, ethnicity, hours_since_icu_admit "
        "FROM cohort.landmarks"
    ).df()
    labels = conn.execute(
        "SELECT stay_id, landmark_time, "
        "       y_stage1_24h, y_stage1_48h, y_stage2_24h, y_stage2_48h, "
        "       y_cr_only_stage1_24h, y_cr_only_stage1_48h "
        "FROM labels.labels"
    ).df()

    # Merge everything
    full = landmarks.merge(demographics, on="stay_id", how="left")
    full = full.merge(wide, on=["stay_id", "landmark_time"], how="left")
    full = full.merge(treatments, on=["stay_id", "landmark_time"], how="left")
    full = full.merge(labels, on=["stay_id", "landmark_time"], how="inner")

    # Add missingness indicators from count columns
    full = _add_missingness_indicators(full)

    # Export per family
    out_dir = cfg.curated_dir / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    # "combined" is always written so the minimal-family derivation step
    # has a source to mine EBM importance from, even if the training
    # config skipped it.
    families = list(dict.fromkeys(cfg.eval["feature_families_to_train"] + ["combined"]))
    # Skip "minimal" here, itt is derived after the first training pass.
    families = [f for f in families if f != "minimal"]

    for family_name in families:
        family_cfg = cfg.features["feature_families"][family_name]
        subset = _select_family(full, family_cfg, cfg)
        path = out_dir / f"{family_name}.parquet"
        subset.to_parquet(path, index=False, compression="zstd")
        logger.info(f"  {family_name}: {len(subset):,} rows × {subset.shape[1]} cols -> {path}")


# Helpers

def _pivot_rolling_to_wide(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Pivot features.rolling_aggregations long -> wide."""
    long_df = conn.execute(
        "SELECT stay_id, landmark_time, window_h, feature_name, "
        "       latest_val, delta_val, min_val, max_val, mean_val, std_val, n_count "
        "FROM features.rolling_aggregations"
    ).df()

    # Rename columns and pivot
    long_df = long_df.rename(columns={
        "latest_val": "latest",
        "delta_val":  "delta",
        "min_val":    "min",
        "max_val":    "max",
        "mean_val":   "mean",
        "std_val":    "std",
        "n_count":    "count",
    })

    melted = long_df.melt(
        id_vars=["stay_id", "landmark_time", "window_h", "feature_name"],
        value_vars=_AGGS,
        var_name="agg",
        value_name="val",
    )
    melted["col"] = (
        melted["feature_name"] + "_" + melted["agg"] + "_" + melted["window_h"].astype(str) + "h"
    )
    wide = melted.pivot_table(
        index=["stay_id", "landmark_time"],
        columns="col",
        values="val",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def _build_demographic_columns(conn: duckdb.DuckDBPyConnection, cfg: Config) -> pd.DataFrame:
    """Encode demographics into ML-ready columns keyed by stay_id."""
    c = conn.execute(
        "SELECT stay_id, age, sex, ethnicity FROM cohort.cohort"
    ).df()

    # Binary sex
    c["sex_male"] = (c["sex"] == "M").astype(int)

    # Ethnicity grouping
    eth_groups_cfg = next(
        d for d in cfg.features["demographics"] if d["name"] == "ethnicity_group"
    )["groups"]
    eth_map = {val: grp for grp, vals in eth_groups_cfg.items() for val in vals}
    c["ethnicity_group"] = c["ethnicity"].map(eth_map).fillna("Other")

    # Keep all categories so fairness evaluation can index them
    eth_ohe = pd.get_dummies(c["ethnicity_group"], prefix="eth", dtype=int)
    # ``cohort.landmarks`` already carries age/sex/ethnicity for each row.
    # Keep only encoded columns here to avoid pandas creating age_x/age_y.
    out = pd.concat([c[["stay_id", "sex_male"]], eth_ohe], axis=1)
    return out


def _add_missingness_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """For every ``{feature}_count_{window}h`` column, add a ``_missing_`` flag."""
    count_cols = [c for c in df.columns if "_count_" in c]
    for c in count_cols:
        missing_col = c.replace("_count_", "_missing_")
        df[missing_col] = (df[c].fillna(0) == 0).astype(int)
    return df


def _select_family(
    full: pd.DataFrame,
    family_cfg: dict,
    cfg: Config,
) -> pd.DataFrame:
    """Subset columns per feature family definition."""
    meta_cols = [
        "stay_id", "subject_id", "landmark_time",
        "anchor_year_group", "age", "age_group", "sex", "ethnicity",
        "hours_since_icu_admit",
        "y_stage1_24h", "y_stage1_48h", "y_stage2_24h", "y_stage2_48h",
        "y_cr_only_stage1_24h", "y_cr_only_stage1_48h",
    ]

    include = set(family_cfg.get("include", []))

    # Demographics are always included (for modeling + subgroup analysis)
    keep_cols = list(meta_cols)
    demo_cols = ["age", "sex_male"] + [c for c in full.columns if c.startswith("eth_")]
    keep_cols.extend(c for c in demo_cols if c not in keep_cols)

    vital_names = {v["name"] for v in cfg.features["vitals"]}
    vital_names.discard("temperature_f")  # merged into temperature_c
    lab_names = {lab["name"] for lab in cfg.features["labs"]}

    renal_flagged = {lab["name"] for lab in cfg.features["labs"] if lab.get("renal")}
    exclude_renal = family_cfg.get("exclude_renal_flagged", False)

    def _feat_cols(names: set[str]) -> list[str]:
        cols: list[str] = []
        for n in names:
            cols.extend(c for c in full.columns if c.startswith(f"{n}_"))
        return cols

    if "vitals" in include:
        keep_cols.extend(_feat_cols(vital_names))

    if "labs" in include:
        lab_set = lab_names - renal_flagged if exclude_renal else lab_names
        keep_cols.extend(_feat_cols(lab_set))

    if "treatments" in include:
        tx_cols = [
            "vasopressor_any_24h", "loop_diuretic_24h", "mech_vent_24h",
            "fluid_input_ml_6h", "fluid_input_ml_12h", "fluid_input_ml_24h",
        ]
        if not exclude_renal:
            tx_cols += ["urine_output_ml_6h", "urine_output_ml_12h", "urine_output_ml_24h"]
        keep_cols.extend(c for c in tx_cols if c in full.columns)

    # De-dup while preserving order
    seen: set[str] = set()
    final = [c for c in keep_cols if not (c in seen or seen.add(c))]
    return full[final].copy()
