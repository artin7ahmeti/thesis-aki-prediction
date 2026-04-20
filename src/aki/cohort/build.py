"""Run the cohort and landmark SQL, export to parquet."""

from __future__ import annotations

import duckdb
from loguru import logger

from aki.data.db import run_sql_file
from aki.utils.config import Config
from aki.utils.paths import paths


def build_cohort_and_landmarks(conn: duckdb.DuckDBPyConnection, cfg: Config) -> None:
    """Execute sql/cohort/*.sql and export curated parquet files.

    Parameters
    ----------
    conn : DuckDB connection (already staged with raw views + concepts)
    cfg  : project configuration
    """
    cohort_cfg = cfg.cohort["cohort"]
    lm_cfg = cfg.cohort["landmarks"]
    esrd_cfg = cfg.cohort["esrd_icd"]

    max_horizon = max(lm_cfg["horizons_hours"])

    # Cohort
    run_sql_file(
        conn,
        paths.sql / "cohort" / "01_cohort.sql",
        params={
            "min_age": cohort_cfg["min_age"],
            "min_icu_los_hours": cohort_cfg["min_icu_los_hours"],
            "max_icu_los_days": cohort_cfg["max_icu_los_days"],
            "obs_window_hours": lm_cfg["obs_window_hours"],
            "esrd_icd10_list": esrd_cfg["icd10"],
            "esrd_icd9_list": esrd_cfg["icd9"],
        },
    )

    # Landmarks
    run_sql_file(
        conn,
        paths.sql / "cohort" / "02_landmarks.sql",
        params={
            "obs_window_hours": lm_cfg["obs_window_hours"],
            "spacing_hours": lm_cfg["spacing_hours"],
            "max_horizon_hours": max_horizon,
            "min_stay_hours": lm_cfg["min_stay_hours"],
            "exclude_after_aki": lm_cfg["exclude_after_aki"],
        },
    )

    # Export curated parquet
    cfg.curated_dir.mkdir(parents=True, exist_ok=True)
    for tbl, rel in [
        ("cohort.cohort",    "cohort.parquet"),
        ("cohort.landmarks", "landmarks.parquet"),
    ]:
        out_path = cfg.curated_dir / rel
        conn.execute(f"COPY {tbl} TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        logger.info(f"  {tbl}: {n:,} rows -> {out_path}")
