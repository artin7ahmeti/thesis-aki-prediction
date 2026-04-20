"""KDIGO concept builder.

Runs the concept SQLs in order and exports the label table.
The KDIGO logic itself lives in sql/concepts/*.sql — this module only
orchestrates execution and parquet export.
"""

from __future__ import annotations

import duckdb
from loguru import logger

from aki.data.db import run_sql_file
from aki.utils.config import Config
from aki.utils.paths import paths

_CONCEPT_ORDER = [
    "01_weight.sql",
    "02_creatinine.sql",
    "03_creatinine_baseline.sql",
    "04_urine_output_hourly.sql",
    "05_kdigo_creatinine.sql",
    "06_kdigo_uo.sql",
    "07_kdigo_stages.sql",
    "08_aki_onset.sql",
]


def build_kdigo_concepts_and_labels(conn: duckdb.DuckDBPyConnection, cfg: Config) -> None:
    """Execute concept + label SQL and export curated parquet files."""
    # Concepts must run before cohort (cohort excludes prevalent AKI)
    for fname in _CONCEPT_ORDER:
        run_sql_file(conn, paths.sql / "concepts" / fname)

    # Labels depend on cohort.landmarks (built separately)
    run_sql_file(conn, paths.sql / "labels" / "01_labels.sql")

    # Export
    cfg.curated_dir.mkdir(parents=True, exist_ok=True)
    for tbl, rel in [
        ("concepts.aki_onset",      "aki_onset.parquet"),
        ("concepts.kdigo_stages",   "kdigo_stages.parquet"),
        ("labels.labels",           "labels.parquet"),
    ]:
        out_path = cfg.curated_dir / rel
        conn.execute(f"COPY {tbl} TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        logger.info(f"  {tbl}: {n:,} rows -> {out_path}")
