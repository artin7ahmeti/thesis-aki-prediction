"""KDIGO concept and landmark-label builders.

The KDIGO logic itself lives in ``sql/concepts/*.sql`` and
``sql/labels/*.sql``. This module only orchestrates execution and parquet
export so the pipeline order stays explicit:

1. concepts.aki_onset must exist before cohort/landmarks are built
2. labels.labels depends on cohort.landmarks
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

_EXPORTS = [
    ("concepts.aki_onset", "aki_onset.parquet"),
    ("concepts.kdigo_stages", "kdigo_stages.parquet"),
    ("labels.labels", "labels.parquet"),
]


def build_kdigo_concepts(conn: duckdb.DuckDBPyConnection, cfg: Config | None = None) -> None:
    """Execute KDIGO concept SQLs through ``concepts.aki_onset``."""
    for fname in _CONCEPT_ORDER:
        run_sql_file(conn, paths.sql / "concepts" / fname)
    _ = cfg


def build_landmark_labels(conn: duckdb.DuckDBPyConnection, cfg: Config) -> None:
    """Build landmark-level labels and export labels/concept audit tables."""
    run_sql_file(conn, paths.sql / "labels" / "01_labels.sql")
    _export_tables(conn, cfg, _EXPORTS)


def build_kdigo_concepts_and_labels(conn: duckdb.DuckDBPyConnection, cfg: Config) -> None:
    """Execute concept + label SQL and export curated parquet files."""
    build_kdigo_concepts(conn, cfg)
    build_landmark_labels(conn, cfg)


def _export_tables(
    conn: duckdb.DuckDBPyConnection,
    cfg: Config,
    tables: list[tuple[str, str]],
) -> None:
    cfg.curated_dir.mkdir(parents=True, exist_ok=True)
    for tbl, rel in tables:
        out_path = cfg.curated_dir / rel
        conn.execute(f"COPY {tbl} TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        logger.info(f"  {tbl}: {n:,} rows -> {out_path}")
