"""Read-only helpers for inspecting the DuckDB analysis database."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

CORE_TABLES = [
    "concepts.aki_onset",
    "cohort.cohort",
    "cohort.landmarks",
    "labels.labels",
    "features.numeric_events",
    "features.rolling_aggregations",
    "features.treatments",
]

_DEFAULT_ORDER_BY = {
    "cohort.cohort": "subject_id, stay_id",
    "cohort.landmarks": "subject_id, stay_id, landmark_time",
    "labels.labels": "subject_id, stay_id, landmark_time",
}


def core_table_counts(conn: duckdb.DuckDBPyConnection) -> list[tuple[str, int | None]]:
    """Return row counts for the core pipeline tables, preserving order."""
    counts: list[tuple[str, int | None]] = []
    for table in CORE_TABLES:
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            counts.append((table, int(n)))
        except duckdb.Error:
            counts.append((table, None))
    return counts


def preview_table(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    limit: int = 20,
) -> pd.DataFrame:
    """Return a small preview of a schema-qualified table."""
    order_by = _DEFAULT_ORDER_BY.get(table)
    query = f"SELECT * FROM {table}"
    if order_by:
        query += f" ORDER BY {order_by}"
    query += f" LIMIT {int(limit)}"
    return conn.execute(query).df()


def path_status(path: Path) -> str:
    """Human-readable file presence string for reporting."""
    if not path.exists():
        return "missing"
    size_mb = path.stat().st_size / (1024 * 1024)
    return f"exists ({size_mb:.1f} MB)"
