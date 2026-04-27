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

LANDMARK_COMPACT_QUERY = """
SELECT
    l.subject_id,
    l.stay_id,
    l.anchor_year_group,
    l.age,
    l.sex,
    l.ethnicity,
    l.intime,
    l.landmark_time,
    l.hours_since_icu_admit,
    l.onset_stage1_time,
    l.onset_stage2_time,
    CASE
        WHEN l.onset_stage1_time IS NOT NULL
        THEN date_diff('hour', l.landmark_time, l.onset_stage1_time)
        ELSE NULL
    END AS hours_to_stage1_onset,
    CASE
        WHEN l.onset_stage2_time IS NOT NULL
        THEN date_diff('hour', l.landmark_time, l.onset_stage2_time)
        ELSE NULL
    END AS hours_to_stage2_onset,
    y.y_stage1_24h,
    y.y_stage1_48h,
    y.y_stage2_24h,
    y.y_stage2_48h
FROM cohort.landmarks l
LEFT JOIN labels.labels y
  ON l.stay_id = y.stay_id
 AND l.landmark_time = y.landmark_time
ORDER BY l.subject_id, l.stay_id, l.landmark_time
LIMIT {limit}
"""

LANDMARK_SUMMARY_QUERY = """
SELECT
    COUNT(*) AS n_landmarks,
    COUNT(DISTINCT stay_id) AS n_stays,
    COUNT(DISTINCT subject_id) AS n_patients,
    ROUND(AVG(hours_since_icu_admit), 1) AS mean_hours_since_icu_admit,
    MEDIAN(hours_since_icu_admit) AS median_hours_since_icu_admit,
    ROUND(AVG(CASE WHEN onset_stage1_time IS NOT NULL THEN 1 ELSE 0 END), 4) AS share_with_stage1_onset_recorded
FROM cohort.landmarks
"""

LANDMARKS_PER_STAY_QUERY = """
SELECT
    stay_id,
    subject_id,
    COUNT(*) AS n_landmarks,
    MIN(hours_since_icu_admit) AS first_landmark_hour,
    MAX(hours_since_icu_admit) AS last_landmark_hour
FROM cohort.landmarks
GROUP BY stay_id, subject_id
ORDER BY n_landmarks DESC, subject_id, stay_id
LIMIT {limit}
"""


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


def landmark_compact_preview(
    conn: duckdb.DuckDBPyConnection,
    limit: int = 20,
) -> pd.DataFrame:
    """Compact human-readable preview of landmarks joined with labels."""
    return conn.execute(LANDMARK_COMPACT_QUERY.format(limit=int(limit))).df()


def landmark_summary(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One-row landmark summary."""
    return conn.execute(LANDMARK_SUMMARY_QUERY).df()


def landmarks_per_stay(
    conn: duckdb.DuckDBPyConnection,
    limit: int = 20,
) -> pd.DataFrame:
    """Preview of stays with the most generated landmarks."""
    return conn.execute(LANDMARKS_PER_STAY_QUERY.format(limit=int(limit))).df()
