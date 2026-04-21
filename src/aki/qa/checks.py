"""Run SQL QA views and enforce hard data-integrity invariants."""

from __future__ import annotations

from typing import Any

import duckdb
import pandas as pd
from loguru import logger

from aki.data.db import run_sql_file
from aki.utils.config import Config
from aki.utils.paths import paths

QA_VIEWS = (
    "qa.cohort_summary",
    "qa.landmark_summary",
    "qa.label_prevalence",
    "qa.leakage_check",
    "qa.baseline_coverage",
    "qa.cohort_baseline_coverage",
)


def run_qa_checks(conn: duckdb.DuckDBPyConnection, cfg: Config) -> dict[str, pd.DataFrame]:
    """Create QA views, export them to CSV, and fail on critical violations."""
    run_sql_file(conn, paths.sql / "qa" / "01_qa_checks.sql")

    tables: dict[str, pd.DataFrame] = {}
    for view in QA_VIEWS:
        df = conn.execute(f"SELECT * FROM {view}").df()
        out = paths.tables / f"{view.replace('.', '__')}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        tables[view] = df
        logger.info(f"  {view}: {len(df):,} rows -> {out}")

    assert_qa_invariants(tables, cfg)
    return tables


def assert_qa_invariants(tables: dict[str, pd.DataFrame], cfg: Config | None = None) -> None:
    """Raise ``AssertionError`` for critical leakage/empty-output violations."""
    failures: list[str] = []

    cohort = _row(tables, "qa.cohort_summary")
    if _number(cohort, "n_stays") <= 0:
        failures.append("cohort.cohort is empty")
    if _number(cohort, "n_patients") <= 0:
        failures.append("cohort has zero patients")
    if cfg is not None and _number(cohort, "min_age") < cfg.cohort["cohort"]["min_age"]:
        failures.append("cohort contains patients below configured minimum age")

    landmarks = _row(tables, "qa.landmark_summary")
    if _number(landmarks, "n_landmarks") <= 0:
        failures.append("cohort.landmarks is empty")
    if _number(landmarks, "n_stays_with_landmarks") <= 0:
        failures.append("no stays have usable landmarks")

    labels = _row(tables, "qa.label_prevalence")
    if _number(labels, "n_landmarks") <= 0:
        failures.append("labels.labels is empty")
    for col in (
        "prev_stage1_24h",
        "prev_stage1_48h",
        "prev_stage2_24h",
        "prev_stage2_48h",
        "prev_cr_only_24h",
        "prev_cr_only_48h",
    ):
        val = _number(labels, col)
        if not 0.0 <= val <= 1.0:
            failures.append(f"{col} outside [0, 1]: {val}")
    if _number(labels, "prev_stage1_48h") < _number(labels, "prev_stage1_24h"):
        failures.append("stage1 48h prevalence is below stage1 24h prevalence")
    if _number(labels, "prev_stage2_48h") < _number(labels, "prev_stage2_24h"):
        failures.append("stage2 48h prevalence is below stage2 24h prevalence")
    if _number(labels, "prev_stage2_24h") > _number(labels, "prev_stage1_24h"):
        failures.append("stage2 24h prevalence exceeds stage1 24h prevalence")
    if _number(labels, "prev_stage2_48h") > _number(labels, "prev_stage1_48h"):
        failures.append("stage2 48h prevalence exceeds stage1 48h prevalence")

    leakage = _row(tables, "qa.leakage_check")
    if _number(leakage, "n_future_feature_rows") != 0:
        failures.append("rolling features include events after landmark_time")
    if _number(leakage, "n_before_window_feature_rows") != 0:
        failures.append("rolling features include events before their lookback window")

    baseline = _row(tables, "qa.cohort_baseline_coverage")
    if _number(baseline, "n_stays") != _number(cohort, "n_stays"):
        failures.append("cohort baseline QA row count does not match cohort size")

    if failures:
        msg = "QA invariant failure(s): " + "; ".join(failures)
        logger.error(msg)
        raise AssertionError(msg)
    logger.info("QA invariants passed")


def _row(tables: dict[str, pd.DataFrame], name: str) -> dict[str, Any]:
    if name not in tables:
        raise AssertionError(f"missing QA table: {name}")
    if tables[name].empty:
        raise AssertionError(f"QA table is empty: {name}")
    return tables[name].iloc[0].to_dict()


def _number(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if pd.isna(value):
        return 0.0
    return float(value)
