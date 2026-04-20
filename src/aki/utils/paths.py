"""Canonical project paths.

Resolves absolute paths for every directory referenced by the pipeline,
starting from the importable package location.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class _Paths:
    root: Path = PROJECT_ROOT
    configs: Path = PROJECT_ROOT / "configs"
    sql: Path = PROJECT_ROOT / "sql"
    data: Path = PROJECT_ROOT / "data"
    raw: Path = PROJECT_ROOT / "data" / "raw"
    curated: Path = PROJECT_ROOT / "data" / "curated"
    duckdb: Path = PROJECT_ROOT / "data" / "mimic.duckdb"
    reports: Path = PROJECT_ROOT / "reports"
    figures: Path = PROJECT_ROOT / "reports" / "figures"
    tables: Path = PROJECT_ROOT / "reports" / "tables"
    artifacts: Path = PROJECT_ROOT / "reports" / "artifacts"
    mlruns: Path = PROJECT_ROOT / "mlruns"


paths = _Paths()


def ensure_output_dirs() -> None:
    """Create all output directories if they don't already exist."""
    for p in (paths.curated, paths.figures, paths.tables, paths.artifacts):
        p.mkdir(parents=True, exist_ok=True)
