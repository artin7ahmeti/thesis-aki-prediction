"""DuckDB session and SQL-file runner.

Design choices
* One database file at ``data/mimic.duckdb`` holds views over the raw
  CSV.gz files and materialized concept/cohort/label/feature tables.
* SQL files live in ``sql/`` as plain-text templates with
  ``{{variable}}`` placeholders. Substitution is deliberately simple
  (no Jinja) — parameters come from YAML configs.
* A context-manager session guarantees the connection is closed even
  if a query fails.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import duckdb
from loguru import logger

from aki.utils.paths import paths


class DuckDBSession:
    """Thin wrapper around ``duckdb.connect`` with memory/threads settings."""

    def __init__(
        self,
        db_path: Path | None = None,
        memory_limit: str = "16GB",
        threads: int = 8,
        temp_directory: Path | None = None,
        read_only: bool = False,
    ):
        self.db_path = db_path or paths.duckdb
        self.memory_limit = memory_limit
        self.threads = threads
        self.temp_directory = temp_directory
        self.read_only = read_only
        self._conn: duckdb.DuckDBPyConnection | None = None

    def __enter__(self) -> duckdb.DuckDBPyConnection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.db_path), read_only=self.read_only)
        self._conn.execute(f"SET memory_limit='{self.memory_limit}'")
        self._conn.execute(f"SET threads={self.threads}")
        if self.temp_directory is not None:
            self.temp_directory.mkdir(parents=True, exist_ok=True)
            self._conn.execute(f"SET temp_directory='{self.temp_directory}'")
        return self._conn

    def __exit__(self, *exc: Any) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

# SQL template loading

_VAR_PATTERN = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def render_sql(sql: str, params: dict[str, Any]) -> str:
    """Substitute ``{{name}}`` placeholders using ``params``.

    Unknown placeholders raise ``KeyError`` — fail fast rather than
    silently leaving ``{{x}}`` literals in the executed SQL.
    """

    def _sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in params:
            raise KeyError(f"SQL placeholder {{{{{key}}}}} has no value")
        value = params[key]
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, (list, tuple)):
            return ", ".join(f"'{v}'" for v in value)
        return str(value)

    return _VAR_PATTERN.sub(_sub, sql)


def _strip_line_comments(sql: str) -> str:
    """Remove ``--`` comments without touching quoted string literals."""
    out: list[str] = []
    in_single = False
    in_double = False
    i = 0
    while i < len(sql):
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < len(sql) else ""

        if in_single:
            out.append(ch)
            if ch == "'" and nxt == "'":
                out.append(nxt)
                i += 2
                continue
            if ch == "'":
                in_single = False
            i += 1
            continue

        if in_double:
            out.append(ch)
            if ch == '"' and nxt == '"':
                out.append(nxt)
                i += 2
                continue
            if ch == '"':
                in_double = False
            i += 1
            continue

        if ch == "'":
            in_single = True
            out.append(ch)
        elif ch == '"':
            in_double = True
            out.append(ch)
        elif ch == "-" and nxt == "-":
            while i < len(sql) and sql[i] not in "\r\n":
                i += 1
            continue
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def split_sql_statements(sql: str) -> list[str]:
    """Split SQL into statements, ignoring semicolons in comments/strings."""
    sql = _strip_line_comments(sql)
    statements: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    i = 0
    while i < len(sql):
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < len(sql) else ""

        if in_single:
            current.append(ch)
            if ch == "'" and nxt == "'":
                current.append(nxt)
                i += 2
                continue
            if ch == "'":
                in_single = False
            i += 1
            continue

        if in_double:
            current.append(ch)
            if ch == '"' and nxt == '"':
                current.append(nxt)
                i += 2
                continue
            if ch == '"':
                in_double = False
            i += 1
            continue

        if ch == "'":
            in_single = True
            current.append(ch)
        elif ch == '"':
            in_double = True
            current.append(ch)
        elif ch == ";":
            stmt = "".join(current).strip()
            if stmt:
                statements.append(stmt)
            current = []
        else:
            current.append(ch)
        i += 1

    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return statements


def run_sql_file(
    conn: duckdb.DuckDBPyConnection,
    sql_path: Path,
    params: dict[str, Any] | None = None,
) -> None:
    """Read, render, and execute a SQL file.

    DuckDB's ``execute`` accepts multi-statement strings — we split on
    semicolons at end-of-line to get per-statement logging.
    """
    try:
        log_path = sql_path.relative_to(paths.root)
    except ValueError:
        log_path = sql_path
    logger.info(f"SQL: {log_path}")
    text = sql_path.read_text(encoding="utf-8")
    if params:
        text = render_sql(text, params)

    statements = split_sql_statements(text)
    for i, stmt in enumerate(statements):
        try:
            conn.execute(stmt)
        except Exception as e:
            snippet = stmt[:200].replace("\n", " ")
            logger.error(f"  statement {i} failed: {snippet!r}")
            raise RuntimeError(f"{sql_path}: statement {i} failed") from e
    logger.debug(f"  {len(statements)} statement(s) executed")
