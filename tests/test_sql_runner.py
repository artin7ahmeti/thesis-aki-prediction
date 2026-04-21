"""SQL runner parsing edge cases."""

import duckdb

from aki.data.db import run_sql_file, split_sql_statements


def test_split_sql_ignores_semicolons_in_comments_and_strings():
    sql = """
    -- Comment with a semicolon; this must not become a statement.
    CREATE TABLE t AS SELECT 'a;b' AS value;
    -- Another comment;
    INSERT INTO t VALUES ('c'';d');
    """

    statements = split_sql_statements(sql)

    assert statements == [
        "CREATE TABLE t AS SELECT 'a;b' AS value",
        "INSERT INTO t VALUES ('c'';d')",
    ]


def test_run_sql_file_handles_comment_semicolons(tmp_path):
    sql_path = tmp_path / "example.sql"
    sql_path.write_text(
        """
        -- The old runner failed here; comment semicolon split this line.
        CREATE TABLE t AS SELECT 1 AS x;
        INSERT INTO t VALUES (2);
        """,
        encoding="utf-8",
    )

    conn = duckdb.connect(":memory:")
    run_sql_file(conn, sql_path)

    assert conn.execute("SELECT SUM(x) FROM t").fetchone()[0] == 3
