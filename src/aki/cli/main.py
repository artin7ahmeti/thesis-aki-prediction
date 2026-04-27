"""``aki`` CLI - Typer app with one command per pipeline stage.

Install via ``pip install -e .`` (entry point declared in pyproject.toml)
then run e.g. ``aki pipeline``.
"""

from __future__ import annotations

from pathlib import Path

import typer

from aki.data.db import DuckDBSession, run_sql_file
from aki.utils.config import Config, load_configs
from aki.utils.logging import configure_logging
from aki.utils.paths import ensure_output_dirs, paths
from aki.utils.seed import seed_everything

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "AKI prediction pipeline - stages are run in this order: "
        "stage -> cohort -> labels -> features -> train -> evaluate -> explain."
    ),
)


def _duckdb_session(cfg: Config, read_only: bool = False) -> DuckDBSession:
    """Create a DuckDB session using YAML/env-resolved Komondor paths."""
    db_cfg = cfg.data.get("duckdb", {})
    temp_raw = db_cfg.get("temp_directory")
    temp_dir = Path(temp_raw) if temp_raw else None
    if temp_dir is not None and not temp_dir.is_absolute():
        temp_dir = paths.root / temp_dir

    return DuckDBSession(
        db_path=cfg.duckdb_path,
        memory_limit=db_cfg.get("memory_limit", "16GB"),
        threads=int(db_cfg.get("threads", 8)),
        temp_directory=temp_dir,
        read_only=read_only,
    )


@app.callback()
def _bootstrap() -> None:
    """Configure logging + seed + ensure output dirs exist before every command."""
    configure_logging()
    ensure_output_dirs()
    try:
        cfg = load_configs()
        seed_everything(cfg.random_seed)
    except Exception:  # pragma: no cover - config missing is caught per-command
        pass


# Data-pipeline stages


@app.command("stage")
def cmd_stage() -> None:
    """Create DuckDB views over raw MIMIC CSV.gz files + write manifest."""
    from aki.data.manifest import write_manifest

    cfg = load_configs()
    write_manifest(cfg.raw_dir)

    params = {"raw_dir": str(cfg.raw_dir).replace("\\", "/")}
    with _duckdb_session(cfg) as conn:
        run_sql_file(conn, paths.sql / "staging" / "01_schemas.sql")
        run_sql_file(conn, paths.sql / "staging" / "02_stage_hosp.sql", params=params)
        run_sql_file(conn, paths.sql / "staging" / "03_stage_icu.sql", params=params)


@app.command("cohort")
def cmd_cohort() -> None:
    """Build KDIGO onset concepts, cohort.cohort, and cohort.landmarks."""
    from aki.cohort.build import build_cohort_and_landmarks
    from aki.labels.kdigo import build_kdigo_concepts

    cfg = load_configs()
    with _duckdb_session(cfg) as conn:
        # Cohort SQL excludes prevalent AKI and landmark SQL excludes times after onset.
        build_kdigo_concepts(conn, cfg)
        build_cohort_and_landmarks(conn, cfg)


@app.command("labels")
def cmd_labels() -> None:
    """Run landmark label SQL and export curated parquet."""
    from aki.labels.kdigo import build_landmark_labels

    cfg = load_configs()
    with _duckdb_session(cfg) as conn:
        build_landmark_labels(conn, cfg)


@app.command("features")
def cmd_features() -> None:
    """Build per-family feature matrices (one parquet per family)."""
    from aki.features.engineer import build_features

    cfg = load_configs()
    with _duckdb_session(cfg) as conn:
        build_features(conn, cfg)


@app.command("qa")
def cmd_qa() -> None:
    """Run QA views and log row counts."""
    from aki.qa.checks import run_qa_checks

    cfg = load_configs()
    with _duckdb_session(cfg) as conn:
        run_qa_checks(conn, cfg)


@app.command("inspect-db")
def cmd_inspect_db(
    table: str = typer.Option("cohort.landmarks", help="Schema-qualified table to preview."),
    limit: int = typer.Option(20, help="Number of rows to preview."),
    summary_only: bool = typer.Option(False, "--summary-only", help="Only print core table counts."),
) -> None:
    """Inspect the configured DuckDB database in read-only mode."""
    from aki.data.inspect import core_table_counts, path_status, preview_table

    cfg = load_configs()
    typer.echo(f"DuckDB: {cfg.duckdb_path} [{path_status(cfg.duckdb_path)}]")
    typer.echo(f"Curated: {cfg.curated_dir}")

    if not cfg.duckdb_path.exists():
        raise typer.Exit(code=1)

    with _duckdb_session(cfg, read_only=True) as conn:
        typer.echo("\nCore table counts:")
        for name, count in core_table_counts(conn):
            count_text = "missing" if count is None else f"{count:,}"
            typer.echo(f"  {name:<30} {count_text}")

        if summary_only:
            return

        typer.echo(f"\nPreview: {table} (limit {limit})")
        df = preview_table(conn, table=table, limit=limit)
        if df.empty:
            typer.echo("  [empty result]")
            return
        typer.echo(df.to_string(index=False))


@app.command("inspect-landmarks")
def cmd_inspect_landmarks(
    limit: int = typer.Option(20, help="Number of rows to preview."),
    mode: str = typer.Option(
        "compact",
        help="One of: compact, summary, per-stay.",
    ),
) -> None:
    """Human-friendly inspection views for the landmark dataset."""
    from aki.data.inspect import landmark_compact_preview, landmark_summary, landmarks_per_stay, path_status

    cfg = load_configs()
    typer.echo(f"DuckDB: {cfg.duckdb_path} [{path_status(cfg.duckdb_path)}]")
    if not cfg.duckdb_path.exists():
        raise typer.Exit(code=1)

    with _duckdb_session(cfg, read_only=True) as conn:
        if mode == "summary":
            typer.echo("\nLandmark summary:")
            typer.echo(landmark_summary(conn).to_string(index=False))
            return

        if mode == "per-stay":
            typer.echo(f"\nLandmarks per stay (top {limit}):")
            typer.echo(landmarks_per_stay(conn, limit=limit).to_string(index=False))
            return

        if mode != "compact":
            raise typer.BadParameter("mode must be one of: compact, summary, per-stay")

        typer.echo(f"\nCompact landmark preview (limit {limit}):")
        typer.echo(landmark_compact_preview(conn, limit=limit).to_string(index=False))


# Modeling stages


@app.command("train")
def cmd_train(
    tune: bool = typer.Option(False, help="Run Optuna HPO before fitting."),
    n_trials: int = typer.Option(60, help="Optuna trials per model."),
    family: str | None = typer.Option(None, help="Restrict to one feature family."),
    task: str | None = typer.Option(None, help="Restrict to one task (e.g. aki_stage1_24h)."),
    model: str | None = typer.Option(None, help="Restrict to one model name."),
) -> None:
    """Fit EBM + sparse-logistic + LightGBM across every (task, family)."""
    from aki.models.train import train_all

    cfg = load_configs()
    train_all(
        cfg,
        tune=tune,
        n_trials=n_trials,
        families=[family] if family else None,
        tasks=[task] if task else None,
        models=[model] if model else None,
    )


@app.command("tune")
def cmd_tune(
    n_trials: int = typer.Option(120, help="Optuna trials per model."),
    family: str | None = typer.Option(None),
    task: str | None = typer.Option(None),
    model: str | None = typer.Option(None),
) -> None:
    """Alias for ``aki train --tune`` - tunes + fits + logs."""
    cmd_train(tune=True, n_trials=n_trials, family=family, task=task, model=model)


@app.command("minimal")
def cmd_minimal(
    source_family: str = typer.Option("combined", help="Family to mine EBM importance from."),
) -> None:
    """Build the minimal feature family (<= k features) for the scorecard."""
    from aki.features.minimal import derive_minimal_family

    cfg = load_configs()
    derive_minimal_family(cfg, source_family=source_family)


@app.command("drift")
def cmd_drift(
    family: str = typer.Option("combined", help="Feature family to analyze."),
) -> None:
    """Prevalence + SMD drift across temporal splits."""
    from aki.eval.drift import compute_drift_report

    cfg = load_configs()
    compute_drift_report(cfg, family=family)


@app.command("evaluate")
def cmd_evaluate(
    task: list[str] | None = typer.Option(None, help="Repeat to restrict to one or more tasks."),
    family: list[str] | None = typer.Option(None, help="Repeat to restrict to one or more families."),
    model: list[str] | None = typer.Option(None, help="Repeat to restrict to one or more model names."),
    output_tag: str | None = typer.Option(None, help="Suffix for summary outputs when running a subset."),
) -> None:
    """Score every trained artifact on the held-out test split."""
    from aki.eval.evaluate import evaluate_all

    cfg = load_configs()
    evaluate_all(
        cfg,
        tasks=task,
        families=family,
        models=model,
        output_tag=output_tag,
    )


@app.command("explain")
def cmd_explain(
    task: list[str] | None = typer.Option(None, help="Repeat to restrict to one or more tasks."),
    family: list[str] | None = typer.Option(None, help="Repeat to restrict to one or more families."),
    model: list[str] | None = typer.Option(None, help="Repeat to restrict to one or more model names."),
) -> None:
    """Emit global-importance tables, shape plots, and scorecard artifacts."""
    from aki.explain.report import run_explanations

    cfg = load_configs()
    run_explanations(cfg, tasks=task, families=family, models=model)


@app.command("report")
def cmd_report(
    task: list[str] | None = typer.Option(None, help="Repeat to restrict to one or more tasks."),
    family: list[str] | None = typer.Option(None, help="Repeat to restrict to one or more families."),
    model: list[str] | None = typer.Option(None, help="Repeat to restrict to one or more model names."),
    output_tag: str | None = typer.Option(None, help="Suffix for report outputs when running a subset."),
) -> None:
    """Aggregate per-model tables into final thesis-ready results."""
    from aki.eval.aggregate import build_final_results

    cfg = load_configs()
    build_final_results(
        cfg,
        tasks=task,
        families=family,
        models=model,
        output_tag=output_tag,
    )


@app.command("refresh")
def cmd_refresh(
    task: list[str] | None = typer.Option(None, help="Repeat to restrict to one or more tasks."),
    family: list[str] | None = typer.Option(None, help="Repeat to restrict to one or more families."),
    model: list[str] | None = typer.Option(None, help="Repeat to restrict to one or more model names."),
    output_tag: str | None = typer.Option(None, help="Suffix for evaluate/report summary outputs."),
) -> None:
    """Run evaluate -> explain -> report, optionally on a filtered subset."""
    cmd_evaluate(task=task, family=family, model=model, output_tag=output_tag)
    cmd_explain(task=task, family=family, model=model)
    cmd_report(task=task, family=family, model=model, output_tag=output_tag)


# All-in-one


@app.command("pipeline")
def cmd_pipeline(
    skip_stage: bool = typer.Option(False, help="Skip the staging step."),
    tune: bool = typer.Option(False, help="Run Optuna HPO in the train step."),
    n_trials: int = typer.Option(60, help="Optuna trials per model (when --tune)."),
) -> None:
    """End-to-end: stage -> cohort -> labels -> features -> train -> minimal -> evaluate."""
    from aki.features.minimal import derive_minimal_family
    from aki.models.train import train_all

    cfg = load_configs()
    if not skip_stage:
        cmd_stage()
    cmd_cohort()
    cmd_labels()
    cmd_features()
    cmd_qa()
    cmd_drift(family="combined")

    train_all(cfg, tune=tune, n_trials=n_trials)

    # Derive minimal family then train on it using cached tuned params where possible.
    derive_minimal_family(cfg, source_family="combined")
    train_all(cfg, tune=False, families=["minimal"])

    cmd_evaluate()
    cmd_explain()
    cmd_report()


if __name__ == "__main__":
    app()
