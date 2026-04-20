"""CLI imports + help text sanity."""

from typer.testing import CliRunner

from aki.cli.main import app

runner = CliRunner()


def test_cli_help_lists_all_stages():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("stage", "cohort", "labels", "features", "qa",
                "train", "tune", "minimal", "drift",
                "evaluate", "explain", "report", "pipeline"):
        assert cmd in result.output


def test_cli_train_help_has_tune_flag():
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "--tune" in result.output
    assert "--n-trials" in result.output
