"""MLflow helpers.

Wraps ``mlflow.start_run`` with project-standard tags (config hash,
git commit, python version) so every run is reproducible.
"""

from __future__ import annotations

import platform
import subprocess
from contextlib import contextmanager
from typing import Any

import mlflow

from aki.utils.config import Config
from aki.utils.paths import paths


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=paths.root, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def init_mlflow(config: Config, experiment: str) -> None:
    """Set the tracking URI and active experiment."""
    mlflow.set_tracking_uri(config.project["paths"]["mlflow_uri"])
    mlflow.set_experiment(experiment)


@contextmanager
def run(config: Config, run_name: str, tags: dict[str, Any] | None = None):
    """Context manager wrapping ``mlflow.start_run`` with standard metadata."""
    std_tags = {
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "config_hash": config.config_hash(),
    }
    if tags:
        std_tags.update(tags)
    with mlflow.start_run(run_name=run_name, tags=std_tags) as r:
        yield r
