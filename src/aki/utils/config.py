"""Configuration loading.

All tunable parameters live in ``configs/*.yaml``. This module loads them
into a single ``Config`` namespace and exposes a ``config_hash`` that is
attached to MLflow runs for reproducibility.

String values in the YAML support ``${VAR}`` and ``${VAR:-default}``
expansion so paths can be redirected per host (local dev vs. Komondor)
without editing the config.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from aki.utils.paths import paths

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in string values."""
    if isinstance(value, str):
        def repl(match: re.Match) -> str:
            var, default = match.group(1), match.group(2)
            return os.environ.get(var, default if default is not None else "")
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return _expand_env(yaml.safe_load(f) or {})


@dataclass(frozen=True)
class Config:
    """Aggregated read-only view of every configs/*.yaml file."""

    project:  dict[str, Any]
    data:     dict[str, Any]
    cohort:   dict[str, Any]
    features: dict[str, Any]
    models:   dict[str, Any]
    eval:     dict[str, Any]

    @property
    def random_seed(self) -> int:
        return int(self.project.get("random_seed", 42))

    @property
    def raw_dir(self) -> Path:
        p = Path(self.project["paths"]["raw_dir"])
        return p if p.is_absolute() else paths.root / p

    @property
    def duckdb_path(self) -> Path:
        p = Path(self.project["paths"]["duckdb_path"])
        return p if p.is_absolute() else paths.root / p

    @property
    def curated_dir(self) -> Path:
        p = Path(self.project["paths"]["curated_dir"])
        return p if p.is_absolute() else paths.root / p

    def config_hash(self) -> str:
        """Stable SHA-256 hash of every config dict (for MLflow/audit trails)."""
        blob = json.dumps(
            {
                "project": self.project,
                "data": self.data,
                "cohort": self.cohort,
                "features": self.features,
                "models": self.models,
                "eval": self.eval,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_configs(configs_dir: Path | None = None) -> Config:
    """Load all six YAML configs and return them as a single ``Config``."""
    d = configs_dir or paths.configs
    return Config(
        project=_load_yaml(d / "project.yaml"),
        data=_load_yaml(d / "data.yaml"),
        cohort=_load_yaml(d / "cohort.yaml"),
        features=_load_yaml(d / "features.yaml"),
        models=_load_yaml(d / "models.yaml"),
        eval=_load_yaml(d / "eval.yaml"),
    )
