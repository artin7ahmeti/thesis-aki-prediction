"""Helpers for subset-aware evaluate / explain / report passes."""

from __future__ import annotations

import re
from pathlib import Path


def matches_selector(
    *,
    task: str,
    family: str,
    model: str,
    tasks: list[str] | None = None,
    families: list[str] | None = None,
    models: list[str] | None = None,
) -> bool:
    """Return True when a model triple matches the optional filters."""
    if tasks and task not in tasks:
        return False
    if families and family not in families:
        return False
    if models and model not in models:
        return False
    return True


def artifact_triple_from_path(path: Path) -> tuple[str, str, str]:
    """Parse ``task__family__model`` from an artifact/per-model path."""
    stem = path.stem if path.suffix else path.name
    parts = stem.split("__")
    if len(parts) != 3:
        raise ValueError(f"Expected task__family__model tag, got {stem!r}")
    return parts[0], parts[1], parts[2]


def output_path(base: Path, output_tag: str | None = None) -> Path:
    """Return ``base`` or ``base`` with a sanitized ``__tag`` suffix."""
    if not output_tag:
        return base
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", output_tag.strip())
    safe = safe.strip("._-") or "subset"
    return base.with_name(f"{base.stem}__{safe}{base.suffix}")
