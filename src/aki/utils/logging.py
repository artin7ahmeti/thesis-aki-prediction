"""Logging setup via loguru.

Calling ``configure_logging()`` once at process start installs a console
handler with colored level-aware formatting and a rotating file handler
in ``reports/artifacts/``.
"""

from __future__ import annotations

import sys

from loguru import logger

from aki.utils.paths import paths

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Idempotent logger setup — safe to call many times."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level:<7}</level> | "
            "<cyan>{name}:{function}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    logger.add(
        paths.artifacts / "pipeline_{time:YYYYMMDD}.log",
        level="DEBUG",
        rotation="50 MB",
        retention="14 days",
    )
    _CONFIGURED = True
