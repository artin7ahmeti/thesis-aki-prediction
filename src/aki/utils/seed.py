"""Deterministic-seed utilities.

Call :func:`seed_everything` once at the start of every training script
(or via the CLI callback) to seed Python, NumPy, scikit-learn's default
RNG, LightGBM, PyTorch (if installed), and any child processes.
"""

from __future__ import annotations

import os
import random

import numpy as np
from loguru import logger


def seed_everything(seed: int = 42) -> None:
    """Seed every RNG we can reach."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")
    random.seed(seed)
    np.random.seed(seed)

    # sklearn picks up np.random; nothing extra needed.

    try:  # pragma: no cover - optional
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:  # pragma: no cover - optional
        import lightgbm

        _ = lightgbm  # LightGBM picks up random_state per-call
    except ImportError:
        pass

    logger.debug(f"seed_everything({seed})")
