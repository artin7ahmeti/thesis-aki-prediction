"""Single (task, family, model) driver for HPC array jobs.

Designed to be invoked as::

    python scripts/run_single.py \
        --task aki_stage1_24h \
        --family combined \
        --model lightgbm \
        --tune --n-trials 120

Each invocation runs independently (separate MLflow run, separate tuned
params file) so an HPC can parallelize across the full (task × family ×
model) grid with one process per combination.
"""

from __future__ import annotations

import argparse

from aki.models.train import train_all
from aki.utils.config import load_configs
from aki.utils.logging import configure_logging
from aki.utils.paths import ensure_output_dirs
from aki.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task",    required=True, help="e.g. aki_stage1_24h")
    p.add_argument("--family",  required=True, help="e.g. combined")
    p.add_argument("--model",   required=True, choices=["ebm", "scorecard", "lightgbm"])
    p.add_argument("--tune",    action="store_true")
    p.add_argument("--n-trials", type=int, default=60)
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    ensure_output_dirs()
    seed_everything(args.seed)

    cfg = load_configs()
    train_all(
        cfg,
        tune=args.tune,
        n_trials=args.n_trials,
        families=[args.family],
        tasks=[args.task],
        models=[args.model],
    )


if __name__ == "__main__":
    main()
