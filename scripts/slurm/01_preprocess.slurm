#!/usr/bin/env bash
# Komondor: stage MIMIC-IV -> cohort -> labels -> features -> QA -> drift.
# Single-node, high-memory; DuckDB is happiest on one big box.
# Raw CSVs are read from /project/pr_2026/aki/data/raw/ and the DuckDB
# file + curated parquet land on /scratch/pr_2026/aki/.

#SBATCH --job-name=aki-prep
#SBATCH --account=pr_2026
#SBATCH --partition=cpu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2000

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/komondor.env"
aki_activate

aki stage
aki cohort
aki labels
aki features
aki qa
aki drift --family combined

echo "[prep] done — curated parquet at $AKI_CURATED_DIR"
