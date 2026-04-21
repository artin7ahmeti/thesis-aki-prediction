#!/usr/bin/env bash
# Komondor SMOKE TEST: validate the end-to-end pipeline
#
# Fits one (task x family x model), optionally with a tiny tuning budget,
# runs evaluate + report, and writes a final CSV you can sanity-check.
#
# Requires: 01_preprocess.slurm has already finished so curated
# parquet exists under $AKI_CURATED_DIR.
#SBATCH --job-name=aki-smoke
#SBATCH --account=pr_2026
#SBATCH --partition=cpu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2000

# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/slurm/komondor.env"
aki_activate

# Override by exporting before sbatch, e.g.:
#   SMOKE_MODEL=ebm sbatch scripts/slurm/smoke_test.slurm
SMOKE_TASK="${SMOKE_TASK:-aki_stage1_24h}"
SMOKE_FAMILY="${SMOKE_FAMILY:-combined}"
SMOKE_MODEL="${SMOKE_MODEL:-lightgbm}"
SMOKE_TUNE="${SMOKE_TUNE:-true}"
SMOKE_TRIALS="${SMOKE_TRIALS:-1}"

echo "[smoke] task=$SMOKE_TASK family=$SMOKE_FAMILY model=$SMOKE_MODEL tune=$SMOKE_TUNE trials=$SMOKE_TRIALS"

if [[ "$SMOKE_TUNE" == "true" ]]; then
    python scripts/run_single.py \
        --task    "$SMOKE_TASK" \
        --family  "$SMOKE_FAMILY" \
        --model   "$SMOKE_MODEL" \
        --tune --n-trials "$SMOKE_TRIALS"
else
    python scripts/run_single.py \
        --task    "$SMOKE_TASK" \
        --family  "$SMOKE_FAMILY" \
        --model   "$SMOKE_MODEL"
fi

# Evaluate just this one artifact on the held out test split + aggregate.
aki evaluate
aki report

echo "[smoke] done — inspect $AKI_REPORTS_DIR/tables/final_results.csv"
