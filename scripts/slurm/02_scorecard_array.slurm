#!/usr/bin/env bash
# Komondor: sparse scorecard jobs for small, clinically readable families.
#
# Scorecard fitting does nested patient-grouped CV over the L1 path and is much
# slower on wide matrices. Keep this array deliberately narrow; use EBM
# importance plus 03_evaluate.slurm for the final minimal scorecard.
#
# Default: 4 tasks x vitals_only = 4 array slots.

#SBATCH --job-name=aki-score
#SBATCH --account=pr_2026
#SBATCH --partition=cpu
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000
#SBATCH --array=0-3%2

# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/slurm/komondor.env"
aki_activate

TASKS=(aki_stage1_24h aki_stage1_48h aki_stage2_24h aki_stage2_48h)
FAMILIES=(vitals_only)

N_F=${#FAMILIES[@]}

i=${SLURM_ARRAY_TASK_ID}
t_i=$(( i / N_F ))
f_i=$(( i % N_F ))

TASK=${TASKS[$t_i]}
FAMILY=${FAMILIES[$f_i]}

echo "[scorecard] array=$i  task=$TASK  family=$FAMILY  tune=${SCORECARD_TUNE:-false}"

if [[ "${SCORECARD_TUNE:-false}" == "true" ]]; then
    python scripts/run_single.py \
        --task    "$TASK" \
        --family  "$FAMILY" \
        --model   scorecard \
        --tune --n-trials "${N_TRIALS:-1}"
else
    python scripts/run_single.py \
        --task    "$TASK" \
        --family  "$FAMILY" \
        --model   scorecard
fi

echo "[scorecard] done: ${TASK}__${FAMILY}__scorecard"
