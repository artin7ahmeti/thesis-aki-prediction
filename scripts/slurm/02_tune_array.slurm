#!/usr/bin/env bash
# Komondor: Optuna HPO array for the main tabular comparison.
#
# This intentionally tunes only EBM and LightGBM across the full feature-family
# grid. The sparse scorecard is handled by 02_scorecard_array.slurm because wide
# L1-logistic CV is CPU-expensive and the clinically useful scorecard should be
# built from small/interpretable feature sets, not every 600-column family.
#
# 4 tasks x 4 families x 2 models = 32 array slots.
# Minimal family is derived and trained separately by 03_evaluate.slurm.

#SBATCH --job-name=aki-tune
#SBATCH --account=pr_2026
#SBATCH --partition=cpu
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2000
#SBATCH --array=0-31%8

# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/slurm/komondor.env"
aki_activate

TASKS=(aki_stage1_24h aki_stage1_48h aki_stage2_24h aki_stage2_48h)
FAMILIES=(vitals_only labs_only combined renal_excluded)
MODELS=(ebm lightgbm)

N_F=${#FAMILIES[@]}
N_M=${#MODELS[@]}

i=${SLURM_ARRAY_TASK_ID}
t_i=$(( i / (N_F * N_M) ))
f_i=$(( (i / N_M) % N_F ))
m_i=$(( i % N_M ))

TASK=${TASKS[$t_i]}
FAMILY=${FAMILIES[$f_i]}
MODEL=${MODELS[$m_i]}

echo "[tune] array=$i  task=$TASK  family=$FAMILY  model=$MODEL"

python scripts/run_single.py \
    --task    "$TASK" \
    --family  "$FAMILY" \
    --model   "$MODEL" \
    --tune --n-trials "${N_TRIALS:-20}"

echo "[tune] done: ${TASK}__${FAMILY}__${MODEL}"
