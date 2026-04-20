#!/usr/bin/env bash
# Komondor: Optuna HPO array for every (task x family x model).
# 4 tasks x 4 families x 3 models = 48 array slots.
# Minimal family is trained separately by 03_evaluate.slurm.

#SBATCH --job-name=aki-tune
#SBATCH --account=pr_2026
#SBATCH --partition=cpu
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2000
#SBATCH --array=0-47%12

# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/slurm/komondor.env"
aki_activate

TASKS=(aki_stage1_24h aki_stage1_48h aki_stage2_24h aki_stage2_48h)
FAMILIES=(vitals_only labs_only combined renal_excluded)
MODELS=(ebm scorecard lightgbm)

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
    --tune --n-trials "${N_TRIALS:-80}"

echo "[tune] done: ${TASK}__${FAMILY}__${MODEL}"
