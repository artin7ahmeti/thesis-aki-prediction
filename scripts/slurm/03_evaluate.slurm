#!/usr/bin/env bash

# Komondor: derive minimal family -> train on it -> evaluate ->
# explain -> aggregate final results. Runs after 02_tune_array.

#SBATCH --job-name=aki-eval
#SBATCH --account=pr_2026
#SBATCH --partition=cpu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2000

# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/slurm/komondor.env"
aki_activate

# Derive minimal feature family from the EBM importance on `combined`.
aki minimal

# Train EBM/scorecard/LightGBM on the minimal family using cached tuned params
# where available. EBM/LightGBM params come from the main array; scorecard uses
# the configured sparse-logistic path unless you have separately cached params.
python - <<'PY'
from aki.models.train import train_all
from aki.utils.config import load_configs
train_all(load_configs(), tune=False, families=["minimal"])
PY

aki evaluate
aki explain
aki report

echo "[eval] final results at $AKI_REPORTS_DIR/tables/"
