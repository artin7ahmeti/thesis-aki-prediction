#!/usr/bin/env bash
# Komondor: venv bootstrap.
# Creates $AKI_VENV from cray-python/3.11.7 and installs the package in editable
# mode with dev tools. Sequence-model extras are intentionally not installed
# for the tabular smoke/full pipeline.
# Re-run any time pyproject.toml changes.

#SBATCH --job-name=aki-setup
#SBATCH --account=pr_2026
#SBATCH --partition=cpu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2000

set -euo pipefail

# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/slurm/komondor.env"

cd "$AKI_PROJECT_DIR"
mkdir -p "$(dirname "$AKI_VENV")" logs

module purge
module load "$AKI_PY_MODULE"

echo "[setup] python: $(which python)  version: $(python --version)"
echo "[setup] creating isolated venv at $AKI_VENV"

python -m venv --clear "$AKI_VENV"
# shellcheck disable=SC1091
source "$AKI_VENV/bin/activate"

python -m pip install --upgrade pip wheel setuptools

# Editable install + dev tools. --no-cache-dir keeps the venv inode footprint down.
python -m pip install --no-cache-dir -e ".[dev]"

python - <<'PY'
import scipy
import scipy.sparse

import aki
import duckdb
import interpret
import lightgbm
import mlflow
import optuna
import shap

assert hasattr(scipy.sparse, "sparray"), scipy.__version__
print("imports ok")
PY

echo "[setup] done — venv ready at $AKI_VENV"
