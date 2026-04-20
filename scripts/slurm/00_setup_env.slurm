#!/usr/bin/env bash
# Komondor: venv bootstrap.
# Creates $AKI_VENV from cray-python/3.11.7 and installs the package in editable 
# mode with all extras.
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
source "$(dirname "${BASH_SOURCE[0]}")/komondor.env"

cd "$AKI_PROJECT_DIR"
mkdir -p "$(dirname "$AKI_VENV")" logs

module purge
module load "$AKI_PY_MODULE"

echo "[setup] python: $(which python)  version: $(python --version)"
echo "[setup] creating venv at $AKI_VENV"

python -m venv --system-site-packages "$AKI_VENV"
# shellcheck disable=SC1091
source "$AKI_VENV/bin/activate"

python -m pip install --upgrade pip wheel setuptools

# Editable install + all extras (notebooks/dev/sequence).
# --no-cache-dir keeps the venv inode footprint down.
python -m pip install --no-cache-dir -e ".[all]"

python -c "import aki, duckdb, lightgbm, interpret, optuna, shap, mlflow; print('imports ok')"

echo "[setup] done — venv ready at $AKI_VENV"
