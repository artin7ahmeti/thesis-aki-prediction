# Komondor HPC: launch recipes

All scripts source `komondor.env`, which pins the billing account
(`pr_2026`), the Python module (`cray-python/3.11.7`), and the storage
layout. Override by exporting the relevant `AKI_*` variable before
`sbatch`.

## Storage layout

| Path | Purpose | Quota |
|---|---|---|
| `/project/pr_2026/aki/` | repo, venv, MLflow, reports | 250 GB |
| `/project/pr_2026/aki/data/raw/` | raw MIMIC-IV CSVs | shared with above |
| `/scratch/pr_2026/aki/` | DuckDB file + curated parquet (hot I/O) | 62.5 GB |

Hot I/O lives on scratch for speed; durable artifacts on project for persistence.

## One-time deploy (login node)

```bash
ssh <your_user>@komondor.hpc.dkf.hu

# Clone the repo under /project (not /home - only 20 GB there).
mkdir -p /project/pr_2026
cd /project/pr_2026
git clone https://github.com/artin7ahmeti/thesis-aki-prediction.git aki
cd aki

# Upload raw MIMIC-IV CSVs to /project/pr_2026/aki/data/raw/
# (use scp/rsync from your workstation)

# Build the venv.
sbatch scripts/slurm/00_setup_env.slurm
# Wait for it to finish: `squeue -u $USER`, then check logs/aki-setup-*.out
```

## Smoke test first

```bash
cd /project/pr_2026/aki

# 1. preprocess
prep=$(sbatch --parsable scripts/slurm/01_preprocess.slurm)

# 2. smoke test - 10 Optuna trials on one (task, family, model)
sbatch --dependency=afterok:$prep scripts/slurm/smoke_test.slurm
```

Default combo is `aki_stage1_24h + combined + lightgbm` with 10 trials.
Override via env:

```bash
SMOKE_MODEL=ebm SMOKE_TRIALS=5 \
    sbatch --dependency=afterok:$prep scripts/slurm/smoke_test.slurm
```

When it finishes (~30-60 min), check:

```bash
cat logs/aki-smoke-*.out | tail -30
cat reports/tables/final_results.csv        # should show one row with AUROC > 0.75
ls reports/artifacts/models/                # should list one .joblib
ls reports/artifacts/tune/                  # should list one .json with best params
```

## Run the full pipeline

```bash
cd /project/pr_2026/aki

# 1. Preprocess: stage -> cohort -> labels -> features -> qa -> drift
prep_id=$(sbatch --parsable scripts/slurm/01_preprocess.slurm)

# 2. Optuna HPO sweep (48 array slots, 12 concurrent)
tune_id=$(sbatch --parsable \
    --dependency=afterok:$prep_id \
    scripts/slurm/02_tune_array.slurm)

# 3. Minimal family -> minimal train -> evaluate -> explain -> report
sbatch --dependency=afterok:$tune_id scripts/slurm/03_evaluate.slurm
```

Override the trial budget:

```bash
sbatch --export=ALL,N_TRIALS=200 scripts/slurm/02_tune_array.slurm
```

## Monitoring

```bash
squeue -u $USER                       # your queued / running jobs
squeue -j <jobid>                     # one job
scontrol show job <jobid>             # full detail
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
tail -f logs/aki-tune-<jobid>_0.out   # stream a running array slot
scancel <jobid>                       # kill a job (or whole array)
sbalance                              # CPU-hour budget left
squota                                # storage usage
```

## Updating the deployment

```bash
cd /project/pr_2026/aki
git pull
# Re-run setup only if pyproject.toml changed:
sbatch scripts/slurm/00_setup_env.slurm
```