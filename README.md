# Interpretable ML for Early AKI Prediction (MIMIC-IV)

Master's thesis pipeline for predicting incident Acute Kidney Injury (AKI) in
ICU patients 24–48 hours ahead, using transparent models (EBM, sparse logistic
regression) on the credentialed MIMIC-IV v3.1 dataset.

## Quickstart

```bash
# 1. Install (needs Python 3.10+)
pip install -e ".[all]"
pre-commit install

# 2. Place MIMIC-IV data
#    data/raw/hosp/*.csv.gz
#    data/raw/icu/*.csv.gz

# 3. Run the pipeline
make pipeline
```

## Pipeline

| Step | Command            | Output                                         |
|------|--------------------|------------------------------------------------|
| 1    | `aki stage`        | `data/mimic.duckdb` with staged raw tables     |
| 2    | `aki cohort`       | Cohort + landmarks table (parquet)             |
| 3    | `aki labels`       | KDIGO AKI labels per landmark (parquet)        |
| 4    | `aki features`     | Rolling-window feature matrices (parquet)      |
| 5    | `aki train`        | EBM / sparse LR / LightGBM models -> MLflow     |
| 6    | `aki evaluate`     | Metrics, calibration, decision curves, fairness|
| 7    | `aki explain`      | Shape plots, scorecard, patient explanations   |

See `CLAUDE.md` for design decisions and `configs/` for all tunable parameters.

## Clinical framing

- **Task:** dynamic landmark prediction every 6 h during ICU stay
- **Horizons:** 24 h and 48 h
- **Labels:** KDIGO stage ≥ 1 (primary), ≥ 2 (secondary)
- **Baseline creatinine:** MIMIC-Code convention (lowest in 7 d prior to ICU admit)
- **Splits:** patient-level, temporal via `anchor_year_group`

## Data use

All MIMIC-IV data stays local. This repo never transmits patient data to
third-party services. See PhysioNet DUA for your obligations as a
credentialed user.
