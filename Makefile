# AKI Prediction Thesis Makefile
# Convenience targets for the full reproducible pipeline.
# Requires: `pip install -e ".[all]"` then `pre-commit install`.

.PHONY: help install install-dev lint format typecheck test test-all \
        stage cohort labels features qa inspect-db inspect-landmarks train tune minimal drift \
        evaluate explain report pipeline \
        clean-cache clean-reports clean-all

help:
	@echo "Setup:"
	@echo "  install        Install package (runtime deps only)"
	@echo "  install-dev    Install with dev/test/notebook/sequence extras"
	@echo ""
	@echo "Code quality:"
	@echo "  lint           Run ruff linter"
	@echo "  format         Auto-format with ruff"
	@echo "  typecheck      Run mypy"
	@echo "  test           Run pytest (fast unit tests)"
	@echo "  test-all       Run all tests including slow/integration"
	@echo ""
	@echo "Pipeline (run in order):"
	@echo "  stage          Stage raw MIMIC CSVs into DuckDB"
	@echo "  cohort         Build cohort and landmark table"
	@echo "  labels         Compute KDIGO AKI labels at landmarks"
	@echo "  features       Build rolling-window feature matrix"
	@echo "  qa             Run QA checks on curated data"
	@echo "  inspect-db     Read-only DB summary + table preview"
	@echo "  inspect-landmarks Compact landmark-specific inspection views"
	@echo "  tune           Optuna HPO for every (task x family x model)"
	@echo "  train          Train EBM / sparse LR / LightGBM (reads tuned params if cached)"
	@echo "  minimal        Derive minimal feature family from EBM importance"
	@echo "  drift          Train/val/test prevalence + SMD drift report"
	@echo "  evaluate       Metrics, calibration, decision curves, fairness, bootstrap CIs"
	@echo "  explain        Global importance, EBM shapes, SHAP, scorecard, patient trace"
	@echo "  report         Aggregate per-model results into thesis-ready tables"
	@echo "  pipeline       Full pipeline end-to-end (stage -> report)"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean-cache    Remove __pycache__, .pytest_cache, etc."
	@echo "  clean-reports  Remove generated figures/tables/artifacts"
	@echo "  clean-all      All of the above + remove DuckDB files"

# Setup
install:
	pip install -e .

install-dev:
	pip install -e ".[all]"
	pre-commit install

# Code quality
lint:
	ruff check src tests

format:
	ruff format src tests
	ruff check --fix src tests

typecheck:
	mypy src

test:
	pytest -m "not slow and not integration"

test-all:
	pytest

# Pipeline steps
stage:
	aki stage

cohort:
	aki cohort

labels:
	aki labels

features:
	aki features

qa:
	aki qa

inspect-db:
	aki inspect-db

inspect-landmarks:
	aki inspect-landmarks

tune:
	aki train --tune --n-trials $(N_TRIALS)

train:
	aki train

minimal:
	aki minimal

drift:
	aki drift

evaluate:
	aki evaluate

explain:
	aki explain

report:
	aki report

# Full end-to-end pipeline. Delegates to the CLI so tune + minimal
# are orchestrated in one place. Override: `make pipeline TUNE=1 N_TRIALS=200`.
pipeline:
	aki pipeline $(if $(filter 1 true yes,$(TUNE)),--tune,) --n-trials $(N_TRIALS)

# Shortcut used by SLURM preprocess job (no training)
preprocess: stage cohort labels features qa drift

# Default trial budget for `make tune` / `make pipeline TUNE=1`
N_TRIALS ?= 120

# Cleanup 
clean-cache:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .ruff_cache .mypy_cache .pytest_cache

clean-reports:
	rm -rf reports/figures/* reports/tables/* reports/artifacts/*

clean-all: clean-cache clean-reports
	rm -f data/mimic.duckdb data/mimic.duckdb.wal
	rm -rf mlruns mlartifacts
