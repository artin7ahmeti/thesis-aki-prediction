from __future__ import annotations

from dataclasses import replace

import pandas as pd

from aki.eval import aggregate
from aki.utils.config import load_configs
from aki.utils.paths import paths as real_paths


def _bootstrap_ci_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "auroc", "point": 0.71, "ci_lower": 0.68, "ci_upper": 0.74},
            {"metric": "auprc", "point": 0.31, "ci_lower": 0.28, "ci_upper": 0.34},
            {"metric": "brier", "point": 0.12, "ci_lower": 0.11, "ci_upper": 0.13},
            {"metric": "calibration_slope", "point": 0.98, "ci_lower": 0.90, "ci_upper": 1.05},
            {"metric": "calibration_intercept", "point": -0.05, "ci_lower": -0.10, "ci_upper": 0.00},
            {"metric": "ece", "point": 0.03, "ci_lower": 0.02, "ci_upper": 0.04},
            {"metric": "sensitivity_at_spec_90", "point": 0.25, "ci_lower": 0.20, "ci_upper": 0.30},
            {"metric": "specificity_at_sens_90", "point": 0.18, "ci_lower": 0.15, "ci_upper": 0.21},
        ]
    )


def test_build_final_results_supports_subset_filters_and_output_tag(tmp_path, monkeypatch):
    tables = tmp_path / "tables"
    per_model = tables / "per_model"
    for tag in [
        "aki_stage1_24h__scorecard_primary__scorecard",
        "aki_stage1_48h__scorecard_primary__scorecard",
        "aki_stage1_24h__combined__ebm",
    ]:
        out = per_model / tag
        out.mkdir(parents=True)
        _bootstrap_ci_frame().to_csv(out / "bootstrap_ci.csv", index=False)

    monkeypatch.setattr(
        aggregate,
        "paths",
        replace(real_paths, tables=tables),
    )

    cfg = load_configs()
    df = aggregate.build_final_results(
        cfg,
        tasks=["aki_stage1_24h"],
        families=["scorecard_primary"],
        models=["scorecard"],
        output_tag="bedside_scorecards",
    )

    assert list(df["task"].unique()) == ["aki_stage1_24h"]
    assert list(df["family"].unique()) == ["scorecard_primary"]
    assert list(df["model"].unique()) == ["scorecard"]
    assert (tables / "final_results__bedside_scorecards.csv").exists()
    assert (tables / "final_results__bedside_scorecards.md").exists()
    assert (tables / "best_per_task__bedside_scorecards.csv").exists()
