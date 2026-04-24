from __future__ import annotations

from pathlib import Path

from aki.utils.subset import artifact_triple_from_path, matches_selector, output_path


def test_matches_selector_filters_task_family_and_model():
    assert matches_selector(
        task="aki_stage1_24h",
        family="scorecard_primary",
        model="scorecard",
        tasks=["aki_stage1_24h"],
        families=["scorecard_primary"],
        models=["scorecard"],
    )
    assert not matches_selector(
        task="aki_stage1_48h",
        family="scorecard_primary",
        model="scorecard",
        tasks=["aki_stage1_24h"],
    )
    assert not matches_selector(
        task="aki_stage1_24h",
        family="combined",
        model="scorecard",
        families=["scorecard_primary"],
    )
    assert not matches_selector(
        task="aki_stage1_24h",
        family="scorecard_primary",
        model="ebm",
        models=["scorecard"],
    )


def test_artifact_triple_from_path_parses_file_and_directory_names():
    assert artifact_triple_from_path(
        Path("aki_stage1_24h__scorecard_primary__scorecard.joblib")
    ) == ("aki_stage1_24h", "scorecard_primary", "scorecard")
    assert artifact_triple_from_path(
        Path("aki_stage1_24h__scorecard_primary__scorecard")
    ) == ("aki_stage1_24h", "scorecard_primary", "scorecard")


def test_output_path_adds_sanitized_suffix():
    out = output_path(Path("reports/tables/final_results.csv"), "bedside scorecards")
    assert out.as_posix().endswith("final_results__bedside_scorecards.csv")
