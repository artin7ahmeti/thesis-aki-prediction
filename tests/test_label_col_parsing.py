"""Task-name -> label-column parsing (evaluate module)."""

import pytest

from aki.eval.evaluate import _label_col_from_task


@pytest.mark.parametrize("task, expected", [
    ("aki_stage1_24h", "y_stage1_24h"),
    ("aki_stage1_48h", "y_stage1_48h"),
    ("aki_stage2_24h", "y_stage2_24h"),
    ("aki_stage2_48h", "y_stage2_48h"),
])
def test_label_col_roundtrip(task, expected):
    assert _label_col_from_task(task) == expected
