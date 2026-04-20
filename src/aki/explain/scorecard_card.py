"""Printable scorecard artifact.

Converts a fitted sparse-logistic ``ModelArtifact`` into:
- a CSV of (feature, coefficient, points) rows,
- a Markdown table suitable for pasting into the thesis.

Points are coefficients rounded to the nearest integer on the *log-odds*
scale (then scaled so the smallest |coef| ~ 1), which is a standard
convention for clinical scorecards.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aki.models.base import ModelArtifact


def build_scorecard_artifact(
    art: ModelArtifact,
    out_dir: Path,
) -> dict[str, Path]:
    """Write ``scorecard.csv`` + ``scorecard.md`` into ``out_dir``."""
    if art.name != "scorecard":
        raise ValueError("build_scorecard_artifact requires a scorecard artifact")

    out_dir.mkdir(parents=True, exist_ok=True)
    clf = art.estimator.named_steps["clf"]
    coef = clf.coef_.ravel()
    intercept = float(clf.intercept_.ravel()[0])

    if np.abs(coef).max() == 0:
        scale = 1.0
    else:
        scale = 1.0 / np.abs(coef[coef != 0]).min()
    points = np.round(coef * scale).astype(int)

    table = pd.DataFrame({
        "feature":     art.feature_names,
        "coefficient": coef,
        "points":      points,
    }).sort_values("points", key=np.abs, ascending=False, ignore_index=True)

    csv_path = out_dir / "scorecard.csv"
    table.to_csv(csv_path, index=False)

    md_lines = [
        f"# Scorecard — {art.task} ({art.family})",
        "",
        f"- Intercept: **{intercept:+.3f}** (log-odds)",
        f"- Scaling: 1 point ≈ {1/scale:.3f} log-odds",
        "",
        "| Feature | Coefficient | Points |",
        "|---------|-------------:|-------:|",
    ]
    for _, r in table.iterrows():
        md_lines.append(f"| `{r['feature']}` | {r['coefficient']:+.4f} | {r['points']:+d} |")
    md_path = out_dir / "scorecard.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return {"csv": csv_path, "md": md_path}
