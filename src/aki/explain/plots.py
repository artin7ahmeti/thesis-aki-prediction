"""Matplotlib plots: EBM shape functions + reliability curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aki.models.base import ModelArtifact


def plot_ebm_shapes(
    art: ModelArtifact,
    out_dir: Path,
    top_k: int = 12,
) -> list[Path]:
    """One figure per term (shape function with confidence band)."""
    if art.name != "ebm":
        raise ValueError("plot_ebm_shapes requires an EBM artifact")

    out_dir.mkdir(parents=True, exist_ok=True)
    expl = art.estimator.explain_global()
    terms = art.estimator.term_names_
    scores = expl.data()["scores"]
    order = np.argsort(-np.abs(scores))[:top_k]

    files: list[Path] = []
    for i in order:
        d = expl.data(int(i))
        if d is None:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.5))
        names  = np.asarray(d.get("names", []))
        values = np.asarray(d.get("scores", []))
        lower  = np.asarray(d.get("lower_bounds", values))
        upper  = np.asarray(d.get("upper_bounds", values))

        if values.size == 0:
            plt.close(fig)
            continue

        x = np.arange(len(names)) if names.dtype.kind in ("U", "O") else names.astype(float)
        ax.plot(x, values, color="tab:blue")
        if lower.shape == values.shape and upper.shape == values.shape:
            ax.fill_between(x, lower, upper, alpha=0.2, color="tab:blue")
        ax.axhline(0.0, color="gray", linewidth=0.5)
        ax.set_title(terms[i])
        ax.set_ylabel("score contribution")
        ax.set_xlabel("feature value")
        fig.tight_layout()

        safe_name = terms[i].replace(" ", "_").replace("/", "_").replace("&", "and")
        out = out_dir / f"ebm_shape_{safe_name}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        files.append(out)
    return files


def plot_reliability(
    rc_df: pd.DataFrame,
    out_path: Path,
    title: str = "Reliability curve",
) -> Path:
    """Scatter of predicted vs observed rate per probability bin."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect")
    valid = rc_df.dropna(subset=["pred_mean", "obs_rate"])
    ax.scatter(
        valid["pred_mean"], valid["obs_rate"],
        s=valid["n"].clip(5, 200),
        alpha=0.7,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
