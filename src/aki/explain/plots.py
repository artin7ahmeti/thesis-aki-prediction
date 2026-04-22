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
        names = d.get("names", [])
        values = np.asarray(d.get("scores", []), dtype=float)
        lower = np.asarray(d.get("lower_bounds", values), dtype=float)
        upper = np.asarray(d.get("upper_bounds", values), dtype=float)

        if values.size == 0:
            plt.close(fig)
            continue

        if values.ndim == 2:
            im = ax.imshow(values, aspect="auto", origin="lower", cmap="coolwarm")
            fig.colorbar(im, ax=ax, label="score contribution")
            _set_interaction_ticks(ax, names)
            ax.set_title(terms[i])
        else:
            x, tick_labels = _shape_x_values(names, values)
            ax.plot(x, values, color="tab:blue")
            if lower.shape == values.shape and upper.shape == values.shape:
                ax.fill_between(x, lower, upper, alpha=0.2, color="tab:blue")
            ax.axhline(0.0, color="gray", linewidth=0.5)
            if tick_labels is not None:
                ax.set_xticks(x)
                ax.set_xticklabels(tick_labels, rotation=45, ha="right")
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


def _shape_x_values(names, values: np.ndarray) -> tuple[np.ndarray, list[str] | None]:
    """Return x coordinates for an EBM univariate shape.

    Interpret returns continuous-term bin edges in some versions, so the
    x-axis can be one element longer than the per-bin score vector. Plot the
    score at bin centers in that case. Categorical terms keep integer
    positions plus readable tick labels.
    """
    x_raw = np.asarray(names)
    n = len(values)
    if x_raw.size == 0:
        return np.arange(n), None

    try:
        x_num = x_raw.astype(float)
    except (TypeError, ValueError):
        labels = [str(v) for v in x_raw[:n]]
        labels.extend([""] * (n - len(labels)))
        return np.arange(n), labels

    if len(x_num) == n + 1:
        return (x_num[:-1] + x_num[1:]) / 2.0, None
    if len(x_num) == n:
        return x_num, None

    return np.arange(n), None


def _set_interaction_ticks(ax, names) -> None:
    """Best-effort tick labels for 2D EBM interaction heatmaps."""
    if not isinstance(names, (list, tuple)) or len(names) != 2:
        ax.set_xlabel("feature 1 bin")
        ax.set_ylabel("feature 2 bin")
        return

    x_names = [str(v) for v in names[0]]
    y_names = [str(v) for v in names[1]]
    if x_names:
        x_idx = np.linspace(0, len(x_names) - 1, min(6, len(x_names))).astype(int)
        ax.set_xticks(x_idx)
        ax.set_xticklabels([x_names[j] for j in x_idx], rotation=45, ha="right")
    if y_names:
        y_idx = np.linspace(0, len(y_names) - 1, min(6, len(y_names))).astype(int)
        ax.set_yticks(y_idx)
        ax.set_yticklabels([y_names[j] for j in y_idx])
    ax.set_xlabel("feature 1 value")
    ax.set_ylabel("feature 2 value")


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
