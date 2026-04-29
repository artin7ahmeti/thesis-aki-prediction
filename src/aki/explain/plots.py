"""Matplotlib plots: EBM shapes, patient explanations, and calibration curves."""

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


def plot_patient_contributions(
    contrib_df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Patient-level explanation",
    subtitle: str | None = None,
    top_n: int = 10,
) -> Path:
    """Horizontal bar chart of one patient's additive feature contributions."""
    df = _prepare_patient_contribution_frame(contrib_df, top_n=top_n)
    fig_height = max(4.8, 0.55 * len(df) + 2.1)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    _draw_patient_contribution_panel(ax, df, panel_title=None)

    fig.suptitle(title, x=0.07, y=0.98, ha="left", fontsize=13)
    if subtitle:
        fig.text(
            0.07,
            0.945,
            subtitle,
            ha="left",
            va="top",
            fontsize=9,
            color="#555555",
            wrap=True,
        )

    fig.subplots_adjust(left=0.43, right=0.96, top=0.88 if subtitle else 0.92, bottom=0.14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_patient_contribution_comparison(
    ebm_df: pd.DataFrame,
    scorecard_df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Patient-level explanation",
    subtitle: str | None = None,
    ebm_title: str = "Flagship EBM (combined)",
    scorecard_title: str = "Final bedside scorecard (augmented v2)",
    top_n: int = 8,
) -> Path:
    """Two-panel local explanation for the same held-out landmark."""
    ebm = _prepare_patient_contribution_frame(ebm_df, top_n=top_n)
    scorecard = _prepare_patient_contribution_frame(scorecard_df, top_n=top_n)
    shared_limit = max(
        float(ebm["contribution_logodds"].abs().max()),
        float(scorecard["contribution_logodds"].abs().max()),
        0.5,
    )

    fig_height = max(5.6, 0.55 * max(len(ebm), len(scorecard)) + 2.4)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16.0, fig_height),
        sharex=True,
    )
    _draw_patient_contribution_panel(axes[0], ebm, panel_title=ebm_title, shared_limit=shared_limit)
    _draw_patient_contribution_panel(axes[1], scorecard, panel_title=scorecard_title, shared_limit=shared_limit)
    axes[1].set_ylabel("")

    fig.suptitle(title, x=0.05, y=0.98, ha="left", fontsize=14)
    if subtitle:
        fig.text(
            0.05,
            0.945,
            subtitle,
            ha="left",
            va="top",
            fontsize=9.5,
            color="#555555",
            wrap=True,
        )

    fig.subplots_adjust(left=0.12, right=0.98, top=0.84, bottom=0.12, wspace=0.28)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _prepare_patient_contribution_frame(
    contrib_df: pd.DataFrame,
    *,
    top_n: int,
) -> pd.DataFrame:
    required = {"feature", "contribution_logodds"}
    missing = required.difference(contrib_df.columns)
    if missing:
        raise ValueError(f"contribution frame missing columns: {sorted(missing)}")

    df = contrib_df.copy()
    df = df.loc[df["feature"].astype(str).str.len() > 0].copy()
    if df.empty:
        raise ValueError("no patient contributions available to plot")

    if "value" not in df.columns:
        df["value"] = np.nan

    df["abs_contribution"] = df["contribution_logodds"].abs()
    df = df.sort_values("abs_contribution", ascending=False).head(top_n).copy()
    df = df.sort_values("contribution_logodds", ascending=True).reset_index(drop=True)
    df["label"] = [_patient_feature_label(row) for _, row in df.iterrows()]
    return df


def _draw_patient_contribution_panel(
    ax,
    df: pd.DataFrame,
    *,
    panel_title: str | None,
    shared_limit: float | None = None,
) -> None:
    colors = np.where(df["contribution_logodds"] >= 0, "#C44E52", "#4E79A7")
    y = np.arange(len(df))
    bars = ax.barh(
        y,
        df["contribution_logodds"],
        color=colors,
        alpha=0.92,
        height=0.72,
        linewidth=0,
    )
    ax.axvline(0.0, color="#333333", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlabel("Contribution to predicted log-odds of AKI")
    if panel_title:
        ax.set_title(panel_title, fontsize=11, loc="left", pad=8)
    ax.grid(axis="x", alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if shared_limit is None:
        x_min = float(df["contribution_logodds"].min())
        x_max = float(df["contribution_logodds"].max())
        pad = max(0.08, 0.12 * max(abs(x_min), abs(x_max), 0.5))
        ax.set_xlim(x_min - pad, x_max + pad)
    else:
        pad = max(0.08, 0.12 * max(shared_limit, 0.5))
        ax.set_xlim(-(shared_limit + pad), shared_limit + pad)

    for bar, value in zip(bars, df["contribution_logodds"], strict=True):
        x = float(bar.get_width())
        y_mid = bar.get_y() + bar.get_height() / 2.0
        if x >= 0:
            ax.text(x + pad * 0.08, y_mid, f"+{x:.2f}", ha="left", va="center", fontsize=9)
        else:
            ax.text(x - pad * 0.08, y_mid, f"{x:.2f}", ha="right", va="center", fontsize=9)


def _patient_feature_label(row: pd.Series) -> str:
    feature = _pretty_feature_name(row["feature"])
    if pd.notna(row.get("active_level", np.nan)):
        return f"{feature} = {row['active_level']}"
    value = row.get("value", np.nan)
    return f"{feature} = {_format_feature_value(value)}"


def _pretty_feature_name(name: object) -> str:
    text = str(name)
    replacements = {
        "map": "MAP",
        "sbp": "SBP",
        "dbp": "DBP",
        "spo2": "SpO2",
        "gcs": "GCS",
        "bun": "BUN",
        "aki": "AKI",
    }
    parts = text.split("_")
    pretty_parts = []
    for part in parts:
        lower = part.lower()
        pretty_parts.append(replacements.get(lower, part))
    return " ".join(pretty_parts)


def _format_feature_value(value: object) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        flat = np.asarray(value).ravel().tolist()
        if not flat:
            return "interaction"
        return " / ".join(_format_feature_value(v) for v in flat[:3])
    if isinstance(value, str):
        text = value.strip()
        return text or "interaction"
    if pd.isna(value):
        return "missing"
    try:
        numeric = float(value)
    except Exception:
        return str(value)

    if abs(numeric) >= 100:
        return f"{numeric:.0f}"
    if abs(numeric - round(numeric)) < 1e-9:
        return f"{numeric:.0f}"
    if abs(numeric) >= 10:
        return f"{numeric:.1f}"
    return f"{numeric:.2f}"
