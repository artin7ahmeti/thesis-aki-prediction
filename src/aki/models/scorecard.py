"""Sparse-logistic and bedside-binned scorecards.

The scorecard path now supports two related representations:

1. ``linear`` (default): the original L1-selected sparse logistic model on
   median-imputed, z-scored raw features.
2. ``binned``: a clinically oriented bedside scorecard that keeps a fixed
   small raw feature set, bins continuous inputs into explicit ranges, and
   fits a logistic model on those additive terms.

The linear path remains useful as a baseline. The binned path is the more
thesis-ready bedside artifact because it yields explicit point rules and is
better aligned with threshold-heavy ICU predictors such as MAP, creatinine,
and urine output.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aki.models.base import BaseModel


class ScorecardDesignTransformer(BaseEstimator, TransformerMixin):
    """Transform raw scorecard features into additive bedside design terms."""

    def __init__(self, bin_edges: dict[str, list[float]] | None = None) -> None:
        self.bin_edges = dict(bin_edges or {})

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> ScorecardDesignTransformer:
        X_df = self._coerce_frame(X)
        y_series = None if y is None else pd.Series(y).reset_index(drop=True).astype(float)
        self.input_features_ = list(X_df.columns)
        self.bin_edges_ = {
            feature: [float(v) for v in values]
            for feature, values in self.bin_edges.items()
        }

        self.medians_: dict[str, float] = {}
        self.feature_profiles_: dict[str, dict[str, Any]] = {}
        self.term_names_: list[str] = []
        self.term_metadata_: list[dict[str, Any]] = []

        for feature in self.input_features_:
            series = pd.to_numeric(X_df[feature], errors="coerce")
            observed = series.dropna().astype(float)
            median = float(observed.median()) if not observed.empty else 0.0
            self.medians_[feature] = median
            filled = series.fillna(median).astype(float).to_numpy()
            missing_count = int(series.isna().sum())

            quantiles = {
                f"{q:.2f}": (
                    float(observed.quantile(q)) if not observed.empty else float("nan")
                )
                for q in (0.1, 0.25, 0.5, 0.75, 0.9)
            }
            unique_values = (
                sorted(float(v) for v in observed.unique().tolist())
                if not observed.empty
                else []
            )
            is_binary = len(unique_values) <= 2 and set(unique_values).issubset({0.0, 1.0})

            if feature in self.bin_edges_:
                if is_binary:
                    raise ValueError(
                        f"Binary feature {feature!r} should not be listed in scorecard bin_edges"
                    )
                edges = sorted(set(self.bin_edges_[feature]))
                if not edges:
                    raise ValueError(f"scorecard bin_edges for {feature!r} cannot be empty")
                labels = _bin_labels(edges)
                reference_idx = int(np.digitize([median], edges, right=False)[0])
                bins: list[dict[str, Any]] = []
                for idx, label in enumerate(labels):
                    lower, upper = _bin_bounds(edges, idx)
                    reference = idx == reference_idx
                    term_name = f"{feature}__bin_{idx}" if not reference else None
                    range_display = _format_range(lower, upper)
                    mask = np.digitize(filled, edges, right=False) == idx
                    train_n = int(mask.sum())
                    train_events = (
                        int(y_series.loc[mask].sum()) if y_series is not None and train_n else 0
                    )
                    train_event_rate = (
                        float(y_series.loc[mask].mean()) if y_series is not None and train_n else float("nan")
                    )
                    bin_meta = {
                        "index": idx,
                        "label": label,
                        "lower": lower,
                        "upper": upper,
                        "range_display": range_display,
                        "reference": reference,
                        "term_name": term_name,
                        "train_n": train_n,
                        "train_events": train_events,
                        "train_event_rate": train_event_rate,
                    }
                    bins.append(bin_meta)
                    if not reference:
                        self.term_names_.append(term_name)
                        self.term_metadata_.append(
                            {
                                "feature": feature,
                                "kind": "binned_continuous",
                                "term_name": term_name,
                                "level_label": label,
                                "range_display": range_display,
                                "lower": lower,
                                "upper": upper,
                                "reference": False,
                                "bin_index": idx,
                            }
                        )

                self.feature_profiles_[feature] = {
                    "kind": "binned_continuous",
                    "impute_median": median,
                    "quantiles": quantiles,
                    "unique_values": unique_values[:5],
                    "bin_edges": edges,
                    "bins": bins,
                    "reference_bin_index": reference_idx,
                    "reference_bin_label": labels[reference_idx],
                    "missing_count": missing_count,
                }
                continue

            if not is_binary:
                raise ValueError(
                    f"Binned scorecard requires explicit bin_edges for continuous feature {feature!r}"
                )

            self.term_names_.append(feature)
            self.term_metadata_.append(
                {
                    "feature": feature,
                    "kind": "binary",
                    "term_name": feature,
                    "level_label": "1 / present",
                    "range_display": "1 / present",
                    "lower": float("nan"),
                    "upper": float("nan"),
                    "reference": False,
                    "bin_index": 1,
                }
            )
            self.feature_profiles_[feature] = {
                "kind": "binary",
                "impute_median": median,
                "quantiles": quantiles,
                "unique_values": unique_values[:5],
                "missing_count": missing_count,
                "levels": [
                    {
                        "label": "0 / absent",
                        "reference": True,
                        "value": 0.0,
                        "train_n": int((filled == 0.0).sum()),
                        "train_events": (
                            int(y_series.loc[filled == 0.0].sum())
                            if y_series is not None and int((filled == 0.0).sum())
                            else 0
                        ),
                        "train_event_rate": (
                            float(y_series.loc[filled == 0.0].mean())
                            if y_series is not None and int((filled == 0.0).sum())
                            else float("nan")
                        ),
                    },
                    {
                        "label": "1 / present",
                        "reference": False,
                        "value": 1.0,
                        "train_n": int((filled == 1.0).sum()),
                        "train_events": (
                            int(y_series.loc[filled == 1.0].sum())
                            if y_series is not None and int((filled == 1.0).sum())
                            else 0
                        ),
                        "train_event_rate": (
                            float(y_series.loc[filled == 1.0].mean())
                            if y_series is not None and int((filled == 1.0).sum())
                            else float("nan")
                        ),
                    },
                ],
                "reference_level": "0 / absent",
            }

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        self._ensure_fitted()
        X_df = self._coerce_frame(X)
        arrays: list[np.ndarray] = []

        for feature in self.input_features_:
            values = (
                pd.to_numeric(X_df[feature], errors="coerce")
                .astype(float)
                .fillna(self.medians_[feature])
                .to_numpy()
            )
            profile = self.feature_profiles_[feature]

            if profile["kind"] == "binned_continuous":
                edges = [float(v) for v in profile["bin_edges"]]
                bin_idx = np.digitize(values, edges, right=False)
                for bin_meta in profile["bins"]:
                    if bin_meta["reference"]:
                        continue
                    arrays.append(
                        (bin_idx == int(bin_meta["index"])).astype(float).reshape(-1, 1)
                    )
                continue

            arrays.append(values.reshape(-1, 1))

        if not arrays:
            return np.empty((len(X_df), 0), dtype=float)
        return np.hstack(arrays)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        self._ensure_fitted()
        return np.asarray(self.term_names_, dtype=object)

    @staticmethod
    def _coerce_frame(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ScorecardDesignTransformer expects a pandas DataFrame")
        return X.copy()

    def _ensure_fitted(self) -> None:
        if not hasattr(self, "input_features_"):
            raise RuntimeError("ScorecardDesignTransformer has not been fit yet")


class ScorecardModel(BaseModel):
    """Sparse linear or bedside-binned logistic scorecard."""

    name = "scorecard"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series | np.ndarray | None = None,
    ) -> ScorecardModel:
        representation = str(self.params.get("representation", "linear")).lower()
        if representation == "binned":
            return self._fit_binned(X, y, groups)
        return self._fit_linear(X, y, groups)

    def coefficients(self) -> pd.DataFrame:
        """Return the fitted model's additive terms and coefficients."""
        self._ensure_fitted()
        clf = self.estimator_.named_steps["clf"]
        term_names = self.extra_.get("scorecard_terms", self.feature_names_)
        term_meta = self.extra_.get("term_metadata", [])
        features = [m.get("feature", term) for m, term in zip(term_meta, term_names, strict=False)]
        if len(features) != len(term_names):
            features = list(term_names)
        return pd.DataFrame(
            {
                "feature": features,
                "term": term_names,
                "coefficient": clf.coef_.ravel(),
            }
        ).sort_values("coefficient", key=np.abs, ascending=False, ignore_index=True)

    def intercept(self) -> float:
        self._ensure_fitted()
        return float(self.estimator_.named_steps["clf"].intercept_.ravel()[0])

    def _fit_linear(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series | np.ndarray | None,
    ) -> ScorecardModel:
        target_k = int(self.params.get("target_features", 10))
        max_iter = int(self.params.get("max_iter", 5000))
        class_weight = self.params.get("class_weight")
        C_grid = _resolve_c_grid(self.params)
        seed = int(self.params.get("random_state", 42))
        selection_tol = float(self.params.get("selection_tolerance", 0.01))
        refit_C = float(self.params.get("refit_C", 1.0))

        self.feature_names_ = list(X.columns)
        if len(C_grid) == 1:
            best_C = float(C_grid[0])
            cv_summary = pd.DataFrame()
        else:
            best_C, cv_summary = self._select_C(
                X,
                y,
                groups,
                C_grid,
                max_iter,
                class_weight,
                seed,
                target_k=target_k,
                selection_tolerance=selection_tol,
            )

        path_pipe = self._make_linear_pipeline(
            penalty="l1",
            C=best_C,
            max_iter=max_iter,
            class_weight=class_weight,
            seed=seed,
        )
        path_pipe.fit(X, y)
        coefs = path_pipe.named_steps["clf"].coef_.ravel()

        order = np.argsort(np.abs(coefs))[::-1]
        kept_idx = [i for i in order if coefs[i] != 0][:target_k]
        if not kept_idx:
            kept_idx = list(order[:target_k])
        kept_names = [self.feature_names_[i] for i in kept_idx]

        refit_pipe = self._make_linear_pipeline(
            penalty="l2",
            C=refit_C,
            max_iter=max_iter,
            class_weight=class_weight,
            seed=seed,
        )
        refit_pipe.fit(X[kept_names], y)

        self.feature_names_ = kept_names
        self.estimator_ = refit_pipe
        self.extra_ = {
            "selected_C": best_C,
            "C_grid": C_grid,
            "cv_summary": cv_summary.to_dict(orient="records"),
            "n_selected": len(kept_names),
            "target_features": target_k,
            "selection_tolerance": selection_tol,
            "base_rate": float(y.mean()),
            "refit_C": refit_C,
            "feature_profiles": _linear_feature_profiles(X[kept_names], refit_pipe),
            "scorecard_representation": "linear",
            "selection_mode": "sparse_path",
            "scorecard_terms": kept_names,
            "term_metadata": [],
        }
        return self

    def _fit_binned(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series | np.ndarray | None,
    ) -> ScorecardModel:
        selection_mode = str(self.params.get("selection_mode", "fixed")).lower()
        if selection_mode != "fixed":
            raise ValueError(
                "Binned bedside scorecards currently support selection_mode='fixed' only"
            )

        max_iter = int(self.params.get("max_iter", 5000))
        class_weight = self.params.get("class_weight")
        C_grid = _resolve_c_grid(self.params)
        seed = int(self.params.get("random_state", 42))
        selection_tol = float(self.params.get("selection_tolerance", 0.01))
        bin_edges = {
            feature: [float(v) for v in values]
            for feature, values in dict(self.params.get("bin_edges", {})).items()
        }

        self.feature_names_ = list(X.columns)
        if len(C_grid) == 1:
            best_C = float(C_grid[0])
            cv_summary = pd.DataFrame()
        else:
            best_C, cv_summary = self._select_binned_C(
                X,
                y,
                groups,
                C_grid,
                max_iter,
                class_weight,
                seed,
                bin_edges=bin_edges,
                selection_tolerance=selection_tol,
            )

        pipe = self._make_binned_pipeline(
            bin_edges=bin_edges,
            C=best_C,
            max_iter=max_iter,
            class_weight=class_weight,
            seed=seed,
        )
        pipe.fit(X, y)

        design = pipe.named_steps["design"]
        self.estimator_ = pipe
        self.extra_ = {
            "selected_C": best_C,
            "C_grid": C_grid,
            "cv_summary": cv_summary.to_dict(orient="records"),
            "n_selected": len(self.feature_names_),
            "n_terms": len(design.term_names_),
            "target_features": len(self.feature_names_),
            "selection_tolerance": selection_tol,
            "base_rate": float(y.mean()),
            "refit_C": None,
            "feature_profiles": design.feature_profiles_,
            "scorecard_representation": "binned",
            "selection_mode": selection_mode,
            "scorecard_terms": list(design.term_names_),
            "term_metadata": list(design.term_metadata_),
        }
        return self

    def _select_C(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series | np.ndarray | None,
        C_grid: list[float],
        max_iter: int,
        class_weight: Any,
        seed: int,
        *,
        target_k: int,
        selection_tolerance: float,
    ) -> tuple[float, pd.DataFrame]:
        """Patient-grouped CV over ``C_grid`` with a sparsity-aware selector."""
        from sklearn.metrics import roc_auc_score

        folds = _make_cv_folds(X, y, groups, seed)
        rows: list[dict[str, float | int | bool | str]] = []
        for C in C_grid:
            fold_aucs: list[float] = []
            fold_auprcs: list[float] = []
            fold_briers: list[float] = []
            fold_nonzero: list[int] = []
            for tr_idx, va_idx in folds:
                pipe = self._make_linear_pipeline(
                    penalty="l1",
                    C=C,
                    max_iter=max_iter,
                    class_weight=class_weight,
                    seed=seed,
                )
                pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                fold_nonzero.append(int(np.count_nonzero(pipe.named_steps["clf"].coef_)))
                p = pipe.predict_proba(X.iloc[va_idx])[:, 1]
                if y.iloc[va_idx].nunique() < 2:
                    continue
                fold_aucs.append(roc_auc_score(y.iloc[va_idx], p))
                fold_auprcs.append(average_precision_score(y.iloc[va_idx], p))
                fold_briers.append(brier_score_loss(y.iloc[va_idx], p))

            rows.append(
                {
                    "C": float(C),
                    "auroc": float(np.mean(fold_aucs)) if fold_aucs else float("nan"),
                    "auprc": float(np.mean(fold_auprcs)) if fold_auprcs else float("nan"),
                    "brier": float(np.mean(fold_briers)) if fold_briers else float("nan"),
                    "mean_nonzero": float(np.mean(fold_nonzero)) if fold_nonzero else float("nan"),
                    "max_nonzero": int(max(fold_nonzero)) if fold_nonzero else 0,
                }
            )

        summary = pd.DataFrame(rows).dropna(subset=["auroc"]).reset_index(drop=True)
        if summary.empty:
            return float(C_grid[0]), pd.DataFrame(rows)

        best_auroc = float(summary["auroc"].max())
        near_best = summary["auroc"] >= (best_auroc - selection_tolerance)
        within_budget = summary["mean_nonzero"] <= float(target_k)

        candidates = summary.loc[near_best & within_budget].copy()
        selection_rule = "within_tolerance_and_within_budget"
        if candidates.empty:
            candidates = summary.loc[near_best].copy()
            selection_rule = "within_tolerance"
        if candidates.empty:
            candidates = summary.copy()
            selection_rule = "best_auroc_fallback"

        candidates = candidates.sort_values(
            ["mean_nonzero", "brier", "auprc", "C"],
            ascending=[True, True, False, True],
            ignore_index=True,
        )
        best_C = float(candidates.loc[0, "C"])
        summary["selected"] = np.isclose(summary["C"], best_C)
        summary["selection_rule"] = selection_rule
        return best_C, summary

    def _select_binned_C(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series | np.ndarray | None,
        C_grid: list[float],
        max_iter: int,
        class_weight: Any,
        seed: int,
        *,
        bin_edges: dict[str, list[float]],
        selection_tolerance: float,
    ) -> tuple[float, pd.DataFrame]:
        """Patient-grouped CV over ``C_grid`` for fixed bedside scorecards."""
        from sklearn.metrics import roc_auc_score

        folds = _make_cv_folds(X, y, groups, seed)
        rows: list[dict[str, float | int | bool | str]] = []
        for C in C_grid:
            fold_aucs: list[float] = []
            fold_auprcs: list[float] = []
            fold_briers: list[float] = []
            for tr_idx, va_idx in folds:
                pipe = self._make_binned_pipeline(
                    bin_edges=bin_edges,
                    C=C,
                    max_iter=max_iter,
                    class_weight=class_weight,
                    seed=seed,
                )
                pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                p = pipe.predict_proba(X.iloc[va_idx])[:, 1]
                if y.iloc[va_idx].nunique() < 2:
                    continue
                fold_aucs.append(roc_auc_score(y.iloc[va_idx], p))
                fold_auprcs.append(average_precision_score(y.iloc[va_idx], p))
                fold_briers.append(brier_score_loss(y.iloc[va_idx], p))

            rows.append(
                {
                    "C": float(C),
                    "auroc": float(np.mean(fold_aucs)) if fold_aucs else float("nan"),
                    "auprc": float(np.mean(fold_auprcs)) if fold_auprcs else float("nan"),
                    "brier": float(np.mean(fold_briers)) if fold_briers else float("nan"),
                }
            )

        summary = pd.DataFrame(rows).dropna(subset=["auroc"]).reset_index(drop=True)
        if summary.empty:
            return float(C_grid[0]), pd.DataFrame(rows)

        best_auroc = float(summary["auroc"].max())
        candidates = summary.loc[
            summary["auroc"] >= (best_auroc - selection_tolerance)
        ].copy()
        selection_rule = "within_tolerance_prefer_shrinkage"
        if candidates.empty:
            candidates = summary.copy()
            selection_rule = "best_auroc_fallback"

        candidates = candidates.sort_values(
            ["brier", "auprc", "C"],
            ascending=[True, False, True],
            ignore_index=True,
        )
        best_C = float(candidates.loc[0, "C"])
        summary["selected"] = np.isclose(summary["C"], best_C)
        summary["selection_rule"] = selection_rule
        return best_C, summary

    @staticmethod
    def _make_linear_pipeline(
        penalty: str,
        C: float,
        max_iter: int,
        class_weight: Any,
        seed: int,
    ) -> Pipeline:
        clf_kwargs: dict[str, Any] = {
            "C": C,
            "solver": "saga",
            "max_iter": max_iter,
            "class_weight": class_weight,
            "random_state": seed,
        }
        if _sklearn_ge_18():
            clf_kwargs["l1_ratio"] = 1.0 if penalty == "l1" else 0.0
        else:
            clf_kwargs["penalty"] = penalty

        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=True, with_std=True)),
                ("clf", LogisticRegression(**clf_kwargs)),
            ]
        )

    @staticmethod
    def _make_binned_pipeline(
        *,
        bin_edges: dict[str, list[float]],
        C: float,
        max_iter: int,
        class_weight: Any,
        seed: int,
    ) -> Pipeline:
        clf_kwargs: dict[str, Any] = {
            "C": C,
            "solver": "lbfgs",
            "max_iter": max_iter,
            "class_weight": class_weight,
            "random_state": seed,
        }
        if _sklearn_ge_18():
            clf_kwargs["l1_ratio"] = 0.0
        else:
            clf_kwargs["penalty"] = "l2"

        return Pipeline(
            [
                ("design", ScorecardDesignTransformer(bin_edges=bin_edges)),
                (
                    "clf",
                    LogisticRegression(**clf_kwargs),
                ),
            ]
        )


def _linear_feature_profiles(X: pd.DataFrame, pipe: Pipeline) -> dict[str, dict[str, Any]]:
    """Return raw-space summaries for the selected linear scorecard features."""
    imputer = pipe.named_steps["impute"]
    scaler = pipe.named_steps["scale"]
    profiles: dict[str, dict[str, Any]] = {}

    for idx, feature in enumerate(X.columns):
        series = X[feature]
        observed = series.dropna()
        if observed.empty:
            quantiles = {f"{q:.2f}": float("nan") for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
            unique_values: list[float] = []
        else:
            observed_num = observed.astype(float)
            quantiles = {
                f"{q:.2f}": float(observed_num.quantile(q))
                for q in (0.1, 0.25, 0.5, 0.75, 0.9)
            }
            unique_values = sorted(float(v) for v in observed_num.unique().tolist())

        is_binary = len(unique_values) <= 2 and set(unique_values).issubset({0.0, 1.0})
        scale = float(scaler.scale_[idx]) if float(scaler.scale_[idx]) != 0.0 else 1.0
        profiles[feature] = {
            "kind": "binary" if is_binary else "continuous",
            "impute_median": float(imputer.statistics_[idx]),
            "mean": float(scaler.mean_[idx]),
            "scale": scale,
            "quantiles": quantiles,
            "unique_values": unique_values[:5],
        }

    return profiles


def _make_cv_folds(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | np.ndarray | None,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if groups is None:
        from sklearn.model_selection import KFold

        splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y)
    else:
        splitter = GroupKFold(n_splits=5)
        split_iter = splitter.split(X, y, groups=np.asarray(groups))
    return list(split_iter)


def _resolve_c_grid(params: dict[str, Any]) -> list[float]:
    """Return the effective regularization path for scorecard fitting."""
    if params.get("C") is not None:
        return [float(params["C"])]
    raw_grid = params.get("C_grid", [0.01, 0.1, 1.0])
    return [float(c) for c in raw_grid]


def _bin_labels(edges: list[float]) -> list[str]:
    labels: list[str] = []
    lower = float("-inf")
    for edge in edges:
        labels.append(_format_range(lower, edge))
        lower = edge
    labels.append(_format_range(lower, float("inf")))
    return labels


def _bin_bounds(edges: list[float], idx: int) -> tuple[float, float]:
    lower = float("-inf") if idx == 0 else float(edges[idx - 1])
    upper = float("inf") if idx == len(edges) else float(edges[idx])
    return lower, upper


def _format_range(lower: float, upper: float) -> str:
    if np.isneginf(lower):
        return f"< {_fmt_number(upper)}"
    if np.isposinf(upper):
        return f">= {_fmt_number(lower)}"
    return f"{_fmt_number(lower)} to < {_fmt_number(upper)}"


def _fmt_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _sklearn_ge_18() -> bool:
    """Return True for scikit-learn >= 1.8 without adding a packaging dep."""
    parts: list[int] = []
    for token in sklearn_version.split(".")[:2]:
        digits = "".join(ch for ch in token if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 2:
        parts.append(0)
    return tuple(parts) >= (1, 8)
