"""Sparse-logistic scorecard.

Workflow:
1. Median-impute + standard-scale numeric features.
2. Sweep the L1 ``C_grid`` via patient-level CV.
3. Prefer the *sparsest* candidate whose CV AUROC stays within a small
   tolerance of the best-performing candidate.
4. If the non-zero coefficient count still exceeds ``target_features``,
   keep the top-|coef| features and re-fit with a mild L2 so the final
   scorecard has a stable <= ``target_features`` term set.

The model remains a plain logistic regression, but it now stores enough
raw-value metadata to build a thesis-grade score sheet: imputation
medians, scaler moments, quartiles, and simple anchor-point rules.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aki.models.base import BaseModel


class ScorecardModel(BaseModel):
    """L1-selected, refit-L2 logistic regression."""

    name = "scorecard"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series | np.ndarray | None = None,
    ) -> ScorecardModel:
        target_k = int(self.params.get("target_features", 10))
        max_iter = int(self.params.get("max_iter", 5000))
        class_weight = self.params.get("class_weight")
        C_grid = list(self.params.get("C_grid", [0.01, 0.1, 1.0]))
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

        # Path fit at the selected C -> identify non-zero features.
        path_pipe = self._make_pipeline(
            penalty="l1",
            C=best_C,
            max_iter=max_iter,
            class_weight=class_weight,
            seed=seed,
        )
        path_pipe.fit(X, y)
        coefs = path_pipe.named_steps["clf"].coef_.ravel()

        # Truncate to target_features by |coef|.
        order = np.argsort(np.abs(coefs))[::-1]
        kept_idx = [i for i in order if coefs[i] != 0][:target_k]
        if not kept_idx:
            # Fall back to top-k by |coef| even if they are zero.
            kept_idx = list(order[:target_k])
        kept_names = [self.feature_names_[i] for i in kept_idx]

        # Refit mild-L2 on the selected columns -> stable coefficients.
        refit_pipe = self._make_pipeline(
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
            "feature_profiles": _feature_profiles(X[kept_names], refit_pipe),
        }
        return self

    def coefficients(self) -> pd.DataFrame:
        """Return the refit model's coefficients as a small DataFrame."""
        self._ensure_fitted()
        clf = self.estimator_.named_steps["clf"]
        return pd.DataFrame(
            {
                "feature": self.feature_names_,
                "coefficient": clf.coef_.ravel(),
            }
        ).sort_values("coefficient", key=np.abs, ascending=False, ignore_index=True)

    def intercept(self) -> float:
        self._ensure_fitted()
        return float(self.estimator_.named_steps["clf"].intercept_.ravel()[0])

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

        if groups is None:
            # Fall back to row-level KFold; caller should pass subject_id groups.
            from sklearn.model_selection import KFold

            splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
            split_iter = splitter.split(X, y)
        else:
            splitter = GroupKFold(n_splits=5)
            split_iter = splitter.split(X, y, groups=np.asarray(groups))

        folds = list(split_iter)
        rows: list[dict[str, float | int | bool | str]] = []
        for C in C_grid:
            fold_aucs: list[float] = []
            fold_auprcs: list[float] = []
            fold_briers: list[float] = []
            fold_nonzero: list[int] = []
            for tr_idx, va_idx in folds:
                pipe = self._make_pipeline(
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

    @staticmethod
    def _make_pipeline(
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
            # scikit-learn 1.8 deprecates `penalty=` in favor of l1_ratio.
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


def _feature_profiles(X: pd.DataFrame, pipe: Pipeline) -> dict[str, dict[str, Any]]:
    """Return raw-space summaries for the selected scorecard features."""
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


def _sklearn_ge_18() -> bool:
    """Return True for scikit-learn >= 1.8 without adding a packaging dep."""
    parts: list[int] = []
    for token in sklearn_version.split(".")[:2]:
        digits = "".join(ch for ch in token if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 2:
        parts.append(0)
    return tuple(parts) >= (1, 8)
