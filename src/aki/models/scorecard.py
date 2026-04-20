"""Sparse-logistic scorecard.

Workflow:
1. Median-impute + standard-scale numeric features.
2. Sweep the L1 ``C_grid`` via patient-level CV -> pick ``C`` by ROC-AUC.
3. If the non-zero coefficient count exceeds ``target_features``, keep the
   top-|coef| features and re-fit with a mild L2 so the final scorecard
   has less or equal to ``target_features`` terms and stable coefficients.

The output is a small, inspectable logistic model suitable for a
printed scorecard (see ``aki.explain.scorecard``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
        target_k  = int(self.params.get("target_features", 10))
        max_iter  = int(self.params.get("max_iter", 5000))
        class_weight = self.params.get("class_weight", "balanced")
        C_grid    = list(self.params.get("C_grid", [0.01, 0.1, 1.0]))
        seed      = int(self.params.get("random_state", 42))

        self.feature_names_ = list(X.columns)
        best_C = self._select_C(X, y, groups, C_grid, max_iter, class_weight, seed)

        # Path fit at the selected C -> identify non-zero features
        path_pipe = self._make_pipeline(
            penalty="l1", C=best_C, max_iter=max_iter,
            class_weight=class_weight, seed=seed,
        )
        path_pipe.fit(X, y)
        coefs = path_pipe.named_steps["clf"].coef_.ravel()

        # Truncate to target_features by |coef|
        order = np.argsort(np.abs(coefs))[::-1]
        kept_idx = [i for i in order if coefs[i] != 0][:target_k]
        if not kept_idx:
            # Fall back to top-k by |coef| even if they are zero
            kept_idx = list(order[:target_k])
        kept_names = [self.feature_names_[i] for i in kept_idx]

        # Refit mild-L2 on the selected columns -> stable coefficients
        refit_pipe = self._make_pipeline(
            penalty="l2", C=1.0, max_iter=max_iter,
            class_weight=class_weight, seed=seed,
        )
        refit_pipe.fit(X[kept_names], y)

        self.feature_names_ = kept_names
        self.estimator_ = refit_pipe
        self.extra_ = {
            "selected_C":      best_C,
            "C_grid":          C_grid,
            "n_selected":      len(kept_names),
            "target_features": target_k,
        }
        return self

    def coefficients(self) -> pd.DataFrame:
        """Return the refit model's coefficients as a small DataFrame."""
        self._ensure_fitted()
        clf = self.estimator_.named_steps["clf"]
        return pd.DataFrame({
            "feature":     self.feature_names_,
            "coefficient": clf.coef_.ravel(),
        }).sort_values("coefficient", key=np.abs, ascending=False, ignore_index=True)

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
    ) -> float:
        """Patient-grouped CV over ``C_grid`` — ROC-AUC average."""
        from sklearn.metrics import roc_auc_score

        if groups is None:
            # Fall back to row-level KFold; caller should pass subject_id groups
            from sklearn.model_selection import KFold
            splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
            split_iter = splitter.split(X, y)
        else:
            splitter = GroupKFold(n_splits=5)
            split_iter = splitter.split(X, y, groups=np.asarray(groups))

        folds = list(split_iter)
        scores: list[float] = []
        for C in C_grid:
            fold_aucs: list[float] = []
            for tr_idx, va_idx in folds:
                pipe = self._make_pipeline(
                    penalty="l1", C=C, max_iter=max_iter,
                    class_weight=class_weight, seed=seed,
                )
                pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                p = pipe.predict_proba(X.iloc[va_idx])[:, 1]
                if y.iloc[va_idx].nunique() < 2:
                    continue
                fold_aucs.append(roc_auc_score(y.iloc[va_idx], p))
            scores.append(float(np.mean(fold_aucs)) if fold_aucs else float("nan"))
        return float(C_grid[int(np.nanargmax(scores))])

    @staticmethod
    def _make_pipeline(
        penalty: str, C: float, max_iter: int, class_weight: Any, seed: int,
    ) -> Pipeline:
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(
                penalty=penalty, C=C, solver="saga",
                max_iter=max_iter, class_weight=class_weight,
                random_state=seed,
            )),
        ])
