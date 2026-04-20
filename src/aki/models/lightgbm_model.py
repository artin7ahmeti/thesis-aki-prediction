"""LightGBM opaque baseline.

Used strictly as a *reference* for the interpretable models — the thesis
claims that the glass-box models should come within a small AUROC delta
of LightGBM while remaining fully inspectable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from aki.models.base import BaseModel


class LightGBMModel(BaseModel):
    """``LGBMClassifier`` wrapper with optional early-stopping on a val split."""

    name = "lightgbm"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> LightGBMModel:
        params: dict[str, Any] = {
            "objective":         self.params.get("objective", "binary"),
            "metric":            self.params.get("metric", "auc"),
            "n_estimators":      self.params.get("n_estimators", 2000),
            "learning_rate":     self.params.get("learning_rate", 0.05),
            "num_leaves":        self.params.get("num_leaves", 31),
            "max_depth":         self.params.get("max_depth", -1),
            "min_child_samples": self.params.get("min_child_samples", 20),
            "subsample":         self.params.get("subsample", 0.8),
            "colsample_bytree":  self.params.get("colsample_bytree", 0.8),
            "reg_alpha":         self.params.get("reg_alpha", 0.0),
            "reg_lambda":        self.params.get("reg_lambda", 0.0),
            "class_weight":      self.params.get("class_weight", "balanced"),
            "random_state":      self.params.get("random_state", 42),
            "n_jobs":            self.params.get("n_jobs", -1),
            "verbosity":         -1,
        }
        self.feature_names_ = list(X.columns)
        self.estimator_ = LGBMClassifier(**params)

        fit_kwargs: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val[self.feature_names_], y_val)]
            fit_kwargs["callbacks"] = [
                early_stopping(self.params.get("early_stopping_rounds", 100)),
                log_evaluation(0),
            ]
        self.estimator_.fit(X, y, **fit_kwargs)
        return self

    def feature_importance(self) -> pd.DataFrame:
        self._ensure_fitted()
        imp = np.asarray(self.estimator_.booster_.feature_importance(importance_type="gain"))
        return (
            pd.DataFrame({"feature": self.feature_names_, "importance": imp})
            .sort_values("importance", ascending=False, ignore_index=True)
        )
