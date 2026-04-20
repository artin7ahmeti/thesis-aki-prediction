"""Explainable Boosting Machine (EBM) wrapper.

EBM is the primary *glass-box* model — each feature contributes an
additive shape function which is plotted directly (see ``aki.explain``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier

from aki.models.base import BaseModel


class EBMModel(BaseModel):
    """Wraps ``ExplainableBoostingClassifier`` for the AKI pipeline."""

    name = "ebm"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> EBMModel:
        params: dict[str, Any] = {
            "max_bins":              self.params.get("max_bins", 256),
            "max_interaction_bins":  self.params.get("max_interaction_bins", 32),
            "interactions":          self.params.get("interactions", 10),
            "learning_rate":         self.params.get("learning_rate", 0.01),
            "max_rounds":            self.params.get("max_rounds", 5000),
            "early_stopping_rounds": self.params.get("early_stopping_rounds", 50),
            "min_samples_leaf":      self.params.get("min_samples_leaf", 2),
            "outer_bags":            self.params.get("outer_bags", 8),
            "random_state":          self.params.get("random_state", 42),
        }
        self.feature_names_ = list(X.columns)
        self.estimator_ = ExplainableBoostingClassifier(**params)
        self.estimator_.fit(X, y.values)
        return self

    def global_importance(self) -> pd.DataFrame:
        """Mean absolute contribution per term (features + interactions)."""
        self._ensure_fitted()
        expl = self.estimator_.explain_global()
        data = expl.data()
        return (
            pd.DataFrame({"term": data["names"], "importance": data["scores"]})
            .sort_values("importance", ascending=False, ignore_index=True)
        )

    def term_shapes(self) -> list[dict[str, Any]]:
        """Extract per-term shape functions for plotting."""
        self._ensure_fitted()
        expl = self.estimator_.explain_global()
        out: list[dict[str, Any]] = []
        for i, name in enumerate(self.estimator_.term_names_):
            d = expl.data(i)
            if d is None:
                continue
            out.append({
                "term":   name,
                "type":   d.get("type", "univariate"),
                "names":  np.asarray(d.get("names",  [])),
                "scores": np.asarray(d.get("scores", [])),
                "upper":  np.asarray(d.get("upper_bounds", [])),
                "lower":  np.asarray(d.get("lower_bounds", [])),
            })
        return out
