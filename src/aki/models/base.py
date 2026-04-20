"""Shared model interface and feature-matrix helpers.

All AKI models expose a uniform :class:`BaseModel` interface so the
training / evaluation / explanation code can treat them polymorphically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

# Columns that are metadata or label columns and must never be fed to a model.
META_COLS: tuple[str, ...] = (
    "stay_id", "subject_id", "landmark_time",
    "anchor_year_group", "age_group", "sex", "ethnicity",
    "hours_since_icu_admit", "split",
)
LABEL_PREFIX = "y_"


@dataclass
class ModelArtifact:
    """Serialized payload saved for every trained model."""

    name: str
    task: str
    family: str
    feature_names: list[str]
    estimator: Any
    calibrator: Any | None
    extra: dict[str, Any]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> ModelArtifact:
        return joblib.load(path)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Columns eligible as model features (drops meta + label columns)."""
    return [
        c for c in df.columns
        if c not in META_COLS and not c.startswith(LABEL_PREFIX)
    ]


def split_xy(
    df: pd.DataFrame,
    label_col: str,
    feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return (X, y, feature_names) with rows having a valid label."""
    mask = df[label_col].notna()
    sub = df.loc[mask]
    feats = feature_names or feature_columns(df)
    X = sub[feats].copy()
    y = sub[label_col].astype(int)
    return X, y, feats


class BaseModel(ABC):
    """Uniform interface over EBM / scorecard / LightGBM."""

    name: str = "base"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = dict(params or {})
        self.estimator_: Any | None = None
        self.calibrator_: Any | None = None
        self.feature_names_: list[str] = []
        self.extra_: dict[str, Any] = {}

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseModel: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(y=1). Uses the calibrator if one was fitted."""
        self._ensure_fitted()
        X = X[self.feature_names_]
        if self.calibrator_ is not None:
            return self.calibrator_.predict_proba(X)[:, 1]
        return self.estimator_.predict_proba(X)[:, 1]

    def calibrate(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        method: str = "isotonic",
    ) -> BaseModel:
        """Prefit isotonic/Platt calibration on the validation split."""
        self._ensure_fitted()
        cal = CalibratedClassifierCV(self.estimator_, method=method, cv="prefit")
        cal.fit(X_val[self.feature_names_], y_val)
        self.calibrator_ = cal
        return self

    def artifact(self, task: str, family: str) -> ModelArtifact:
        self._ensure_fitted()
        return ModelArtifact(
            name=self.name,
            task=task,
            family=family,
            feature_names=self.feature_names_,
            estimator=self.estimator_,
            calibrator=self.calibrator_,
            extra=self.extra_,
        )

    def _ensure_fitted(self) -> None:
        if self.estimator_ is None:
            raise RuntimeError(f"{self.name} model has not been fit yet")
