"""Interpretable + baseline model wrappers.

Primary interpretable: :class:`EBMModel`, :class:`ScorecardModel`.
Opaque baseline: :class:`LightGBMModel`.
"""

from aki.models.base import BaseModel, ModelArtifact
from aki.models.ebm import EBMModel
from aki.models.lightgbm_model import LightGBMModel
from aki.models.scorecard import ScorecardModel
from aki.models.train import train_all

__all__ = [
    "BaseModel",
    "ModelArtifact",
    "EBMModel",
    "ScorecardModel",
    "LightGBMModel",
    "train_all",
]
