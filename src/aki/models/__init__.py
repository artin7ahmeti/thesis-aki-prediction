"""Interpretable + baseline model wrappers.

Heavy optional model dependencies are imported lazily so data-only CLI
commands can run even before the full modeling stack is loaded.
"""

from aki.models.base import BaseModel, ModelArtifact

__all__ = [
    "BaseModel",
    "ModelArtifact",
    "EBMModel",
    "ScorecardModel",
    "LightGBMModel",
    "train_all",
]


def __getattr__(name: str):
    if name == "EBMModel":
        from aki.models.ebm import EBMModel

        return EBMModel
    if name == "LightGBMModel":
        from aki.models.lightgbm_model import LightGBMModel

        return LightGBMModel
    if name == "ScorecardModel":
        from aki.models.scorecard import ScorecardModel

        return ScorecardModel
    if name == "train_all":
        from aki.models.train import train_all

        return train_all
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
