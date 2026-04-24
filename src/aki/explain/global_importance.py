"""Global importance tables from a :class:`ModelArtifact`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aki.models.base import ModelArtifact


def global_importance_table(art: ModelArtifact) -> pd.DataFrame:
    """Return a ``(term, importance)`` frame, model-appropriate."""
    est = art.estimator
    name = art.name

    if name == "ebm":
        expl = est.explain_global()
        data = expl.data()
        return (
            pd.DataFrame({"term": data["names"], "importance": data["scores"]})
            .sort_values("importance", ascending=False, ignore_index=True)
        )

    if name == "scorecard":
        clf = est.named_steps["clf"]
        term_names = art.extra.get("scorecard_terms", art.feature_names)
        term_meta = art.extra.get("term_metadata", [])
        if term_meta and len(term_meta) == len(term_names):
            return (
                pd.DataFrame({
                    "feature": [meta.get("feature") for meta in term_meta],
                    "term": [meta.get("range_display", term) for meta, term in zip(term_meta, term_names, strict=True)],
                    "coefficient": clf.coef_.ravel(),
                    "importance": np.abs(clf.coef_.ravel()),
                })
                .sort_values("importance", ascending=False, ignore_index=True)
            )
        return (
            pd.DataFrame({
                "term":        term_names,
                "coefficient": clf.coef_.ravel(),
                "importance":  np.abs(clf.coef_.ravel()),
            })
            .sort_values("importance", ascending=False, ignore_index=True)
        )

    if name == "lightgbm":
        imp = np.asarray(est.booster_.feature_importance(importance_type="gain"))
        return (
            pd.DataFrame({"term": art.feature_names, "importance": imp})
            .sort_values("importance", ascending=False, ignore_index=True)
        )

    raise ValueError(f"Unsupported model name for importance: {name}")
