"""Patient-level additive contributions.

For the two glass-box models (EBM, scorecard), decomposes an individual
prediction into per-feature contributions on the log-odds scale. This is
what a clinician sees on a case card: "feature X pushed the probability
up by Y".

Important: these are associations, not causal effects. The returned
frame should be labeled as such in any figures/text.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aki.models.base import ModelArtifact


def patient_additive_contributions(
    art: ModelArtifact,
    x_row: pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    """Return (feature, value, contribution_logodds) for one landmark row."""
    if isinstance(x_row, pd.DataFrame):
        if len(x_row) != 1:
            raise ValueError("x_row must be a single row")
        x_row = x_row.iloc[0]

    if art.name == "ebm":
        return _ebm_contributions(art, x_row)
    if art.name == "scorecard":
        return _scorecard_contributions(art, x_row)
    raise ValueError(f"patient contributions not supported for {art.name}")


def select_representative_patient_case(
    art: ModelArtifact,
    df: pd.DataFrame,
    label_col: str,
    *,
    quantile: float = 0.90,
    reference_art: ModelArtifact | None = None,
    reference_df: pd.DataFrame | None = None,
    reference_label_col: str | None = None,
) -> pd.Series:
    """Pick one deterministic held-out landmark for a patient-level figure."""
    ranked = rank_representative_patient_cases(
        art,
        df,
        label_col,
        quantile=quantile,
        reference_art=reference_art,
        reference_df=reference_df,
        reference_label_col=reference_label_col,
        top_k=1,
    )
    return ranked.iloc[0].copy()


def rank_representative_patient_cases(
    art: ModelArtifact,
    df: pd.DataFrame,
    label_col: str,
    *,
    quantile: float = 0.90,
    reference_art: ModelArtifact | None = None,
    reference_df: pd.DataFrame | None = None,
    reference_label_col: str | None = None,
    top_k: int = 10,
) -> pd.DataFrame:
    """Rank deterministic held-out landmarks for patient-level review.

    The ranking prefers high-risk positive test landmarks. When a bedside
    reference scorecard is available, the shortlist is refined toward cases
    where the bedside explanation uses clinically central terms such as
    creatinine, MAP, and loop diuretic exposure.
    """
    ranked = _rank_candidate_cases(
        art,
        df,
        label_col,
        quantile=quantile,
        reference_art=reference_art,
        reference_df=reference_df,
        reference_label_col=reference_label_col,
    )
    return ranked.head(top_k).copy()


def _rank_candidate_cases(
    art: ModelArtifact,
    df: pd.DataFrame,
    label_col: str,
    *,
    quantile: float,
    reference_art: ModelArtifact | None,
    reference_df: pd.DataFrame | None,
    reference_label_col: str | None,
) -> pd.DataFrame:
    """Return the fully ranked candidate pool for one explanation target."""
    if label_col not in df.columns:
        raise ValueError(f"label column {label_col!r} missing from candidate frame")

    candidates = df.dropna(subset=[label_col]).copy()
    if candidates.empty:
        raise ValueError("no rows with non-missing labels are available")

    X = candidates[art.feature_names]
    probs = artifact_predict_proba(art, X)
    candidates["_pred_proba"] = probs
    candidates["_missing_features"] = X.isna().sum(axis=1).to_numpy()

    positives = candidates[candidates[label_col].astype(int) == 1].copy()
    if not positives.empty:
        pool = positives
        selection_pool = "positive"
    else:
        pool = candidates
        selection_pool = "all"

    target_prob = float(pool["_pred_proba"].quantile(quantile))
    pool["_selection_pool"] = selection_pool
    pool["_selection_quantile"] = float(quantile)
    pool["_selection_target_prob"] = target_prob
    pool["_selection_strategy"] = "primary_quantile_match"

    reference_candidates = None
    reference_key_cols: list[str] = []

    if reference_art is not None and reference_df is not None:
        ref_label = reference_label_col or label_col
        ref_candidates = reference_df.dropna(subset=[ref_label]).copy()
        if not ref_candidates.empty:
            ref_X = ref_candidates[reference_art.feature_names]
            ref_candidates["_reference_pred_proba"] = artifact_predict_proba(reference_art, ref_X)

            key_cols = [c for c in ("stay_id", "subject_id", "landmark_time") if c in pool.columns and c in ref_candidates.columns]
            if key_cols:
                join_cols = key_cols + ["_reference_pred_proba"]
                pool = pool.merge(ref_candidates[join_cols], on=key_cols, how="inner")
                ref_pool = pool if positives.empty else pool[pool[label_col].astype(int) == 1].copy()
                if not ref_pool.empty:
                    ref_target_prob = float(ref_pool["_reference_pred_proba"].quantile(quantile))
                    pool["_reference_target_prob"] = ref_target_prob
                    pool["_distance_to_target"] = (
                        (pool["_pred_proba"] - target_prob).abs()
                        + 0.75 * (pool["_reference_pred_proba"] - ref_target_prob).abs()
                    )
                    pool["_agreement_gap"] = (pool["_pred_proba"] - pool["_reference_pred_proba"]).abs()
                    pool["_reference_model_tag"] = (
                        f"{reference_art.task}__{reference_art.family}__{reference_art.name}"
                    )
                    pool["_selection_strategy"] = "primary_plus_bedside_agreement"
                else:
                    pool["_distance_to_target"] = (pool["_pred_proba"] - target_prob).abs()
                    pool["_agreement_gap"] = np.nan
            else:
                pool["_distance_to_target"] = (pool["_pred_proba"] - target_prob).abs()
                pool["_agreement_gap"] = np.nan
            reference_candidates = ref_candidates
            reference_key_cols = key_cols
        else:
            pool["_distance_to_target"] = (pool["_pred_proba"] - target_prob).abs()
            pool["_agreement_gap"] = np.nan
    else:
        pool["_distance_to_target"] = (pool["_pred_proba"] - target_prob).abs()
        pool["_agreement_gap"] = np.nan

    sort_cols = ["_distance_to_target", "_agreement_gap", "_missing_features", "_pred_proba"]
    ascending = [True, True, True, False]
    for maybe_col in ("hours_since_icu_admit", "subject_id", "stay_id", "landmark_time"):
        if maybe_col in pool.columns:
            sort_cols.append(maybe_col)
            ascending.append(True)

    if (
        reference_art is not None
        and reference_art.name == "scorecard"
        and reference_candidates is not None
        and reference_key_cols
    ):
        shortlisted = _shortlist_legible_reference_cases(
            pool=pool,
            reference_art=reference_art,
            reference_candidates=reference_candidates,
            key_cols=reference_key_cols,
            sort_cols=sort_cols,
            ascending=ascending,
        )
        if shortlisted is not None and not shortlisted.empty:
            return shortlisted.reset_index(drop=True)

    return pool.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def _shortlist_legible_reference_cases(
    *,
    pool: pd.DataFrame,
    reference_art: ModelArtifact,
    reference_candidates: pd.DataFrame,
    key_cols: list[str],
    sort_cols: list[str],
    ascending: list[bool],
    shortlist_size: int = 200,
) -> pd.DataFrame | None:
    """Refine patient selection toward clinically legible bedside examples.

    The base selection already enforces held-out, high-risk, concordant cases.
    This refinement only reorders a small deterministic shortlist so the
    bedside scorecard explanation is not dominated by neutral reference bins.
    """
    prelim = pool.sort_values(sort_cols, ascending=ascending).head(shortlist_size).copy()
    if prelim.empty:
        return None

    ref_indexed = reference_candidates.set_index(key_cols, drop=False)
    summaries: list[dict[str, float | str | bool]] = []
    for _, row in prelim.iterrows():
        key = tuple(row[col] for col in key_cols)
        try:
            ref_row = ref_indexed.loc[key]
        except KeyError:
            summaries.append(_empty_legibility_summary())
            continue

        if isinstance(ref_row, pd.DataFrame):
            ref_row = ref_row.iloc[0]
        contrib = patient_additive_contributions(reference_art, ref_row)
        summaries.append(_scorecard_legibility_summary(contrib))

    summary_df = pd.DataFrame(summaries, index=prelim.index)
    prelim = prelim.join(summary_df)

    has_legible_case = bool((prelim["_reference_priority_hits"] > 0).any())
    if not has_legible_case:
        return None

    prelim["_selection_strategy"] = "primary_plus_bedside_legibility"
    refined_sort = [
        "_reference_priority_hits",
        "_reference_renal_hit",
        "_reference_hemodynamic_hit",
        "_reference_nonzero_terms",
        "_reference_context_hits",
        "_distance_to_target",
        "_agreement_gap",
        "_missing_features",
        "_pred_proba",
    ]
    refined_ascending = [False, False, False, False, False, True, True, True, False]
    for maybe_col in ("hours_since_icu_admit", "subject_id", "stay_id", "landmark_time"):
        if maybe_col in prelim.columns:
            refined_sort.append(maybe_col)
            refined_ascending.append(True)
    return prelim.sort_values(refined_sort, ascending=refined_ascending)


def _empty_legibility_summary() -> dict[str, float | str | bool]:
    return {
        "_reference_nonzero_terms": 0.0,
        "_reference_priority_hits": 0.0,
        "_reference_context_hits": 0.0,
        "_reference_renal_hit": False,
        "_reference_hemodynamic_hit": False,
        "_reference_active_terms": "",
    }


def _scorecard_legibility_summary(contrib_df: pd.DataFrame) -> dict[str, float | str | bool]:
    """Summarize how thesis-friendly a bedside scorecard explanation is."""
    active = contrib_df.loc[contrib_df["contribution_logodds"].abs() > 1e-9].copy()
    active_features = active["feature"].astype(str).tolist()

    priority_terms = {
        "creatinine_max_24h",
        "map_mean_12h",
        "loop_diuretic_24h",
    }
    context_terms = {
        "fluid_input_ml_12h",
        "glucose_std_24h",
        "age",
    }
    renal_hit = "creatinine_max_24h" in active_features
    hemodynamic_hit = "map_mean_12h" in active_features

    return {
        "_reference_nonzero_terms": float(len(active_features)),
        "_reference_priority_hits": float(sum(feature in priority_terms for feature in active_features)),
        "_reference_context_hits": float(sum(feature in context_terms for feature in active_features)),
        "_reference_renal_hit": renal_hit,
        "_reference_hemodynamic_hit": hemodynamic_hit,
        "_reference_active_terms": ",".join(active_features),
    }


def _ebm_contributions(art: ModelArtifact, x_row: pd.Series) -> pd.DataFrame:
    """Use EBM's local explanation API."""
    X = pd.DataFrame([x_row[art.feature_names].values], columns=art.feature_names)
    local = art.estimator.explain_local(X, y=None)
    d = local.data(0)
    return (
        pd.DataFrame({
            "feature": d["names"],
            "value": d.get("values", [np.nan] * len(d["names"])),
            "contribution_logodds": d["scores"],
        })
        .sort_values("contribution_logodds", key=np.abs, ascending=False, ignore_index=True)
    )


def _scorecard_contributions(art: ModelArtifact, x_row: pd.Series) -> pd.DataFrame:
    """Return per-feature scorecard contributions for one patient row."""
    if art.extra.get("scorecard_representation") == "binned":
        return _binned_scorecard_contributions(art, x_row)
    return _linear_scorecard_contributions(art, x_row)


def _linear_scorecard_contributions(art: ModelArtifact, x_row: pd.Series) -> pd.DataFrame:
    pipeline = art.estimator
    raw = x_row[art.feature_names].values.reshape(1, -1)
    imputed = pipeline.named_steps["impute"].transform(raw)
    scaled = pipeline.named_steps["scale"].transform(imputed)
    coef = pipeline.named_steps["clf"].coef_.ravel()

    contribs = coef * scaled.ravel()
    return (
        pd.DataFrame({
            "feature": art.feature_names,
            "value": raw.ravel(),
            "scaled_value": scaled.ravel(),
            "coefficient": coef,
            "contribution_logodds": contribs,
        })
        .sort_values("contribution_logodds", key=np.abs, ascending=False, ignore_index=True)
    )


def _binned_scorecard_contributions(art: ModelArtifact, x_row: pd.Series) -> pd.DataFrame:
    pipeline = art.estimator
    X = pd.DataFrame([x_row[art.feature_names]], columns=art.feature_names)
    transformed = pipeline.named_steps["design"].transform(X)
    coef = pipeline.named_steps["clf"].coef_.ravel()
    term_meta = art.extra.get("term_metadata", [])
    profiles = art.extra.get("feature_profiles", {})

    coeff_by_term = {
        meta.get("term_name"): float(c)
        for meta, c in zip(term_meta, coef, strict=True)
    }
    value_by_term = {
        meta.get("term_name"): float(v)
        for meta, v in zip(term_meta, transformed.ravel(), strict=True)
    }

    rows: list[dict[str, object]] = []
    for feature in art.feature_names:
        raw_value = float(x_row[feature])
        profile = profiles.get(feature, {})
        if profile.get("kind") == "binary":
            active = bool(round(raw_value))
            coefficient = coeff_by_term.get(feature, 0.0)
            rows.append(
                {
                    "feature": feature,
                    "value": raw_value,
                    "active_level": "1 / present" if active else "0 / absent",
                    "coefficient": coefficient if active else 0.0,
                    "contribution_logodds": coefficient if active else 0.0,
                }
            )
            continue

        active_level = profile.get("reference_bin_label", "reference")
        coefficient = 0.0
        for level in profile.get("bins", []):
            if level.get("reference"):
                continue
            term_name = level.get("term_name")
            if float(value_by_term.get(term_name, 0.0)) >= 0.5:
                active_level = level.get("range_display", level.get("label"))
                coefficient = coeff_by_term.get(term_name, 0.0)
                break
        rows.append(
            {
                "feature": feature,
                "value": raw_value,
                "active_level": active_level,
                "coefficient": coefficient,
                "contribution_logodds": coefficient,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("contribution_logodds", key=np.abs, ascending=False, ignore_index=True)
    )


def artifact_predict_proba(art: ModelArtifact, X: pd.DataFrame) -> np.ndarray:
    """Return calibrated probabilities for a saved model artifact."""
    X = X[art.feature_names]
    if art.calibrator is not None:
        return art.calibrator.predict_proba(X)[:, 1]
    return art.estimator.predict_proba(X)[:, 1]
