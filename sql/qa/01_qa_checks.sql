-- qa.qa_report

-- Sanity checks executed after each pipeline stage. All queries
-- return one row; the Python layer logs these values and fails
-- loudly if any zero/NULL invariants are violated.

CREATE OR REPLACE VIEW qa.cohort_summary AS
SELECT
    COUNT(*)                                    AS n_stays,
    COUNT(DISTINCT subject_id)                  AS n_patients,
    COUNT(DISTINCT hadm_id)                     AS n_admissions,
    MIN(age)                                    AS min_age,
    MAX(age)                                    AS max_age,
    AVG(age)                                    AS mean_age,
    SUM(CASE WHEN sex = 'F' THEN 1 ELSE 0 END)  AS n_female,
    MIN(anchor_year_group)                      AS earliest_year_group,
    MAX(anchor_year_group)                      AS latest_year_group
FROM cohort.cohort;

CREATE OR REPLACE VIEW qa.landmark_summary AS
SELECT
    COUNT(*)                         AS n_landmarks,
    COUNT(DISTINCT stay_id)          AS n_stays_with_landmarks,
    COUNT(DISTINCT subject_id)       AS n_patients_with_landmarks,
    AVG(hours_since_icu_admit)       AS mean_hours_since_admit,
    MAX(hours_since_icu_admit)       AS max_hours_since_admit
FROM cohort.landmarks;

CREATE OR REPLACE VIEW qa.label_prevalence AS
SELECT
    COUNT(*)                                           AS n_landmarks,
    AVG(y_stage1_24h)                                  AS prev_stage1_24h,
    AVG(y_stage1_48h)                                  AS prev_stage1_48h,
    AVG(y_stage2_24h)                                  AS prev_stage2_24h,
    AVG(y_stage2_48h)                                  AS prev_stage2_48h,
    AVG(y_cr_only_stage1_24h)                          AS prev_cr_only_24h,
    AVG(y_cr_only_stage1_48h)                          AS prev_cr_only_48h
FROM labels.labels;

-- Leakage guard: rolling feature rows must be built only from events inside
-- their declared retrospective window. Raw future events after a landmark are
-- expected; only events used in features would be leakage.
CREATE OR REPLACE VIEW qa.leakage_check AS
SELECT
    COUNT(*) AS n_rolling_rows,
    SUM(CASE WHEN latest_event_time > landmark_time THEN 1 ELSE 0 END)
        AS n_future_feature_rows,
    SUM(
        CASE
            WHEN earliest_event_time <= landmark_time - (window_h || ' hours')::INTERVAL
            THEN 1 ELSE 0
        END
    ) AS n_before_window_feature_rows,
    MAX(latest_event_time - landmark_time) AS max_future_offset
FROM features.rolling_aggregations;

-- Baseline-creatinine coverage (what fraction of stays have a baseline)
CREATE OR REPLACE VIEW qa.baseline_coverage AS
SELECT
    COUNT(*)                                                AS n_stays,
    SUM(CASE WHEN baseline_mg_dl IS NOT NULL THEN 1 END)    AS n_with_baseline,
    SUM(CASE WHEN baseline_7d   IS NOT NULL THEN 1 END)     AS n_7d_baseline,
    SUM(CASE WHEN baseline_365d IS NOT NULL THEN 1 END)     AS n_365d_baseline
FROM concepts.creatinine_baseline;

-- Same baseline coverage restricted to the eligible modeling cohort.
CREATE OR REPLACE VIEW qa.cohort_baseline_coverage AS
SELECT
    COUNT(*)                                                AS n_stays,
    SUM(CASE WHEN b.baseline_mg_dl IS NOT NULL THEN 1 END)  AS n_with_baseline,
    SUM(CASE WHEN b.baseline_7d   IS NOT NULL THEN 1 END)   AS n_7d_baseline,
    SUM(CASE WHEN b.baseline_365d IS NOT NULL THEN 1 END)   AS n_365d_baseline
FROM cohort.cohort c
LEFT JOIN concepts.creatinine_baseline b USING (stay_id);

-- Patient overlap between splits (must be zero)
CREATE OR REPLACE VIEW qa.split_patient_overlap AS
SELECT
    'placeholder' AS note
-- Populated by Python: queries each split DataFrame for subject_id intersection.
;
