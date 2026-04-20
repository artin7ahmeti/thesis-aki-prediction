-- labels.labels

-- Per-landmark binary outcomes for each (stage, horizon) combination:
--   - y_stage1_24h : incident KDIGO>=1 within (landmark, landmark + 24h]
--   - y_stage1_48h : incident KDIGO>=1 within (landmark, landmark + 48h]
--   - y_stage2_24h : incident KDIGO>=2 within (landmark, landmark + 24h]
--   - y_stage2_48h : incident KDIGO>=2 within (landmark, landmark + 48h]
--
-- Landmark-level semantics ("incident"):
--   - Positive iff onset time falls strictly AFTER landmark and
--     on-or-before landmark + horizon.
--   - Negative iff no onset in horizon AND patient is still at
--     risk at landmark (landmark excluded after onset upstream).
--
-- Censoring:
--   - ICU discharge or death during horizon with no AKI -> treated
--     as negative (standard convention for short-horizon AKI
--     prediction; sensitivity analyses can toggle this).

CREATE OR REPLACE TABLE labels.labels AS
SELECT
    l.stay_id,
    l.subject_id,
    l.landmark_time,

    -- Primary: stage-1 incident AKI
    CASE
        WHEN l.onset_stage1_time IS NOT NULL
         AND l.onset_stage1_time > l.landmark_time
         AND l.onset_stage1_time <= l.landmark_time + INTERVAL 24 HOUR
        THEN 1
        ELSE 0
    END AS y_stage1_24h,

    CASE
        WHEN l.onset_stage1_time IS NOT NULL
         AND l.onset_stage1_time > l.landmark_time
         AND l.onset_stage1_time <= l.landmark_time + INTERVAL 48 HOUR
        THEN 1
        ELSE 0
    END AS y_stage1_48h,

    -- Secondary: stage-2 incident AKI
    CASE
        WHEN l.onset_stage2_time IS NOT NULL
         AND l.onset_stage2_time > l.landmark_time
         AND l.onset_stage2_time <= l.landmark_time + INTERVAL 24 HOUR
        THEN 1
        ELSE 0
    END AS y_stage2_24h,

    CASE
        WHEN l.onset_stage2_time IS NOT NULL
         AND l.onset_stage2_time > l.landmark_time
         AND l.onset_stage2_time <= l.landmark_time + INTERVAL 48 HOUR
        THEN 1
        ELSE 0
    END AS y_stage2_48h,

    -- Sensitivity: creatinine-only labels ignore UO-driven onset
    CASE
        WHEN EXISTS (
            SELECT 1 FROM concepts.kdigo_creatinine k
            WHERE k.stay_id = l.stay_id
              AND k.charttime > l.landmark_time
              AND k.charttime <= l.landmark_time + INTERVAL 24 HOUR
              AND k.kdigo_cr_stage >= 1
        ) THEN 1 ELSE 0
    END AS y_cr_only_stage1_24h,

    CASE
        WHEN EXISTS (
            SELECT 1 FROM concepts.kdigo_creatinine k
            WHERE k.stay_id = l.stay_id
              AND k.charttime > l.landmark_time
              AND k.charttime <= l.landmark_time + INTERVAL 48 HOUR
              AND k.kdigo_cr_stage >= 1
        ) THEN 1 ELSE 0
    END AS y_cr_only_stage1_48h
FROM cohort.landmarks l;

CREATE INDEX IF NOT EXISTS idx_labels_stay_time
    ON labels.labels(stay_id, landmark_time);
