-- concepts.kdigo_creatinine

-- KDIGO creatinine-based AKI stage evaluated at each creatinine
-- measurement during the ICU stay.
--
-- Criteria:
--   Stage 3: Cr >= 3 x baseline  OR  Cr >= 4.0 mg/dL
--   Stage 2: Cr >= 2 x baseline
--   Stage 1: Cr >= 1.5 x baseline (within 7 days)
--            OR absolute rise >= 0.3 mg/dL over trailing 48 h
--   Stage 0: none of the above
--
-- When baseline is missing, we can only apply:
--   Stage 1: abs rise >= 0.3 over 48h
--   Stage 3: Cr >= 4.0 (absolute criterion)
-- All stages using ratio-to-baseline become NULL -> effectively
-- Stage 0 unless the absolute criteria trigger.

CREATE OR REPLACE TABLE concepts.kdigo_creatinine AS
WITH stay_cr AS (
    -- Every creatinine measurement during an ICU stay
    SELECT
        s.stay_id,
        s.subject_id,
        s.hadm_id,
        s.intime,
        c.charttime,
        c.creatinine_mg_dl AS cr
    FROM mimic_icu.icustays s
    JOIN concepts.creatinine c
      ON c.subject_id = s.subject_id
     AND c.charttime BETWEEN s.intime AND s.intime + INTERVAL 30 DAY
),
cr_with_baseline AS (
    SELECT
        sc.*,
        b.baseline_mg_dl AS baseline
    FROM stay_cr sc
    LEFT JOIN concepts.creatinine_baseline b USING (stay_id)
),
cr_with_deltas AS (
    SELECT
        sc.*,
        -- Trailing 48h minimum creatinine (for 0.3 absolute-rise criterion)
        MIN(cr) OVER (
            PARTITION BY stay_id
            ORDER BY charttime
            RANGE BETWEEN INTERVAL 48 HOUR PRECEDING AND CURRENT ROW
        ) AS cr_min_48h,
        -- Trailing 7d minimum creatinine (for 1.5x-baseline criterion when no pre-ICU baseline)
        MIN(cr) OVER (
            PARTITION BY stay_id
            ORDER BY charttime
            RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND CURRENT ROW
        ) AS cr_min_7d
    FROM cr_with_baseline sc
)
SELECT
    stay_id,
    subject_id,
    hadm_id,
    charttime,
    cr AS creatinine_mg_dl,
    baseline,
    cr_min_48h,
    cr_min_7d,
    -- Effective baseline for ratio comparison: pre-ICU baseline if available,
    -- else the trailing-7d minimum during ICU (MIMIC-Code fallback).
    COALESCE(baseline, cr_min_7d) AS ref_baseline,
    -- Stage determination
    CASE
        WHEN cr >= 4.0
             AND (cr - cr_min_48h) >= 0.3  -- avoids labeling stable chronic >=4
            THEN 3
        WHEN COALESCE(baseline, cr_min_7d) IS NOT NULL
             AND cr / NULLIF(COALESCE(baseline, cr_min_7d), 0) >= 3.0
            THEN 3
        WHEN COALESCE(baseline, cr_min_7d) IS NOT NULL
             AND cr / NULLIF(COALESCE(baseline, cr_min_7d), 0) >= 2.0
            THEN 2
        WHEN COALESCE(baseline, cr_min_7d) IS NOT NULL
             AND cr / NULLIF(COALESCE(baseline, cr_min_7d), 0) >= 1.5
            THEN 1
        WHEN (cr - cr_min_48h) >= 0.3
            THEN 1
        ELSE 0
    END AS kdigo_cr_stage
FROM cr_with_deltas;

CREATE INDEX IF NOT EXISTS idx_kcr_stay_time
    ON concepts.kdigo_creatinine(stay_id, charttime);
