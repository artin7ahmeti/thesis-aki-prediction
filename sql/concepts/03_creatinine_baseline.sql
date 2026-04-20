-- concepts.creatinine_baseline

-- Baseline creatinine per ICU stay, following MIMIC-Code logic:
--   Primary  : lowest creatinine measurement in the 7 days
--              preceding ICU admission (and before intime).
--   Fallback : lowest creatinine in the 365 days before ICU admit
--              (captures chronic baseline).
--   If none  : NULL — patient has no pre-ICU creatinine; KDIGO
--              will fall back to absolute + 48h-delta criteria only.
--
-- Rationale: using the lowest prior creatinine avoids falsely
-- labeling chronically elevated patients as acute; and excluding
-- the intime point itself prevents leakage from the first ICU lab.

CREATE OR REPLACE TABLE concepts.creatinine_baseline AS
WITH stay_base AS (
    SELECT
        s.subject_id,
        s.hadm_id,
        s.stay_id,
        s.intime
    FROM mimic_icu.icustays s
)
SELECT
    sb.stay_id,
    sb.subject_id,
    sb.hadm_id,
    sb.intime,
    -- Primary: 7-day lookback
    MIN(CASE
        WHEN c.charttime BETWEEN sb.intime - INTERVAL 7 DAY AND sb.intime - INTERVAL 1 MINUTE
        THEN c.creatinine_mg_dl
    END) AS baseline_7d,
    -- Fallback: 365-day lookback
    MIN(CASE
        WHEN c.charttime BETWEEN sb.intime - INTERVAL 365 DAY AND sb.intime - INTERVAL 1 MINUTE
        THEN c.creatinine_mg_dl
    END) AS baseline_365d
FROM stay_base sb
LEFT JOIN concepts.creatinine c
       ON c.subject_id = sb.subject_id
      AND c.charttime < sb.intime
      AND c.charttime >= sb.intime - INTERVAL 365 DAY
GROUP BY sb.stay_id, sb.subject_id, sb.hadm_id, sb.intime;

-- Resolved baseline: prefer 7d, fall back to 365d, else NULL
ALTER TABLE concepts.creatinine_baseline
    ADD COLUMN baseline_mg_dl DOUBLE;

UPDATE concepts.creatinine_baseline
SET baseline_mg_dl = COALESCE(baseline_7d, baseline_365d);
