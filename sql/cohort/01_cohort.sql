-- cohort.cohort
-- Eligible ICU stays for AKI prediction.
--
-- Inclusion:
--   - age >= {{min_age}} at ICU admission
--   - ICU LOS between {{min_icu_los_hours}} h and {{max_icu_los_days}} d
--   - first ICU stay per hospital admission (earliest by intime)
-- Exclusion:
--   - ESRD (ICD-10 N18.6 / ICD-9 585.6) on any prior admission
--   - Prevalent AKI within first {{obs_window_hours}} of ICU stay
--     (i.e., AKI onset detected before the first landmark)
--
-- Template variables ({{ ... }}) are substituted by
-- aki.cohort.build at runtime from configs/cohort.yaml.

CREATE OR REPLACE TABLE cohort.cohort AS
WITH first_icu AS (
    -- Earliest ICU stay per (subject_id, hadm_id)
    SELECT
        s.*,
        ROW_NUMBER() OVER (
            PARTITION BY s.subject_id, s.hadm_id
            ORDER BY s.intime
        ) AS stay_rank
    FROM mimic_icu.icustays s
),
base AS (
    SELECT
        s.subject_id,
        s.hadm_id,
        s.stay_id,
        s.first_careunit,
        s.intime,
        s.outtime,
        s.los,
        p.gender AS sex,
        p.anchor_age AS age,
        p.anchor_year,
        p.anchor_year_group,
        p.dod,
        a.admittime,
        a.dischtime,
        a.deathtime,
        a.race AS ethnicity
    FROM first_icu s
    JOIN mimic_hosp.patients p USING (subject_id)
    JOIN mimic_hosp.admissions a ON s.hadm_id = a.hadm_id
    WHERE s.stay_rank = 1
      AND p.anchor_age >= {{min_age}}
      AND s.los >= {{min_icu_los_hours}} / 24.0
      AND s.los <= {{max_icu_los_days}}
),
esrd_patients AS (
    -- Any patient with ESRD diagnosis on any admission
    SELECT DISTINCT subject_id
    FROM mimic_hosp.diagnoses_icd
    WHERE (icd_version = 10 AND icd_code IN ({{esrd_icd10_list}}))
       OR (icd_version =  9 AND icd_code IN ({{esrd_icd9_list}}))
),
prevalent_aki AS (
    -- AKI onset falls within the observation window of the ICU stay
    SELECT
        b.stay_id
    FROM base b
    JOIN concepts.aki_onset o USING (stay_id)
    WHERE o.onset_stage1_time IS NOT NULL
      AND o.onset_stage1_time <= b.intime + INTERVAL {{obs_window_hours}} HOUR
)
SELECT
    b.*,
    -- Convenient age bucket for fairness analysis
    CASE
        WHEN b.age < 45 THEN '18-44'
        WHEN b.age < 65 THEN '45-64'
        WHEN b.age < 85 THEN '65-84'
        ELSE '85+'
    END AS age_group
FROM base b
WHERE b.subject_id NOT IN (SELECT subject_id FROM esrd_patients)
  AND b.stay_id     NOT IN (SELECT stay_id     FROM prevalent_aki);

CREATE INDEX IF NOT EXISTS idx_cohort_stay    ON cohort.cohort(stay_id);
CREATE INDEX IF NOT EXISTS idx_cohort_subject ON cohort.cohort(subject_id);
CREATE INDEX IF NOT EXISTS idx_cohort_year    ON cohort.cohort(anchor_year_group);
