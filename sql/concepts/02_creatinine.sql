-- concepts.creatinine
-- Per-patient creatinine measurements, de-duplicated and filtered.
-- Unit: mg/dL. Valid physiological range: 0.1 – 30.

CREATE OR REPLACE TABLE concepts.creatinine AS
SELECT
    le.subject_id,
    le.hadm_id,
    le.charttime,
    le.valuenum AS creatinine_mg_dl
FROM mimic_hosp.labevents le
WHERE le.itemid = 50912
  AND le.valuenum IS NOT NULL
  AND le.valuenum BETWEEN 0.1 AND 30.0;

CREATE INDEX IF NOT EXISTS idx_cr_subject ON concepts.creatinine(subject_id);
CREATE INDEX IF NOT EXISTS idx_cr_hadm ON concepts.creatinine(hadm_id);
CREATE INDEX IF NOT EXISTS idx_cr_time ON concepts.creatinine(charttime);
