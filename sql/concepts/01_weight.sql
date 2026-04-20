-- concepts.patient_weight_kg
-- Best-available patient weight per ICU stay, in kg.
-- Priority order:
--   1. inputevents.patientweight (recorded during med administration)
--   2. chartevents weight items (daily weight 224639, admit weight 226512)
--   3. NULL  (caller must handle, UO /kg/h features will be missing)

CREATE OR REPLACE TABLE concepts.patient_weight_kg AS
WITH
weight_input AS (
    SELECT
        stay_id,
        AVG(patientweight) AS weight_kg
    FROM mimic_icu.inputevents
    WHERE patientweight IS NOT NULL
      AND patientweight BETWEEN 30 AND 300
    GROUP BY stay_id
),
weight_chart AS (
    SELECT
        stay_id,
        AVG(valuenum) AS weight_kg
    FROM mimic_icu.chartevents
    WHERE itemid IN (224639, 226512, 226531)   -- Daily Weight, Admit Wt, Admission Wt (lb)
      AND valuenum IS NOT NULL
      AND valuenum BETWEEN 30 AND 300
    GROUP BY stay_id
)
SELECT
    s.stay_id,
    COALESCE(wi.weight_kg, wc.weight_kg) AS weight_kg
FROM mimic_icu.icustays s
LEFT JOIN weight_input wi ON s.stay_id = wi.stay_id
LEFT JOIN weight_chart wc ON s.stay_id = wc.stay_id;

CREATE INDEX IF NOT EXISTS idx_weight_stay ON concepts.patient_weight_kg(stay_id);
