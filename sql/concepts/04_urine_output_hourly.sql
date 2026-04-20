-- concepts.urine_output_hourly

-- Hourly urine output (mL) per ICU stay, binned to the start of
-- each hour since ICU admission. Used for KDIGO UO staging.
--
-- MIMIC outputevents charttime represents the end of the interval
-- over which urine was collected; we attribute it to the hour
-- ending at that timestamp.

CREATE OR REPLACE TABLE concepts.urine_output_hourly AS
WITH uo_raw AS (
    SELECT
        stay_id,
        date_trunc('hour', charttime) AS uo_hour,
        SUM(value) AS uo_ml
    FROM mimic_icu.outputevents
    WHERE itemid IN (
        226559,  -- Foley
        226560,  -- Void
        226561,  -- Condom Cath
        226584,  -- Ileoconduit
        226563,  -- Suprapubic
        226564,  -- R Nephrostomy
        226565,  -- L Nephrostomy
        226567,  -- Straight Cath
        226557,  -- R Ureteral Stent
        227488,  -- GU Irrigant Volume In (subtract)
        227489   -- GU Irrigant/Urine Volume Out
    )
      AND value IS NOT NULL
      AND value >= 0
    GROUP BY stay_id, date_trunc('hour', charttime)
)
SELECT
    stay_id,
    uo_hour,
    uo_ml
FROM uo_raw;

CREATE INDEX IF NOT EXISTS idx_uo_stay_time
    ON concepts.urine_output_hourly(stay_id, uo_hour);
