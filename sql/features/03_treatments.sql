-- features.treatments

-- Per-landmark treatment exposure flags and continuous sums
-- within the observation window (24 h before landmark).
--
-- Emits:
--   vasopressor_any_24h     : 1 if any vasopressor was running
--                              at any point in [lm-24h, lm]
--   loop_diuretic_24h       : 1 if loop diuretic administered
--   mech_vent_24h           : 1 if invasive/NI vent setting present
--   fluid_input_ml_6h/12h/24h : summed inputevents.amount (mL-like)
--   urine_output_ml_6h/12h/24h: net summed outputevents.value
--
-- All windowed strictly to times <= landmark.

CREATE OR REPLACE TABLE features.treatments AS
WITH vaso AS (
    SELECT
        l.stay_id,
        l.landmark_time,
        MAX(CASE WHEN i.starttime <= l.landmark_time
                  AND i.endtime   >  l.landmark_time - INTERVAL 24 HOUR
                THEN 1 ELSE 0 END) AS vasopressor_any_24h
    FROM cohort.landmarks l
    LEFT JOIN mimic_icu.inputevents i
           ON i.stay_id = l.stay_id
          AND i.itemid IN (221906, 222315, 221289, 221662, 221749, 221653)
    GROUP BY l.stay_id, l.landmark_time
),
loop_diuretic AS (
    SELECT
        l.stay_id,
        l.landmark_time,
        MAX(CASE WHEN i.starttime BETWEEN l.landmark_time - INTERVAL 24 HOUR AND l.landmark_time
                THEN 1 ELSE 0 END) AS loop_diuretic_24h
    FROM cohort.landmarks l
    LEFT JOIN mimic_icu.inputevents i
           ON i.stay_id = l.stay_id
          AND i.itemid = 221794
    GROUP BY l.stay_id, l.landmark_time
),
mech_vent AS (
    SELECT
        l.stay_id,
        l.landmark_time,
        MAX(CASE WHEN ce.charttime BETWEEN l.landmark_time - INTERVAL 24 HOUR AND l.landmark_time
                THEN 1 ELSE 0 END) AS mech_vent_24h
    FROM cohort.landmarks l
    LEFT JOIN mimic_icu.chartevents ce
           ON ce.stay_id = l.stay_id
          AND ce.itemid IN (225792, 225794)
    GROUP BY l.stay_id, l.landmark_time
),
fluid_in AS (
    SELECT
        l.stay_id,
        l.landmark_time,
        SUM(CASE WHEN i.starttime BETWEEN l.landmark_time - INTERVAL  6 HOUR AND l.landmark_time
                 THEN i.amount END) AS fluid_input_ml_6h,
        SUM(CASE WHEN i.starttime BETWEEN l.landmark_time - INTERVAL 12 HOUR AND l.landmark_time
                 THEN i.amount END) AS fluid_input_ml_12h,
        SUM(CASE WHEN i.starttime BETWEEN l.landmark_time - INTERVAL 24 HOUR AND l.landmark_time
                 THEN i.amount END) AS fluid_input_ml_24h
    FROM cohort.landmarks l
    LEFT JOIN mimic_icu.inputevents i
           ON i.stay_id = l.stay_id
          AND i.amountuom IN ('ml', 'mL')
    GROUP BY l.stay_id, l.landmark_time
),
urine_out AS (
    SELECT
        l.stay_id,
        l.landmark_time,
        GREATEST(
            COALESCE(SUM(CASE WHEN o.charttime BETWEEN l.landmark_time - INTERVAL  6 HOUR AND l.landmark_time
                     THEN CASE WHEN o.itemid = 227488 THEN -o.value ELSE o.value END END), 0),
            0
        ) AS urine_output_ml_6h,
        GREATEST(
            COALESCE(SUM(CASE WHEN o.charttime BETWEEN l.landmark_time - INTERVAL 12 HOUR AND l.landmark_time
                     THEN CASE WHEN o.itemid = 227488 THEN -o.value ELSE o.value END END), 0),
            0
        ) AS urine_output_ml_12h,
        GREATEST(
            COALESCE(SUM(CASE WHEN o.charttime BETWEEN l.landmark_time - INTERVAL 24 HOUR AND l.landmark_time
                     THEN CASE WHEN o.itemid = 227488 THEN -o.value ELSE o.value END END), 0),
            0
        ) AS urine_output_ml_24h
    FROM cohort.landmarks l
    LEFT JOIN mimic_icu.outputevents o
           ON o.stay_id = l.stay_id
          AND o.itemid IN (226559,226560,226561,226584,226563,226564,226565,226567,226557,227488,227489)
    GROUP BY l.stay_id, l.landmark_time
)
SELECT
    v.stay_id,
    v.landmark_time,
    v.vasopressor_any_24h,
    d.loop_diuretic_24h,
    m.mech_vent_24h,
    f.fluid_input_ml_6h,  f.fluid_input_ml_12h,  f.fluid_input_ml_24h,
    u.urine_output_ml_6h, u.urine_output_ml_12h, u.urine_output_ml_24h
FROM vaso v
JOIN loop_diuretic d USING (stay_id, landmark_time)
JOIN mech_vent     m USING (stay_id, landmark_time)
JOIN fluid_in      f USING (stay_id, landmark_time)
JOIN urine_out     u USING (stay_id, landmark_time);

CREATE INDEX IF NOT EXISTS idx_tx_stay_time
    ON features.treatments(stay_id, landmark_time);
