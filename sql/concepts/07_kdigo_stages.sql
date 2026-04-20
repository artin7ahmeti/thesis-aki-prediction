-- concepts.kdigo_stages

-- Hourly KDIGO stage per ICU stay, combining creatinine- and
-- urine-output-derived stages by taking the MAXIMUM.
--
-- For hours without a creatinine measurement, the creatinine
-- stage is carried forward from the last observed measurement
-- (up to 7 days). This matches standard clinical practice:
-- once an AKI stage is reached it persists until a new
-- measurement resolves it.

CREATE OR REPLACE TABLE concepts.kdigo_stages AS
WITH hours AS (
    -- Hourly grid per stay (reuse from concepts.kdigo_uo)
    SELECT DISTINCT stay_id, uo_hour AS t_hour
    FROM concepts.kdigo_uo
),
cr_stage_per_hour AS (
    -- Carry-forward the most recent creatinine stage up to 7 days
    SELECT
        h.stay_id,
        h.t_hour,
        (SELECT k.kdigo_cr_stage
         FROM concepts.kdigo_creatinine k
         WHERE k.stay_id = h.stay_id
           AND k.charttime <= h.t_hour
           AND k.charttime >= h.t_hour - INTERVAL 7 DAY
         ORDER BY k.charttime DESC
         LIMIT 1) AS kdigo_cr_stage
    FROM hours h
)
SELECT
    h.stay_id,
    h.t_hour,
    cr.kdigo_cr_stage,
    uo.kdigo_uo_stage,
    GREATEST(
        COALESCE(cr.kdigo_cr_stage, 0),
        COALESCE(uo.kdigo_uo_stage, 0)
    ) AS kdigo_stage
FROM hours h
LEFT JOIN cr_stage_per_hour cr
       ON cr.stay_id = h.stay_id AND cr.t_hour = h.t_hour
LEFT JOIN concepts.kdigo_uo uo
       ON uo.stay_id = h.stay_id AND uo.uo_hour = h.t_hour;

CREATE INDEX IF NOT EXISTS idx_kstage_stay_hour
    ON concepts.kdigo_stages(stay_id, t_hour);
