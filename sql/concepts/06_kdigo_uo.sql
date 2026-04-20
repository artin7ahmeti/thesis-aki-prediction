-- concepts.kdigo_uo

-- KDIGO urine-output-based AKI stage, evaluated hourly during
-- the ICU stay. Units: mL/kg/h.
--
-- Criteria (require sustained low UO):
--   Stage 3: UO < 0.3 mL/kg/h for 24 h  OR  anuria >= 12 h
--   Stage 2: UO < 0.5 mL/kg/h for 12 h
--   Stage 1: UO < 0.5 mL/kg/h for 6 h
--
-- Implementation: for each hour, compute total UO and total hours
-- over trailing 6h / 12h / 24h. A sustained low UO must have
-- enough measurement coverage (we require ≥ floor(hours/2)
-- observed hours to avoid false positives from gaps).

CREATE OR REPLACE TABLE concepts.kdigo_uo AS
WITH hourly_grid AS (
    -- Expand every stay to an hourly grid (up to 30 days)
    SELECT
        s.stay_id,
        s.subject_id,
        s.intime,
        s.outtime,
        generate_series(
            date_trunc('hour', s.intime),
            LEAST(date_trunc('hour', s.outtime), date_trunc('hour', s.intime + INTERVAL 30 DAY)),
            INTERVAL 1 HOUR
        ) AS uo_hour_list
    FROM mimic_icu.icustays s
),
hourly AS (
    SELECT
        g.stay_id,
        g.subject_id,
        unnest(g.uo_hour_list) AS uo_hour
    FROM hourly_grid g
),
hourly_uo AS (
    SELECT
        h.stay_id,
        h.subject_id,
        h.uo_hour,
        COALESCE(uo.uo_ml, 0) AS uo_ml,
        CASE WHEN uo.uo_ml IS NOT NULL THEN 1 ELSE 0 END AS has_uo
    FROM hourly h
    LEFT JOIN concepts.urine_output_hourly uo
      ON uo.stay_id = h.stay_id AND uo.uo_hour = h.uo_hour
),
hourly_windows AS (
    SELECT
        h.stay_id,
        h.uo_hour,
        w.weight_kg,
        -- 6h window
        SUM(uo_ml) OVER w6 AS uo_6h_ml,
        SUM(has_uo) OVER w6 AS n_obs_6h,
        -- 12h window
        SUM(uo_ml) OVER w12 AS uo_12h_ml,
        SUM(has_uo) OVER w12 AS n_obs_12h,
        -- 24h window
        SUM(uo_ml) OVER w24 AS uo_24h_ml,
        SUM(has_uo) OVER w24 AS n_obs_24h
    FROM hourly_uo h
    LEFT JOIN concepts.patient_weight_kg w USING (stay_id)
    WINDOW
        w6 AS (PARTITION BY stay_id ORDER BY uo_hour
               ROWS BETWEEN 5 PRECEDING AND CURRENT ROW),
        w12 AS (PARTITION BY stay_id ORDER BY uo_hour
                ROWS BETWEEN 11 PRECEDING AND CURRENT ROW),
        w24 AS (PARTITION BY stay_id ORDER BY uo_hour
                ROWS BETWEEN 23 PRECEDING AND CURRENT ROW)
)
SELECT
    stay_id,
    uo_hour,
    weight_kg,
    uo_6h_ml,  uo_12h_ml,  uo_24h_ml,
    n_obs_6h,  n_obs_12h,  n_obs_24h,
    -- mL/kg/h
    CASE WHEN weight_kg IS NOT NULL AND n_obs_6h  >= 3 THEN uo_6h_ml  / (weight_kg * 6 ) END AS uo_rate_6h,
    CASE WHEN weight_kg IS NOT NULL AND n_obs_12h >= 6 THEN uo_12h_ml / (weight_kg * 12) END AS uo_rate_12h,
    CASE WHEN weight_kg IS NOT NULL AND n_obs_24h >= 12 THEN uo_24h_ml / (weight_kg * 24) END AS uo_rate_24h,
    -- KDIGO UO stage
    CASE
        WHEN weight_kg IS NULL THEN NULL
        WHEN n_obs_12h >= 12 AND uo_12h_ml = 0 THEN 3
        WHEN n_obs_24h >= 12 AND uo_24h_ml / (weight_kg * 24) < 0.3 THEN 3
        WHEN n_obs_12h >= 6  AND uo_12h_ml / (weight_kg * 12) < 0.5 THEN 2
        WHEN n_obs_6h  >= 3  AND uo_6h_ml  / (weight_kg *  6) < 0.5 THEN 1
        ELSE 0
    END AS kdigo_uo_stage
FROM hourly_windows;

CREATE INDEX IF NOT EXISTS idx_kuo_stay_hour
    ON concepts.kdigo_uo(stay_id, uo_hour);
