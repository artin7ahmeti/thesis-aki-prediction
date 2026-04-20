-- concepts.aki_onset

-- First time each stay crosses KDIGO stage thresholds, and the
-- corresponding source (creatinine vs. urine output).
--
-- Used to:
--   1. Exclude landmarks that fall after AKI onset (incident framing)
--   2. Label each landmark, did AKI onset occur within the horizon?
--   3. Exclude stays with prevalent AKI during the observation window

CREATE OR REPLACE TABLE concepts.aki_onset AS
SELECT
    stay_id,
    -- Stage >=1 onset (primary outcome)
    MIN(CASE WHEN kdigo_stage >= 1 THEN t_hour END) AS onset_stage1_time,
    -- Stage >=2 onset (secondary outcome)
    MIN(CASE WHEN kdigo_stage >= 2 THEN t_hour END) AS onset_stage2_time,
    -- Stage >=3 onset
    MIN(CASE WHEN kdigo_stage >= 3 THEN t_hour END) AS onset_stage3_time,
    -- Source of stage-1 onset: was it driven by creatinine or UO?
    (SELECT CASE
        WHEN kdigo_cr_stage >= 1 AND COALESCE(kdigo_uo_stage, 0) < 1 THEN 'creatinine'
        WHEN COALESCE(kdigo_cr_stage, 0) < 1 AND kdigo_uo_stage >= 1 THEN 'urine_output'
        WHEN kdigo_cr_stage >= 1 AND kdigo_uo_stage >= 1 THEN 'both'
        ELSE NULL
     END
     FROM concepts.kdigo_stages inner_ks
     WHERE inner_ks.stay_id = outer_ks.stay_id
       AND inner_ks.kdigo_stage >= 1
     ORDER BY inner_ks.t_hour
     LIMIT 1) AS stage1_source
FROM concepts.kdigo_stages outer_ks
GROUP BY stay_id;

CREATE INDEX IF NOT EXISTS idx_onset_stay ON concepts.aki_onset(stay_id);
