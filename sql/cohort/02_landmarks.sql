-- cohort.landmarks
-- One row per (stay_id, landmark_time) for eligible prediction
-- times during each ICU stay.
--
-- Landmark generation rules:
--   - First landmark at intime + {{obs_window_hours}}
--   - New landmark every {{spacing_hours}} h thereafter
--   - Landmark must leave at least max(horizon) hours before stay end
--     (so both the 24 h and 48 h outcomes are evaluable)
--   - If {{exclude_after_aki}} = true: drop landmarks at or after
--     KDIGO stage-1 onset (incident-AKI framing)
--
-- Every landmark inherits static cohort columns for convenient joins.

CREATE OR REPLACE TABLE cohort.landmarks AS
WITH hours_grid AS (
    SELECT
        c.stay_id,
        c.subject_id,
        c.hadm_id,
        c.anchor_year_group,
        c.age,
        c.age_group,
        c.sex,
        c.ethnicity,
        c.intime,
        c.outtime,
        unnest(generate_series(
            c.intime + INTERVAL {{obs_window_hours}} HOUR,
            c.outtime - INTERVAL {{max_horizon_hours}} HOUR,
            INTERVAL {{spacing_hours}} HOUR
        )) AS landmark_time
    FROM cohort.cohort c
    WHERE c.outtime - c.intime >= INTERVAL {{min_stay_hours}} HOUR
),
with_onset AS (
    SELECT
        g.*,
        o.onset_stage1_time,
        o.onset_stage2_time,
        o.onset_stage3_time
    FROM hours_grid g
    LEFT JOIN concepts.aki_onset o USING (stay_id)
)
SELECT
    stay_id,
    subject_id,
    hadm_id,
    anchor_year_group,
    age,
    age_group,
    sex,
    ethnicity,
    intime,
    outtime,
    landmark_time,
    onset_stage1_time,
    onset_stage2_time,
    onset_stage3_time,
    -- Hours since ICU admission (useful feature / stratifier)
    date_diff('hour', intime, landmark_time) AS hours_since_icu_admit
FROM with_onset
WHERE
    -- Incident-AKI framing: drop landmarks at or after stage-1 onset
    {{exclude_after_aki}} = FALSE
 OR onset_stage1_time IS NULL
 OR landmark_time < onset_stage1_time;

CREATE INDEX IF NOT EXISTS idx_lm_stay_time
    ON cohort.landmarks(stay_id, landmark_time);
CREATE INDEX IF NOT EXISTS idx_lm_subject
    ON cohort.landmarks(subject_id);
