-- features.numeric_events

-- Long-format union of all numeric events (vitals + labs),
-- mapped to our feature names via a lookup table that Python
-- builds from configs/features.yaml.
--
-- The lookup table features.signal_map has columns:
--   feature_name TEXT, source TEXT, itemid INTEGER,
--   valid_low DOUBLE, valid_high DOUBLE, unit_multiplier DOUBLE,
--   unit_offset DOUBLE
-- (e.g. temperature_f rows get multiplier=5/9, offset=-32*5/9
--  to convert to °C under feature_name='temperature_c')

CREATE OR REPLACE TABLE features.numeric_events AS
WITH vitals_raw AS (
    SELECT
        ce.stay_id,
        ce.charttime AS t,
        m.feature_name,
        (ce.valuenum + m.unit_offset) * m.unit_multiplier AS value
    FROM mimic_icu.chartevents ce
    JOIN features.signal_map m
      ON m.source = 'chartevents' AND m.itemid = ce.itemid
    WHERE ce.valuenum IS NOT NULL
      AND (ce.valuenum + m.unit_offset) * m.unit_multiplier
          BETWEEN m.valid_low AND m.valid_high
),
labs_raw AS (
    SELECT
        c.stay_id,
        le.charttime AS t,
        m.feature_name,
        (le.valuenum + m.unit_offset) * m.unit_multiplier AS value
    FROM mimic_hosp.labevents le
    JOIN cohort.cohort c USING (hadm_id)
    JOIN features.signal_map m
      ON m.source = 'labevents' AND m.itemid = le.itemid
    WHERE le.valuenum IS NOT NULL
      AND (le.valuenum + m.unit_offset) * m.unit_multiplier
          BETWEEN m.valid_low AND m.valid_high
)
SELECT stay_id, t, feature_name, value FROM vitals_raw
UNION ALL
SELECT stay_id, t, feature_name, value FROM labs_raw;

CREATE INDEX IF NOT EXISTS idx_numev_stay_feat_t
    ON features.numeric_events(stay_id, feature_name, t);
