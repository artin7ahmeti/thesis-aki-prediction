-- features.rolling_aggregations

-- For every (landmark, feature_name, window_hours) combination,
-- compute: latest, delta, min, max, mean, std, count.
-- "Missing" indicator is derived downstream from count == 0.
--
-- Strict leakage guard: only events with t <= landmark_time AND
-- t > landmark_time - window_hours are included.
--
-- Implementation: an INNER JOIN between landmarks and numeric
-- events on stay_id, filtered by the time-window condition,
-- then GROUP BY (stay_id, landmark_time, feature_name, window_h).

CREATE OR REPLACE TABLE features.rolling_aggregations AS
WITH
windows AS (SELECT UNNEST([6, 12, 24]) AS window_h),
events_in_window AS (
    SELECT
        l.stay_id,
        l.landmark_time,
        w.window_h,
        e.feature_name,
        e.t,
        e.value,
        -- Rank within window: latest = rank 1 descending by t,
        --                     earliest = rank 1 ascending by t
        ROW_NUMBER() OVER (
            PARTITION BY l.stay_id, l.landmark_time, w.window_h, e.feature_name
            ORDER BY e.t DESC
        ) AS rn_desc,
        ROW_NUMBER() OVER (
            PARTITION BY l.stay_id, l.landmark_time, w.window_h, e.feature_name
            ORDER BY e.t ASC
        ) AS rn_asc
    FROM cohort.landmarks l
    CROSS JOIN windows w
    JOIN features.numeric_events e
      ON e.stay_id = l.stay_id
     AND e.t <= l.landmark_time
     AND e.t >  l.landmark_time - (w.window_h || ' hours')::INTERVAL
)
SELECT
    stay_id,
    landmark_time,
    window_h,
    feature_name,
    MIN(t) AS earliest_event_time,
    MAX(t) AS latest_event_time,
    COUNT(value) AS n_count,
    AVG(value) AS mean_val,
    STDDEV_SAMP(value) AS std_val,
    MIN(value) AS min_val,
    MAX(value) AS max_val,
    MAX(CASE WHEN rn_desc = 1 THEN value END) AS latest_val,
    MAX(CASE WHEN rn_asc  = 1 THEN value END) AS earliest_val,
    MAX(CASE WHEN rn_desc = 1 THEN value END)
        - MAX(CASE WHEN rn_asc = 1 THEN value END) AS delta_val
FROM events_in_window
GROUP BY stay_id, landmark_time, window_h, feature_name;

CREATE INDEX IF NOT EXISTS idx_rollagg_stay_time
    ON features.rolling_aggregations(stay_id, landmark_time);
