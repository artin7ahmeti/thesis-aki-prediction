-- Create logical schemas in the DuckDB database

-- mimic_hosp   : raw hospital module
-- mimic_icu    : raw ICU module
-- concepts     : derived clinical concepts (weight, KDIGO, etc.)
-- cohort       : analysis cohort and landmarks
-- labels       : AKI labels at each landmark
-- features     : rolling-window feature matrices
-- qa           : sanity-check views

CREATE SCHEMA IF NOT EXISTS mimic_hosp;
CREATE SCHEMA IF NOT EXISTS mimic_icu;
CREATE SCHEMA IF NOT EXISTS concepts;
CREATE SCHEMA IF NOT EXISTS cohort;
CREATE SCHEMA IF NOT EXISTS labels;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS qa;
