-- Stage hospital-module tables as DuckDB views over raw CSV.gz

-- Using VIEWs keeps the DuckDB file small; DuckDB reads CSV.gz
-- directly with parallel decompression. For heavier queries,
-- replace with CREATE TABLE ... AS to materialize.
-- Template variables are substituted by aki.data.db at runtime:
--   {{raw_dir}}  -> absolute path to data/raw

CREATE OR REPLACE VIEW mimic_hosp.patients AS
SELECT
    subject_id,
    gender,
    anchor_age,
    anchor_year,
    anchor_year_group,
    dod
FROM read_csv_auto('{{raw_dir}}/hosp/patients.csv.gz');

CREATE OR REPLACE VIEW mimic_hosp.admissions AS
SELECT
    subject_id,
    hadm_id,
    admittime::TIMESTAMP AS admittime,
    dischtime::TIMESTAMP AS dischtime,
    deathtime::TIMESTAMP AS deathtime,
    admission_type,
    admission_location,
    discharge_location,
    insurance,
    language,
    marital_status,
    race,
    hospital_expire_flag
FROM read_csv_auto('{{raw_dir}}/hosp/admissions.csv.gz');

CREATE OR REPLACE VIEW mimic_hosp.labevents AS
SELECT
    labevent_id,
    subject_id,
    hadm_id,
    specimen_id,
    itemid,
    charttime::TIMESTAMP AS charttime,
    storetime::TIMESTAMP AS storetime,
    valuenum,
    valueuom,
    ref_range_lower,
    ref_range_upper,
    flag
FROM read_csv_auto('{{raw_dir}}/hosp/labevents.csv.gz', sample_size=-1);

CREATE OR REPLACE VIEW mimic_hosp.d_labitems AS
SELECT itemid, label, fluid, category
FROM read_csv_auto('{{raw_dir}}/hosp/d_labitems.csv.gz');

CREATE OR REPLACE VIEW mimic_hosp.diagnoses_icd AS
SELECT subject_id, hadm_id, seq_num, icd_code, icd_version
FROM read_csv_auto('{{raw_dir}}/hosp/diagnoses_icd.csv.gz');

CREATE OR REPLACE VIEW mimic_hosp.omr AS
SELECT
    subject_id,
    chartdate::DATE AS chartdate,
    seq_num,
    result_name,
    result_value
FROM read_csv_auto('{{raw_dir}}/hosp/omr.csv.gz');
