-- Stage ICU-module tables as DuckDB views over raw CSV.gz

CREATE OR REPLACE VIEW mimic_icu.icustays AS
SELECT
    subject_id,
    hadm_id,
    stay_id,
    first_careunit,
    last_careunit,
    intime::TIMESTAMP AS intime,
    outtime::TIMESTAMP AS outtime,
    los
FROM read_csv_auto('{{raw_dir}}/icu/icustays.csv.gz');

CREATE OR REPLACE VIEW mimic_icu.d_items AS
SELECT itemid, label, abbreviation, linksto, category, unitname, param_type
FROM read_csv_auto('{{raw_dir}}/icu/d_items.csv.gz');

CREATE OR REPLACE VIEW mimic_icu.chartevents AS
SELECT
    subject_id,
    hadm_id,
    stay_id,
    caregiver_id,
    charttime::TIMESTAMP AS charttime,
    storetime::TIMESTAMP AS storetime,
    itemid,
    value,
    valuenum,
    valueuom,
    warning
FROM read_csv_auto('{{raw_dir}}/icu/chartevents.csv.gz', sample_size=-1);

CREATE OR REPLACE VIEW mimic_icu.inputevents AS
SELECT
    subject_id,
    hadm_id,
    stay_id,
    caregiver_id,
    starttime::TIMESTAMP AS starttime,
    endtime::TIMESTAMP AS endtime,
    storetime::TIMESTAMP AS storetime,
    itemid,
    amount,
    amountuom,
    rate,
    rateuom,
    orderid,
    linkorderid,
    ordercategoryname,
    secondaryordercategoryname,
    ordercomponenttypedescription,
    ordercategorydescription,
    patientweight,
    totalamount,
    totalamountuom,
    isopenbag,
    continueinnextdept,
    statusdescription,
    originalamount,
    originalrate
FROM read_csv_auto('{{raw_dir}}/icu/inputevents.csv.gz', sample_size=-1);

CREATE OR REPLACE VIEW mimic_icu.outputevents AS
SELECT
    subject_id,
    hadm_id,
    stay_id,
    caregiver_id,
    charttime::TIMESTAMP AS charttime,
    storetime::TIMESTAMP AS storetime,
    itemid,
    value,
    valueuom
FROM read_csv_auto('{{raw_dir}}/icu/outputevents.csv.gz', sample_size=-1);

CREATE OR REPLACE VIEW mimic_icu.procedureevents AS
SELECT
    subject_id,
    hadm_id,
    stay_id,
    starttime::TIMESTAMP AS starttime,
    endtime::TIMESTAMP AS endtime,
    itemid,
    value,
    valueuom
FROM read_csv_auto('{{raw_dir}}/icu/procedureevents.csv.gz');
