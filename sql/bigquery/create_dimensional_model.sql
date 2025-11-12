CREATE or REPLACE TABLE `{project_id}.core.dim_date` AS
WITH date_range AS (
    SELECT date
    FROM UNNEST(GENERATE_DATE_ARRAY('2020-01-01', '2025-12-31', INTERVAL 1 DAY)) AS date
)

SELECT
    FORMAT_DATE('%Y%m%d', date) AS date_key,
    date as date_value,
    EXTRACT(YEAR FROM date) AS year,
    FORMAT_DATE('%Y', date) AS year_name,
    EXTRACT(QUARTER FROM date) AS quarter,
    CONCAT('Q', CAST(EXTRACT(QUARTER FROM date) AS STRING), " ", CAST(EXTRACT(YEAR FROM date) AS STRING)) AS quarter_name,
    EXTRACT(MONTH FROM date) AS month,
    FORMAT_DATE('%B', date) AS month_name,
    FORMAT_DATE('%b', date) AS month_name_short,
    FORMAT_DATE('%Y-%m', date) AS year_month,
    EXTRACT(WEEK FROM date) AS week_of_year,
    EXTRACT(ISOWEEK FROM date) AS iso_week,
    EXTRACT(DAY FROM date) AS day_of_month,
    EXTRACT(DAYOFWEEK FROM date) AS day_of_week,
    FORMAT_DATE('%A', date) AS day_name,
    FORMAT_DATE('%a', date) AS day_name_short,
    EXTRACT(DAYOFYEAR FROM date) AS day_of_year,
    CASE 
        WHEN EXTRACT(DAYOFWEEK FROM date) IN (1, 7) THEN TRUE  
        ELSE FALSE 
    END AS is_weekend,
    CASE 
        WHEN EXTRACT(MONTH FROM date) IN (3, 4, 5, 10, 11) THEN TRUE  
        ELSE FALSE 
    END AS is_rainy_season
FROM date_range;

CREATE OR REPLACE TABLE `{project_id}.core.dim_geography` AS
SELECT
    geography_id,
    country_code,
    country_name,
    region_name,
    district_name,
    sub_district_name,
    population,
    urban_rural,
    latitude,
    longitude,
    area_sq_km,
    population_density,
    elevation,
    healthcare_access_index,
    
    CURRENT_TIMESTAMP() AS created_at
FROM `{project_id}.staging.geography`;

CREATE OR REPLACE TABLE `{project_id}.core.dim_facilities` AS
SELECT
    facility_id,
    facility_name,
    facility_type,
    facility_level,
    geography_id,
    country_code,
    bed_capacity,
    staff_count,
    has_lab,
    has_isolation_ward,
    has_xray,
    ambulance_count,
    operational_status,
    established_year,

    CURRENT_TIMESTAMP() AS created_at
FROM `{project_id}.staging.facilities`;

CREATE OR REPLACE TABLE `{project_id}.core.dim_disease` AS
SELECT
    disease AS disease_id,
    disease AS disease_name,
    CASE 
        WHEN disease = 'Malaria' THEN 'Vector_borne'
        WHEN disease = 'Cholera' THEN 'Waterborne'
        WHEN disease = 'Tuberculosis' THEN 'Airborne'  
    END AS disease_category,
    CASE 
        WHEN disease = 'Malaria' THEN TRUE
        WHEN disease = 'Cholera' THEN TRUE  
        ELSE FALSE
    END AS is_seasonal,
    CASE 
        WHEN disease = 'Malaria' THEN TRUe
        ELSE FALSE  
    END AS is_climate_sensitive,

    CURRENT_TIMESTAMP() as created_at
FROM (
    SELECT DISTINCT disease
    FROM `{project_id}.staging.reports`
);

CREATE OR REPLACE TABLE `{project_id}.core.dim_weather`
PARTITION BY observation_date
CLUSTER BY geography_id, observation_date AS
SELECT
    TO_BASE64(MD5(CONCAT(
        geography_id,
        CAST(date as STRING)
    ))) AS weather_key,
    date AS observation_date,
    geography_id,
    FORMAT_DATE('%Y%m%d', date) as date_key,
    temperature_min_c,
    temperature_max_c,
    temperature_avg_c,
    rainfall_mm,
    humidity_pct,

    CURRENT_TIMESTAMP() AS created_at
FROM `{project_id}.staging.weather`;

CREATE OR REPLACE TABLE `{project_id}.core.fact_reports`
PARTITION BY report_date
CLUSTER BY country_code, disease_id AS
SELECT
    TO_BASE64(MD5(CONCAT(
        r.report_id,
        CAST(r.report_date AS string),
        r.facility_id,
        r.geography_id
    ))) AS reports_key,
    r.report_id,
    r.report_date,
    r.case_date,
    r.facility_id,
    r.geography_id,
    r.disease as disease_id,
    w.weather_key,
    g.country_code,
    r.cases,
    r.deaths,
    r.recoveries,
    r.age_group,
    r.gender,
    SAFE_DIVIDE(CAST(r.deaths AS FLOAT64), r.cases) AS case_fatality_rate,
    DATE_DIFF(r.report_date, r.case_date, DAY) AS reporting_delay_days,
    CURRENT_TIMESTAMP() AS created_at
FROM `{project_id}.staging.reports` r
LEFT JOIN `{project_id}.staging.geography` g ON r.geography_id = g.geography_id
LEFT JOIN `{project_id}.core.dim_weather` w ON r.geography_id = w.geography_id AND r.report_date = w.observation_date;

SELECT
    'dim_date' AS table_name,
    COUNT(*) AS row_count
FROM `{project_id}.core.dim_date`

UNION ALL

SELECT
    'dim_geography' as table_name,
    COUNT(*) as row_count
FROM `{project_id}.core.dim_geography`

UNION ALL

SELECT
    'dim_facilities' as table_name,
    COUNT(*) as row_count
FROM `{project_id}.core.dim_facilities`

UNION ALL

SELECT
    'dim_disease' as table_name,
    COUNT(*) as row_count
FROM `{project_id}.core.dim_disease`

UNION ALL

SELECT
    'dim_weather' as table_name,
    COUNT(*) as row_count
FROM `{project_id}.core.dim_weather`

UNION ALL

SELECT
    'fact_reports' as table_name,
    COUNT(*) as row_count
FROM `{project_id}.core.fact_reports`;