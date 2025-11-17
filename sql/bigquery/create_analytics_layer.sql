CREATE OR REPLACE TABLE `{project_id}.analytics.daily_reports`
PARTITION BY report_date
CLUSTER BY country_code, disease_id AS

SELECT
    fr.report_date,
    dd.year,
    dd.quarter,
    dd.month,
    dd.month_name,
    dd.week_of_year,
    dd.day_name,
    dd.is_weekend,
    dd.is_rainy_season,
    dg.country_code,
    dg.country_name,
    dg.region_name,
    dg.urban_rural,
    ddis.disease_id,
    COUNT(DISTINCT fr.report_id) AS reports,
    SUM(fr.cases) AS total_cases,
    SUM(fr.deaths) AS total_deaths,
    SUM(fr.recoveries) AS total_recoveries,
    SAFE_DIVIDE(SUM(fr.deaths), SUM(fr.cases)) AS case_fatality_rate,
    COUNTIF(fr.gender = 'Male') AS cases_male,
    COUNTIF(fr.gender = 'Female') AS cases_female,
    SUM(CASE WHEN age_group IN ('0-5', '6-17') THEN 1 ELSE 0 END) AS cases_children,
    SUM(CASE WHEN age_group IN ('18-49', '50-64') THEN 1 ELSE 0 END) AS cases_adults,
    SUM(CASE WHEN age_group = '65+' THEN 1 ELSE 0 END) AS cases_elderly,

    AVG(dw.temperature_avg_c) AS avg_temperature_c,
    AVG(dw.rainfall_mm) AS avg_rainfall_mm,
    AVG(dw.humidity_pct) AS avg_humidity_pct,

    CURRENT_TIMESTAMP() as created_at

FROM `{project_id}.core.fact_reports` fr
LEFT JOIN `{project_id}.core.dim_date` dd ON fr.report_date = dd.date_value
LEFT JOIN `{project_id}.core.dim_geography` dg ON fr.geography_id = dg.geography_id
LEFT JOIN `{project_id}.core.dim_disease` ddis ON fr.disease_id = ddis.disease_id
LEFT JOIN `{project_id}.core.dim_weather` dw ON fr.weather_key = dw.weather_key

GROUP BY
    fr.report_date, dd.year, dd.quarter, dd.month, dd.month_name,
    dd.week_of_year, dd.day_name, dd.is_weekend, dd.is_rainy_season,
    dg.country_code, dg.country_name, dg.region_name, dg.urban_rural,
    ddis.disease_id, ddis.disease_name;

CREATE OR REPLACE TABLE `{project_id}.analytics.monthly_reports`
CLUSTER BY country_code, disease_id, year, month AS
SELECT
    dd.year,
    dd.quarter,
    dd.quarter_name,
    dd.month,
    dd.month_name,
    dd.year_month,
    dg.country_code,
    dg.country_name,
    dg.region_name,
    dg.urban_rural,
    ddis.disease_id,
    COUNT(DISTINCT fr.report_id) AS reports,
    SUM(fr.cases) AS total_cases,
    SUM(fr.deaths) AS total_deaths,
    SUM(fr.recoveries) AS total_recoveries,
    SAFE_DIVIDE(SUM(fr.deaths), SUM(fr.cases)) AS case_fatality_rate,
    AVG(fr.cases) AS avg_daily_cases,
    MAX(fr.cases) AS max_daily_cases,
    STDDEV(fr.cases) AS stddev_daily_cases,
    SAFE_DIVIDE(COUNTIF(fr.gender = 'Male'), COUNT(*)) as pct_male,
    SAFE_DIVIDE(COUNTIF(fr.gender = 'Female'), COUNT(*)) as pct_female,
    SAFE_DIVIDE(COUNTIF(age_group IN ('0-5', '6-17')), COUNT(*)) AS pct_children,
    AVG(dw.temperature_avg_c) AS avg_temperature_c,
    MIN(dw.temperature_min_c) AS min_temperature_c,
    MAX(dw.temperature_max_c) AS max_temperature_c,
    SUM(dw.rainfall_mm) AS total_rainfall_mm,
    AVG(dw.humidity_pct) AS avg_humidity_pct,
    AVG(fr.reporting_delay_days) AS avg_reporting_delay_days,

    CURRENT_TIMESTAMP() AS created_at

FROM `{project_id}.core.fact_reports` fr
LEFT JOIN `{project_id}.core.dim_date` dd ON fr.report_date = dd.date_value
LEFT JOIN `{project_id}.core.dim_geography` dg ON fr.geography_id = dg.geography_id
LEFT JOIN `{project_id}.core.dim_disease` ddis ON fr.disease_id = ddis.disease_id
LEFT JOIN `{project_id}.core.dim_weather` dw ON fr.weather_key = dw.weather_key
WHERE dg.country_name IS NOT NULL
GROUP BY 
    dd.year, dd.quarter, dd.quarter_name, dd.month, dd.month_name, dd.year_month, dg.country_code,
    dg.country_name, dg.region_name, dg.urban_rural, ddis.disease_id;

CREATE OR REPLACE TABLE `{project_id}.analytics.geography_summary` AS
WITH reports_stats AS (
    SELECT
        geography_id,
        COUNT(DISTINCT report_id) AS total_reports,
        SUM(cases) AS total_cases,
        SUM(deaths) AS total_deaths,
        SUM(recoveries) AS total_recoveries,
        MIN(report_date) AS first_case_date,
        MAX(report_date) AS last_case_date
    FROM `{project_id}.core.fact_reports`
    GROUP BY geography_id
),

facility_stats AS (
    SELECT
        geography_id,
        COUNT(*) AS facility_count,
        SUM(bed_capacity) AS total_bed_capacity,
        SUM(staff_count) AS total_staff,
        COUNTIF(has_lab) AS facilities_with_lab,
        COUNTIF(has_isolation_ward) AS facilities_with_isolation
    FROM `{project_id}.core.dim_facilities`
    GROUP BY geography_id
)

SELECT
    dg.geography_id,
    dg.country_code,
    dg.country_name,
    dg.region_name,
    dg.district_name,
    dg.sub_district_name,
    dg.urban_rural,
    dg.population,
    dg.population_density,
    dg.latitude,
    dg.longitude,
    COALESCE(rs.total_reports, 0) AS total_reports,
    COALESCE(rs.total_cases, 0) AS total_cases,
    COALESCE(rs.total_deaths, 0) AS total_deaths,
    COALESCE(rs.total_recoveries, 0) AS total_recoveries,
    SAFE_DIVIDE(rs.total_deaths, rs.total_cases) AS case_fatality_rate,
    rs.first_case_date,
    rs.last_case_date,
    SAFE_DIVIDE(rs.total_cases, dg.population) * 100000 AS cases_per_100k_population,
    COALESCE(fs.facility_count, 0) AS facility_count,
    COALESCE(fs.total_bed_capacity, 0) AS total_bed_capacity,
    COALESCE(fs.total_staff, 0) AS total_healthcare_staff,
    COALESCE(fs.facilities_with_lab, 0) AS facilities_with_lab,
    COALESCE(fs.facilities_with_isolation, 0) AS facilities_with_isolation,
    SAFE_DIVIDE(fs.total_bed_capacity, dg.population) * 100000 AS beds_per_100k_population,
    SAFE_DIVIDE(fs.total_staff, dg.population) * 100000 AS staff_per_100k_population,
    
    CURRENT_TIMESTAMP() as created_at
FROM `{project_id}.core.dim_geography` dg
LEFT JOIN reports_stats rs ON dg.geography_id = rs.geography_id
LEFT JOIN facility_stats fs ON dg.geography_id = fs.geography_id
WHERE dg.country_name IS NOT NULL;

CREATE OR REPLACE TABLE `{project_id}.analytics.weather_disease_correlation` AS
SELECT
    ddis.disease_id,
    CASE 
        WHEN dw.temperature_avg_c < 15 THEN '< 15°C'
        WHEN dw.temperature_avg_c < 20 THEN '15-20°C'
        WHEN dw.temperature_avg_c < 25 THEN '20-25°C'
        WHEN dw.temperature_avg_c < 30 THEN '25-30°C'  
        ELSE '> 30°C'
    END as temperature_bin,
    CASE 
        WHEN dw.rainfall_mm = 0 THEN 'No Rain'
        WHEN dw.rainfall_mm < 5 THEN '< 5mm'
        WHEN dw.rainfall_mm < 20 THEN '5-20mm'
        ELSE '> 20mm'
    END AS rainfall_bin,
    COUNT(DISTINCT fr.report_id) AS reports_count,
    SUM(fr.cases) AS total_cases,
    SUM(fr.deaths) AS total_deaths,
    AVG(dw.temperature_avg_c) AS avg_temperature_c,
    AVG(dw.rainfall_mm) AS avg_rainfall_mm,
    AVG(dw.humidity_pct) AS avg_humidity_pct,

    CURRENT_TIMESTAMP() as created_at
FROM `{project_id}.core.fact_reports` fr
JOIN `{project_id}.core.dim_disease` ddis ON fr.disease_id = ddis.disease_id
JOIN `{project_id}.core.dim_weather` dw ON fr.weather_key = dw.weather_key
GROUP BY
    ddis.disease_id, temperature_bin, rainfall_bin;

CREATE OR REPLACE TABLE `{project_id}.analytics.facility_performance` AS
WITH facility_cases AS (
    SELECT
        facility_id,
        COUNT(DISTINCT report_id) AS case_reports_handled,
        SUM(cases) AS total_cases_handled,
        SUM(deaths) AS total_deaths,
        MIN(report_date) AS first_case_date,
        MAX(report_date) AS last_case_date,
        COUNT(DISTINCT disease_id) as diseases_treated
    FROM `{project_id}.core.fact_reports`
    WHERE facility_id IS NOT NULL
    GROUP BY facility_id
)

SELECT
    df.facility_id,
    df.facility_name,
    df.facility_type,
    df.facility_level,
    df.country_code,
    df.bed_capacity,
    df.staff_count,
    df.has_lab,
    df.has_isolation_ward,
    df.has_xray,
    df.ambulance_count,
    df.operational_status,
    COALESCE(fc.case_reports_handled, 0) AS case_reports_handled,
    COALESCE(fc.total_cases_handled, 0) AS total_cases_handled,
    COALESCE(fc.total_deaths, 0) AS total_deaths,
    fc.first_case_date,
    fc.last_case_date,
    fc.diseases_treated,
    SAFE_DIVIDE(fc.total_cases_handled, df.bed_capacity) AS cases_per_bed,
    SAFE_DIVIDE(fc.total_cases_handled, df.staff_count) AS cases_per_staff_member,
    DATE_DIFF(fc.last_case_date, fc.first_case_date, DAY) + 1 AS days_in_operation,

    CURRENT_TIMESTAMP() AS created_at

FROM `{project_id}.core.dim_facilities` df
LEFT JOIN facility_cases fc ON df.facility_id = fc.facility_id;

SELECT 
  'daily_reports' AS table_name,
  COUNT(*) AS row_count,
  MIN(report_date) AS min_date,
  MAX(report_date) AS max_date
FROM `{project_id}.analytics.daily_reports`

UNION ALL

SELECT 
  'monthly_reports' AS table_name,
  COUNT(*) AS row_count,
  MIN(CAST(year_month AS STRING)) AS min_date,
  MAX(CAST(year_month AS STRING)) AS max_date
FROM `{project_id}.analytics.monthly_reports`

UNION ALL

SELECT 
  'geography_summary' AS table_name,
  COUNT(*) AS row_count,
  NULL AS min_date,
  NULL AS max_date
FROM `{project_id}.analytics.geography_summary`

UNION ALL

SELECT 
  'weather_disease_correlation' AS table_name,
  COUNT(*) AS row_count,
  NULL AS min_date,
  NULL AS max_date
FROM `{project_id}.analytics.weather_disease_correlation`

UNION ALL

SELECT 
  'facility_performance' AS table_name,
  COUNT(*) AS row_count,
  NULL AS min_date,
  NULL AS max_date
FROM `{project_id}.analytics.facility_performance`;

