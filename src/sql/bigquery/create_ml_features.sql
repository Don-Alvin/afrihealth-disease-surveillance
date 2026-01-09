CREATE OR REPLACE TABLE `{project_id}.ml_features.outbreak_prediction_features`
PARTITION BY report_date
CLUSTER BY geography_id, disease_id AS

WITH daily_cases AS (
    SELECT
        report_date,
        geography_id,
        disease_id,
        SUM(cases) AS daily_cases,
        SUM(deaths) AS daily_deaths,
        AVG(SAFE_DIVIDE(deaths, NULLIF(cases, 0))) AS daily_cfr
    FROM `{project_id}.core.fact_reports`
    GROUP BY report_date, geography_id, disease_id
),

enrinched_daily AS (
    SELECT
        dc.*,
        dg.country_code,
        dg.region_name,
        dg.population,
        dg.population_density,
        dg.urban_rural,
        dw.temperature_avg_c,
        dw.temperature_min_c,
        dw.temperature_max_c,
        dw.rainfall_mm,
        dw.humidity_pct,
        EXTRACT(YEAR FROM dc.report_date) AS year,
        EXTRACT(MONTH FROM dc.report_date) AS month,
        EXTRACT(QUARTER FROM dc.report_date) AS quarter,
        EXTRACT(DAYOFWEEK FROM dc.report_date) AS day_of_week,
        EXTRACT(DAYOFYEAR FROM dc.report_date) AS day_of_year,
        CASE 
            WHEN EXTRACT(MONTH FROM dc.report_date) IN (3, 4, 5, 10, 11) THEN 1  
            ELSE 0 
        END AS is_rainy_season
    FROM daily_cases dc
    LEFT JOIN `{project_id}.core.dim_geography` dg ON dc.geography_id = dg.geography_id
    LEFT JOIN `{project_id}.core.dim_weather` dw ON dc.geography_id = dw.geography_id AND dc.report_date = dw.observation_date
),

lag_features AS (
    SELECT
        *,
        LAG(daily_cases, 1) OVER w AS cases_lag_1d,
        LAG(daily_cases, 3) OVER w AS cases_lag_3d,
        LAG(daily_cases, 7) OVER w AS cases_lag_7d,
        LAG(daily_cases, 14) OVER w AS cases_lag_14d,
        LAG(daily_cases, 30) OVER w AS cases_lag_30d,
        LAG(daily_deaths, 1) OVER w AS deaths_lag_1d,
        LAG(daily_deaths, 7) OVER w AS deaths_lag_7d,
        LAG(temperature_avg_c, 1) OVER w AS temp_lag_1d,
        LAG(temperature_avg_c, 7) OVER w AS temp_lag_7d,
        LAG(rainfall_mm, 1) OVER w AS rainfall_lag_1d,
        LAG(rainfall_mm, 7) OVER w AS rainfall_lag_7d
    FROM enrinched_daily

    WINDOW w AS (
        PARTITION BY geography_id, disease_id
        ORDER BY report_date
    )
),

rolling_features AS (
    SELECT
        *,
        AVG(daily_cases) OVER w7 AS cases_rolling_avg_7d,
        AVG(daily_cases) OVER w14 AS cases_rolling_avg_14d,
        AVG(daily_cases) OVER w30 AS cases_rolling_avg_30d,
        SUM(daily_cases) OVER w7 AS cases_rolling_sum_7d,
        SUM(daily_cases) OVER w14 AS cases_rolling_sum_14d,
        SUM(daily_cases) OVER w30 AS cases_rolling_sum_30d,
        STDDEV(daily_cases) OVER w7 AS cases_rolling_std_7d,
        STDDEV(daily_cases) OVER w14 AS cases_rolling_std_14d,
        MAX(daily_cases) OVER w7 AS cases_rolling_max_7d,
        MAX(daily_cases) OVER w14 AS cases_rolling_max_14d,
        AVG(temperature_avg_c) OVER w7 AS temp_rolling_avg_7d,
        AVG(rainfall_mm) OVER w7 AS rainfall_rolling_avg_7d,
        AVG(humidity_pct) OVER w7 AS humidity_rolling_avg_7d
    FROM lag_features

    WINDOW
        w7 AS (
            PARTITION BY geography_id, disease_id
            ORDER BY report_date
            ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
        ),
        w14 AS (
            PARTITION BY geography_id, disease_id
            ORDER BY report_date
            ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING
        ),
        w30 AS (
            PARTITION BY geography_id, disease_id
            ORDER BY report_date
            ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
        )
),

target_variables AS (
    SELECT
        *,
        CASE 
            WHEN LEAD(cases_rolling_avg_7d, 7) OVER w > cases_rolling_avg_7d * 1.5 THEN 1  
            ELSE 0
        END AS outbreak_next_7d,
        CASE 
            WHEN LEAD(cases_rolling_avg_14d, 14) OVER w > cases_rolling_avg_14d * 1.5 THEN 1 
            ELSE 0 
        END AS outbreak_next_14d,
        LEAD(daily_cases, 7) OVER w AS cases_next_7d,
        LEAD(daily_cases, 14) OVER w AS cases_next_14d,
        LEAD(daily_cases, 30) OVER w AS cases_next_30d
    FROM rolling_features

    WINDOW w AS (
        PARTITION BY geography_id, disease_id
        ORDER BY report_date
    )
),

derived_features AS (
    SELECT
        *,
        SAFE_DIVIDE(daily_cases, NULLIF(cases_lag_7d, 0)) AS cases_growth_7d,
        SAFE_DIVIDE(daily_cases, NULLIF(cases_lag_14d, 0)) AS cases_growth_14d,
        SAFE_DIVIDE(daily_cases, NULLIF(cases_lag_30d, 0)) AS cases_growth_30d,
        (daily_cases - cases_lag_7d) - (cases_lag_7d - cases_lag_14d) AS cases_acceleration_7d,
        SAFE_DIVIDE(cases_rolling_std_7d, NULLIF(cases_rolling_avg_7d, 0)) AS cases_volatility_7d,
        temperature_avg_c - temp_rolling_avg_7d AS temp_anomaly_7d,
        rainfall_mm - (rainfall_rolling_avg_7d / 7) AS rainfall_anomaly_7d,
        CASE 
            WHEN temperature_avg_c BETWEEN 20 AND 30
                AND rainfall_mm > 0
                AND humidity_pct > 60
            THEN 1  
            ELSE 0
        END AS mosquitoe_favorable_conditions,
        DATE_DIFF(
            report_date,
            LAG(CASE WHEN daily_cases > cases_rolling_avg_30d * 2 THEN report_date END)
                OVER (PARTITION BY geography_id, disease_id 
                ORDER BY report_date
                ),
            DAY
        ) AS days_since_last_outbreak,
        SAFE_DIVIDE(daily_cases, NULLIF(population, 0)) * 100000 AS cases_per_100k
    FROM target_variables
)

SELECT
    report_date,
    geography_id,
    disease_id,
    country_code,
    region_name,
    outbreak_next_7d,
    outbreak_next_14d,
    cases_next_7d,
    cases_next_14d,
    cases_next_30d,
    daily_cases,
    daily_deaths,
    daily_cfr,
    cases_lag_1d,
    cases_lag_3d,
    cases_lag_7d,
    cases_lag_14d,
    cases_lag_30d,
    deaths_lag_1d,
    deaths_lag_7d,
    cases_rolling_avg_7d,
    cases_rolling_avg_14d,
    cases_rolling_avg_30d,
    cases_rolling_sum_7d,
    cases_rolling_sum_14d,
    cases_rolling_sum_30d,
    cases_rolling_std_7d,
    cases_rolling_std_14d,
    cases_rolling_max_7d,
    cases_rolling_max_14d,
    cases_growth_7d,
    cases_growth_14d,
    cases_growth_30d,
    cases_acceleration_7d,
    cases_volatility_7d,
    days_since_last_outbreak,
    cases_per_100k,
    temperature_avg_c,
    temperature_min_c,
    temperature_max_c,
    rainfall_mm,
    humidity_pct,
    temp_lag_1d,
    temp_lag_7d,
    rainfall_lag_1d,
    rainfall_lag_7d,
    temp_rolling_avg_7d,
    rainfall_rolling_avg_7d,
    humidity_rolling_avg_7d,
    temp_anomaly_7d,
    rainfall_anomaly_7d,
    mosquitoe_favorable_conditions,
    population,
    population_density,
    urban_rural,
    year,
    month,
    quarter,
    day_of_week,
    day_of_year,
    is_rainy_season,

    CURRENT_TIMESTAMP() as created_at
FROM derived_features
WHERE cases_lag_30d IS NOT NULL
    AND cases_next_7d IS NOT NULL;

SELECT
    'Date range' as metric,
    CONCAT(
        CAST(MIN(report_date) AS string),
        'to',
        CAST(MAX(report_date) AS string)
    ) AS value
FROM `{project_id}.ml_features.outbreak_prediction_features`

UNION ALL

SELECT
    'Outbreak rate (7d)' AS metric,
    CONCAT(
        CAST(ROUND(AVG(outbreak_next_7d) * 100, 2) AS STRING),
        '%'
    ) AS value
FROM `{project_id}.ml_features.outbreak_prediction_features`

UNION ALL

SELECT
    'Outbreak rate (14d)' AS metric,
    CONCAT(
        CAST(ROUND(AVG(outbreak_next_14d) * 100, 2) AS STRING),
        '%'
    ) AS value
FROM `{project_id}.ml_features.outbreak_prediction_features`

UNION ALL

SELECT
    'Unique diseases' AS metric,
    CAST(COUNT(DISTINCT disease_id) AS STRING) AS VALUES
FROM `{project_id}.ml_features.outbreak_prediction_features`

UNION ALL

SELECT
    'Unique locations' AS metric,
    CAST(COUNT(DISTINCT geography_id) AS STRING) AS value 
FROM `{project_id}.ml_features.outbreak_prediction_features`;