import os
from google.cloud import bigquery
from dotenv import load_dotenv
from pathlib import Path
import time

load_dotenv()

project_id = os.getenv('GCP_PROJECT_ID')
key_path = os.getenv('GCP_SERVICE_ACCOUNT_KEY_PATH')

client = bigquery.Client.from_service_account_json(
    key_path,
    project=project_id
)

def create_ml_dataset():
    dataset_id = f"{project_id}.ml_features"

    try:
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US"
        dataset.description = "Machine learning feature store for outbreak prediction"

        dataset = client.create_dataset(dataset, exists_ok=True)
        print(f"Dataset created: {dataset.dataset_id}")
    except Exception as e:
        print(f"Failed to create dataset: {e}")

def create_feature_store(sql_file: str):

    with open(sql_file, 'r') as f:
        sql = f.read()
    
    sql = sql.replace('{project_id}', project_id)
    statements = [s.strip() for s in sql.split(';') if s.strip()]
    
    try:
        for i, statement in enumerate(statements, 1):
            if 'CREATE' in statement.upper():
                print('Creating feature table')
            else:
                print('Running verification query')
            
            try:
                start_time = time.time()
                query_job = client.query(statement)
                result = query_job.result()
                elapsed = time.time() - start_time
                print(f"Success, took {elapsed:.1f}s")

                if 'SELECT' in statement.upper() and 'metric' in statement.lower():
                    for row in result:
                        metric = row.get('metric', 'unknown')
                        value = row.get('value', "")
                        print(f" {metric:30} {value}")
            except Exception as e:
                print(f"Failed: {e}")
                return False

            print()

        return True
    except Exception as e:
        print(f"Error: {e}")

def analyse_features():
    analyses = [
        {
            'name': 'Feature completeness check',
            'query': f"""
                SELECT
                    COUNT(*) AS total_rows,
                    COUNTIF(cases_lag_7d IS NULL) as missing_lag_7d,
                    COUNTIF(cases_rolling_avg_7d IS NULL) AS missing_rolling_7d,
                    COUNTIF(temperature_avg_c IS NULL) AS missing_temperature,
                    ROUND(100 * COUNTIF(cases_lag_7d IS NOT NULL) / COUNT(*), 2) AS pct_complete
                FROM `{project_id}.ml_features.outbreak_prediction_features`
            """
        },
        {
            'name': 'Target variable distribution',
            'query': f"""
                SELECT
                    disease_id,
                    COUNT(*) AS total_samples,
                    SUM(outbreak_next_7d) AS outbreaks_7d,
                    SUM(outbreak_next_14d) AS outbreaks_14d,
                    ROUND(100 * AVG(outbreak_next_7d), 2) AS outbreak_rate_7d_pct,
                    ROUND(100 * AVG(outbreak_next_14d), 2) AS outbreak_rate_14d_pct
                FROM `{project_id}.ml_features.outbreak_prediction_features`
                GROUP BY disease_id
                ORDER BY total_samples DESC
            """
        },
        {
            'name': 'Feature summary statistics',
            'query': f"""
                SELECT
                    'daily_cases' AS feature,
                    ROUND(AVG(daily_cases), 2) AS avg_value,
                    ROUND(STDDEV(daily_cases), 2) AS std_dev,
                    MIN(daily_cases) AS min_value,
                    MAX(daily_cases) AS max_value
                FROM `{project_id}.ml_features.outbreak_prediction_features`

                UNION ALL

                SELECT
                    'cases_rolling_avg_7d' AS feature,
                    ROUND(AVG(cases_rolling_avg_7d), 2) AS avg_value,
                    ROUND(STDDEV(cases_rolling_avg_7d), 2) AS std_dev,
                    MIN(cases_rolling_avg_7d) AS min_value,
                    MAX(cases_rolling_avg_7d) AS max_value
                FROM `{project_id}.ml_features.outbreak_prediction_features`

                UNION ALL

                SELECT
                    'temperature_avg_c' AS feature,
                    ROUND(AVG(temperature_avg_c), 2) AS avg_value,
                    ROUND(STDDEV(temperature_avg_c), 2) AS std_dev,
                    MIN(temperature_avg_c) AS min_value,
                    MAX(temperature_avg_c) AS max_value
                FROM `{project_id}.ml_features.outbreak_prediction_features`
                
                UNION ALL

                SELECT
                    'rainfall_mm' AS feature,
                    ROUND(AVG(rainfall_mm), 2) AS avg_value,
                    ROUND(STDDEV(rainfall_mm), 2) AS std_dev,
                    MIN(rainfall_mm) AS min_value,
                    MAX(rainfall_mm) AS max_value
                FROM `{project_id}.ml_features.outbreak_prediction_features`
            """
        },
        {
            'name': 'Top locations for outbreak frequency',
            'query': f"""
                SELECT
                    geography_id,
                    country_code,
                    region_name,
                    COUNT(*) AS total_days,
                    SUM(outbreak_next_7d) AS outbreak_7d,
                    ROUND(100 * AVG(outbreak_next_7d), 2) AS outbreak_rate_pct
                FROM `{project_id}.ml_features.outbreak_prediction_features`
                GROUP BY geography_id, country_code, region_name
                HAVING COUNT(*) > 100
                ORDER BY outbreak_rate_pct DESC
                LIMIT 10
            """
        }
    ]

    for analysis in analyses:
        print(f"Analysis: {analysis['name']}")
        print()

        try:
            query_job = client.query(analysis['query'])
            results = query_job.result()

            for row in results:
                print(f"{dict(row)}")
        except Exception as e:
            print(f"Analysis failed: {e}")

def export_sample_for_local_ml():
    query = f"""
        SELECT * 
        FROM `{project_id}.ml_features.outbreak_prediction_features`
        WHERE report_date >= '2024-01-01'
        ORDER BY RAND()
        LIMIT 10000
    """
    
    print("Export 10000 records for local machine learning developement")

    try:
        df=client.query(query).to_dataframe()
        output_path = Path('data/ml/training_sample.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return True
    except Exception as e:
        print(f"Export failed: {e}")
        return False

def main():
    sql_file = 'sql/bigquery/create_ml_features.sql'
    create_ml_dataset()
    success = create_feature_store(sql_file)

    if not success:
        print("Feature store creation failed")
        return
    
    analyse_features()

    # Export sample
    export_sample_for_local_ml()

if __name__ == "__main__":
    main()
