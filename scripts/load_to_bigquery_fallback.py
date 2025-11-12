"""
Our create_external_tables.py fails to work. We will use this fallback query to load cleaned data directly to bigquery
"""

import os
import pandas as pd
from google.cloud import bigquery
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
project_id = os.getenv('GCP_PROJECT_ID')
key_path = os.getenv('GCP_SERVICE_ACCOUNT_KEY_PATH')
iceberg_bucket = os.getenv('GCS_BUCKET_ICEBERG')

# Initialize client
client = bigquery.Client.from_service_account_json(
    key_path,
    project=project_id
)

def load_csv_to_bigquery(
    dataset_id: str,
    table_id: str, 
    csv_path: str,
    description: str,
    partition_field: str = None
):
    """
    Load csvs directly into bigquery
    """

    full_table_id = f"{project_id}.{dataset_id}.{table_id}"

    if csv_path.endswith('.gz'):
        df = pd.read_csv(csv_path, compression='gzip')
    else:
        df = pd.read_csv(csv_path)
    
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
    )

    if partition_field:
        job_config.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field=partition_field
        )

        print(f"Partitioned by: {partition_field}")
    
    # Load csv file
    with open(csv_path, 'rb') as source_file:
        job = client.load_table_from_file(
            source_file,
            full_table_id,
            job_config=job_config
        )
    
    job.result()

    # Update description
    table = client.get_table(full_table_id)
    table.description = description
    client.update_table(table, ['description'])

    return table.num_rows

def load_all_staging_tables():
    data_dir = Path("data/processed")
    total_rows = 0

    # Geographical data (no partitioning)
    rows = load_csv_to_bigquery(
        dataset_id='staging',
        table_id='geography',
        csv_path=str(data_dir/'geographical_clean.csv'),
        description="Geographical areas with demographics",
        partition_field=None
    )

    total_rows += rows

    # Facilities table (no partitioning)
    rows = load_csv_to_bigquery(
        dataset_id='staging',
        table_id='facilities',
        csv_path=str(data_dir/'facilities_clean.csv'),
        description='Healthcare facilities and capacities',
        partition_field=None
    )

    total_rows += rows

    # Weather table (partition by date)
    rows = load_csv_to_bigquery(
        dataset_id='staging',
        table_id='weather',
        csv_path=str(data_dir/'weather_clean.csv'),
        description='Daily weather observations',
        partition_field='date'
    )

    total_rows += rows

    # Reports table (partition by date report_date and disease)
    rows = load_csv_to_bigquery(
        dataset_id='staging',
        table_id='reports',
        csv_path=str(data_dir/'reports_clean.csv'),
        description='Reports table showing daily disease cases',
        partition_field='report_date'
    )
    
    total_rows += rows
    return total_rows

def test_staging_tables():
    tests = [
        {
            'name': 'Disease Cases Count',
            'query': f"SELECT COUNT(*) as total FROM `{project_id}.staging.reports`"
        },
        {
            'name': 'Geography Count',
            'query': f"SELECT COUNT(*) as total FROM `{project_id}.staging.geography`"
        },
        {
            'name': 'Sample Data',
            'query': f"""
                SELECT report_id, report_date, disease, cases
                FROM `{project_id}.staging.reports`
                LIMIT 3
            """
        }
]
    
    for test in tests:
        print(f"Test: {test['name']}")
        try:
            results = client.query(test['query']).result()
            for row in results:
                print(f"  {dict(row)}")
            print("✓ Success")
        except Exception as e:
            print(f"✗ Failed: {e}")
        print()


if __name__ == "__main__":
    total = load_all_staging_tables()
    test_staging_tables()
    

