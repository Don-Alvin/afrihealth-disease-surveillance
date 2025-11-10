"""
Create external tables linked to Iceberg
"""

import os
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()
project_id = os.getenv('GCP_PROJECT_ID')
key_path = os.getenv('GCP_SERVICE_ACCOUNT_KEY_PATH')
iceberg_bucket = os.getenv('GCS_BUCKET_ICEBERG')

# Initialize client
client = bigquery.Client.from_service_account_json(
    key_path,
    project=project_id
)

def create_external_table(
        dataset_id: str,
        table_id: str,
        iceberg_namespace: str,
        iceberg_table: str,
        description: str
):
    """
    This function creates external table that will link to the Iceberg.

    Args:
        dataset_id: Bigquery dataset.
        table_id: BigQuery table.
        iceberg_namespace: Iceberg namespace (bronze, silver).
        iceberg_table: Iceberg table name.
        description: Table description.
    """

    full_table_id = f"{project_id}.{dataset_id}.{table_id}"
    iceberg_path = f"gs://{iceberg_bucket}/warehouse/{iceberg_namespace}.db/{iceberg_table}"

    try:
        # Create external table configuration
        external_config = bigquery.ExternalConfig("ICEBERG")
        external_config.source_uris = [iceberg_path]

        # Define table
        table = bigquery.Table(full_table_id)
        table.external_data_configuration = external_config
        table.description = description

        # Create table
        table = client.create_table(table, exists_ok=True)
        print(f"Created table {table.table_id}")
    except Exception as e:
        print(f"Failed to create table")
        print(f"Error: {e}")
    
def create_all_external_tables():
    # Reports
    create_external_table(
        dataset_id='staging',
        table_id='reports',
        iceberg_namespace='silver',
        iceberg_table='reports_validated',
        description="Reports (external link to Iceberg silver layer)"
    )

    # Geographical
    create_external_table(
        dataset_id='staging',
        table_id='geography',
        iceberg_namespace='silver',
        iceberg_table='geography_validated',
        description="Geographical (external link to Iceberg silver layer)"
    )

    # Facilities
    create_external_table(
        dataset_id='staging',
        table_id='facilities',
        iceberg_namespace='silver',
        iceberg_table='facilities_validated',
        description="Facilities (external link to Iceberg silver layer)"
    )

    # Weather
    create_external_table(
        dataset_id='staging',
        table_id='weather',
        iceberg_namespace='silver',
        iceberg_table='weather_validated',
        description="Weather (external link to Iceberg silver layer)"
    )

    print('External tables created')

def test_external_tables():
    test_queries = [
        {
            'name': 'Disease Cases Count',
            'query': f"""
                SELECT COUNT(*) as total_rows
                FROM `{project_id}.staging.reports`
            """
        },
        {
            'name': 'Geography Count',
            'query': f"""
                SELECT COUNT(*) as total_rows
                FROM `{project_id}.staging.geography`
            """
        },
        {
            'name': 'Sample Disease Data',
            'query': f"""
                SELECT 
                    report_id,
                    report_date,
                    disease,
                    cases
                FROM `{project_id}.staging.reports`
                LIMIT 5
            """
        }
    ]
    
    for test in test_queries:
        print(f"Test: {test['name']}")
        print(f"Query: {test['query'].strip()}")
        
        try:
            query_job = client.query(test['query'])
            results = query_job.result()
            
            print("Results:")
            for row in results:
                print(f"  {dict(row)}")
            
            print(f"✓ Query successful")
            
        except Exception as e:
            print(f"✗ Query failed: {e}")
        
        print()

if __name__ == "__main__":
    create_all_external_tables()
    import time
    time.sleep(10)
    test_external_tables()
