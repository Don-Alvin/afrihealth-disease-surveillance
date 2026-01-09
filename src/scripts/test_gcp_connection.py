import os
from google.cloud import storage
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()

project_id = os.getenv('GCP_PROJECT_ID')
key_path = os.getenv('GCP_SERVICE_ACCOUNT_KEY_PATH')

try:
    print('Testing Cloud Storage connection...')
    storage_client = storage.Client.from_service_account_json(
        key_path,
        project=project_id
    )
    buckets = list(storage_client.list_buckets())
    print(f"Found {len(buckets)} buckets")

    print("Testing BigQueryConnection")
    bq_client = bigquery.Client.from_service_account_json(
        key_path,
        project=project_id
    )
    datasets = list(bq_client.list_datasets())
    print(f'Found {len(datasets)} found')
    print("All connections successfull")

except Exception as e:
    print(f"Connection failed: {e}")