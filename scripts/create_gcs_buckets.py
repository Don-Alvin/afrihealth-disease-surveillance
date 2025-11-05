import os
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()
project_id = os.getenv('GCP_PROJECT_ID')
key_path = os.getenv('GCP_SERVICE_ACCOUNT_KEY_PATH')

storage_client = storage.Client.from_service_account_json(
    key_path,
    project=project_id
)

BUCKET_NAMES = [
    'afrihealth-surveillance-iceberg-da',
    'afrihealth-surveillance-models-da'
]

def create_buckets():
    print("Creating storage buckets")

    for bucket_name in BUCKET_NAMES:
        try:
            bucket = storage_client.bucket(bucket_name)
            if bucket.exists():
                print(f"Bucket already exists: gs//{bucket_name}")
            else:
                bucket = storage_client.create_bucket(
                    bucket.name,
                    location='us-central1'
                )
                print(f"Created bucket: gs//{bucket_name}")

        except Exception as e:
            print(f"Failed to create {bucket_name}")
            print(f"Error: {e}")

    
    for bucket in storage_client.list_buckets():
        print(f"gs://{bucket.name}")

if __name__ == "__main__":
    create_buckets()
