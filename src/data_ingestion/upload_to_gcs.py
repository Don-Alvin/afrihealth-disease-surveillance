import os
from pathlib import Path
from google.cloud import storage
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
import requests

load_dotenv()

project_id = os.getenv('GCP_PROJECT_ID')
key_path = os.getenv('GCP_SERVICE_ACCOUNT_KEY_PATH')
bucket_name = os.getenv('GCS_BUCKET_PROCESSED')

storage_client = storage.Client.from_service_account_json(
    key_path,
    project=project_id
)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def upload_file(local_path: Path, gcs_path: str, bucket_name:str):
    """
    Upload a single file to GCS

    Args:
        local_path: Path to local file
        gcs_path: Destination path in GCS (within bucket)
        bucket_name: GCS bucket name
    """
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)

    # Get file size
    file_size = local_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)

    print(f"Uploading: {local_path.name} ({file_size_mb:.2f}mbs)")

    # Upload with progress
    blob.upload_from_filename(str(local_path), timeout=600)
    print(f"Uploaded to: gs//{bucket_name}/{gcs_path}")

def upload_processed_data():
    """
    Upload all processed datasets to gcs
    """

    processed_dir = Path('data/processed')
    files_to_upload = [
        'geographical_clean.csv',
        'weather_clean.csv',
        'facilities_clean.csv',
        'reports_clean.csv'
    ]

    for filename in files_to_upload:
        local_path = processed_dir/filename

        if not local_path.exists():
            print(f"File {filename} not found!")
            continue
            
        try:
            gcs_path = filename
            upload_file(local_path, gcs_path, bucket_name)
        except Exception as e:
            print(f"Failed to upload {filename}")
            print(f"Error: {e}")

def verify_uploads():
    files = [
        'geographical_clean.csv',
        'weather_clean.csv',
        'facilities_clean.csv',
        'reports_clean.csv'
    ]

    bucket = storage_client.bucket(bucket_name)
    
    all_good = True

    for filename in files:
        blob = bucket.blob(filename)
        if blob.exists():
            print(f"{filename} exists")
        else:
            print(f"{filename} does not exist")
            all_good = False
    
    if all_good:
        print("All files uploaded")
    else:
        print('Some files are missing')

if __name__ == "__main__":
    upload_processed_data()
    verify_uploads()