"""
Create BigQuery datasets for Afrihealth project

Datasets to be created:
    - staging:  External tables linked to Iceberg
    - core: Dimensional model (fact + dimensions)
    - analytics: Aggregated marts
    - ml_features: Feature store for ml
"""

import os
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()
project_id = os.getenv('GCP_PROJECT_ID')
key_path = os.getenv('GCP_SERVICE_ACCOUNT_KEY_PATH')

# Initialize client
client = bigquery.Client.from_service_account_json(
    key_path,
    project=project_id
)

def create_dataset(dataset_id: str, description: str, location: str = 'US'):
    """
    This functions creates a BigQuery dataset
    

    Args:
        dataset_id: Dataset name
        description: Dataset description
        location: Data location (US)
    """

    full_dataset_id = f"{project_id}.{dataset_id}"

    # Create dataset
    dataset = bigquery.Dataset(full_dataset_id)
    dataset.location = location
    dataset.description = description

    try:
        dataset = client.create_dataset(dataset, exists_ok=True)
        print(f"Created {dataset.dataset_id} dataset")
    except Exception as e:
        print(f"Failed to create {dataset.dataset_id}: {e}")

def create_all_datasets():
    """
    This function creates all the datasets required (staging, core, analytics, and ml_features)
    """

    # Staging (external tables to iceberg)
    create_dataset(
        dataset_id='staging',
        description="External table linking to Iceberg (read-only)"
    )

    # Core (dimensional model)
    create_dataset(
        dataset_id='core',
        description="Dimensional model: fact and dimension tables"
    )

    # Analytics
    create_dataset(
        dataset_id='analytics',
        description="Pre-aggregated tables for dashboards"
    )

    # ML Features
    create_dataset(
        dataset_id='ml_features',
        description="Feature store for machine learning models"
    )

    datasets = list(client.list_datasets())
    print(datasets)


if __name__ == "__main__":
    create_all_datasets()  

    