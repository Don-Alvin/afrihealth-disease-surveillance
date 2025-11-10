import os
import sys
import time
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from src.data_ingestion.iceberg_manager import IcebergManager


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def load_bronze_layer():
    manager = IcebergManager()
    data_dir = Path('data/raw')

    start = time.time()
    manager.load_data_to_table(
        namespace='bronze',
        table_name='geography',
        data_path=str(data_dir/'geographical.csv')
    )

    manager.load_data_to_table(
        namespace='bronze',
        table_name='facilities',
        data_path=str(data_dir/'facilities.csv')
    )

    manager.load_data_to_table(
        namespace='bronze',
        table_name='reports',
        data_path=str(data_dir/'reports.csv')
    )

    manager.load_data_to_table(
        namespace='bronze',
        table_name='weather',
        data_path=str(data_dir/'weather.csv')
    )

    print(f"Bronze layer data loaded in {time.time() - start:.2f} seconds")

def load_silver_layer():
    manager = IcebergManager()
    data_dir = Path('data/processed')

    start = time.time()
    manager.load_data_to_table(
        namespace='silver',
        table_name='geography_validated',
        data_path=str(data_dir/'geographical_clean.csv')
    )

    manager.load_data_to_table(
        namespace='silver',
        table_name='facilities_validated',
        data_path=str(data_dir/'facilities_clean.csv')
    )

    manager.load_data_to_table(
        namespace='silver',
        table_name='reports_validated',
        data_path=str(data_dir/'reports_clean.csv')
    )

    manager.load_data_to_table(
        namespace='silver',
        table_name='weather_validated',
        data_path=str(data_dir/'weather_clean.csv')
    )

    print(f"Silver layer data loaded in {time.time() - start:.2f} seconds")

def verify_data_loaded():
    manager = IcebergManager()

    tables_to_verify = [
        ('bronze', 'geography'),
        ('bronze', 'facilities'),
        ('bronze', 'weather'),
        ('bronze', 'reports'),
        ('silver', 'geography_validated'),
        ('silver', 'facilities_validated'),
        ('silver', 'weather_validated'),
        ('silver', 'reports_validated')
    ]

    for namespace, table_name in tables_to_verify:
        full_name = f"{namespace}.{table_name}"

        try:
            table = manager.catalog.load_table(full_name)

            snapshots = list(table.history())
            if snapshots:
                latest_snapshot = snapshots[-1]
                print(latest_snapshot.snapshot_id)
                print(f"Data loaded into {full_name}")
            else:
                print("No data loaded")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    total_start = time.time()

    # load_bronze_layer()
    # load_silver_layer()
    verify_data_loaded()

    total_time = time.time() - total_start