import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data_ingestion.iceberg_manager import IcebergManager
from pyiceberg.partitioning import PartitionSpec, PartitionField
from pyiceberg.transforms import DayTransform, IdentityTransform


# Create bronze layer tables
def create_bronze_tables():
    manager = IcebergManager()
    manager.create_namespace('bronze')

    # Reports table
    reports_partition = PartitionSpec(
        PartitionField(
            source_id=2,
            field_id=1000,
            transform=DayTransform(),
            name='report_date_day'
        ),
        PartitionField(
            source_id=6,
            field_id=1001,
            transform=IdentityTransform(),
            name='disease'
        )
    )

    manager.create_table(
        namespace='bronze',
        table_name='reports',
        schema=manager.get_reports_schema(),
        partition_spec=PartitionSpec()
    )

    # Geography table
    manager.create_table(
        namespace='bronze',
        table_name='geography',
        schema=manager.get_geography_schema(),
        partition_spec=PartitionSpec()
    )

    # Facilities table
    manager.create_table(
        namespace='bronze',
        table_name='facilities',
        schema=manager.get_facilities_schema(),
        partition_spec=PartitionSpec()
    )

    # Weather
    weather_partition = PartitionSpec(
        PartitionField(
            source_id=1,
            field_id=1000,
            transform=DayTransform(),
            name="date_day"
        )
    )

    manager.create_table(
        namespace='bronze',
        table_name='weather',
        schema=manager.get_weather_schema(),
        partition_spec=PartitionSpec()
    )

# Create silver layer table
def create_silver_tables():
    manager = IcebergManager()
    manager.create_namespace('silver')

    # Reports validated table
    reports_partition = PartitionSpec(
        PartitionField(
            source_id=2,
            field_id=1000,
            transform=DayTransform(),
            name="report_date_day"
        ),
        PartitionField(
            source_id=6,
            field_id=1001,
            transform=IdentityTransform(),
            name="disease"
        )
    )

    manager.create_table(
        namespace='silver',
        table_name='reports_validated',
        schema=manager.get_reports_schema(),
        partition_spec=PartitionSpec()
    )

    # Geography validated
    manager.create_table(
        namespace='silver',
        table_name='geography_validated',
        schema=manager.get_geography_schema(),
        partition_spec=PartitionSpec()
    )

    # Facilities validated
    manager.create_table(
        namespace='silver',
        table_name='facilities_validated',
        schema=manager.get_facilities_clean_schema(),
        partition_spec=PartitionSpec()
    )

    # Weather validated
    weather_partition = PartitionSpec(
        PartitionField(
            source_id=1,
            field_id=1000,
            transform=DayTransform(),
            name="date_day"
        )
    )
    manager.create_table(
        namespace='silver',
        table_name='weather_validated',
        schema=manager.get_weather_schema(),
        partition_spec=PartitionSpec()
    )

def drop_tables():
    manager = IcebergManager()
    
    # Tables to recreate
    tables_to_drop = [
        ('bronze', 'reports'),
        ('bronze', 'geography'),
        ('bronze', 'facilities'),
        ('bronze', 'weather'),
        ('silver', 'reports_validated'),
        ('silver', 'geography_validated'),
        ('silver', 'facilities_validated'),
        ('silver', 'weather_validated')
    ]
    
    # Drop tables
    for namespace, table_name in tables_to_drop:
        full_name = f"{namespace}.{table_name}"
        try:
            manager.catalog.drop_table(full_name)
            print(f"Dropped table: {full_name}")
        except Exception as e:
            print(f"Could not drop {full_name}: {e}")

def list_tables():
    manager = IcebergManager()
    for namespace in ['bronze', 'silver']:
        print(f"{namespace.upper()} Layer")
        try:
            tables = manager.catalog.list_tables(namespace)
            for table in tables:
                print(table)
        except Exception as e:
            print(f"No table in {namespace}")

if __name__ == "__main__":
    # drop_tables()
    # create_bronze_tables()
    # create_silver_tables()

    list_tables()
    