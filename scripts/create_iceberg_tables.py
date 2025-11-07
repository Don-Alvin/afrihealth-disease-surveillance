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
        partition_spec=reports_partition
    )

    # Geography table
    manager.create_table(
        namespace='bronze',
        table_name='geography',
        schema=manager.get_geography_schema(),
        partition_spec=None
    )

    # Facilities table
    manager.create_table(
        namespace='bronze',
        table_name='facilities',
        schema=manager.get_facilities_schema(),
        partition_spec=None
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
        partition_spec=weather_partition
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
        partition_spec=reports_partition
    )

    # Geography validated
    manager.create_table(
        namespace='silver',
        table_name='geography_validated',
        schema=manager.get_geography_schema(),
        partition_spec=None
    )

    # Facilities validated
    manager.create_table(
        namespace='silver',
        table_name='facilities_validated',
        schema=manager.get_facilities_schema(),
        partition_spec=None
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
        partition_spec=weather_partition
    )

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
    create_bronze_tables()
    create_silver_tables

    list_tables()
    