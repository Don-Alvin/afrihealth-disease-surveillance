import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data_ingestion.iceberg_manager import IcebergManager
from pyiceberg.partitioning import PartitionSpec, PartitionField
from pyiceberg.transforms import DayTransform, IdentityTransform

manager = IcebergManager()

def create_partition_specs():
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
    
    # Weather
    weather_partition = PartitionSpec(
        PartitionField(
            source_id=1,
            field_id=1000,
            transform=DayTransform(),
            name="date_day"
        )
    )

    return {
        'bronze.reports': reports_partition,
        'silver.reports_validated': reports_partition,
        'bronze.weather': weather_partition,
        'silver.weather_validated': weather_partition
    }

def add_partitions_to_tables():
    partition_specs = create_partition_specs()

    for full_table_name, partition_spec in partition_specs.items():
        try:
            table = manager.catalog.load_table(full_table_name)
            current_spec = table.spec()
            if not current_spec.is_unpartitioned:
                print(f"Table {full_table_name} is already partitioned")
                continue
            with table.update_spec() as update:
                for field in partition_spec.fields:
                    col_name = table.schema().find_column_name(field.source_id)
                    update.add_field(col_name, field.transform, field.name)
            print(f"Successfully added partitions to {full_table_name}")
            # print(table.specs())
        except Exception as e:
            print(f"Failed to add partitions to {full_table_name}: {e}")

def verify_partitions():
    tables_to_check = [
        'bronze.reports',
        'bronze.weather',
        'silver.reports_validated',
        'silver.weather_validated'
    ]

    for full_table_name in tables_to_check:
        try:
            table = manager.catalog.load_table(full_table_name)
            spec = table.spec()

            if spec.is_unpartitioned:
                print(f"{full_table_name} is not partitioned!")
            else:
                partition_fields = [f.name for f in spec.fields]
                print(f"{full_table_name}: Partitions = {partition_fields}")
        except Exception as e:
            print(f"{full_table_name}: Error - {e}")

def show_table_metadata():
    tables = ['bronze.reports', 'bronze.weather', 'silver.reports_validated', 'silver.weather_validated']
    for name in tables:
        try:
            table = manager.catalog.load_table(name)
            spec = table.spec()
            print(f"\n=== {name} ===")
            print(f"Partitioned: {len(spec.fields) > 0}")
            if len(spec.fields) > 0:
                print(f"Partitions: {[f.name for f in spec.fields]}")
            print(f"Specs IDs: {list(table.specs().keys())}")
            print("Schema (first 3):")
            for f in table.schema().fields[:3]:
                print(f"  {f.field_id}: {f.name} ({type(f.field_type).__name__})")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    show_table_metadata()
    add_partitions_to_tables()
    verify_partitions()
    show_table_metadata()
