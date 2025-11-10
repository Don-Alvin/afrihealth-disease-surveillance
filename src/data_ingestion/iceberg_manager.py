import os
import yaml
import pandas as pd
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.schema import Schema
from pyiceberg.types import (
    NestedField, StringType, IntegerType, DoubleType, DateType, BooleanType, LongType
)
from pyiceberg.partitioning import PartitionSpec
from dotenv import load_dotenv
import pyarrow as pa

load_dotenv()

class IcebergManager:
    def __init__(self, config_path: str = 'config/iceberg_config.yaml'):
        """
        Initialize Iceberg Manager

        Args:
            config_path: Path to iceberg configuration file
        """

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up GCS credentials
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.config['gcs']['credentials_path']

        # Initialize catalog
        self.catalog = SqlCatalog(
            'afrihealth',
            **{
                'uri':self.config['catalog']['uri'],
                'warehouse': self.config['catalog']['warehouse']
            }
        )

        print('Iceberg catalog initialized.')

    def create_namespace(self, namespace:str):
        """
        Create a namespace (like a database/schema)

        Args:
            namespace: Namespace name (eg. bronze, silver)
        """

        try:
            self.catalog.create_namespace(namespace)
            print(f"Created namespace: {namespace}")

        except Exception as e:
            if 'already exists' in str(e).lower():
                print(f"{namespace} already exists")
            else:
                raise
    
    def get_reports_schema(self) -> Schema:
        """
        Define schema for reports table

        Returns:
            Iceberg Schema Object
        """

        return Schema(
            NestedField(1, "report_id", StringType(), required=False),
            NestedField(2, "report_date", DateType(), required=False),
            NestedField(3, "case_date", DateType(), required=False),
            NestedField(4, "facility_id", StringType(), required=False),
            NestedField(5, "geography_id", StringType(), required=False),
            NestedField(6, "disease", StringType(), required=False),
            NestedField(7, "cases", DoubleType(), required=False),
            NestedField(8, "deaths", DoubleType(), required=False),
            NestedField(9, "recoveries", LongType(), required=False),
            NestedField(10, "age_group", StringType(), required=False),
            NestedField(11, "gender", StringType(), required=False)
        )

    def get_geography_schema(self) -> Schema:
        """
        Define schema for geography table

        Returns:
            Iceberg Schema Object
        """

        return Schema(
            NestedField(1, "geography_id", StringType(), required=False),
            NestedField(2, "country_code", StringType(), required=False),
            NestedField(3, "country_name", StringType(), required=False),
            NestedField(4, "region_name", StringType(), required=False),
            NestedField(5, "district_name", StringType(), required=False),
            NestedField(6, "sub_district_name", StringType(), required=False),
            NestedField(7, "population", DoubleType(), required=False),
            NestedField(8, "urban_rural", StringType(), required=False),
            NestedField(9, "latitude", DoubleType(), required=False),
            NestedField(10, "longitude", DoubleType(), required=False),
            NestedField(11, "area_sq_km", LongType(), required=False),
            NestedField(12, "population_density", DoubleType(), required=False),
            NestedField(13, "elevation", LongType(), required=False),
            NestedField(14, "healthcare_access_index", DoubleType(), required=False)
        )

    def get_facilities_schema(self) -> Schema:
        """
        Define schema for facilities table

        Returns:
            Iceberg Schema Object
        """

        return Schema(
            NestedField(1, "facility_id", StringType(), required=False),
            NestedField(2, "facility_name", StringType(), required=False),
            NestedField(3, "facility_type", StringType(), required=False),
            NestedField(4, "facility_level", LongType(), required=False),
            NestedField(5, "geography_id", StringType(), required=False),
            NestedField(6, "country_code", StringType(), required=False),
            NestedField(7, "region_name", StringType(), required=False),
            NestedField(8, "district_name", StringType(), required=False),
            NestedField(9, "capacity", DoubleType(), required=False),
            NestedField(10, "staff_count", LongType(), required=False),
            NestedField(11, "has_lab", BooleanType(), required=False),
            NestedField(12, "has_isolation_ward", BooleanType(), required=False),
            NestedField(13, "has_xray", BooleanType(), required=False),
            NestedField(14, "ambulance_count", LongType(), required=False),
            NestedField(15, "latitude", DoubleType(), required=False),
            NestedField(16, "longitude", DoubleType(), required=False),
            NestedField(17, "operational_status", StringType(), required=False),
            NestedField(18, "established_year", LongType(), required=False),
        )

    def get_facilities_clean_schema(self) -> Schema:
        """
        Define schema for facilities table

        Returns:
            Iceberg Schema Object
        """

        return Schema(
            NestedField(1, "facility_id", StringType(), required=False),
            NestedField(2, "facility_name", StringType(), required=False),
            NestedField(3, "facility_type", StringType(), required=False),
            NestedField(4, "facility_level", LongType(), required=False),
            NestedField(5, "geography_id", StringType(), required=False),
            NestedField(6, "country_code", StringType(), required=False),
            NestedField(7, "region_name", StringType(), required=False),
            NestedField(8, "district_name", StringType(), required=False),
            NestedField(9, "bed_capacity", DoubleType(), required=False),
            NestedField(10, "staff_count", LongType(), required=False),
            NestedField(11, "has_lab", BooleanType(), required=False),
            NestedField(12, "has_isolation_ward", BooleanType(), required=False),
            NestedField(13, "has_xray", BooleanType(), required=False),
            NestedField(14, "ambulance_count", LongType(), required=False),
            NestedField(15, "latitude", DoubleType(), required=False),
            NestedField(16, "longitude", DoubleType(), required=False),
            NestedField(17, "operational_status", StringType(), required=False),
            NestedField(18, "established_year", LongType(), required=False),
        )
    
    def get_weather_schema(self) -> Schema:
        """
        Define schema for weather table

        Returns:
            Iceberg Schema Object
        """

        return Schema(
            NestedField(1, "date", DateType(), required=False),
            NestedField(2, "geography_id", StringType(), required=False),
            NestedField(3, "temperature_min_c", DoubleType(), required=False),
            NestedField(4, "temperature_max_c", DoubleType(), required=False),
            NestedField(5, "temperature_avg_c", DoubleType(), required=False),
            NestedField(6, "rainfall_mm", DoubleType(), required=False),
            NestedField(7, "humidity_pct", DoubleType(), required=False),
        )

    def create_table(
            self,
            namespace: str,
            table_name: str,
            schema: Schema,
            partition_spec: PartitionSpec = None
    ):
        """
        Create an Iceberg table.

        Args:
            namespace: Namespace
            table_name: Table name
            schema: Table schema
            partition_spec: Partitioning specification
        """

        full_name = f"{namespace}.{table_name}"

        try:
            self.catalog.create_table(
                identifier=full_name,
                schema=schema,
                partition_spec=partition_spec
            )
            print(f"Created table: {full_name}")
            if partition_spec:
                print(f"Partitioned by {[f.name for f in partition_spec.fields]}")
        except Exception as e:
            if 'already exists' in str(e).lower():
                print(f"Table '{full_name}' already exists")
            else:
                print(f"Failed to create {full_name}: {e}")
                raise
        
    def load_data_to_table(
            self, 
            namespace: str,
            table_name: str,
            data_path: str
    ):
        """
        Load data into Iceberg table

        Args:
            namespace: Namespace
            table_name: Table name
            data_path: Path to data file
        """
        full_name = f"{namespace}.{table_name}"

        # Load data
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.csv.gz'):
            df = pd.read_csv(data_path, compression='gzip')
        elif data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            raise ValueError('Unsupported file format')
        
        # Convert date columns
        date_columns = ['report_date', 'case_date', 'date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
        
        # Get table
        table = self.catalog.load_table(full_name)

        # Append data
        pa_table = pa.Table.from_pandas(df)
        table.append(pa_table)


if __name__ == "__main__":
    manager = IcebergManager()
    print("Iceberg manager initialized successfully!")