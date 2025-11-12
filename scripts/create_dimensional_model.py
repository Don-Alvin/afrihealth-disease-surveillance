"""
This script executes sql for dimensional modeling.
It creates:
    - Dimension tables (dim_date, dim_facilities, dim_weather, dim_disease, dim_geography)
    - Fact table (fact_reports)
"""

import os
from google.cloud import bigquery
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
project_id = os.getenv('GCP_PROJECT_ID')
key_path = os.getenv('GCP_SERVICE_ACCOUNT_KEY_PATH')

client = bigquery.Client.from_service_account_json(
    key_path,
    project=project_id
)

def run_sql_file(sql_file: str):
    """
    This function execute sql file

    Args:
        sql_file: Path to SQL file
    """

    with open(sql_file, 'r') as f:
        sql = f.read()

    # Substitute project_id
    sql = sql.replace('{project_id}', project_id)

    # Split the sql file into individual statements
    statements = [s.strip() for s in sql.split(';') if s.strip()]
    try:
        for i, statement in enumerate(statements, 1):
            if statement.startswith('--'):
                continue

            if 'CREATE' in statement.upper():
                if 'dim_date' in statement:
                    table_name = 'dim_date'
                elif 'dim_geography' in statement:
                    table_name = 'dim_geography'
                elif 'dim_facilities' in statement:
                    table_name = 'dim_facilities'
                elif 'dim_disease' in statement:
                    table_name = 'dim_disease'
                elif 'dim_weather' in statement:
                    table_name = 'dim_weather'
                elif 'fact_reports' in statement:
                    table_name = 'fact_reports'
                elif 'SELECT' in statement.upper() and 'UNION' in statement.upper():
                    table_name = 'verification_query'
                else:
                    table_name = f'statement {i}'
                
                print(f"Creating: {table_name}")
            else:
                print(f'Executing statement {i}')

            try:
                query_job = client.query(statement)
                query_job.result()
                print(f"{table_name} creation success")

                if 'SELECT' in statement.upper() and 'row_count' in statement:
                    print("\nRow counts:")
                    for row in query_job:
                        print(f"  {row['table_name']}: {row['row_count']:,}")
            except Exception as e:
                print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    print("Dimensional modeling...")
    sql_file = 'sql/bigquery/create_dimensional_model.sql'
    if not Path(sql_file).exists():
        print(f'Error: SQL file "{sql_file}" not found.')
    
    run_sql_file(sql_file)

if __name__ == "__main__":
    main()
