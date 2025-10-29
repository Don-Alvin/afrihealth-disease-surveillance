"""
This script when run will generate all data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generate_geographical_data import(
    generate_geographical_data, save_geographical_data
)

from generate_facility_data import (
    generate_facility_data, save_facility_data
)

from generate_weather_data import (
    generate_weather_data, save_weather_data
)

from generate_reports_data import (
    generate_reports_data, save_reports_data
)

def main():
    print('Afrihealth data generation.')
    print()

    # Configuration
    NUM_DISTRICTS = 100
    NUM_FACILITIES = 500
    START_DATE = '2020-01-01'
    END_DATE = '2024-12-31'
    RANDOM_STATE = 42

    # Generate data
    # Generate geographical data
    print('Generating geographical data....')
    print()
    geo_df = generate_geographical_data(
        num_districts=NUM_DISTRICTS,
        random_state=RANDOM_STATE
    )

    geo_path = save_geographical_data(geo_df)
    print()

    # Generate facilities data
    print('Generating facilities data...')
    print()
    facility_df = generate_facility_data(
        geography_df=geo_df,
        num_facilities=NUM_FACILITIES,
        random_state=RANDOM_STATE
    )

    facility_path = save_facility_data(facility_df)
    print()

    # Generate weather data
    print('Generating weather data....')
    weather_df = generate_weather_data(
        geography_df=geo_df,
        start_date=START_DATE,
        end_date=END_DATE,
        random_state=RANDOM_STATE
    )

    weather_path = save_weather_data(weather_df)
    print()

    # Generate reports data
    print('Generating reports data....')
    reports_df = generate_reports_data(
        geography_df=geo_df,
        facility_df=facility_df,
        weather_df=weather_df,
        start_date=START_DATE,
        end_date=END_DATE,
        random_state=RANDOM_STATE
    )

    reports_path = save_reports_data(reports_df)

    print('Data generation complete!')

if __name__ == "__main__":
        main()

