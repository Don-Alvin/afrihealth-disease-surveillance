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

from generate_climate_data import (
    generate_climate_data, save_climate_data
)

from generate_disease_data import (
    generate_disease_data, save_disease_data
)

def main():
    print('Afrihealth data generation.')
    print()

    # Configuration
    NUM_DISTRICTS = 30
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

    # Generate climate data
    print('Generating climate data....')
    climate_df = generate_climate_data(
        geography_df=geo_df,
        start_date=START_DATE,
        end_date=END_DATE,
        random_state=RANDOM_STATE
    )

    climate_path = save_climate_data(climate_df)
    print()

    # Generate disease data
    print('Generating disease data....')
    disease_df = generate_disease_data(
        geography_df=geo_df,
        facility_df=facility_df,
        climate_df=climate_df,
        start_date=START_DATE,
        end_date=END_DATE,
        random_state=RANDOM_STATE
    )

    disease_path = save_disease_data(disease_df)

    print('Data generation complete!')

if __name__ == "__main__":
        main()

