"""
This module generates healthcare facility data.

This module creates synthetic data for various healthcare facilities with varying geographical locations,
capacities, and types. The generated data can be used for simulations, testing algorithms, or populating databases during development.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

def generate_facility_data(geography_df: pd.DataFrame, num_facilities: int = 500, random_state=42) -> pd.DataFrame:
    """
    This function generates synthetic healthcare facility data.

    Healthcare facilities are distributed based on provided geographical data provided.
    Urban areas have a higher density of facilities compared to rural areas.

    Args:
        geography_df (pd.DataFrame): DataFrame generated from generate_geographical_data.py
        num_facilities (int): Number of healthcare facilities to generate.
        random_state (int): Seed for random number generator for reproducibility.
    
    Returns: pd.DataFrame
        A DataFrame containing generated healthcare facility data with columns:
        - facility_id: Unique identifier for the facility.
        - facility_name: Name of the healthcare facility.
        - facility_type: Type of the facility e.g Hospital, Clinic, Health Center.
        - capacity: Number of beds or patient capacity.
        - facility-level: 1 - 6 (1 being primary care, 6 being specialized care).
        - geography_id: Foreign key to the geography DataFrame.
        - staff_count: Number of staff members.
        - has_lab: Boolean indicating if the facility has a lab.
        - has_isolation_ward: Boolean indicating if the facility has an isolation ward.
        - has_xray: Boolean indicating if the facility has X-ray facilities.
        - ambulance_count: Number of ambulances available.
        - latitude: Latitude of the facility location.
        - longitude: Longitude of the facility location.
        - operational_status: Status of the facility (Operational, Under Construction, Closed).
        - established_year: Year the facility was established.

    Example:
        >>> geography_df = pd.read_csv('geography_data.csv')
        >>> facility_df = generate_facility_data(geography_df, num_facilities=500)
        >>> facility_df.to_csv('data/raw/facility_data.csv', index=False)
        """
    np.random.seed(random_state)

    # Facility types and their corresponding levels
    facility_types = {
        'National Hospital': {
            'level': 6,
            'capacity_range': (500, 2000),
            'staff_range': (200, 1000),
            'min_pop_density': 1000,
            'weight': 1
        },
        'Regional Hospital': {
            'level': 5,
            'capacity_range': (200, 800),
            'staff_range': (100, 300),
            'min_pop_density': 500,
            'weight': 2
        },
        'District Hospital': {
            'level': 4,
            'capacity_range': (100, 200),
            'staff_range': (50, 150),
            'min_pop_density': 200,
            'weight': 3
        },

        'Health Center': {
            'level': 3,
            'capacity_range': (20, 100),
            'staff_range': (10, 50),
            'min_pop_density': 50,
            'weight': 5
        },
        'Clinic': {
            'level': 2,
            'capacity_range': (5, 20),
            'staff_range': (5, 20),
            'min_pop_density': 10,
            'weight': 8
        },
        'Dispensary': {
            'level': 1,
            'capacity_range': (1, 5),
            'staff_range': (2, 10),
            'min_pop_density': 0,
            'weight': 10
        }
    }

    data = []

    # Calculate facilities per geography based on population density
    geography_df = geography_df.copy()
    geography_df['population_density'] = geography_df['population_density'].fillna(0)

    # Assign weights based on urban/rural classification
    geography_df['weight'] = geography_df['population_density']
    geography_df.loc[geography_df['urban_rural'] == 'Urban', 'weight'] *= 2

    # Normalize weights
    geography_df['weight'] = geography_df['weight'] / geography_df['weight'].sum()
    geography_df['num_facilities'] = (geography_df['weight'] * num_facilities).round().astype(int)

    # Ensure at least one facility in each geography
    geography_df.loc[geography_df['num_facilities'] == 0, 'num_facilities'] = 1

    facility_counter = 1

    for _, row in geography_df.iterrows():
        num_facilities_in_geo = row['num_facilities']
        population_density = row['population_density']
        is_urban = row['urban_rural'] == 'Urban'

        for _ in range(num_facilities_in_geo):
            # Determine facility type based on population density and urban/rural classification
            possible_types = []
            weights = []

            for ftype, finfo, in facility_types.items():
                if population_density > finfo['min_pop_density']:
                    possible_types.append(ftype)
                    weight = finfo['weight']
                    if is_urban and finfo['level'] <= 3:
                        weight *=3
                    weights.append(weight)
                
            if not possible_types:
                possible_types = ['Health Center', 'Clinic', 'Dispensary']
                weights = [1, 1, 1]
            
            # Normalize weights
            weights = np.array(weights) / sum(weights)
            facility_type = np.random.choice(possible_types, p=weights)
            facility_info = facility_types[facility_type]

            # Generate facility details
            facility_id = f"FAC-{facility_counter:05d}"
            facility_name = f"{row['district_name']} {facility_type}-{facility_counter}"
            district_name = row['district_name']
            level = facility_info['level']

            if level >= 5:
                capacity = np.random.randint(500, 2000)
                staff_count = np.random.randint(200, 1000)
                ambulance_count = np.random.randint(5, 20)
                has_lab = True
                has_isolation_ward = True
                has_xray = True
            elif level == 4:
                capacity = np.random.randint(200, 800)
                staff_count = np.random.randint(100, 300)
                ambulance_count = np.random.randint(2, 10)
                has_lab = True
                has_isolation_ward = True
                has_xray = True
            elif level < 4:
                capacity = np.random.randint(5, 100)
                staff_count = np.random.randint(5, 50)
                ambulance_count = np.random.randint(0, 3)
                has_lab = np.random.rand() < 0.3
                has_isolation_ward = np.random.rand() < 0.2
                has_xray = np.random.rand() < 0.4
            
            lat = row['latitude'] + np.random.uniform(-0.05, 0.05)
            lon = row['longitude'] + np.random.uniform(-0.05, 0.05)
            
            status_prob = np.random.rand()
            if status_prob < 0.85:
                operational_status = 'Operational'
            elif status_prob < 0.95:
                operational_status = 'Under Construction'
            else:
                operational_status = 'Closed'

            established_year = np.random.randint(1990, 2024)

            data.append({
                'facility_id': facility_id,
                'facility_name': facility_name,
                'facility_type': facility_type,
                'facility_level': level,
                'geography_id': row['geography_id'],
                'country_code': row['country_code'],
                'region_name': row['region_name'],
                'district_name': district_name,
                'capacity': capacity,
                'staff_count': staff_count,
                'has_lab': has_lab,
                'has_isolation_ward': has_isolation_ward,
                'has_xray': has_xray,
                'ambulance_count': ambulance_count,
                'latitude': round(lat, 6),
                'longitude': round(lon, 6),
                'operational_status': operational_status,
                'established_year': established_year
            })

            facility_counter += 1
    
    df = pd.DataFrame(data)

    # Add data quality issues to mimic real-world data
    # Introduce some missing values for capacity 
    missing_indices = np.random.choice(df.index, size=int(0.01 * len(df)), replace=False)
    df.loc[missing_indices, 'capacity'] = np.nan

    # Some facilities have inconsistent data (beds but no staff)
    inconsistent_indices = np.random.choice(df.index, size=3, replace=False)
    df.loc[inconsistent_indices, 'staff_count'] = 0

    # Add some duplicate entries
    duplicate_indices = np.random.choice(df.index, size=int(0.005 * len(df)), replace=False)
    duplicates = df.loc[duplicate_indices]

    print(f"Generated {len(df)} healthcare facilities.")

    return df

def save_facility_data(df: pd.DataFrame, output_path: str = "data/raw") -> Path:
    """
    Saves the generated facility data to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame containing the facility data.
        output_path (str): Directory path where the CSV file will be saved.
    
    Returns: Path
        The path to the saved CSV file.
    """

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / "facilities.csv"
    df.to_csv(file_path, index=False)
    print(f"Facility data saved to {file_path}")
    print(f"File size: {file_path.stat().st_size / 1024:.2f} KB")
    return file_path



