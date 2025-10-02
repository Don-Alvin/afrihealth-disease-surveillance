"""
Generate geographical data for the afrihealth-disease-surveillance project.
This module creates synthetic geographical data for Kenya, Nigeria, and South Africa
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

def generate_geographical_data(num_districts: int = 30, random_state: int = 32) -> pd.DataFrame:
    """
    Generate synthetic geographical data for Kenya, Nigeria, and South Africa.
    This creates a DataFrame with country, region, district and sub-district information.

    Args:
        num_districts (int): Number of districts to generate per country.
        random_state (int): Seed for random number generator for reproducibility.

    Returns:
        pd.DataFrame: DataFrame containing the generated geographical data with the following columns:
            - geography_id: Unique identifier for each geographical entry.
            - country_code: Country code (e.g., 'KE', 'NG', 'ZA').
            - country_name: Name of the country.
            - region_name: Name of the region within the country.
            - district_name: Name of the district within the region.
            - sub_district_name: Name of the sub-district within the district.
            - population: Population of the sub-district.
            - urban_rural: Classification of the area as 'Urban' or 'Rural'.
            - latitude: Latitude coordinate of the sub-district.
            - longitude: Longitude coordinate of the sub-district.
    
    Example:
        >>> df = generate_geography_data(num_districts=10)
        >>> print(df.head())
        >>> df.to_csv('data/raw/geography_data.csv', index=False)
    """

    np.random.seed(random_state)

    # Define countries with their information
    countries = {
        'KE': {
            'name': 'Kenya',
            'regions': ['Nairobi', 'Coast', 'Rift Valley', 'Central', 'Eastern', 'Western', 'Nyanza'],
            'lat_range': (-4.7, 4.6),
            'lon_range': (33.9, 41.9),
            'population': 56_000_000,
            'disrict_share': 0.30,
            'urban_share': 0.27
        },
        'NG': {
            'name': 'Nigeria',
            'regions': ['Lagos', 'Kano', 'Kaduna', 'Rivers', 'Oyo', 'Borno', 'Enugu'],
            'lat_range': (4.3, 13.9),
            'lon_range': (2.7, 14.6),
            'population': 232_000_000,
            'disrict_share': 0.25,
            'urban_share': 0.52
        },
        'ZA': {
            'name': 'South Africa',
            'regions': ['Gauteng', 'KwaZulu-Natal', 'Western Cape', 'Eastern Cape', 'Limpopo', 'Mpumalanga', 'Free State'],
            'lat_range': (-34.0, -22.0),
            'lon_range': (16.5, 32.9),
            'population': 64_000_000,
            'disrict_share': 0.20,
            'urban_share': 0.67
        }
    }

    data = []
    district_counter = 1

    for country_code, country_info in countries.items():

        # Distribute districts among regions
        num_regions = len(country_info['regions'])
        country_districts = int(num_districts * country_info['disrict_share'])
        districts_per_region = max(1, country_districts // num_regions)

        for region_idx, region_name in enumerate(country_info['regions']):
            region_districts = districts_per_region
            if region_idx == 0:
                region_districts = int(country_districts * 0.4)

            for district_num in range(region_districts):
                # District name
                district_name = f"{region_name} District {district_num + 1}"

                # 2 - 3 sub-districts per district
                num_sub_districts = np.random.randint(2, 4)

                for sub_district_num in range(num_sub_districts):
                    # Generate a unique geography ID
                    geography_id = f"{country_code}-{region_name[:3].upper()}-{district_counter:03d}"
                    
                    # Generate coordinates
                    lat = np.random.uniform(*country_info['lat_range'])
                    lon = np.random.uniform(*country_info['lon_range'])

                    # Population distribution
                    # Population varies depending on country and urban/rural classification
                    # Urban areas have higher population density(50k - 500k)
                    # Rural areas have lower density (5k - 50k)
                    urban_prob = 0.3 if region_idx == 0 else 0.15
                    is_urban = np.random.rand() < urban_prob

                    if is_urban:
                        population = np.random.randint(50_000, 500_000)
                        urban_rural = 'Urban'
                    else:
                        population = np.random.randint(5_000, 50_000)
                        urban_rural = 'Rural'
                    
                    # Population density
                    area_sq_km = population / (np.random.randint(100, 1000) if is_urban else np.random.randint(10, 100))
                    population_density = population / area_sq_km

                    data.append({
                        'geography_id': geography_id,
                        'country_code': country_code,
                        'country_name': country_info['name'],
                        'region_name': region_name,
                        'district_name': district_name,
                        'sub_district_name': f"{district_name} Sub-district {sub_district_num + 1}",
                        'population': population,
                        'urban_rural': urban_rural,
                        'latitude': round(lat, 6),
                        'longitude': round(lon, 6),
                        'population_density': round(population_density, 2)
                    })

                    district_counter += 1
            
    df = pd.DataFrame(data)

    # Adding some data quality issues to mimic real-world data
    # Introduce some missing values
    missing_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
    df.loc[missing_indices, 'population'] = np.nan

    # Duplicate geography_id for some entries (very few cases)
    duplicate_indices = np.random.choice(df.index, size=int(0.01 * len(df)), replace=False)
    for idx in duplicate_indices:
        if idx > 0:
            df.at[idx, 'geography_id'] = df.at[idx - 1, 'geography_id']
    
    print(f"Generated {len(df)} geographical entries across {len(countries)} countries.")
    print(f" Countries: {df['country_code'].nunique()}")
    print(f" Regions: {df['region_name'].nunique()}")
    print(f" Districts: {df['district_name'].nunique()}")
    print(f" Total Population: {df['population'].sum():,.0f}")

    return df

def save_geographical_data(df: pd.DataFrame, output_path: str = "data/raw") -> Path:
    """
    Save the generated geographical data to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame containing the geographical data.
        output_path (str): Directory path where the CSV file will be saved.

    Returns:
        Path to saved CSV file.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / "geographical.csv"
    df.to_csv(file_path, index=False)
    print(f"Geographical data saved to {file_path}")
    print(f"File size: {file_path.stat().st_size / 1024:.2f} KB")
    return file_path


        


                    
                

