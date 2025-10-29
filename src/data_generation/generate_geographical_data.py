"""
Generate geographical data for the afrihealth-disease-surveillance project.
This module creates synthetic geographical data for Kenya, Nigeria, and South Africa
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeographyDataGenerator:
    """
    A class to generate synthetic geographical data for African countries.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the geography data generator.
        
        Args:
            random_state (int): Seed for random number generator for reproducibility.
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Define countries with their detailed information
        self.countries = {
            'KE': {
                'name': 'Kenya',
                'regions': {
                    'Nairobi': {'urban_share': 0.95, 'population_share': 0.15},
                    'Coast': {'urban_share': 0.35, 'population_share': 0.12},
                    'Rift Valley': {'urban_share': 0.25, 'population_share': 0.20},
                    'Central': {'urban_share': 0.30, 'population_share': 0.15},
                    'Eastern': {'urban_share': 0.20, 'population_share': 0.13},
                    'Western': {'urban_share': 0.18, 'population_share': 0.12},
                    'Nyanza': {'urban_share': 0.22, 'population_share': 0.13}
                },
                'lat_range': (-4.7, 4.6),
                'lon_range': (33.9, 41.9),
                'total_population': 56_000_000,
                'district_share': 0.30
            },
            'NG': {
                'name': 'Nigeria',
                'regions': {
                    'Lagos': {'urban_share': 0.90, 'population_share': 0.18},
                    'Kano': {'urban_share': 0.60, 'population_share': 0.15},
                    'Kaduna': {'urban_share': 0.45, 'population_share': 0.12},
                    'Rivers': {'urban_share': 0.70, 'population_share': 0.10},
                    'Oyo': {'urban_share': 0.50, 'population_share': 0.11},
                    'Borno': {'urban_share': 0.25, 'population_share': 0.08},
                    'Enugu': {'urban_share': 0.40, 'population_share': 0.08},
                    'Others': {'urban_share': 0.35, 'population_share': 0.18}
                },
                'lat_range': (4.3, 13.9),
                'lon_range': (2.7, 14.6),
                'total_population': 232_000_000,
                'district_share': 0.25
            },
            'ZA': {
                'name': 'South Africa',
                'regions': {
                    'Gauteng': {'urban_share': 0.95, 'population_share': 0.25},
                    'KwaZulu-Natal': {'urban_share': 0.60, 'population_share': 0.20},
                    'Western Cape': {'urban_share': 0.85, 'population_share': 0.15},
                    'Eastern Cape': {'urban_share': 0.35, 'population_share': 0.12},
                    'Limpopo': {'urban_share': 0.25, 'population_share': 0.10},
                    'Mpumalanga': {'urban_share': 0.40, 'population_share': 0.08},
                    'Free State': {'urban_share': 0.45, 'population_share': 0.06},
                    'North West': {'urban_share': 0.30, 'population_share': 0.04}
                },
                'lat_range': (-34.0, -22.0),
                'lon_range': (16.5, 32.9),
                'total_population': 64_000_000,
                'district_share': 0.20
            }
        }

    def _generate_coordinates(self, country_code: str, region_name: str) -> Tuple[float, float]:
        """
        Generate realistic coordinates based on country and region.
        
        Args:
            country_code (str): Country code
            region_name (str): Region name
            
        Returns:
            Tuple[float, float]: Latitude and longitude coordinates
        """
        country_info = self.countries[country_code]
        lat_range = country_info['lat_range']
        lon_range = country_info['lon_range']
        
        # Add some regional bias to coordinates
        region_bias = hash(region_name) % 100 / 100.0  # Deterministic bias based on region name
        
        lat = np.random.uniform(lat_range[0], lat_range[1]) 
        lon = np.random.uniform(lon_range[0], lon_range[1])
        
        # Apply slight regional clustering
        lat += (region_bias - 0.5) * (lat_range[1] - lat_range[0]) * 0.3
        lon += (region_bias - 0.5) * (lon_range[1] - lon_range[0]) * 0.3
        
        # Ensure coordinates stay within country bounds
        lat = max(lat_range[0], min(lat_range[1], lat))
        lon = max(lon_range[0], min(lon_range[1], lon))
        
        return round(lat, 6), round(lon, 6)

    def _generate_population(self, country_code: str, region_name: str, is_urban: bool) -> int:
        """
        Generate realistic population numbers based on location and urban/rural classification.
        
        Args:
            country_code (str): Country code
            region_name (str): Region name
            is_urban (bool): Whether the area is urban
            
        Returns:
            int: Population number
        """
        country_info = self.countries[country_code]
        region_info = country_info['regions'][region_name]
        
        if is_urban:
            # Urban areas have higher population
            if country_code == 'NG':  # Nigeria has very dense urban areas
                population = np.random.randint(1_000_000, 10_000_000)
            elif country_code == 'ZA':  # South Africa
                population = np.random.randint(500_000, 5_000_000)
            else:  # Kenya
                population = np.random.randint(500_000, 3_000_000)
        else:
            # Rural areas
            if country_code == 'NG':
                population = np.random.randint(100_000, 1_000_000)
            elif country_code == 'ZA':
                population = np.random.randint(50_000, 500_000)
            else:  # Kenya
                population = np.random.randint(50_000, 400_000)
                
        return population

    def generate_geographical_data(self, total_districts: int = 100) -> pd.DataFrame:
        """
        Generate synthetic geographical data for Kenya, Nigeria, and South Africa.
        
        Args:
            total_districts (int): Total number of districts to generate across all countries.
            
        Returns:
            pd.DataFrame: DataFrame containing the generated geographical data.
        """
        data = []
        district_counter = 1  # Changed from geography_id_counter
        
        for country_code, country_info in self.countries.items():
            # Calculate number of districts for this country
            num_districts = int(total_districts * country_info['district_share'])
            regions = list(country_info['regions'].keys())
            
            # Distribute districts among regions based on population share
            region_districts = {}
            remaining_districts = num_districts
            
            for i, region in enumerate(regions):
                if i == len(regions) - 1:
                    region_districts[region] = remaining_districts
                else:
                    region_share = country_info['regions'][region]['population_share']
                    region_district_count = max(1, int(num_districts * region_share))
                    region_districts[region] = region_district_count
                    remaining_districts -= region_district_count
            
            # Generate data for each region
            for region_name, district_count in region_districts.items():
                region_info = country_info['regions'][region_name]
                
                for district_num in range(district_count):
                    district_name = f"{region_name} District {district_num + 1}"
                    
                    # Generate 2-4 sub-districts per district
                    num_sub_districts = np.random.randint(2, 5)
                    
                    for sub_district_num in range(num_sub_districts):
                        # Determine if urban or rural
                        is_urban = np.random.random() < region_info['urban_share']
                        urban_rural = 'Urban' if is_urban else 'Rural'
                        
                        # Generate data
                        population = self._generate_population(country_code, region_name, is_urban)
                        lat, lon = self._generate_coordinates(country_code, region_name)
                        
                        # Calculate population density (people per sq km)
                        area_sq_km = np.random.randint(10, 1000) if is_urban else np.random.randint(50, 5000)
                        population_density = population / area_sq_km
                        
                        # Generate geography_id in the original format: "KE-NAI-001"
                        region_code = region_name[:3].upper()
                        geography_id = f"{country_code}-{region_code}-{district_counter:03d}"
                        
                        data.append({
                            'geography_id': geography_id,
                            'country_code': country_code,
                            'country_name': country_info['name'],
                            'region_name': region_name,
                            'district_name': district_name,
                            'sub_district_name': f"{district_name} Sub-district {sub_district_num + 1}",
                            'population': population,
                            'urban_rural': urban_rural,
                            'latitude': lat,
                            'longitude': lon,
                            'area_sq_km': area_sq_km,
                            'population_density': round(population_density, 2),
                            'elevation': np.random.randint(0, 2500),  # meters above sea level
                            'healthcare_access_index': round(np.random.uniform(0.1, 0.9), 3)  # 0-1 scale
                        })
                        
                        district_counter += 1
        
        df = pd.DataFrame(data)
    
        # VALIDATION: Check if we're close to target populations
        country_targets = {
            'KE': 56_000_000,
            'NG': 232_000_000, 
            'ZA': 64_000_000
        }
        
        for country_code, target in country_targets.items():
            country_pop = df[df['country_code'] == country_code]['population'].sum()
            ratio = country_pop / target
            
            if ratio < 0.8 or ratio > 1.2:  # If we're more than 20% off
                logger.warning(f"{country_code} population is {country_pop:,} vs target {target:,} (ratio: {ratio:.2f})")
                
                # Auto-adjust populations to match target
                adjustment_factor = target / country_pop
                mask = df['country_code'] == country_code
                df.loc[mask, 'population'] = (df.loc[mask, 'population'] * adjustment_factor).astype(int)
                logger.info(f"Adjusted {country_code} populations by factor {adjustment_factor:.2f}")
        
        # Introduce realistic data quality issues
        df = self._introduce_data_issues(df)
        
        self._log_generation_stats(df)
        return df

    def _introduce_data_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Introduce realistic data quality issues found in real-world datasets.
        
        Args:
            df (pd.DataFrame): Clean geographical data
            
        Returns:
            pd.DataFrame: Data with introduced quality issues
        """
        # 1. Missing values (2% of population data)
        missing_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
        df.loc[missing_indices, 'population'] = np.nan
        
        # 2. Duplicate geography IDs (1% of entries) - using the original format
        duplicate_indices = np.random.choice(df.index, size=int(0.01 * len(df)), replace=False)
        for idx in duplicate_indices:
            if idx > 0:
                df.at[idx, 'geography_id'] = df.at[idx - 1, 'geography_id']
        
        # 3. Inconsistent formatting in some names
        format_indices = np.random.choice(df.index, size=int(0.03 * len(df)), replace=False)
        for idx in format_indices:
            if 'district' in df.at[idx, 'district_name'].lower():
                df.at[idx, 'district_name'] = df.at[idx, 'district_name'].upper()
        
        # 4. Outlier populations (0.5% of entries)
        outlier_indices = np.random.choice(df.index, size=int(0.005 * len(df)), replace=False)
        for idx in outlier_indices:
            df.at[idx, 'population'] = df.at[idx, 'population'] * 10
        
        return df

    def _log_generation_stats(self, df: pd.DataFrame):
        """Log statistics about the generated data."""
        logger.info(f"Generated {len(df)} geographical entries")
        logger.info(f"Countries: {df['country_code'].nunique()}")
        logger.info(f"Regions: {df['region_name'].nunique()}")
        logger.info(f"Districts: {df['district_name'].nunique()}")
        logger.info(f"Total Population: {df['population'].sum():,.0f}")
        
        # Log country-specific stats
        for country_code in df['country_code'].unique():
            country_data = df[df['country_code'] == country_code]
            urban_rural_ratio = (country_data['urban_rural'] == 'Urban').mean()
            logger.info(f"{country_code}: {len(country_data)} entries, "
                       f"Urban: {urban_rural_ratio:.1%}, "
                       f"Pop: {country_data['population'].sum():,.0f}")

def generate_geographical_data(num_districts: int = 100, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic geographical data for Kenya, Nigeria, and South Africa.
    
    This is the main function that creates a DataFrame with country, region, 
    district and sub-district information.
    
    Args:
        num_districts (int): Total number of districts to generate across all countries.
        random_state (int): Seed for random number generator for reproducibility.
        
    Returns:
        pd.DataFrame: DataFrame containing the generated geographical data with the following columns:
            - geography_id: Unique identifier for each geographical entry in format "CC-REG-XXX".
            - country_code: Country code (e.g., 'KE', 'NG', 'ZA').
            - country_name: Name of the country.
            - region_name: Name of the region within the country.
            - district_name: Name of the district within the region.
            - sub_district_name: Name of the sub-district within the district.
            - population: Population of the sub-district.
            - urban_rural: Classification of the area as 'Urban' or 'Rural'.
            - latitude: Latitude coordinate of the sub-district.
            - longitude: Longitude coordinate of the sub-district.
            - area_sq_km: Area in square kilometers.
            - population_density: Population density (people per sq km).
            - elevation: Elevation in meters above sea level.
            - healthcare_access_index: Healthcare access index (0-1 scale).
    
    Example:
        >>> df = generate_geographical_data(num_districts=100)
        >>> print(df.head())
        >>> df.to_csv('data/raw/geographical.csv', index=False)
    """
    generator = GeographyDataGenerator(random_state=random_state)
    return generator.generate_geographical_data(total_districts=num_districts)

def save_geographical_data(df: pd.DataFrame, output_path: str = "data/raw") -> Path:
    """
    Save the generated geographical data to a CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame containing the geographical data.
        output_path (str): Directory path where the CSV file will be saved.
        
    Returns:
        Path: Path to saved CSV file.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / "geographical.csv"
    
    df.to_csv(file_path, index=False)
    
    logger.info(f"Geographical data saved to {file_path}")
    logger.info(f"File size: {file_path.stat().st_size / 1024:.2f} KB")
    logger.info(f"Records saved: {len(df)}")
    
    return file_path



