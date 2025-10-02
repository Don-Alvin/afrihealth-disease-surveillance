"""
This module generates synthetic climate data for the Afrihealth project.
It creates daily climate observations (temperature, humidity, rainfall) for each geograriphica area that mimics African seasonal patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple

def generate_climate_data(
        geography_df: pd.DataFrame, 
        start_date: str = '2020-01-01', 
        end_date: str = '2024-12-31',
        random_state: int = 42
        ) -> pd.DataFrame:
    
    """
    This function generates realistic daily weather data (temperature, humidity, rainfall) for each geography in the provided DataFrame.

    Args:
        geography_df (pd.DataFrame): DataFrame containing geography information with at least 'geography_id' column.
        start_date (str): Start date for the generated data in 'YYYY-MM-DD' format.
        end_date (str): End date for the generated data in 'YYYY-MM-DD' format.
        random_state (int): Seed for random number generator for reproducibility.
    
    Returns: 
        pd.DataFrame: DataFrame containing daily climate data for each geography with columns:
            - date: Date of observation
            - geography_id: Identifier for the geography (links to geographical data)
            - temperature_min_c: Minimum daily temperature in Celsius
            - temperature_max_c: Maximum daily temperature in Celsius
            - rainfall_mm: Daily rainfall in millimeters
            - humidity_pct: Relative humidity percentage
    
    Example:
        >>> geography_df = pd.DataFrame(data/raw/geograohical_data.csv)
        >>> climate_df = generate_climate_data(geography_df, '2020-01-01', '2024-12-31')
        >>> climate_df.to_csv('data/raw/climate_data.csv', index=False)
    """
    np.random.seed(random_state)

    # Create date range
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    date_range = pd.date_range(start, end, freq='D')
    num_days = len(date_range)

    print(f"Generating climate data from {start_date} to {end_date} for {len(geography_df)} geographic areas in {num_days} days.")

    data = []

    for _, row in geography_df.iterrows():
        geography_id = row['geography_id']
        latitude = row['latitude']
        country_code = row['country_code']

        # Location temperature patterns
        distance_from_equator = abs(latitude)
        base_temp = 30 - (distance_from_equator / 3)

        # Coastal vs inland temperature adjustment
        is_coastal = any(word in row['region_name'].lower() for word in ['coast', 'lagos', 'Eastern Cape', 'KwaZulu-Natal', 'Northern Cape', 'Western Cape'])
        if is_coastal:
            temp_variation = 5
        else:
            temp_variation = 10
        
        # Country specific climate patterns
        if country_code == 'KE':  # Kenya
            rainy_months = [3, 4, 5, 10, 11]
            base_rainfall = 80
        elif country_code == 'NG':  # Nigeria
            rainy_months = [4, 5, 6, 7, 8, 9]
            base_rainfall = 120
        elif country_code == 'ZA':  # South Africa
            rainy_months = [11, 12, 1, 2, 3]
            base_rainfall = 100
        else:
            rainy_months = []
            base_rainfall = 50
        
        for date in date_range:
            month = date.month
            day_of_the_year = date.timetuple().tm_yday

            # Seasonal temperature variation
            seasonal_factor = np.sin((day_of_the_year / 365.0) * 2 * np.pi)
            temp_avg = base_temp + (seasonal_factor * temp_variation)

            # Daily variation
            temp_avg += np.random.normal(0, 2)
            temp_min = temp_avg - np.random.uniform(5, 12) / 2
            temp_max = temp_avg + np.random.uniform(5, 12) / 2

            # Rainfall generation
            if month in rainy_months:
                rain_prob = 0.6
                if np.random.rand() < rain_prob:
                    rainfall = np.random.exponential(base_rainfall)
                    rainfall = min(rainfall, 200)
                else:
                    rainfall = 0.0
            else:
                rain_prob = 0.15
                if np.random.rand() < rain_prob:
                    rainfall = np.random.exponential(base_rainfall / 4)
                    rainfall = min(rainfall, 50)
                else:
                    rainfall = 0.0
            
            # Humidity generation
            base_humidity = 70 if month in rainy_months else 50
            if rainfall > 0:
                humidity = min(100, base_humidity + np.random.uniform(10, 30))
            else:
                humidity = max(20, base_humidity - np.random.uniform(5, 15))
            humidity = max(20, min(humidity, 100))
            data.append({
                'date': date,
                'geography_id': geography_id,
                'temperature_min_c': round(temp_min, 1),
                'temperature_max_c': round(temp_max, 1),
                'rainfall_mm': round(rainfall, 1),
                'humidity_pct': round(humidity, 1)
            })
    df = pd.DataFrame(data)

    # Add data quality issues to mimic real-world data
    # Introduce some missing values
    missing_indices = np.random.choice(df.index, size=int(0.005 * len(df)), replace=False)
    df.loc[missing_indices, 'rainfall_mm'] = np.nan

    # Unrealistic outliers - very rare
    outlier_indices = np.random.choice(df.index, size=5, replace=False)
    df.loc[outlier_indices[0], 'temperature_max_c'] = 999.9
    df.loc[outlier_indices[1], 'rainfall_mm'] = -5.0

    # Station offline - missing for one day
    offline_indices = np.random.choice(df.index, size=5, replace=False)
    for idx in offline_indices:
        geo_id = df.loc[idx, 'geography_id']
        date = df.loc[idx, 'date']
        df.loc[(df['geography_id'] == geo_id) & (df['date'] == date), ['temperature_min_c', 'temperature_max_c', 'rainfall_mm', 'humidity_pct']] = np.nan
    
    print(f"Generated climate data with {len(df)} records.")
    return df

def save_climate_data(df: pd.DataFrame, output_path: str = "data/raw") -> Path:
    """
    Saves the climate data DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame containing climate data.
        output_path (str): Directory path where the CSV file will be saved.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / "climate.csv"
    df.to_csv(file_path, index=False)
    return file_path





