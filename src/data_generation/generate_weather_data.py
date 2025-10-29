"""
This module generates synthetic weather data for the Afrihealth project.
It creates daily weather observations (temperature, humidity, rainfall) for each geographical area that mimics African seasonal patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import logging

# Set up logging
logger = logging.getLogger(__name__)

class WeatherDataGenerator:
    """
    A class to generate synthetic weather data for African countries.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the weather data generator.
        
        Args:
            random_state (int): Seed for random number generator for reproducibility.
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Region-specific weather patterns based on actual climate data
        self.region_weather_patterns = {
            'KE': {
                # Nairobi and Central - high rainfall areas
                'Nairobi': {'base_temp': 19, 'base_rainfall': 120, 'temp_variation': 8, 'rainy_months': [3,4,5,10,11]},
                'Central': {'base_temp': 18, 'base_rainfall': 130, 'temp_variation': 7, 'rainy_months': [3,4,5,10,11]},
                
                # Coast - moderate rainfall, hot and humid
                'Coast': {'base_temp': 27, 'base_rainfall': 100, 'temp_variation': 4, 'rainy_months': [4,5,6,10,11]},
                
                # Nyanza - hot and humid due to Lake Victoria, moderate rainfall
                'Nyanza': {'base_temp': 24, 'base_rainfall': 110, 'temp_variation': 5, 'rainy_months': [3,4,5,8,9,10,11]},
                
                # Western - also influenced by lake, high rainfall
                'Western': {'base_temp': 23, 'base_rainfall': 150, 'temp_variation': 6, 'rainy_months': [3,4,5,8,9,10,11]},
                
                # Other regions
                'Rift Valley': {'base_temp': 22, 'base_rainfall': 80, 'temp_variation': 10, 'rainy_months': [3,4,5,10,11]},
                'Eastern': {'base_temp': 23, 'base_rainfall': 60, 'temp_variation': 9, 'rainy_months': [3,4,5,10,11]}
            },
            'NG': {
                # Southern Nigeria - tropical rainforest climate
                'Lagos': {'base_temp': 27, 'base_rainfall': 180, 'temp_variation': 4, 'rainy_months': [3,4,5,6,7,8,9,10]},
                'Rivers': {'base_temp': 27, 'base_rainfall': 220, 'temp_variation': 4, 'rainy_months': [3,4,5,6,7,8,9,10]},
                
                # Middle Belt - tropical savanna
                'Oyo': {'base_temp': 26, 'base_rainfall': 120, 'temp_variation': 6, 'rainy_months': [4,5,6,7,8,9]},
                'Enugu': {'base_temp': 26, 'base_rainfall': 140, 'temp_variation': 6, 'rainy_months': [4,5,6,7,8,9]},
                'Kaduna': {'base_temp': 25, 'base_rainfall': 100, 'temp_variation': 8, 'rainy_months': [5,6,7,8,9]},
                
                # Northern Nigeria - semi-arid
                'Kano': {'base_temp': 28, 'base_rainfall': 60, 'temp_variation': 12, 'rainy_months': [6,7,8,9]},
                'Borno': {'base_temp': 29, 'base_rainfall': 40, 'temp_variation': 14, 'rainy_months': [6,7,8,9]},
                
                # Default for other regions
                'Others': {'base_temp': 26, 'base_rainfall': 100, 'temp_variation': 8, 'rainy_months': [4,5,6,7,8,9]}
            },
            'ZA': {
                # Summer rainfall regions
                'Gauteng': {'base_temp': 17, 'base_rainfall': 70, 'temp_variation': 10, 'rainy_months': [10,11,12,1,2,3]},
                'KwaZulu-Natal': {'base_temp': 22, 'base_rainfall': 100, 'temp_variation': 6, 'rainy_months': [10,11,12,1,2,3]},
                'Limpopo': {'base_temp': 23, 'base_rainfall': 50, 'temp_variation': 8, 'rainy_months': [11,12,1,2,3]},
                'Mpumalanga': {'base_temp': 20, 'base_rainfall': 80, 'temp_variation': 8, 'rainy_months': [10,11,12,1,2,3]},
                'Free State': {'base_temp': 16, 'base_rainfall': 60, 'temp_variation': 12, 'rainy_months': [10,11,12,1,2,3]},
                'North West': {'base_temp': 19, 'base_rainfall': 50, 'temp_variation': 10, 'rainy_months': [10,11,12,1,2,3]},
                
                # Winter rainfall region (Mediterranean climate)
                'Western Cape': {'base_temp': 18, 'base_rainfall': 50, 'temp_variation': 4, 'rainy_months': [5,6,7,8]},
                
                # Year-round rainfall with summer peak
                'Eastern Cape': {'base_temp': 19, 'base_rainfall': 70, 'temp_variation': 6, 'rainy_months': [9,10,11,12,1,2,3]}
            }
        }

    def _get_region_weather_params(self, country_code: str, region_name: str, latitude: float) -> Dict:
        """
        Get weather parameters for a specific region.
        
        Args:
            country_code (str): Country code
            region_name (str): Region name
            latitude (float): Latitude for fine-tuning
            
        Returns:
            Dict: Weather parameters for the region
        """
        if country_code in self.region_weather_patterns:
            country_patterns = self.region_weather_patterns[country_code]
            
            # Direct match for known regions
            if region_name in country_patterns:
                params = country_patterns[region_name].copy()
                
                # Fine-tune based on latitude
                if country_code == 'NG':
                    # Nigeria gets hotter further north
                    if latitude > 10:  # Northern regions
                        params['base_temp'] += 1
                    elif latitude < 6:  # Southern coastal
                        params['base_temp'] -= 1
                elif country_code == 'ZA':
                    # South Africa gets cooler further south
                    if latitude < -30:  # Southern regions
                        params['base_temp'] -= 2
                
                return params
        
        # Default fallback values
        return {
            'base_temp': 25,
            'base_rainfall': 80,
            'temp_variation': 8,
            'rainy_months': [4,5,6,7,8,9]
        }

    def generate_weather_data(
        self,
        geography_df: pd.DataFrame, 
        start_date: str = '2020-01-01', 
        end_date: str = '2024-12-31'
    ) -> pd.DataFrame:
        """
        Generate realistic daily weather data for each geography.
        
        Args:
            geography_df (pd.DataFrame): DataFrame containing geography information
            start_date (str): Start date for the generated data
            end_date (str): End date for the generated data
            
        Returns:
            pd.DataFrame: Daily weather data for each geography
        """
        # Create date range
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        date_range = pd.date_range(start, end, freq='D')
        
        logger.info(f"Generating weather data from {start_date} to {end_date} "
                   f"for {len(geography_df)} geographic areas over {len(date_range)} days.")

        data = []

        for _, row in geography_df.iterrows():
            geography_id = row['geography_id']
            latitude = row['latitude']
            country_code = row['country_code']
            region_name = row['region_name']

            # Get region-specific weather parameters
            region_params = self._get_region_weather_params(country_code, region_name, latitude)
            base_temp = region_params['base_temp']
            base_rainfall = region_params['base_rainfall']
            temp_variation = region_params['temp_variation']
            rainy_months = region_params['rainy_months']
            
            # Adjust base temperature by latitude (cooler further from equator)
            latitude_adjustment = -abs(latitude) * 0.3
            base_temp += latitude_adjustment

            for date in date_range:
                month = date.month
                day_of_year = date.timetuple().tm_yday

                # Seasonal temperature variation
                seasonal_factor = np.sin((day_of_year / 365.0) * 2 * np.pi)
                temp_avg = base_temp + (seasonal_factor * temp_variation)

                # Daily variation
                daily_variation = np.random.normal(0, 1.2)
                temp_avg += daily_variation
                
                # Min-max spread based on region characteristics
                if country_code == 'NG' and region_name in ['Kano', 'Borno']:
                    # Northern Nigeria - large diurnal temperature variation
                    min_max_spread = np.random.uniform(12, 20)
                elif country_code == 'ZA' and region_name in ['Gauteng', 'Free State']:
                    # South African highveld - large diurnal variation
                    min_max_spread = np.random.uniform(10, 16)
                elif country_code == 'KE' and region_name == 'Rift Valley':
                    # Kenyan Rift Valley - moderate to large variation
                    min_max_spread = np.random.uniform(8, 14)
                else:
                    min_max_spread = np.random.uniform(6, 12)
                
                temp_min = temp_avg - (min_max_spread / 2)
                temp_max = temp_avg + (min_max_spread / 2)

                # Ensure realistic temperature bounds
                temp_min = max(5, temp_min)
                temp_max = min(45, temp_max)

                # Rainfall generation with region-specific patterns
                if month in rainy_months:
                    rain_prob = 0.5
                    if np.random.rand() < rain_prob:
                        # Use gamma distribution for more realistic rainfall
                        rainfall = np.random.gamma(2, base_rainfall / 2)
                        
                        # Cap extreme rainfall based on region
                        if country_code == 'NG' and region_name in ['Lagos', 'Rivers']:
                            rainfall = min(rainfall, 300)  # Heavy tropical rains
                        elif country_code == 'KE' and region_name in ['Nairobi', 'Central', 'Western']:
                            rainfall = min(rainfall, 250)  # High rainfall regions
                        else:
                            rainfall = min(rainfall, 200)
                    else:
                        rainfall = 0.0
                else:
                    rain_prob = 0.1
                    if np.random.rand() < rain_prob:
                        # Light occasional rains in dry season
                        rainfall = np.random.exponential(base_rainfall / 8)
                        rainfall = min(rainfall, 30)
                    else:
                        rainfall = 0.0
                
                # Special adjustments for very arid regions
                if country_code == 'NG' and region_name in ['Kano', 'Borno']:
                    rainfall *= 0.7  # Even drier in these regions
                
                # Humidity generation - special handling for humid regions
                base_humidity = 75 if month in rainy_months else 55
                
                # Lake Victoria regions (Nyanza and Western) are hot and humid
                is_lake_region = (country_code == 'KE' and region_name in ['Nyanza', 'Western'])
                
                # Coastal regions are humid but with moderate rainfall
                is_coastal = (country_code == 'NG' and region_name in ['Lagos', 'Rivers']) or \
                           (country_code == 'ZA' and region_name in ['KwaZulu-Natal', 'Eastern Cape', 'Western Cape']) or \
                           (country_code == 'KE' and region_name == 'Coast')
                
                if is_lake_region:
                    base_humidity += 20  # Very humid due to lake effect
                elif is_coastal:
                    base_humidity += 15  # Coastal humidity
                
                # Rainfall increases humidity
                if rainfall > 0:
                    humidity_boost = min(25, rainfall / 3)
                    base_humidity += humidity_boost
                
                # Temperature affects humidity
                if temp_avg > 32:
                    base_humidity -= 8
                elif temp_avg < 15:
                    base_humidity += 5
                
                # Add random variation
                humidity = base_humidity + np.random.normal(0, 4)
                humidity = max(25, min(humidity, 95))  # Higher minimum for humid regions
                
                data.append({
                    'date': date,
                    'geography_id': geography_id,
                    'temperature_min_c': round(temp_min, 1),
                    'temperature_max_c': round(temp_max, 1),
                    'temperature_avg_c': round(temp_avg, 1),
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
            df.loc[(df['geography_id'] == geo_id) & (df['date'] == date), 
                  ['temperature_min_c', 'temperature_max_c', 'rainfall_mm', 'humidity_pct']] = np.nan
        
        logger.info(f"Generated weather data with {len(df)} records.")
        return df

def generate_weather_data(
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
        pd.DataFrame: DataFrame containing daily weather data for each geography with columns:
            - date: Date of observation
            - geography_id: Identifier for the geography (links to geographical data)
            - temperature_min_c: Minimum daily temperature in Celsius
            - temperature_max_c: Maximum daily temperature in Celsius
            - temperature_avg_c: Average daily temperature in Celsius  
            - rainfall_mm: Daily rainfall in millimeters
            - humidity_pct: Relative humidity percentage
    
    Example:
        >>> geography_df = pd.DataFrame(data/raw/geograohical_data.csv)
        >>> weather_df = generate_weather_data(geography_df, '2020-01-01', '2024-12-31')
        >>> weather_df.to_csv('data/raw/weather_data.csv', index=False)
    """
    generator = WeatherDataGenerator(random_state=random_state)
    return generator.generate_weather_data(geography_df, start_date, end_date)

def save_weather_data(df: pd.DataFrame, output_path: str = "data/raw") -> Path:
    """
    Saves the weather data DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame containing weather data.
        output_path (str): Directory path where the CSV file will be saved.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / "weather.csv"
    df.to_csv(file_path, index=False)
    return file_path