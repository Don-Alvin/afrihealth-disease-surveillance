"""
Data cleaning script.
This script fixes identified data quality issues:
COMPLETENESS:
- Reports data new_cases column has 10348 missing values.
- Reports data deaths column has 31046 missing values.
- Facilities data capacity column has 5 missing values.
- Geographical data population column has 1 missing value.
- Weather data temperature_min_c column has 5 missing values.
- Weather data temperature_max_c column has 5 missing values.
- Weather data rainfall_mm column has 571 missing values.
- Weather data rainfall_mm column has 571 missing values.
- Weather data humidity_pct column has 5 missing values.
- Weather data temperature_max_c colimn has one outlier value.
- Weather data rainfall_mm column has one negative value.

VALIDITY:
- Reports data has 1 negative value in new_cases column.
- Reports data has one record where deaths exceeds number of new cases.

CONSISTENCY:
 - 5 geographical locations do not have reports.
 - 65 facilities do not have reports.
 - 5 facilities have bed_capacity but zero staff.

 UNIQUENESS:
 - Reports data has 5 duplicate records.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

class DataCleaner:
    """
    Cleans disease surveillance data.
    """

    def __init__ (self, input_dir: str = 'data/raw', output_dir: str = 'data/processed'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.cleaning_log = []

    def log_action(self, action: str, details: str):
        """
        Log cleaning actions.
        """
        self.cleaning_log.append({
            'timestamp': datetime.now().isoformat(),
            "action": action,
            "details": details
        })
        print(f" {action}: {details}")
    
    def load_data(self):
        """Load raw data"""
        print("Loading raw data...")

        geo_types = {
            "urban_rural": "category"
        }

        facilities_types = {
            "facility_type": "category",
            "facility_level": "category",
            "operation_status": "category",
            "established_year": "category"
        }

        reports_types = {
        "disease": "category",
        "age_group": "category",
        "gender": "category" 
        }

        self.geo = pd.read_csv(self.input_dir / 'geographical.csv', dtype=geo_types)
        self.fac = pd.read_csv(self.input_dir / 'facilities.csv', dtype=facilities_types)
        self.weather = pd.read_csv(self.input_dir / 'weather.csv')
        self.reports = pd.read_csv(self.input_dir / 'reports.csv', dtype=reports_types)

        # Convert dates
        self.reports['report_date'] = pd.to_datetime(self.reports['report_date'], errors='coerce')
        self.reports['case_date'] = pd.to_datetime(self.reports['case_date'], errors='coerce')
        self.weather['date'] = pd.to_datetime(self.weather['date'], errors='coerce')

        # Rename capacity to bed_capacity in facilities data
        self.fac = self.fac.rename(columns={
            'capacity': 'bed_capacity'
            })
    
    def clean_geo(self):
        """Cleans geographical dataset"""
        print("Cleaning geography data...")
        initial_rows = len(self.geo)

        # Remove duplicates
        duplicates = self.geo['geography_id'].duplicated().sum()
        if duplicates > 0:
            self.geo = self.geo.drop_duplicates(subset=['geography_id'], keep='first')
            self.log_action('Remove duplicates', f'{duplicates} duplicate geographical entries.')
        
        # Handle missing population data (impute with median by urban/rural classification)
        missing_pop = self.geo['population'].isnull().sum()
        if missing_pop > 0:
            for urban_rural in self.geo['urban_rural'].unique():
                mask = (self.geo['urban_rural'] == urban_rural) & (self.geo['population'].isnull())
                median_pop = self.geo[self.geo['urban_rural'] == urban_rural]['population'].median()
                self.geo.loc[mask, 'population'] = median_pop
            self.log_action('Impute missing values', f'{missing_pop} missing population values imputed.')
        
        final_rows = len(self.geo)
        print(f'Geography cleaned: {initial_rows} -> {final_rows} rows.')

    def clean_facilities(self):
            """Clean facilities data"""
            print('Cleaning facilities data...')
            initial_rows = len(self.fac)

            # Remove duplicate facilities
            duplicates = self.fac['facility_id'].duplicated().sum()
            if duplicates > 0:
                self.fac = self.fac.drop_duplicates(subset=['facility_id'], keep='first')
                self.log_action('Remove duplicates', f'{duplicates} duplicate facilities entries.')

            # Fix logical inconsistencies
            inconsistent = (self.fac['bed_capacity'] > 0) & (self.fac['staff_count'] == 0)
            if inconsistent.sum() > 0:
                # 1 staff per 3 beds
                self.fac.loc[inconsistent, 'staff_count'] = (
                    self.fac.loc[inconsistent, 'bed_capacity'] / 3
                ).round().astype(int).clip(lower=2)
                self.log_action('Fix inconsistencies', f'{inconsistent.sum()} facilities had staff count adjusted')
            
            # Impute missing bed_capacity
            missing_beds = self.fac['bed_capacity'].isnull().sum()
            if missing_beds > 0:
                for ftype in self.fac['facility_type'].unique():
                    mask = (self.fac['facility_type'] == ftype) & (self.fac['bed_capacity'].isnull())
                    median_beds = self.fac[self.fac['facility_type'] == ftype]['bed_capacity'].median()
                    self.fac.loc[mask, 'bed_capacity'] = median_beds
                self.log_action('Impute missing values', f'{missing_beds} missing bed capacity values imputed')
            
            final_rows = len(self.fac)
            print(f"Facilities data cleaned: {initial_rows} -> {final_rows}")
    
    def clean_weather(self):
        """Clean weather data"""
        print("Cleaning weather data...")
        initial_rows = len(self.weather)

        # Remove outliers (temperature outliers)
        temp_mean = self.weather['temperature_max_c'].mean()
        temp_std = self.weather['temperature_max_c'].std()
        temp_z_score = ((self.weather['temperature_max_c'] - temp_mean) / temp_std).round()

        # Remove extreme tempratures (temp_z_score > 3)
        temp_outliers = abs(temp_z_score) > 3
        outlier_count = temp_outliers.sum()

        if outlier_count > 0:
            self.weather = self.weather[~temp_outliers]
            self.log_action('Remove outliers', f'{outlier_count} extreme temperature outlier removed.')
        
        # Fix negative rainfall
        negative_rain = (self.weather['rainfall_mm'] < 0).sum()
        if negative_rain > 0:
            self.weather.loc[self.weather['rainfall_mm'] < 0, 'rainfall_mm'] = 0
            self.log_action('Fix invalid values', f'{negative_rain} negative rainfall values.')

        # Missing temperature values
        missing_min_temp = self.weather['temperature_min_c'].isnull().sum()
        if missing_min_temp > 0:
            for geo_id in self.weather['geography_id'].unique():
                mask = (self.weather['geography_id'] == geo_id) & (self.weather['temperature_min_c'].isnull())
                median_min_temp = self.weather[self.weather['geography_id'] == geo_id]['temperature_min_c'].median()
                self.weather.loc[mask, 'temperature_min_c'] = median_min_temp
            self.log_action('Impute missing values', f'{missing_min_temp} missing min temperature values.')
        
        missing_max_temp = self.weather['temperature_max_c'].isnull().sum()
        if missing_max_temp > 0:
            for geo_id in self.weather['geography_id'].unique():
                mask = (self.weather['geography_id'] == geo_id) & (self.weather['temperature_max_c'].isnull())
                median_max_temp = self.weather[self.weather['geography_id'] == geo_id]['temperature_max_c'].median()
                self.weather.loc[mask, 'temperature_max_c'] = median_max_temp
            self.log_action('Impute missing values', f'{missing_max_temp} missing max temperature values.')
        
        # Missing rainfall values (Assume no rain)
        missing_rainfall = self.weather['rainfall_mm'].isnull().sum()
        if missing_rainfall > 0:
            self.weather['rainfall_mm'].fillna(0, inplace=True)
            self.log_action('Impute missing values', f'{missing_rainfall} missing rainfall values.')
        
        # Missing humidity percentage
        missing_humidity = self.weather['humidity_pct'].isnull().sum()
        if missing_humidity > 0:
            for geo_id in self.weather['geography_id'].unique():
                mask = (self.weather['geography_id'] == geo_id) & (self.weather['humidity_pct'].isnull())
                median_humidity = self.weather[self.weather['geography_id'] == geo_id]['humidity_pct'].median()
                self.weather.loc[mask, 'humidity_pct'] = median_humidity
            self.log_action('Impute missing values', f'{missing_humidity} missing min humidity values.')
        
        final_rows = len(self.weather)
        print(f'Weather cleaned: {initial_rows} -> {final_rows}')
    
    def clean_reports(self):
        """Cleans the reports dataset"""

        initial_rows = len(self.reports)

        # Remove records with negative new cases
        negative_cases = (self.reports['new_cases'] < 0).sum()
        if negative_cases > 0:
            self.reports = self.reports[self.reports['new_cases'] >= 0]
            self.log_action('Remove invalid records', f'{negative_cases}')
        
        # Fix deaths > cases
        death_exceed = (self.reports['deaths'] > self.reports['new_cases']).sum()
        if death_exceed > 0:
            self.reports.loc[self.reports['deaths'] > self.reports['new_cases'], 'deaths'] = \
            self.reports.loc[self.reports['deaths'] > self.reports['new_cases'], 'new_cases']
            self.log_action('Fix invalid record', f"{death_exceed} where deaths exceeds cases.")
        
        # Remove Duplicates
        duplicate_cols = ['report_date', 'facility_id', 'disease']
        duplicates = self.reports.duplicated(subset=duplicate_cols).sum()
        if duplicates > 0:
            self.reports = self.reports.drop_duplicates(subset=duplicate_cols, keep='first')
            self.log_action('Remove duplicates', f'{duplicates} duplicate case reports removed.')
        
        # Handle missing values
        # Remove records with missing cases
        missing_cases = self.reports['new_cases'].isnull().sum()
        if missing_cases > 0:
            self.reports = self.reports[self.reports['new_cases'].notna()]
            self.log_action('Remove incomplete record', f'{missing_cases} records missing cases removed.')
        
        # Impute records with missing deaths with zero. Assume deaths not reported means no deaths.
        missing_deaths = self.reports['deaths'].isnull().sum()
        if missing_deaths > 0:
            self.reports['deaths'].fillna(0, inplace=True)
            self.log_action('Impute missing death values', f'{missing_deaths} missing deaths imputed with zero.')
        
        final_rows = len(self.reports)
        print(f'Reports data cleaned: {initial_rows:,} -> {final_rows:,} rows.')

    def save_clean_data(self):
        """Saves the clean data to the processed directory"""
        self.geo.to_csv(self.output_dir / 'geographical_clean.csv', index=False)
        self.fac.to_csv(self.output_dir / 'facilities_clean.csv', index=False)
        self.weather.to_csv(self.output_dir / 'weather_clean.csv', index=False)
        self.reports.to_csv(self.output_dir / 'reports_clean.csv', index=False)
    
    def run(self):
        """Runs complete cleaning pipeline"""
        self.load_data()
        self.clean_geo()
        self.clean_facilities()
        self.clean_weather()
        self.clean_reports()
        self.save_clean_data()
    

if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.run()







