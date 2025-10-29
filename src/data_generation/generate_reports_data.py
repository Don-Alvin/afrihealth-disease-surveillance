"""
This module generates disease data for the Afrihealth project.

This module models:
- Three diseases: Malaria, Cholera, and Tuberculosis.
- weather correlations for each disease e.g Malaria and Cholera spike in rainy seasons.
- Outbreak events with geographic spread
- Facility-level reporting with underreporting factors
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logger = logging.getLogger(__name__)

class DiseaseDataGenerator:
    """
    A class to generate synthetic disease surveillance data for African countries.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the disease data generator.
        
        Args:
            random_state (int): Seed for random number generator for reproducibility.
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Enhanced disease configurations with country-specific adjustments
        self.diseases = {
            'Malaria': {
                # Base rates adjusted by country (cases per 100,000 per day)
                'base_rates': {
                    'KE': {'urban': 0.8, 'rural': 3.0},    # High burden
                    'NG': {'urban': 1.2, 'rural': 4.0},    # Very high burden (highest in Africa)
                    'ZA': {'urban': 0.3, 'rural': 1.5}     # Lower burden, mostly northern areas
                },
                'case_fatality_rate': 0.02,
                'peak_months': {
                    'KE': [3, 4, 5, 10, 11],  # Long and short rains
                    'NG': [4, 5, 6, 7, 8, 9],  # Single long rainy season
                    'ZA': [10, 11, 12, 1, 2, 3]  # Summer rains
                },
                'optimal_temp_range': (25, 30),
                'reporting_delay_mean': 2,
                'age_weights': [0.4, 0.2, 0.2, 0.1, 0.1]
            },
            'Cholera': {
                'base_rates': {
                    'KE': {'urban': 0.05, 'rural': 0.08},
                    'NG': {'urban': 0.15, 'rural': 0.20},  # Higher in Nigeria due to population density
                    'ZA': {'urban': 0.02, 'rural': 0.03}   # Lower in South Africa
                },
                'case_fatality_rate': 0.05,
                'outbreak_probability': 0.001,
                'rainfall_threshold': 50,
                'reporting_delay_mean': 1,
                'age_weights': [0.2, 0.2, 0.2, 0.2, 0.2]
            },
            'Tuberculosis': {
                'base_rates': {
                    'KE': {'urban': 1.2, 'rural': 0.8},
                    'NG': {'urban': 1.5, 'rural': 1.0},    # Higher burden
                    'ZA': {'urban': 2.0, 'rural': 1.2}     # Very high burden, especially with HIV
                },
                'case_fatality_rate': 0.01,
                'season_factor_winter': 1.2,
                'reporting_delay_mean': 7,
                'age_weights': [0.1, 0.2, 0.5, 0.1, 0.1]
            }
        }
        
        self.age_groups = ['0-5', '6-17', '18-49', '50-64', '65+']
        
        # High malaria risk regions due to water bodies and swampiness
        self.high_malaria_regions = {
            'KE': ['Nyanza', 'Western', 'Coast', 'Eastern'],
            'NG': ['Rivers', 'Lagos', 'Bayelsa', 'Delta', 'Akwa Ibom', 'Cross River'],
            'ZA': ['Limpopo', 'Mpumalanga', 'KwaZulu-Natal']
        }

    def _merge_facility_geography_data(self, facility_df: pd.DataFrame, geography_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge facility and geography data for reporting calculations.
        
        Args:
            facility_df (pd.DataFrame): Facility data
            geography_df (pd.DataFrame): Geography data
            
        Returns:
            pd.DataFrame: Merged facility-geography data
        """
        # Select only essential columns needed for case generation
        geo_cols = ['geography_id', 'population', 'urban_rural']
        
        facility_geo_df = facility_df.merge(
            geography_df[geo_cols],
            on='geography_id', 
            how='left'
        )
        
        # Filter to active facilities only
        active_facilities = facility_geo_df[
            facility_geo_df['operational_status'] == 'Operational'
        ].copy()
        
        # Add facility capacity factor for reporting completeness
        active_facilities['reporting_factor'] = self._calculate_reporting_factor(active_facilities)
        
        return active_facilities

    def _calculate_reporting_factor(self, facilities_df: pd.DataFrame) -> pd.Series:
        """
        Calculate reporting completeness factor based on facility characteristics.
        
        Args:
            facilities_df (pd.DataFrame): Facilities data
            
        Returns:
            pd.Series: Reporting factors (0-1)
        """
        factors = []
        
        for _, facility in facilities_df.iterrows():
            factor = 1.0
            
            # Higher level facilities report more completely
            level = facility.get('facility_level', 1)
            if level >= 4:  # Hospitals
                factor *= 0.95
            elif level == 3:  # Health Centers
                factor *= 0.85
            else:  # Clinics and dispensaries
                factor *= 0.70
            
            # Urban facilities report better
            if facility['urban_rural'] == 'Urban':
                factor *= 1.1
            
            # Facilities with labs report better
            if facility.get('has_lab', False):
                factor *= 1.05
            
            # Limit factor between 0.5 and 1.0
            factor = max(0.5, min(1.0, factor))
            
            factors.append(factor)
        
        return pd.Series(factors, index=facilities_df.index)

    def _is_high_malaria_region(self, country_code: str, region_name: str) -> bool:
        """
        Check if the region is known for high malaria transmission due to water bodies.
        
        Args:
            country_code (str): Country code
            region_name (str): Region name
            
        Returns:
            bool: True if high malaria risk region
        """
        if country_code in self.high_malaria_regions:
            return region_name in self.high_malaria_regions[country_code]
        return False

    def generate_reports_data(
        self,
        geography_df: pd.DataFrame,
        facility_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31"
    ) -> pd.DataFrame:
        """
        Generate synthetic disease surveillance data.
        
        Args:
            geography_df (pd.DataFrame): Geographic information
            facility_df (pd.DataFrame): Health facility information
            weather_df (pd.DataFrame): Weather data
            start_date (str): Start date for data generation
            end_date (str): End date for data generation
            
        Returns:
            pd.DataFrame: Generated disease surveillance data (facts table)
        """
        # Prepare data
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        active_facilities = self._merge_facility_geography_data(facility_df, geography_df)
        
        logger.info(f"Generating report data for {len(active_facilities)} active facilities "
                   f"over {len(date_range)} days...")

        data = []
        report_counter = 1
        
        # Track active outbreaks
        active_outbreaks = {
            'cholera': [],
            'malaria': []
        }

        # Pre-calculate monthly factors for efficiency
        monthly_factors = self._calculate_monthly_factors()

        for date in date_range:
            # Get weather data for this date
            daily_weather = weather_df[weather_df['date'] == date]
            
            if daily_weather.empty:
                continue
            
            for _, facility in active_facilities.iterrows():
                geography_id = facility['geography_id']
                facility_id = facility['facility_id']
                population = facility['population']
                is_urban = facility['urban_rural'] == 'Urban'
                reporting_factor = facility['reporting_factor']
                country_code = facility['country_code']
                region_name = facility['region_name']

                # Get weather data for this location
                location_weather = daily_weather[daily_weather['geography_id'] == geography_id]
                if location_weather.empty:
                    continue
                
                weather_row = location_weather.iloc[0]
                rainfall = weather_row['rainfall_mm']
                temperature = weather_row['temperature_avg_c']
                humidity = weather_row.get('humidity_pct', 70)

                # Generate cases for each disease
                diseases_to_generate = ['Malaria', 'Cholera', 'Tuberculosis']
                
                for disease in diseases_to_generate:
                    if disease == 'Malaria':
                        cases = self._generate_malaria_cases(
                            date, geography_id, population, is_urban, 
                            rainfall, temperature, humidity, weather_df,
                            active_outbreaks['malaria'], monthly_factors,
                            country_code, region_name
                        )
                    elif disease == 'Cholera':
                        cases = self._generate_cholera_cases(
                            date, geography_id, population, is_urban,
                            rainfall, temperature, active_outbreaks['cholera'],
                            monthly_factors, country_code
                        )
                    elif disease == 'Tuberculosis':
                        cases = self._generate_tb_cases(
                            date, population, is_urban, monthly_factors, country_code
                        )
                    
                    # Apply reporting factor (underreporting)
                    reported_cases = int(cases * reporting_factor) if cases > 0 else 0
                    
                    if reported_cases > 0:
                        case_record = self._create_case_record(
                            report_counter, date, facility_id, geography_id, 
                            disease, reported_cases
                        )
                        data.append(case_record)
                        report_counter += 1
                
                # Check for new outbreaks
                self._check_new_outbreaks(
                    date, geography_id, rainfall, temperature, active_outbreaks,
                    country_code, region_name
                )
            
            # Clean up old outbreaks
            self._cleanup_old_outbreaks(date, active_outbreaks)
            
            # Progress logging
            if date.day == 1 and date.month in [1, 4, 7, 10]:  # Quarterly logging
                logger.info(f"Processed {date.strftime('%Y-%m')}... ({len(data)} reports so far)")
        
        df = pd.DataFrame(data)
        
        # Introduce realistic data quality issues
        df = self._introduce_data_quality_issues(df)
        
        self._log_generation_stats(df)
        return df

    def _generate_malaria_cases(
        self,
        date: pd.Timestamp,
        geography_id: str,
        population: float,
        is_urban: bool,
        rainfall: float,
        temperature: float,
        humidity: float,
        weather_df: pd.DataFrame,
        active_outbreaks: List[Dict],
        monthly_factors: Dict,
        country_code: str,
        region_name: str
    ) -> int:
        """Generate malaria cases with country-specific patterns."""
        disease_config = self.diseases['Malaria']
        
        # Get country-specific base rates
        country_rates = disease_config['base_rates'][country_code]
        base_rate = country_rates['urban'] if is_urban else country_rates['rural']
        
        # HIGH MALARIA REGION BOOST
        high_malaria_boost = 1.0
        if self._is_high_malaria_region(country_code, region_name):
            if country_code == 'KE':
                if region_name in ['Nyanza', 'Western']:
                    high_malaria_boost = 3.5  # Lake Victoria regions
                elif region_name == 'Coast':
                    high_malaria_boost = 2.5  # Coastal areas
                else:
                    high_malaria_boost = 2.0
            elif country_code == 'NG':
                # Niger Delta has very high malaria transmission
                high_malaria_boost = 4.0 if region_name in ['Rivers', 'Bayelsa'] else 3.0
            elif country_code == 'ZA':
                # Lowveld and northern areas have higher malaria
                high_malaria_boost = 2.5
        
        # Country-specific seasonal patterns
        month = date.month
        peak_months = disease_config['peak_months'][country_code]
        season_multiplier = 2.5 if month in peak_months else 0.5
        
        # Rainfall lagged effect (2-3 weeks)
        three_weeks_ago = date - timedelta(days=21)
        recent_weather = weather_df[
            (weather_df['geography_id'] == geography_id) &
            (weather_df['date'] >= three_weeks_ago) &
            (weather_df['date'] < date)
        ]
        
        if not recent_weather.empty:
            recent_rainfall = recent_weather['rainfall_mm'].sum()
            rainfall_factor = 1 + (recent_rainfall / 500)
        else:
            rainfall_factor = 1.0
        
        # Temperature factor
        optimal_temp = disease_config['optimal_temp_range']
        if optimal_temp[0] <= temperature <= optimal_temp[1]:
            temp_factor = 1.5
        elif (optimal_temp[0] - 5) <= temperature <= (optimal_temp[1] + 5):
            temp_factor = 1.0
        else:
            temp_factor = 0.5
        
        # Humidity factor - mosquitoes thrive in high humidity
        humidity_factor = 1 + max(0, (humidity - 60) / 50)  # Boost when humidity > 60%
        
        # Calculate expected cases with regional risk factor
        expected = (population / 100000) * base_rate * season_multiplier 
        expected *= rainfall_factor * temp_factor * humidity_factor * high_malaria_boost
        
        # Check for outbreak conditions
        outbreak_multiplier = 1.0
        for outbreak in active_outbreaks:
            if outbreak['geography_id'] == geography_id:
                days_since_start = (date - outbreak['start_date']).days
                if days_since_start < 90:
                    outbreak_curve = np.exp(-0.5 * ((days_since_start - 30) / 20) ** 2)
                    outbreak_multiplier = max(outbreak_multiplier, outbreak['intensity'] * outbreak_curve)
        
        expected *= outbreak_multiplier
        
        # Ensure expected value is valid
        expected = max(0, expected)
        
        # Generate cases
        cases = np.random.poisson(expected) if expected > 0 else 0
        return int(cases)

    def _generate_cholera_cases(
        self,
        date: pd.Timestamp,
        geography_id: str,
        population: float,
        is_urban: bool,
        rainfall: float,
        temperature: float,
        active_outbreaks: List[Dict],
        monthly_factors: Dict,
        country_code: str
    ) -> int:
        """Generate cholera cases with outbreak patterns."""
        disease_config = self.diseases['Cholera']
        
        # Get country-specific base rates
        country_rates = disease_config['base_rates'][country_code]
        base_rate = country_rates['urban'] if is_urban else country_rates['rural']
        
        # Check for active outbreak
        local_outbreak = None
        for outbreak in active_outbreaks:
            if outbreak['geography_id'] == geography_id:
                days_since_start = (date - outbreak['start_date']).days
                if days_since_start < 60:
                    local_outbreak = outbreak
                    break
        
        if local_outbreak:
            # Outbreak in progress
            days_since_start = (date - local_outbreak['start_date']).days
            peak_day = 18
            outbreak_curve = np.exp(-0.5 * ((days_since_start - peak_day) / 10) ** 2)
            
            outbreak_base_rate = 5.0 if is_urban else 3.0
            expected = (population / 100000) * outbreak_base_rate * outbreak_curve * local_outbreak['intensity']
            
            # Ensure expected value is valid
            expected = max(0, expected)  # Prevent negative values
            cases = np.random.poisson(expected) if expected > 0 else 0
        else:
            # No outbreak - very low baseline
            expected = (population / 100000) * base_rate
            
            # Ensure expected value is valid
            expected = max(0, expected)  # Prevent negative values
            
            # Small chance of random cases, higher with heavy rainfall
            if rainfall > disease_config['rainfall_threshold'] and np.random.random() < 0.001:
                cases = np.random.poisson(max(1, expected * 10))
            else:
                cases = np.random.poisson(expected) if expected > 0 else 0
        
        return int(cases)

    def _generate_tb_cases(
        self,
        date: pd.Timestamp,
        population: float,
        is_urban: bool,
        monthly_factors: Dict,
        country_code: str
    ) -> int:
        """Generate tuberculosis cases (chronic, less seasonal)."""
        disease_config = self.diseases['Tuberculosis']
        
        # Get country-specific base rates
        country_rates = disease_config['base_rates'][country_code]
        base_rate = country_rates['urban'] if is_urban else country_rates['rural']
        
        # Mild seasonal variation
        month = date.month
        if month in [6, 7, 8]:
            season_factor = disease_config['season_factor_winter']
        else:
            season_factor = 1.0
        
        expected = (population / 100000) * base_rate * season_factor
        
        # Ensure expected value is valid
        expected = max(0, expected)
        
        cases = np.random.poisson(expected) if expected > 0 else 0
        return int(cases)

    def _create_case_record(
        self,
        report_id: int,
        report_date: pd.Timestamp,
        facility_id: str,
        geography_id: str,
        disease: str,
        cases: int
    ) -> Dict:
        """
        Create a disease case record - facts table with only essential dimensions.
        """
        disease_config = self.diseases[disease]
        
        # Reporting delay (case date is before report date)
        delay_days = int(np.random.exponential(disease_config['reporting_delay_mean']))
        delay_days = min(delay_days, 30)
        case_date = report_date - timedelta(days=delay_days)
        
        # Deaths based on case fatality rate
        deaths = np.random.binomial(cases, disease_config['case_fatality_rate'])
        
        # Recoveries (some cases still sick)
        recoveries = cases - deaths
        still_sick = int(recoveries * np.random.uniform(0.1, 0.3))
        recoveries = max(0, recoveries - still_sick)
        
        # Age group distribution
        age_weights = disease_config['age_weights']
        age_group = np.random.choice(self.age_groups, p=age_weights)
        
        # Gender distribution
        gender = 'M' if np.random.random() < 0.47 else 'F'
        
        return {
            'report_id': f"REP-{report_id:07d}",
            'report_date': report_date.strftime('%Y-%m-%d'),
            'case_date': case_date.strftime('%Y-%m-%d'),
            'facility_id': facility_id,
            'geography_id': geography_id,
            'disease': disease,
            'cases': cases,
            'deaths': deaths,
            'recoveries': recoveries,
            'age_group': age_group,
            'gender': gender
        }

    def _calculate_monthly_factors(self) -> Dict[int, float]:
        """Calculate monthly adjustment factors for disease incidence."""
        return {
            1: 1.0, 2: 1.1, 3: 1.3, 4: 1.5, 5: 1.4, 6: 1.2,
            7: 1.0, 8: 0.9, 9: 1.0, 10: 1.3, 11: 1.4, 12: 1.2
        }

    def _check_new_outbreaks(
        self,
        date: pd.Timestamp,
        geography_id: str,
        rainfall: float,
        temperature: float,
        active_outbreaks: Dict[str, List],
        country_code: str,
        region_name: str
    ):
        """Check conditions for new disease outbreaks."""
        # Cholera outbreaks
        if (rainfall > 80 and temperature > 20 and 
            np.random.random() < 0.02 and len(active_outbreaks['cholera']) < 10):
            active_outbreaks['cholera'].append({
                'geography_id': geography_id,
                'start_date': date,
                'intensity': np.random.uniform(1.5, 3.0)
            })
        
        # Malaria outbreaks - higher probability in high malaria regions
        malaria_outbreak_prob = 0.01
        if self._is_high_malaria_region(country_code, region_name):
            malaria_outbreak_prob = 0.02  # Double the probability in high risk areas
        
        if (rainfall > 100 and temperature > 25 and 
            np.random.random() < malaria_outbreak_prob and len(active_outbreaks['malaria']) < 5):
            active_outbreaks['malaria'].append({
                'geography_id': geography_id,
                'start_date': date,
                'intensity': np.random.uniform(2.0, 4.0)
            })

    def _cleanup_old_outbreaks(self, date: pd.Timestamp, active_outbreaks: Dict[str, List]):
        """Remove outbreaks that have ended."""
        for disease, outbreaks in active_outbreaks.items():
            max_days = 90 if disease == 'malaria' else 60
            active_outbreaks[disease] = [
                outbreak for outbreak in outbreaks
                if (date - outbreak['start_date']).days < max_days
            ]

    def _introduce_data_quality_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Introduce realistic data quality issues."""
        df = df.copy()
        
        # Missing case counts (1%)
        missing_idx = np.random.choice(df.index, size=int(len(df) * 0.01), replace=False)
        df.loc[missing_idx, 'cases'] = np.nan
        
        # Missing deaths (3%)
        missing_idx = np.random.choice(df.index, size=int(len(df) * 0.03), replace=False)
        df.loc[missing_idx, 'deaths'] = np.nan
        
        # Missing age/gender (2%)
        missing_idx = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
        df.loc[missing_idx, ['age_group', 'gender']] = 'Unknown'
        
        # Duplicate records (0.5%)
        if len(df) > 100:
            dup_idx = np.random.choice(df.index, size=int(len(df) * 0.005), replace=False)
            duplicates = df.loc[dup_idx].copy()
            df = pd.concat([df, duplicates], ignore_index=True)
        
        # Impossible values (very rare)
        if len(df) > 50:
            error_idx = np.random.choice(df.index, size=min(3, len(df)//100), replace=False)
            for idx in error_idx:
                if np.random.random() < 0.5:
                    df.loc[idx, 'cases'] = -abs(df.loc[idx, 'cases'])
                else:
                    df.loc[idx, 'deaths'] = df.loc[idx, 'cases'] + 1
        
        return df

    def _log_generation_stats(self, df: pd.DataFrame):
        """Log statistics about the generated data."""
        logger.info(f"Generated {len(df):,} disease case reports")
        logger.info(f"Disease distribution: {df['disease'].value_counts().to_dict()}")
        logger.info(f"Total cases: {df['cases'].sum():,}")
        logger.info(f"Total deaths: {df['deaths'].sum():,}")
        logger.info(f"Date range: {df['report_date'].min()} to {df['report_date'].max()}")

def generate_reports_data(
    geography_df: pd.DataFrame,
    facility_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic disease surveillance data with realistic patterns.

    Args:
        geography_df (pd.DataFrame): DataFrame containing geographic information.
        facility_df (pd.DataFrame): DataFrame containing health facility information.
        weather_df (pd.DataFrame): DataFrame containing weather data.
        start_date (str): Start date for the data generation in 'YYYY-MM-DD' format.
        end_date (str): End date for the data generation in 'YYYY-MM-DD' format.
        random_state (int): Seed for random number generator for reproducibility.
    
    Returns:
        pd.DataFrame: DataFrame containing generated disease data.
    """
    generator = DiseaseDataGenerator(random_state=random_state)
    return generator.generate_reports_data(
        geography_df, facility_df, weather_df, start_date, end_date
    )

def save_reports_data(
    df: pd.DataFrame,
    output_path: str = "data/raw",
    compress: bool = False
) -> Path:
    """
    Save the generated reports data to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame containing reports data.
        output_path (str): Directory path where the CSV file will be saved.
        compress (bool): Whether to compress the output file.
    
    Returns:
        Path: Path to the saved CSV file.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if compress:
        file_path = output_path / 'reports.csv.gz'
        df.to_csv(file_path, index=False, compression='gzip')
    else:
        file_path = output_path / 'reports.csv'
        df.to_csv(file_path, index=False)

    logger.info(f"Reports data saved to {file_path}")
    logger.info(f"File size: {file_path.stat().st_size / 1024:.2f} KB")
    logger.info(f"Records saved: {len(df)}")
    
    return file_path