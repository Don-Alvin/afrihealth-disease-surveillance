"""
This module generates disease data for the Afrihealth project.

This module models:
- Three diseases: Malaria, Cholera, and Tuberculosis.
- Climate correlations for each disease e.g Malaria and Cholera spike in rainy seasons.
- Outbreak events with geographic spread
- Facility-level reporting with underreporting factors
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

def generate_disease_data(
        geography_df: pd.DataFrame,
        facility_df: pd.DataFrame,
        climate_df: pd.DataFrame,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        random_state: int = 42
) -> pd.DataFrame:
    """
    This function generates synthetic disease data for the Afrihealth project with realistic patterns.

    It models three diseases: Malaria, Cholera, and Tuberculosis, incorporating climate correlations, outbreak events, seasonality and transmission.

    Args:
        geography_df (pd.DataFrame): DataFrame containing geographic information.
        facility_df (pd.DataFrame): DataFrame containing health facility information.
        climate_df (pd.DataFrame): DataFrame containing climate data.
        start_date (str): Start date for the data generation in 'YYYY-MM-DD' format.
        end_date (str): End date for the data generation in 'YYYY-MM-DD' format.
        random_state (int): Seed for random number generator for reproducibility.
    
    Returns:
        pd.DataFrame: DataFrame containing generated disease data with columns:
            - case_id: Unique identifier for each case.
            - report_date: Date when the case was reported.
            - case_date: Date when the symptoms started.
            - facility_id: Identifier for the health facility reporting the case.
            - geography_id: Identifier for the geographic location.
            - disease: Type of disease (Malaria, Cholera, Tuberculosis).
            - new_cases: Number of new cases reported.
            - deaths: Number of deaths reported.
            - recoveries: Number of recoveries reported.
            - age_group: 0-5, 6-17, 18-49, 50-64, 65+
            - gender: M/F
    
    Example:
        >>> geo_df = pd.read_csv('geographical.csv')
        >>> fac_df = pd.read_csv('facilities.csv')
        >>> climate_df = pd.read_csv('climate.csv')
        >>> cases_df = generate_disease_data(geo_df, fac_df, climate_df)
        >>> cases.to_csv('disease.csv', index=False)
    """

    np.random.seed(random_state)

    # Date range
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Climate data preprocessing
    climate_df['date'] = pd.to_datetime(climate_df['date'])
    climate_df['temperature_avg_c'] = (climate_df['temperature_min_c'] + climate_df['temperature_max_c']) / 2

    # Merge facility and geography data
    facility_geo_df = facility_df.merge(
        geography_df[['geography_id', 'population', 'urban_rural']],
        on='geography_id', 
        how='left'
    )

    # Filter to active facilities
    active_facilities = facility_geo_df[facility_geo_df['operational_status'] == 'Operational'].copy()

    print(f"Generating disease data for {len(active_facilities)} active facilities...")

    data = []
    case_counter = 1

    # Track active outbreaks
    active_outbreaks = {
        'cholera': [],
    }

    for date in date_range:
        daily_climate = climate_df[climate_df['date'] == date]
        for _, facility in active_facilities.iterrows():
            geography_id = facility['geography_id']
            facility_id = facility['facility_id']
            population = facility['population']
            is_urban = facility['urban_rural'] == 'Urban'

            # Get climate data for the facility's geography
            location_climate = daily_climate[daily_climate['geography_id'] == geography_id]

            if len(location_climate) == 0:
                continue

            climate_row = location_climate.iloc[0]
            rainfall = climate_row['rainfall_mm']
            temperature = climate_row['temperature_avg_c']

            # Malaria
            malaria_cases = generate_malaria_cases(
                date=date,
                population=population,
                is_urban=is_urban,
                rainfall=rainfall,
                temperature=temperature,
                climate_history=climate_df[
                    (climate_df['geography_id'] == geography_id) &
                    (climate_df['date'] >= date -timedelta(days=21)) &
                    (climate_df['date'] < date)
                ]
            )

            if malaria_cases > 0:
                data.append(create_case_record(
                    case_counter, date, facility_id, geography_id, 'Malaria', malaria_cases
                ))

                case_counter += 1
            
            # Cholera
            cholera_cases = generate_cholera_cases(
                date=date,
                geography_id=geography_id,
                population=population,
                is_urban=is_urban,
                rainfall=rainfall,
                active_outbreaks=active_outbreaks['cholera']
            )

            if cholera_cases > 0:
                data.append(create_case_record(
                    case_counter, date, facility_id, geography_id, 'Cholera', cholera_cases
                ))
                case_counter += 1

                if rainfall > 80 and np.random.random() < 0.02:
                    active_outbreaks['cholera'].append({
                        'geography_id': geography_id,
                        'start_date': date,
                        'intensity': np.random.uniform(1.5, 3.0)
                    })
            
            tb_cases = generate_tb_cases(
                date=date,
                population=population,
                is_urban=is_urban
            )

            if tb_cases > 0:
                data.append(create_case_record(
                    case_counter, date, facility_id, geography_id, 'Tuberculosis', tb_cases
                ))
                case_counter += 1
        
        # Clean up old cholera oubreaks (outbreak abg is 30-60 days)
        active_outbreaks['cholera'] = [
            outbreak for outbreak in active_outbreaks['cholera']
            if (date - outbreak['start_date']).days < 60
        ]

        if date.day == 1:
            print(f"Processing {date.strftime('%Y-%m')}... ({len(data)} cases so far).")
    
    df = pd.DataFrame(data)

    # Add data quality issues to mimic a real world scenario

    df = add_data_quality_issues(df)

    print(f"Generated {len(df): ,} disease case reports")

    return df

def generate_malaria_cases(
        date: pd.Timestamp,
        population: float,
        is_urban: bool,
        rainfall: float,
        temperature: float,
        climate_history: pd.DataFrame
) -> int:
    """
    This function generates malaria cases with respect to climate
    Malaria transmission increases 2-3 weeks after rainfall
    """

    if is_urban:
        base_rate = 0.5
    else:
        base_rate = 2.0
    
    # Seasonal factor (peaks in rainy season)
    month = date.month
    if month in [3, 4, 5, 10, 11]:
        season_multiplier = 2.5
    else:
        season_multiplier = 0.5
    
    # Rainfall lagged effect (2-3 weeks)
    if len(climate_history) > 0:
        recent_rainfall = climate_history['rainfall_mm'].sum()
        rainfall_factor = 1 + (recent_rainfall / 500)
    else:
        rainfall_factor = 1.0
    
    # Temperature factor (25-30)
    if 25 <= temperature <= 30:
        temp_factor = 1.5
    elif 20 <= temperature <=35:
        temp_factor = 1.0
    else:
        temp_factor = 0.5
    
    # Calculate expected cases
    expected = (population / 100000) * base_rate * season_multiplier * rainfall_factor * temp_factor

    # Add randomness
    if expected > 0:
        cases = np.random.poisson(expected)
    else:
        cases = 0
    
    return int(cases)

def generate_cholera_cases(
    date: pd.Timestamp,
    geography_id: str,
    population: float,
    is_urban: bool,
    rainfall: float,
    active_outbreaks: List[Dict]
) -> int:
    """
    This function generates cholera cases with outbreak patterns

    Cholera is outbreak-based, not constant baseline.
    """

    # Check if there's an active outbreak in this location
    local_outbreak = None
    for outbreak in active_outbreaks:
        if outbreak['geography_id'] == geography_id:
            days_since_start = (date - outbreak['start_date']).days
            if days_since_start < 60:
                local_outbreak = outbreak
                break
    
    if local_outbreak:
        # Outbreak peak at day 15-20, then declines
        days_since_start = (date - local_outbreak['start_date']).days

        # Bell curve for outbreak intensity
        peak_day = 18
        outbreak_curve = np.exp(-0.5 * ((days_since_start - peak_day) / 10) ** 2)

        base_outbreak_rate = 5. if is_urban else 3.0
        expected = (population / 100000) * base_outbreak_rate * outbreak_curve * local_outbreak['intensity']

        cases = np.random.poisson(expected)
    else:
        # No outbreak: very low baseline
        # Random small outbreak start (rare)
        if rainfall > 50 and np.random.random() < 0.001:
            cases = np.random.poisson(2)
        else:
            cases = 0
    
    return int(cases)

def generate_tb_cases(
        date: pd.Timestamp,
        population: float,
        is_urban: bool
) -> int:
    """
    This function generates TB cases (chronic, less seasonal).

    TB is constant throughout the year, slightly higher in urban.
    """
    # Base incidence rate per 100,000 per day

    if is_urban:
        base_rate = 0.8
    else:
        base_rate = 0.5

    # Slight seasonal variation (worse in winter, but we're in tropics)
    month = date.month
    if month in [6, 7, 8]:
        season_factor = 1.2
    else:
        season_factor = 1.0

    expected = (population / 100000) * base_rate * season_factor

    if expected > 0:
        cases = np.random.poisson(expected)
    else:
        cases = 0
    
    return int(cases)


def create_case_record(
        case_id: int,
        report_date: pd.Timestamp,
        facility_id: str,
        geography_id: str,
        disease: str,
        new_cases: int
) -> Dict:
    """
    This functions creates disease case records
    """
    # Most times case date is earlier that report date
    delay_days = int(np.random.exponential(3))
    delay_days = min(delay_days, 14)
    case_date = report_date - timedelta(days=delay_days)

    # Deaths (varies by disease)
    if disease == 'Malaria':
        cfr = 0.02
    elif disease == "Cholera":
        cfr = 0.05
    elif disease == 'Tuberculosis':
        cfr = 0.01
    
    deaths = np.random.binomial(new_cases, cfr)

    # Recoveries
    recoveries = new_cases - deaths
    still_sick = int(recoveries * np.random.uniform(0.1, 0.3))
    recoveries = recoveries - still_sick

    # Age group distribution
    age_groups = ['0-5', '6-17', '18-49', '50-64', '65+']
    if disease == 'Malaria':
        age_weights = [0.4, 0.2, 0.2, 0.1, 0.1]
    elif disease == 'Cholera':
        age_weights = [0.20, 0.20, 0.20, 0.20, 0.20]
    else:
        age_weights=[0.1, 0.2, 0.5, 0.1, 0.1]
    
    age_group = np.random.choice(age_groups, p=age_weights)

    # Gender (Almost the same)
    gender = 'M' if np.random.random() < 0.47 else 'F'

    return {
        'case_id': f"CASE-{case_id:07d}",
        'report_date': report_date.strftime('%Y-%m-%d'),
        'case_date': case_date.strftime('%Y-%m-%d'),
        'facility_id': facility_id,
        'geography_id': geography_id,
        'disease': disease,
        'new_cases': new_cases,
        'deaths': deaths,
        'recoveries': recoveries,
        'age_group': age_group,
        'gender':gender
    }

def add_data_quality_issues(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function adds data quality issues to the generated data to mimic a real-world scenario
    """

    df = df.copy()

    # Missing case counts at 1% rate
    missing_idx = np.random.choice(df.index, size=int(len(df) * 0.01), replace=False)
    df.loc[missing_idx, 'new_cases'] = np.nan

    # Missing deaths at 3% rate
    missing_idx = np.random.choice(df.index, size=int(len(df) * 0.03), replace=False)
    df.loc[missing_idx, 'deaths'] = np.nan

    # Duplicate records (very rare)
    if len(df) > 100:
        dup_idx = np.random.choice(df.index[:len(df) // 2], size=5, replace=False)
        duplicates = df.loc[dup_idx].copy()
        df = pd.concat([df, duplicates], ignore_index=True)
    
    # Impossible values
    error_idx = np.random.choice(df.index, size=3, replace=False)
    # Negative number of deaths
    df.loc[error_idx[0], 'new_cases'] = -5

    return df

def save_disease_data(
        df: pd.DataFrame,
        output_path: str = "data/raw",
        compress: bool = False
) -> Path:
    """
    This functions takes the generated dataframe and saves it as csv

    Args:
        df (pd.DataFrame): DataFrame containing climate data.
        output_path (str): Directory path where the CSV file will be saved.
    """

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if compress:
        file_path = output_path / 'disease.csv.gz'
        df.to_csv(file_path, index=False, compression='gzip')
    else:
        file_path = output_path / 'diseases.csv'
        df.to_csv(file_path, index=False)
    
    print(f"Saved diseases case data to {file_path}")
    print(f"Record: {len(df)}")

    return file_path



    