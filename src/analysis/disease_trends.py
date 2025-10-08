"""
This module creates reusable functions for analysing disease surveillance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import matplotlib.pyplot as plt

class DiseaseTrendsAnalyser:
    """
    Analyses disease surveillance trends

    Usage:
        analyser = DiseaseTrendsAnalyser()
        analyser.load_data()
        insights = analyser.generate_insights()
    """

    def __init__(self, data_dir: str = 'data/raw'):
        self.data_dir = Path(data_dir)
        self.geo = None
        self.reports = None
        self.weather = None

    def load_data(self):
        """
        This functions loads data from different csvs
        """

        geo_types = {
            "urban_rural": "category"
        }

        reports_types = {
        "disease": "category",
        "age_group": "category",
        "gender": "category" 
        }

        self.geo = pd.read_csv(self.data_dir/'geographical.csv', dtype=geo_types)
        self.weather = pd.read_csv(self.data_dir/'weather.csv')
        self.reports = pd.read_csv(self.data_dir/'reports.csv', dtype=reports_types)

        self.reports['report_date'] = pd.to_datetime(self.reports['report_date'])
        self.reports['case_date'] = pd.to_datetime(self.reports['case_date'])
        self.weather['date'] = pd.to_datetime(self.weather['date'])

        print(f"Loaded {len(self.reports)} case reports")

    def get_overall_statistics(self):
        stats = {
            'total_reports': len(self.reports),
            'total_cases': self.reports['new_cases'].sum(),
            'total_deaths': self.reports['deaths'].sum(),
            'fatality_rate': ((self.reports['deaths'].sum() / self.reports['new_cases'].sum()) * 100).round(2),
            'date_range': (
                self.reports['report_date'].min().strftime('%Y-%m-%d'),
                self.reports['report_date'].max().strftime('%Y-%m-%d'),
                ),
            'num_facilities': self.reports['facility_id'].nunique()
        }

        return stats
    
    def get_disease_breakdown(self) -> pd.DataFrame:
        """
        Get statistics for diseases
        """
        breakdown = self.reports.groupby('disease').agg({
            'new_cases': 'sum',
            'deaths':  'sum'
        }).round(0)
        breakdown['fatality_rate_pct'] = (breakdown['deaths'] / breakdown['new_cases'] * 100).round(2)
        return breakdown.sort_values('new_cases', ascending=False)
    
    def get_seasonal_patterns(self, disease:str = 'Malaria') -> pd.DataFrame:
        """
        Analyse seasonal patterns for Malaria
        """
        disease_cases = self.reports[self.reports['disease'] == disease].copy()
        disease_cases['month'] = disease_cases['report_date'].dt.month

        monthly = disease_cases.groupby('month')['new_cases'].sum()
        return monthly
    
    def identify_hotspots(self, top_n: int = 10) -> pd.DataFrame:
        """
        Identify high burden area
        """
        reports_geo = self.reports.merge(
            self.geo[['geography_id', 'country_code', 'region_name']],
            on='geography_id'
        )
        hotspots = reports_geo.groupby(['country_code', 'region_name'])['new_cases'].sum().reset_index()
        hotspots = hotspots.sort_values('new_cases', ascending=False)
        return hotspots.head(top_n)
    
    def calculate_weather_correlation(self) -> float:
        """
        This function calculate the correlation between rainy seasons and malaria cases
        """
        self.weather['month'] = self.weather['date'].dt.month
        monthly_rainfall = self.weather.groupby('month')['rainfall_mm'].sum().reset_index()

        self.reports['month'] = self.reports['report_date'].dt.month
        monthly_malaria = self.reports.groupby('month')['new_cases'].sum().reset_index()

        correlation = monthly_rainfall['rainfall_mm'].corr(monthly_malaria['new_cases'])
        return correlation
    
    def generate_insights(self) -> Dict:
        insights = {}

        insights['overall_stats'] = self.get_overall_statistics()
        insights['disease_breakdown'] = self.get_disease_breakdown()
        insights['seasonal_malaria'] = self.get_seasonal_patterns()
        insights['hotspots'] = self.identify_hotspots()
        insights['weather_malaria_corr'] = self.calculate_weather_correlation()

        return insights
    
    def print_insights(self):
        insights = self.generate_insights()

        print("="*70)
        print("DISEASE SURVEILLANCE INSIGHTS")
        print("="*70)
        
        # Overall
        stats = insights['overall_stats']
        print("\n Overall Statistics:")
        print(f"  Total Cases: {stats['total_cases']:,.0f}")
        print(f"  Total Deaths: {stats['total_deaths']:,.0f}")
        print(f"  CFR: {stats['fatality_rate']:.2f}%")
        
        # By disease
        print("\n Disease Breakdown:")
        print(insights['disease_breakdown'])
        
        # Hotspots
        print("\n Top Hotspots:")
        print(insights['hotspots'].head())
        
        # Climate
        print(f"\n  Rainfall-Malaria Correlation: {insights['weather_malaria_corr']:.3f}")


if __name__ == "__main__":
    analyser = DiseaseTrendsAnalyser()
    analyser.load_data()
    analyser.print_insights()





