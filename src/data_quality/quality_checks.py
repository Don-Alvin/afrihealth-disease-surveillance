"""
This module runs data quality checks and generates a report on all data issues
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

class DataQualityChecker:
    """
    Comprehensive data quality assesment for surveillance data.

    Checks for:
        - Missing values
        - Validity - do the values make sense
        - Consistency
        - Timeliness (delays)
        - Duplicates

    Usage:
        checker = DataQualityChecker()
        checker.load_data()
        results = checker.run_all_checks
        checker.print_report()
    """
    def __init__(self, data_dir:str = 'data/raw'):
        self.data_dir = Path(data_dir)
        self.reports = None
        self.weather = None
        self.facilities = None
        self.geo = None

        # Quality thresholds
        self.thresholds = {
            'completeness_score': 5.0,
            'duplicate_acceptable': 1.0,
            'outlier_threshold': 3.0,
            'max_reporting_delay': 14
        }

    def load_data(self):
        """Loads all datasets"""

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

        self.reports = pd.read_csv(self.data_dir/'reports.csv', dtype=reports_types)
        self.facilities = pd.read_csv(self.data_dir/'facilities.csv', dtype=facilities_types)
        self.geo = pd.read_csv(self.data_dir/'geographical.csv', dtype=geo_types)
        self.weather = pd.read_csv(self.data_dir/'weather.csv')

        # Convert dates
        self.reports['report_date'] = pd.to_datetime(self.reports['report_date'])
        self.reports['case_date'] = pd.to_datetime(self.reports['case_date'])
        self.weather['date'] = pd.to_datetime(self.weather['date'])
    
    def check_missing_values(self, df:pd.DataFrame, dataset_name: str) -> Dict:
        """Check for missing values"""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)

        issues = []
        for col in missing[missing > 0].index:
            pct = missing_pct[col]
            severity = 'HIGH' if pct > 10 else 'MEDIUM' if pct > 5 else 'LOW'
            issues.append({
                'column': col,
                'missing_count': int(missing[col]),
                'missing_pct': float(pct),
                'severity': severity
            })
        
        total_cells = len(df) * len(df.columns)
        total_missing = missing.sum()
        completeness_score = ((total_cells - total_missing) / total_cells * 100)

        return{
            'dataset': dataset_name,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'completeness_score': round(completeness_score, 2),
            'issues': issues,
            'status': 'PASS' if completeness_score >=95 else 'WARNING' if completeness_score >=90 else 'FAIL'

        }
    
    def check_validity(self) -> Dict:
        """Check for invalid values"""

        # Check if there are negative values in numerical columns. Recorded values in cases, deaths and recoveries cannot be negative.
        issues = []
        numeric_cols = ['cases', 'deaths', 'recoveries']
        for col in numeric_cols:
            if col in self.reports.columns:
                negative_count = (self.reports[col] < 0).sum()
                if negative_count > 0:
                    issues.append({
                        'check': f"Negative {col}",
                        'issue_count': int(negative_count),
                        'severity': 'HIGH',
                        'description': f"{negative_count} records have negative {col} (impossible)."
                    })

        # Check if number of recorded deaths exceed number of cases (impossible).
        death_exceed_count = (self.reports['deaths'] > self.reports['cases']).sum()
        if death_exceed_count > 0:
            issues.append({
                "check": "Deaths exceed cases",
                "issue_count": int(death_exceed_count),
                "severity": 'HIGH',
                "description": f"{death_exceed_count} records where deaths are more than cases"
            })
        
        # Check if report_date is in the future
        today = pd.Timestamp.now()
        future_reports_count = (self.reports['report_date'] > today).sum()
        if future_reports_count > 0:
            issues.append({
                'check': "Future dates",
                "issue_count": int(future_reports_count),
                "severity": "MEDIUM",
                "description": f"{future_reports_count} records where date of report is in the future."
            })
        
        # Check records where case_date > report_date (impossible)
        inconsistent_date_count = (self.reports['case_date'] > self.reports['report_date']).sum()
        if inconsistent_date_count > 0:
            issues.append({
                "check": "Case date after report date",
                "issue_count": int(inconsistent_date_count),
                "severity": "LOW",
                "description": f"{inconsistent_date_count} records where the case date is after the report date."
            })

        # Check for temperature outliers
        cols_to_check = ['temperature_avg_c', 'temperature_min_c', 'temperature_max_c']
        for col in cols_to_check:
            temp_mean = self.weather[col].mean()
            temp_std = self.weather[col].std()
            temp_z = ((self.weather[col] - temp_mean) / temp_std).round(2)
            temp_outliers_count = (abs(temp_z) > 3).sum()

            if temp_outliers_count > 0:
                issues.append({
                    "check": f"{col} outliers",
                    "issue_count": int(temp_outliers_count),
                    "severity": 'LOW',
                    "description": f"{temp_outliers_count} {col} readings are outliers."
                })
        
        
        # Check negative rainfall values
        negative_rainfall = (self.weather['rainfall_mm'] < 0).sum()
        if negative_rainfall > 0:
            issues.append({
                'check': 'Negative rainfall values',
                'issue_count': int(negative_rainfall),
                'severity': 'MEDIUM',
                'description': f"{negative_rainfall} rainfall values are negative."
            })        
        
        return {
            "total_checks": 6,
            "issues_found": len(issues),
            "issues": issues,
            "status": 'PASS' if len(issues) == 0 else 'WARNING' if len(issues) <= 2 else 'FAIL'
        }
    
    # Consistency Checks
    def check_consistency(self) -> Dict:
        """Check for consistency issues"""
        issues = []
        
        # 1 Check valid geography ids in reports dataset
        reports_geo_ids = set(self.reports['geography_id'].unique())
        valid_geo_ids = set(self.geo['geography_id'].unique())

        invalid_geo_ids = reports_geo_ids - valid_geo_ids

        if invalid_geo_ids:
            issues.append({
                "check": "Invalid geography ids in report dataset",
                "issue_count": len(invalid_geo_ids),
                "severity": "HIGH",
                "description": f"The reports dataset reference {len(invalid_geo_ids)} that do not exist in the geographical dataset."
            })

        # 2 Check geographical location with no reports
        orphaned_geos = valid_geo_ids - reports_geo_ids
        if orphaned_geos:
            issues.append({
                "check": "Geographical locations without reports",
                "issue_count": len(orphaned_geos),
                "severity": 'LOW',
                "description": f"{len(orphaned_geos)} geographical areas have no reports."
            })
        
        # 3 Check valid facility IDs in reports dataset
        reports_fac_ids = set(self.reports['facility_id'].unique())
        valid_fac_ids = set(self.facilities['facility_id'].unique())

        invalid_fac_ids = reports_fac_ids - valid_fac_ids

        if invalid_fac_ids:
            issues.append({
                "check": "Invalid facility ids in report dataset",
                "issue_count": len(invalid_fac_ids),
                "severity": "HIGH",
                "description": f"The reports dataset reference {len(invalid_fac_ids)} that do not exist in the facilities dataset."
            })
        
        # 4 Check facilities with no reports
        orphaned_facs = valid_fac_ids - reports_fac_ids
        if orphaned_facs:
            issues.append({
                "check": "Facilities without reports",
                "issue_count": len(orphaned_facs),
                "severity": 'LOW',
                "description": f"{len(orphaned_facs)} facilities have no reports."
            })
        
        # 5 Check duplicate geography ids
        geo_duplicates = self.geo['geography_id'].duplicated().sum()
        if geo_duplicates > 0:
            issues.append({
                "check": "Duplicate geography IDs",
                "issue_count": int(geo_duplicates),
                "severity": "MEDIUM",
                "description": f"{geo_duplicates} duplicate geography IDs found."
            })
        
        # 6 Duplicate facility IDs
        fac_duplicates = self.facilities['facility_id'].duplicated().sum()
        if fac_duplicates > 0:
            issues.append({
                "check": "Duplicate facility IDs",
                "issue_count": int(fac_duplicates),
                "severity": "MEDIUM",
                "description": f"{fac_duplicates} duplicate facility IDs found."
            })

        # 7 Check for logical consistency (beds but no staff)
        inconsistent_fac = ((self.facilities['capacity'] > 0) & (self.facilities['staff_count'] == 0)).sum()
        if inconsistent_fac > 0:
            issues.append({
                "check": 'Facility logical consistency',
                "issue_count": int(inconsistent_fac),
                "severity": 'LOW',
                'description': f"{inconsistent_fac} facilities have beds but no staff." 
            })
        
        return {
            "total_checks": 7,
            "issues_found": len(issues),
            "issues": issues,
            "status": 'PASS' if len(issues) == 0 else "WARNING" if len(issues) <= 2 else "FAIL"
        }

    # Timeliness
    def check_timeliness(self) -> Dict:
        """
        This function checks for reporting delays
        """
        self.reports['report_delay'] = (self.reports['report_date'] - self.reports['case_date']).dt.days

        delays = self.reports['report_delay'].dropna()

        avg_delay = delays.mean()
        median_delay = delays.median()
        max_delay = delays.max()

        # Count extreme delays (delays above 14 days)
        extreme_delays = delays > 14
        extreme_delays_count = extreme_delays.sum()

        issues = []
        if avg_delay > 7:
            issues.append({
                "check": "Severe delays",
                "value": f"{avg_delay:.1f} days",
                "severity": "MEDIUM",
                "description": "Average delay exceeds one week."
            })
        
        # Check if reports with extreme delays are more that 10% of total delays
        if extreme_delays_count > len(delays) * 0.1:
            issues.append({
                "check": "Severe delays",
                "issue_count": int(extreme_delays_count),
                "severity": "HIGH",
                "description": f"{extreme_delays_count} cases ({extreme_delays_count / len(delays) * 100}) reported more than 14 days late."
            })
        
        return {
            'average_delay_days': round(avg_delay, 0),
            "median_delay_days": round(median_delay, 0),
            "max_delay_days": int(max_delay),
            "extreme_delays": int(extreme_delays_count),
            "issues": issues,
            "status": "PASS" if len(issues) == 0 else "WARNING" if avg_delay <=7 else "FAIL"
        }
    
    # Uniqueness
    def check_uniqueness(self) -> Dict:
        """
        Checks for duplicate records
        """
        # Check duplicates in reports dataset
        duplicate_cols = ['report_date', 'facility_id', 'disease']
        duplicates_count = self.reports[duplicate_cols].duplicated().sum()
        duplicate_pct = (duplicates_count / len(self.reports) * 100).round(2)

        issues = []
        if duplicate_pct > self.thresholds['duplicate_acceptable']:
            issues.append({
                "check": "Duplicate records in reports dataset",
                "issue_count": int(duplicates_count),
                "duplicate_pct": float(duplicate_pct),
                "severity": "HIGH",
                "description": f"{duplicates_count} duplicate records found ({duplicate_pct}%)."
            })
        elif duplicates_count > 0:
            issues.append({
                "check": "Duplicate records in reports dataset",
                "issue_count": int(duplicates_count),
                "duplicate_pct": float(duplicate_pct),
                "severity": "LOW",
                "description": f"{duplicates_count} duplicate records found ({duplicate_pct}%)."
            })
        
        return {
            "duplicate_count": int(duplicates_count),
            "duplicate_pct": float(duplicate_pct),
            "issues": issues,
            "status": "PASS" if duplicates_count == 0 else "WARNING" if duplicate_pct <= self.thresholds['duplicate_acceptable'] else "FAIL"

        }
          

    def run_all_checks(self) -> Dict:
        """Run all data quality checks and compile a report"""
        self.results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "completeness": {
                "reports": self.check_missing_values(self.reports, 'reports'),
                "facilities": self.check_missing_values(self.facilities, 'facilities'),
                "geographical": self.check_missing_values(self.geo, 'geographical'),
                "weather": self.check_missing_values(self.weather, 'weather')
            },
            "validity": self.check_validity(),
            "consistency": self.check_consistency(),
            "timeliness": self.check_timeliness(),
            "uniqueness": self.check_uniqueness()
        }

        # Calculate overall scores
        scores = []
        statuses = []

        for category in ['completeness', 'validity', 'consistency', 'timeliness', 'uniqueness']:
            if category == 'completeness':
                for dataset in self.results['completeness'].values():
                    scores.append(dataset['completeness_score'])
                    statuses.append(dataset['status'])
            else:
                result = self.results[category]
                if 'status' in result:
                    statuses.append(result['status'])
        
        overall_score = np.mean(scores).round(2) if scores else 0

        #Status count
        pass_count = statuses.count('PASS')
        warning_count = statuses.count('WARNING')
        fail_count = statuses.count('FAIL')

        if overall_score < 85:
            overall_status = 'FAIL'
        elif overall_score < 90:
            overall_status = 'WARNING'
        else:
            overall_status = 'PASS'
        
        self.results['overall'] = {
            "overall_score": round(overall_score, 2),
            "overall_status": overall_status,
            'checks_passed': pass_count,
            'checks_warning': warning_count,
            'checks_failed': fail_count,
            'total_checks': len(statuses)
        }

        return self.results
        

    def print_report(self):
        """Print a summary report to the console"""
        if not self.results:
            print("No results found. Please run run_all_checks() first.")
            return
        
        print("\nDATA QUALITY REPORT")
        print(f"Generated on: {self.results['timestamp']}")
        print("="*50)

        #Overall summary
        overall = self.results['overall']
        print(f"Overall Data Quality Score: {overall['overall_score']}%")
        print(f"Overall Status: {overall['overall_status']}")
        print(f"Checks Passed: {overall['checks_passed']} / {overall['total_checks']}")
        print("="*50)

        # Completeness
        print("\nCOMPLETENESS CHECKS")
        for dataset, result in self.results['completeness'].items():
            print(f"\nDataset: {dataset}")
            print(f" - Total Rows: {result['total_rows']}")
            print(f" - Total Columns: {result['total_columns']}")
            print(f" - Completeness Score: {result['completeness_score']}%")
            print(f" - Status: {result['status']}")
            if result['issues']:
                print(" - Issues:")
                for issue in result['issues']:
                    print(f"    * Column: {issue['column']}, Missing: {issue['missing_count']} ({issue['missing_pct']}%), Severity: {issue['severity']}")
            else:
                print(" - No missing values found.")
        print("="*50)

        # Validity
        validity = self.results['validity']
        print("\nVALIDITY CHECKS")
        print(f" - Total Checks: {validity['total_checks']}")
        print(f" - Issues Found: {validity['issues_found']}")
        print(f" - Status: {validity['status']}")
        if validity['issues']:
            print(" - Issues:")
            for issue in validity['issues']:
                print(f"    * Check: {issue['check']}, Issues: {issue.get('issue_count', issue.get('issue_count', issue.get('issue_count', 'N/A')))}, Severity: {issue['severity']}")
        else:
            print(" - No validity issues found.")
        print("="*50)

        # Consistency
        consistency = self.results['consistency']
        print("\nCONSISTENCY CHECKS")
        print(f" - Total Checks: {consistency['total_checks']}")
        print(f" - Issues Found: {consistency['issues_found']}")
        print(f" - Status: {consistency['status']}")
        if consistency['issues']:
            print(" - Issues:")
            for issue in consistency['issues']:
                print(f"    * Check: {issue['check']}, Issues: {issue.get('issue_count', issue.get('issue_count', 'N/A'))}, Severity: {issue['severity']}")
        else:
            print(" - No consistency issues found.")
        print("="*50)

        # Timeliness
        timeliness = self.results['timeliness']
        print("\nTIMELINESS CHECKS")
        print(f" - Average Delay (days): {timeliness['average_delay_days']}")
        print(f" - Median Delay (days): {timeliness['median_delay_days']}")
        print(f" - Max Delay (days): {timeliness['max_delay_days']}")
        print(f" - Extreme Delays (>14 days): {timeliness['extreme_delays']}")
        print(f" - Status: {timeliness['status']}")
        if timeliness['issues']:
            print(" - Issues:")
            for issue in timeliness['issues']:
                print(f"    * Check: {issue['check']}, Value/Issues: {issue.get('value', issue.get('issue_count', 'N/A'))}, Severity: {issue['severity']}") 
        else:
            print(" - No timeliness issues found.")
        print("="*50)

        # Uniqueness
        uniqueness = self.results['uniqueness']
        
        print("\nUNIQUENESS CHECKS")
        print(f" - Duplicate Records: {uniqueness['duplicate_count']} ({uniqueness['duplicate_pct']}%)")
        print(f" - Status: {uniqueness['status']}")
        if uniqueness['issues']:
            print(" - Issues:")
            for issue in uniqueness['issues']:
                print(f"    * Check: {issue['check']}, Issues: {issue.get('issue_count', 'N/A')}, Severity: {issue['severity']}")
        else:
            print(" - No duplicate records found.")
        print("="*50)
        print("End of Report\n")
    
if __name__ == "__main__":
    checker = DataQualityChecker()
    checker.load_data()
    checker.run_all_checks()
    checker.print_report()

    import json
    output_path = Path('data/processed')
    output_path.mkdir(exist_ok=True)

    with open(output_path/'data_quality_report.json', 'w') as f:
        json.dump(checker.results, f, indent=2)
    
    print("Data quality report saved to data/processed/data_quality_report.json")

      