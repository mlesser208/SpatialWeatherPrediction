"""
Anomaly Prediction Module

Implements the three-step anomaly prediction process:
1. Baseline Fit (Per Cluster) - Fit sinusoidal curves for spatial clusters
2. Deviation Analysis - Compare recent data to historical baselines
3. Thresholding & Classification - Flag extreme events using statistical thresholds
4. Symbolization - Color-code stations by anomaly level
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import yaml
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from .bigquery_connector import BigQueryConnector
from .sine_curve_fitting import SineCurveFitter
from .spatial_clustering import SpatialClusteringAnalyzer


class AnomalyPredictor:
    """
    Predicts temperature anomalies using historical deviation analysis.
    Implements the three-step process: baseline fit, deviation analysis, thresholding.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize anomaly predictor with configuration."""
        self.config = self._load_config(config_path)
        self.bq_connector = BigQueryConnector(config_path)
        self.sine_fitter = SineCurveFitter()
        self.spatial_analyzer = SpatialClusteringAnalyzer(config_path)
        
        # Analysis parameters
        self.baseline_years = (2005, 2024)  # 20-year baseline period
        self.recent_year = 2024  # Most recent year for deviation analysis
        self.threshold_methods = ['2_std_dev', 'percentile_95', 'iqr_method']
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    def authenticate(self, credentials_path: Optional[str] = None) -> bool:
        """Authenticate with BigQuery."""
        return self.bq_connector.authenticate(credentials_path)
    
    def fit_cluster_baseline(self, station_usaf: str, 
                           neighbor_usafs: List[str]) -> Dict[str, Dict]:
        """
        Step 1: Fit sinusoidal curve for each cluster to establish normal pattern.
        
        Args:
            station_usaf: USAF identifier of target station
            neighbor_usafs: List of neighbor station USAFs
            
        Returns:
            Dict[str, Dict]: Baseline parameters for target station and neighbors
        """
        print(f"Step 1: Fitting baseline curves for station {station_usaf} and {len(neighbor_usafs)} neighbors...")
        
        all_stations = [station_usaf] + neighbor_usafs
        baseline_params = {}
        
        for usaf in all_stations:
            try:
                # Load temperature data for baseline period
                temp_data = self.bq_connector.get_station_daily_temperature(
                    usaf, self.baseline_years[0], self.baseline_years[1]
                )
                
                if len(temp_data) >= 500:  # Minimum data requirement
                    # Fit sine curve to baseline data
                    params = self.sine_fitter.fit_curve(temp_data, 'avg_temp', usaf)
                    baseline_params[usaf] = params
                    print(f"   Fitted baseline for station {usaf}")
                else:
                    print(f"    Insufficient data for station {usaf} ({len(temp_data)} days)")
                    
            except Exception as e:
                print(f"   Failed to fit baseline for station {usaf}: {e}")
                continue
        
        return baseline_params
    
    def analyze_deviations(self, station_usaf: str, 
                         baseline_params: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
        """
        Step 2: Compare recent year actual temperatures to fitted baseline curve.
        
        Args:
            station_usaf: USAF identifier of target station
            baseline_params: Baseline sine curve parameters for all stations
            
        Returns:
            Dict[str, pd.DataFrame]: Deviation analysis results for each station
        """
        print(f"Step 2: Analyzing deviations for station {station_usaf}...")
        
        deviation_results = {}
        
        for usaf, params in baseline_params.items():
            try:
                # Load recent year temperature data
                recent_data = self.bq_connector.get_station_daily_temperature(
                    usaf, self.recent_year, self.recent_year
                )
                
                if len(recent_data) == 0:
                    print(f"    No recent data for station {usaf}")
                    continue
                
                # Calculate predicted temperatures using baseline parameters
                recent_data = recent_data.copy()
                recent_data['data_since_start'] = (
                    recent_data['date'] - recent_data['date'].min()
                ).dt.days
                
                # Predict using baseline sine curve
                predicted_temps = self.sine_fitter.predict_temperatures(
                    recent_data['date'], params
                )
                
                recent_data['predicted_temp'] = predicted_temps
                recent_data['deviation'] = recent_data['avg_temp'] - recent_data['predicted_temp']
                
                # Calculate deviation statistics
                deviation_stats = {
                    'mean_deviation': recent_data['deviation'].mean(),
                    'std_deviation': recent_data['deviation'].std(),
                    'min_deviation': recent_data['deviation'].min(),
                    'max_deviation': recent_data['deviation'].max(),
                    'percentile_5': recent_data['deviation'].quantile(0.05),
                    'percentile_25': recent_data['deviation'].quantile(0.25),
                    'percentile_50': recent_data['deviation'].quantile(0.50),
                    'percentile_75': recent_data['deviation'].quantile(0.75),
                    'percentile_95': recent_data['deviation'].quantile(0.95)
                }
                
                deviation_results[usaf] = {
                    'data': recent_data,
                    'stats': deviation_stats
                }
                
                print(f"   Analyzed deviations for station {usaf}")
                
            except Exception as e:
                print(f"   Failed to analyze deviations for station {usaf}: {e}")
                continue
        
        return deviation_results
    
    def calculate_thresholds(self, deviation_results: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Step 3: Calculate multiple threshold options for anomaly detection.
        
        Args:
            deviation_results: Deviation analysis results for all stations
            
        Returns:
            Dict[str, Dict]: Threshold values for different methods
        """
        print("Step 3: Calculating anomaly detection thresholds...")
        
        # Collect all deviations across all stations
        all_deviations = []
        for usaf, results in deviation_results.items():
            deviations = results['data']['deviation'].dropna()
            all_deviations.extend(deviations.tolist())
        
        all_deviations = np.array(all_deviations)
        
        # Calculate thresholds using different methods
        thresholds = {}
        
        # Method 1: 2 Standard Deviations
        thresholds['2_std_dev'] = {
            'threshold': 2 * np.std(all_deviations),
            'description': '±2 standard deviations (captures ~95% of normal variation)',
            'method': 'statistical'
        }
        
        # Method 2: Percentile-based (5th/95th percentiles)
        thresholds['percentile_95'] = {
            'threshold': np.percentile(np.abs(all_deviations), 95),
            'description': 'Beyond 95th percentile of absolute deviations',
            'method': 'percentile'
        }
        
        # Method 3: Interquartile Range (IQR) method
        q1, q3 = np.percentile(all_deviations, [25, 75])
        iqr = q3 - q1
        thresholds['iqr_method'] = {
            'threshold': 1.5 * iqr,
            'description': '1.5 × Interquartile Range (IQR) method',
            'method': 'iqr'
        }
        
        # Additional statistics
        thresholds['statistics'] = {
            'mean': np.mean(all_deviations),
            'std': np.std(all_deviations),
            'min': np.min(all_deviations),
            'max': np.max(all_deviations),
            'q1': np.percentile(all_deviations, 25),
            'q3': np.percentile(all_deviations, 75),
            'iqr': iqr
        }
        
        print(f"   Calculated {len(thresholds)} threshold methods")
        for method, data in thresholds.items():
            if method != 'statistics':
                print(f"    {method}: {data['threshold']:.2f}°F ({data['description']})")
        
        return thresholds
    
    def classify_extremes(self, deviation_results: Dict[str, Dict], 
                         thresholds: Dict[str, Dict],
                         threshold_method: str = '2_std_dev') -> Dict[str, Dict]:
        """
        Step 3 (continued): Classify stations/dates as extreme events.
        
        Args:
            deviation_results: Deviation analysis results
            thresholds: Calculated threshold values
            threshold_method: Which threshold method to use
            
        Returns:
            Dict[str, Dict]: Classification results for each station
        """
        print(f"Step 3 (continued): Classifying extremes using {threshold_method} method...")
        
        threshold_value = thresholds[threshold_method]['threshold']
        classification_results = {}
        
        for usaf, results in deviation_results.items():
            data = results['data']
            stats = results['stats']
            
            # Classify each day
            data = data.copy()
            data['is_extreme'] = np.abs(data['deviation']) > threshold_value
            data['extreme_type'] = 'normal'
            data.loc[data['is_extreme'] & (data['deviation'] > 0), 'extreme_type'] = 'extreme_hot'
            data.loc[data['is_extreme'] & (data['deviation'] < 0), 'extreme_type'] = 'extreme_cold'
            
            # Calculate summary statistics
            extreme_days = data['is_extreme'].sum()
            total_days = len(data)
            extreme_percentage = (extreme_days / total_days) * 100
            
            extreme_hot_days = (data['extreme_type'] == 'extreme_hot').sum()
            extreme_cold_days = (data['extreme_type'] == 'extreme_cold').sum()
            
            # Overall station classification
            station_classification = 'normal'
            if extreme_percentage > 20:  # More than 20% extreme days
                station_classification = 'highly_irregular'
            elif extreme_percentage > 10:  # More than 10% extreme days
                station_classification = 'moderately_irregular'
            
            classification_results[usaf] = {
                'data': data,
                'extreme_days': extreme_days,
                'total_days': total_days,
                'extreme_percentage': extreme_percentage,
                'extreme_hot_days': extreme_hot_days,
                'extreme_cold_days': extreme_cold_days,
                'station_classification': station_classification,
                'threshold_used': threshold_value,
                'threshold_method': threshold_method
            }
            
            print(f"   Classified station {usaf}: {station_classification} ({extreme_percentage:.1f}% extreme days)")
        
        return classification_results
    
    def generate_symbolization(self, classification_results: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Step 4: Generate symbolization for map visualization.
        
        Args:
            classification_results: Classification results for all stations
            
        Returns:
            Dict[str, Dict]: Symbolization data for each station
        """
        print("Step 4: Generating symbolization for map visualization...")
        
        symbolization = {}
        
        for usaf, results in classification_results.items():
            station_class = results['station_classification']
            extreme_percentage = results['extreme_percentage']
            
            # Color coding
            if station_class == 'normal':
                color = 'green'
                size = 8
            elif station_class == 'moderately_irregular':
                color = 'orange'
                size = 12
            else:  # highly_irregular
                color = 'red'
                size = 16
            
            # Size based on extreme percentage
            size_multiplier = 1 + (extreme_percentage / 100)
            size = int(size * size_multiplier)
            
            symbolization[usaf] = {
                'color': color,
                'size': size,
                'classification': station_class,
                'extreme_percentage': extreme_percentage,
                'symbol': 'circle',
                'opacity': 0.8
            }
            
            print(f"   Symbolized station {usaf}: {color} circle, size {size}")
        
        return symbolization
    
    def run_full_analysis(self, station_usaf: str, 
                         threshold_method: str = '2_std_dev') -> Dict[str, Dict]:
        """
        Run the complete anomaly prediction analysis.
        
        Args:
            station_usaf: USAF identifier of target station
            threshold_method: Which threshold method to use
            
        Returns:
            Dict[str, Dict]: Complete analysis results
        """
        print(f"=== Running Full Anomaly Prediction Analysis ===")
        print(f"Target Station: {station_usaf}")
        print(f"Threshold Method: {threshold_method}")
        print()
        
        # Step 1: Find neighbors
        neighbors = self.spatial_analyzer.find_neighbors(station_usaf)
        print(f"Found {len(neighbors)} neighbors for analysis")
        
        # Step 2: Fit cluster baselines
        baseline_params = self.fit_cluster_baseline(station_usaf, neighbors)
        
        # Step 3: Analyze deviations
        deviation_results = self.analyze_deviations(station_usaf, baseline_params)
        
        # Step 4: Calculate thresholds
        thresholds = self.calculate_thresholds(deviation_results)
        
        # Step 5: Classify extremes
        classification_results = self.classify_extremes(
            deviation_results, thresholds, threshold_method
        )
        
        # Step 6: Generate symbolization
        symbolization = self.generate_symbolization(classification_results)
        
        # Compile results
        results = {
            'target_station': station_usaf,
            'neighbors': neighbors,
            'baseline_params': baseline_params,
            'deviation_results': deviation_results,
            'thresholds': thresholds,
            'classification_results': classification_results,
            'symbolization': symbolization,
            'analysis_summary': self._generate_summary(classification_results)
        }
        
        print("\n=== Analysis Complete ===")
        self._print_summary(results)
        
        return results
    
    def _generate_summary(self, classification_results: Dict[str, Dict]) -> Dict:
        """Generate analysis summary statistics."""
        total_stations = len(classification_results)
        normal_stations = sum(1 for r in classification_results.values() 
                            if r['station_classification'] == 'normal')
        irregular_stations = total_stations - normal_stations
        
        avg_extreme_percentage = np.mean([
            r['extreme_percentage'] for r in classification_results.values()
        ])
        
        return {
            'total_stations_analyzed': total_stations,
            'normal_stations': normal_stations,
            'irregular_stations': irregular_stations,
            'irregularity_rate': (irregular_stations / total_stations) * 100,
            'average_extreme_percentage': avg_extreme_percentage
        }
    
    def _print_summary(self, results: Dict[str, Dict]):
        """Print analysis summary."""
        summary = results['analysis_summary']
        print(f"Stations Analyzed: {summary['total_stations_analyzed']}")
        print(f"Normal Stations: {summary['normal_stations']}")
        print(f"Irregular Stations: {summary['irregular_stations']}")
        print(f"Irregularity Rate: {summary['irregularity_rate']:.1f}%")
        print(f"Average Extreme Days: {summary['average_extreme_percentage']:.1f}%")
        
        # Show target station results
        target_station = results['target_station']
        if target_station in results['classification_results']:
            target_results = results['classification_results'][target_station]
            print(f"\nTarget Station ({target_station}) Results:")
            print(f"  Classification: {target_results['station_classification']}")
            print(f"  Extreme Days: {target_results['extreme_days']}/{target_results['total_days']} ({target_results['extreme_percentage']:.1f}%)")
            print(f"  Extreme Hot Days: {target_results['extreme_hot_days']}")
            print(f"  Extreme Cold Days: {target_results['extreme_cold_days']}")
    
    def get_station_anomaly_data(self, station_usaf: str) -> Optional[pd.DataFrame]:
        """
        Get anomaly data for a specific station for visualization.
        
        Args:
            station_usaf: USAF identifier of target station
            
        Returns:
            pd.DataFrame: Anomaly data with classifications
        """
        if station_usaf in self.classification_results:
            return self.classification_results[station_usaf]['data']
        return None
