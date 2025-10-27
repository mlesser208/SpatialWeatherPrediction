"""
Integration Test Script

Tests that all modules work together correctly without requiring BigQuery authentication.
Validates module imports, class initialization, method calls, and data flow.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Test imports
def test_imports():
    """Test that all modules can be imported without errors."""
    print("=== Testing Module Imports ===")
    
    try:
        from modules.bigquery_connector import BigQueryConnector
        print("BigQueryConnector imported successfully")
    except Exception as e:
        print(f"BigQueryConnector import failed: {e}")
        return False
    
    try:
        from modules.sine_curve_fitting import SineCurveFitter, sin_function, fit_sine_curve_to_daily_temp_data
        print("SineCurveFitter and functions imported successfully")
    except Exception as e:
        print(f"SineCurveFitter import failed: {e}")
        return False
    
    try:
        from modules.spatial_clustering import SpatialClusteringAnalyzer
        print("SpatialClusteringAnalyzer imported successfully")
    except Exception as e:
        print(f"SpatialClusteringAnalyzer import failed: {e}")
        return False
    
    try:
        from modules.anomaly_predictor import AnomalyPredictor
        print("AnomalyPredictor imported successfully")
    except Exception as e:
        print(f"AnomalyPredictor import failed: {e}")
        return False
    
    try:
        from modules.visualizer import WeatherVisualizer
        print("WeatherVisualizer imported successfully")
    except Exception as e:
        print(f"WeatherVisualizer import failed: {e}")
        return False
    
    print()
    return True


def test_configuration():
    """Test that configuration loads correctly."""
    print("=== Testing Configuration Loading ===")
    
    try:
        import yaml
        with open("config/config.yaml", 'r') as file:
            config = yaml.safe_load(file)
        
        # Check required sections
        required_sections = ['project_id', 'spatial_clustering', 'anomaly_detection']
        for section in required_sections:
            if section not in config:
                print(f" Missing configuration section: {section}")
                return False
        
        print(" Configuration loaded successfully")
        print(f"   Project ID: {config['project_id']}")
        print(f"   Spatial clustering neighbors: {config['spatial_clustering']['neighbor_count']}")
        print(f"   Irregularity threshold: {config['spatial_clustering']['irregularity_threshold']}")
        
    except Exception as e:
        print(f" Configuration loading failed: {e}")
        return False
    
    print()
    return True


def test_class_initialization():
    """Test that all classes can be instantiated."""
    print("=== Testing Class Initialization ===")
    
    try:
        from modules.bigquery_connector import BigQueryConnector
        from modules.sine_curve_fitting import SineCurveFitter
        from modules.spatial_clustering import SpatialClusteringAnalyzer
        from modules.anomaly_predictor import AnomalyPredictor
        from modules.visualizer import WeatherVisualizer
        
        # Initialize classes
        bq_connector = BigQueryConnector()
        print(" BigQueryConnector initialized")
        
        sine_fitter = SineCurveFitter()
        print(" SineCurveFitter initialized")
        
        spatial_analyzer = SpatialClusteringAnalyzer()
        print(" SpatialClusteringAnalyzer initialized")
        
        anomaly_predictor = AnomalyPredictor()
        print(" AnomalyPredictor initialized")
        
        visualizer = WeatherVisualizer()
        print(" WeatherVisualizer initialized")
        
    except Exception as e:
        print(f" Class initialization failed: {e}")
        return False
    
    print()
    return True


def test_sine_curve_fitting():
    """Test sine curve fitting with sample data."""
    print("=== Testing Sine Curve Fitting ===")
    
    try:
        from modules.sine_curve_fitting import SineCurveFitter, sin_function, fit_sine_curve_to_daily_temp_data
        
        # Create sample temperature data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        # Create sinusoidal temperature pattern with noise
        t = np.arange(len(dates))
        temp_data = pd.DataFrame({
            'date': dates,
            'avg_temp': 60 + 20 * np.sin(2 * np.pi * t / 365.25) + np.random.normal(0, 3, len(dates))
        })
        
        # Test sine function
        test_params = {'amp': 20, 'freq': 1/365.25, 'phase_shift': 80, 'mean': 60}
        predicted = sin_function(t, test_params['amp'], test_params['freq'], 
                               test_params['phase_shift'], test_params['mean'])
        
        if len(predicted) == len(t):
            print(" sin_function works correctly")
        else:
            print(" sin_function output length mismatch")
            return False
        
        # Test curve fitting
        fit_info = fit_sine_curve_to_daily_temp_data(temp_data, 'avg_temp', 'sine curve fit info')
        
        if len(fit_info) > 0 and 'est_amp_avg_temp' in fit_info.columns:
            print(" fit_sine_curve_to_daily_temp_data works correctly")
            print(f"   Fitted amplitude: {fit_info['est_amp_avg_temp'].iloc[0]:.2f}")
            print(f"   Fitted frequency: {fit_info['est_freq_avg_temp'].iloc[0]:.6f}")
        else:
            print(" fit_sine_curve_to_daily_temp_data failed")
            return False
        
        # Test SineCurveFitter class
        fitter = SineCurveFitter()
        params = fitter.fit_curve(temp_data, 'avg_temp', 'test_station')
        
        if 'amp' in params and 'freq' in params:
            print(" SineCurveFitter.fit_curve works correctly")
        else:
            print(" SineCurveFitter.fit_curve failed")
            return False
        
    except Exception as e:
        print(f" Sine curve fitting test failed: {e}")
        return False
    
    print()
    return True


def test_spatial_clustering():
    """Test spatial clustering with sample data."""
    print("=== Testing Spatial Clustering ===")
    
    try:
        from modules.spatial_clustering import SpatialClusteringAnalyzer
        
        # Create sample stations data
        stations_data = pd.DataFrame({
            'usaf': ['722920', '722921', '722922', '722923', '722924'],
            'name': ['Catalina Airport', 'Station 1', 'Station 2', 'Station 3', 'Station 4'],
            'lat': [33.405, 33.410, 33.400, 33.415, 33.395],
            'lon': [-118.416, -118.420, -118.410, -118.425, -118.405]
        })
        
        # Initialize analyzer
        analyzer = SpatialClusteringAnalyzer()
        analyzer.stations_data = stations_data
        
        # Test distance calculation
        distances = analyzer.calculate_distances(stations_data)
        
        if distances.shape == (5, 5) and np.allclose(distances, distances.T):  # Should be symmetric
            print(" Distance calculation works correctly")
        else:
            print(" Distance calculation failed")
            return False
        
        # Test neighbor finding
        neighbors = analyzer.find_neighbors('722920', neighbor_count=3)
        
        if len(neighbors) == 3 and '722920' not in neighbors:
            print(" Neighbor finding works correctly")
            print(f"   Found neighbors: {neighbors}")
        else:
            print(" Neighbor finding failed")
            return False
        
    except Exception as e:
        print(f" Spatial clustering test failed: {e}")
        return False
    
    print()
    return True


def test_anomaly_prediction():
    """Test anomaly prediction with sample data."""
    print("=== Testing Anomaly Prediction ===")
    
    try:
        from modules.anomaly_predictor import AnomalyPredictor
        
        # Create sample deviation results
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
        deviation_data = pd.DataFrame({
            'date': dates,
            'avg_temp': 60 + 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25) + np.random.normal(0, 5, len(dates)),
            'predicted_temp': 60 + 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25),
            'deviation': np.random.normal(0, 3, len(dates))
        })
        
        deviation_results = {
            '722920': {
                'data': deviation_data,
                'stats': {
                    'mean_deviation': deviation_data['deviation'].mean(),
                    'std_deviation': deviation_data['deviation'].std(),
                    'percentile_95': deviation_data['deviation'].quantile(0.95)
                }
            }
        }
        
        # Test threshold calculation
        predictor = AnomalyPredictor()
        thresholds = predictor.calculate_thresholds(deviation_results)
        
        if '2_std_dev' in thresholds and 'percentile_95' in thresholds:
            print(" Threshold calculation works correctly")
            print(f"   2 SD threshold: {thresholds['2_std_dev']['threshold']:.2f}")
            print(f"   95th percentile threshold: {thresholds['percentile_95']['threshold']:.2f}")
        else:
            print(" Threshold calculation failed")
            return False
        
        # Test classification
        classification_results = predictor.classify_extremes(deviation_results, thresholds, '2_std_dev')
        
        if '722920' in classification_results and 'station_classification' in classification_results['722920']:
            print(" Classification works correctly")
            print(f"   Station classification: {classification_results['722920']['station_classification']}")
        else:
            print(" Classification failed")
            return False
        
    except Exception as e:
        print(f" Anomaly prediction test failed: {e}")
        return False
    
    print()
    return True


def test_visualization():
    """Test visualization with sample data."""
    print("=== Testing Visualization ===")
    
    try:
        from modules.visualizer import WeatherVisualizer
        
        # Create sample spatial results
        spatial_results = {
            'station_usaf': '722920',
            'station_name': 'Catalina Airport',
            'station_lat': 33.405,
            'station_lon': -118.416,
            'neighbor_count': 3,
            'neighbor_usafs': ['722921', '722922', '722923'],
            'target_params': {'amp': 20, 'freq': 0.0027, 'phase_shift': 80, 'mean': 60},
            'irregularity_score': 0.3,
            'is_irregular': False
        }
        
        # Test visualizer initialization
        visualizer = WeatherVisualizer()
        print(" WeatherVisualizer initialized")
        
        # Test output directory creation
        if os.path.exists(visualizer.output_dir):
            print(" Output directory exists")
        else:
            print(" Output directory not created")
            return False
        
        # Test report generation
        report_path = visualizer.generate_analysis_report(spatial_results, station_usaf='722920')
        
        if os.path.exists(report_path):
            print(" Analysis report generated successfully")
            print(f"   Report saved to: {report_path}")
        else:
            print(" Analysis report generation failed")
            return False
        
    except Exception as e:
        print(f" Visualization test failed: {e}")
        return False
    
    print()
    return True


def test_data_flow():
    """Test data flow between modules."""
    print("=== Testing Data Flow Between Modules ===")
    
    try:
        from modules.sine_curve_fitting import SineCurveFitter
        from modules.spatial_clustering import SpatialClusteringAnalyzer
        from modules.anomaly_predictor import AnomalyPredictor
        
        # Create sample data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        temp_data = pd.DataFrame({
            'date': dates,
            'avg_temp': 60 + 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25) + np.random.normal(0, 3, len(dates))
        })
        
        # Test sine curve fitting
        sine_fitter = SineCurveFitter()
        params = sine_fitter.fit_curve(temp_data, 'avg_temp', 'test_station')
        
        # Test spatial clustering
        spatial_analyzer = SpatialClusteringAnalyzer()
        spatial_analyzer.sine_fitter = sine_fitter
        
        # Test anomaly prediction
        anomaly_predictor = AnomalyPredictor()
        anomaly_predictor.sine_fitter = sine_fitter
        
        print(" Data flow between modules works correctly")
        print(f"   Sine curve parameters: {params}")
        
    except Exception as e:
        print(f" Data flow test failed: {e}")
        return False
    
    print()
    return True


def main():
    """Run all integration tests."""
    print("=== Spatial Weather Prediction Tool - Integration Tests ===")
    print("Testing module integration without BigQuery authentication")
    print()
    
    tests = [
        test_imports,
        test_configuration,
        test_class_initialization,
        test_sine_curve_fitting,
        test_spatial_clustering,
        test_anomaly_prediction,
        test_visualization,
        test_data_flow
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f" {test.__name__} failed")
        except Exception as e:
            print(f" {test.__name__} failed with exception: {e}")
    
    print("=== Integration Test Summary ===")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print(" All integration tests passed! Modules are working correctly together.")
        print("\nNext steps:")
        print("1. Set up Google Cloud authentication")
        print("2. Test with real BigQuery data")
        print("3. Create ArcGIS Python Toolbox")
        return True
    else:
        print(f" {total - passed} tests failed. Please fix the issues before proceeding.")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    else:
        print("\n Ready to proceed with ArcGIS Python Toolbox creation!")
