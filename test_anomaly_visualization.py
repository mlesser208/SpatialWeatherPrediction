"""
Test Anomaly Prediction and Visualization Modules

Tests the anomaly prediction and visualization functionality with real BigQuery data
using Catalina Airport as the test case.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from modules.spatial_clustering import SpatialClusteringAnalyzer
from modules.anomaly_predictor import AnomalyPredictor
from modules.visualizer import WeatherVisualizer


def test_anomaly_prediction_and_visualization():
    """Test anomaly prediction and visualization with Catalina Airport."""
    
    print("=== Testing Anomaly Prediction and Visualization Modules ===")
    print("Using Catalina Airport (USAF: 722920) as test case")
    print()
    
    # Step 1: Spatial Analysis
    print("Step 1: Running spatial clustering analysis...")
    spatial_analyzer = SpatialClusteringAnalyzer()
    
    if not spatial_analyzer.authenticate():
        print("Failed to authenticate. Please check your credentials.")
        return False
    
    # Load stations data first
    print("Loading weather stations data...")
    spatial_analyzer.load_stations_data()
    
    spatial_results = spatial_analyzer.analyze_station_irregularity("722920")
    
    print(f"Station: {spatial_results['station_name']}")
    print(f"Irregularity Score: {spatial_results['irregularity_score']:.2f}")
    print(f"Is Irregular: {spatial_results['is_irregular']}")
    print()
    
    # Step 2: Anomaly Prediction
    print("Step 2: Running anomaly prediction analysis...")
    anomaly_predictor = AnomalyPredictor()
    anomaly_predictor.authenticate()
    
    # Authenticate and load stations data for anomaly predictor's spatial analyzer
    if not anomaly_predictor.spatial_analyzer.bq_connector.client:
        anomaly_predictor.spatial_analyzer.authenticate()
    if anomaly_predictor.spatial_analyzer.stations_data is None:
        anomaly_predictor.spatial_analyzer.load_stations_data()
    
    anomaly_results = anomaly_predictor.run_full_analysis("722920", threshold_method='2_std_dev')
    
    print("Anomaly prediction completed!")
    print(f"Analysis Summary:")
    summary = anomaly_results['analysis_summary']
    print(f"  Total stations analyzed: {summary['total_stations_analyzed']}")
    print(f"  Normal stations: {summary['normal_stations']}")
    print(f"  Irregular stations: {summary['irregular_stations']}")
    print(f"  Irregularity rate: {summary['irregularity_rate']:.1f}%")
    print()
    
    # Step 3: Visualization
    print("Step 3: Generating visualizations...")
    visualizer = WeatherVisualizer()
    
    # Get actual station data from spatial analyzer
    visualizer._get_stations_data = lambda usafs: spatial_analyzer.stations_data[spatial_analyzer.stations_data['usaf'].isin(usafs)]
    
    # Get actual temperature data from spatial analyzer
    visualizer._get_temperature_data = lambda usaf: spatial_analyzer.temperature_data.get(usaf, None)
    
    # Get neighbor temperature data
    def get_neighbor_temp_data(usafs):
        neighbor_data_list = []
        for usaf in usafs:
            if usaf in spatial_analyzer.temperature_data:
                neighbor_data_list.append(spatial_analyzer.temperature_data[usaf])
        if neighbor_data_list:
            return pd.concat(neighbor_data_list, ignore_index=True)
        return None
    visualizer._get_neighbor_temperature_data = get_neighbor_temp_data
    
    # Add sine fitter for predictions
    from modules.sine_curve_fitting import SineCurveFitter
    visualizer.sine_fitter = SineCurveFitter()
    
    # Predict temperatures using fitted parameters
    target_params = spatial_results['target_params']
    def predict_temps(dates, params):
        try:
            return visualizer.sine_fitter.predict_temperatures(pd.Series(dates), params)
        except:
            return np.zeros(len(dates))
    visualizer._predict_temperatures = predict_temps
    
    # Create map
    print("Creating station map...")
    map_fig = visualizer.create_station_map(
        spatial_results,
        anomaly_results,
        title="Weather Station Analysis - Catalina Airport"
    )
    
    # Save map
    map_path = visualizer.save_visualization(map_fig, "catalina_station_map", "html")
    print(f"Map saved to: {map_path}")
    
    # Create temperature graph
    print("Creating temperature graph...")
    graph_fig = visualizer.create_temperature_graph(
        spatial_results,
        anomaly_results,
        title="Temperature Analysis - Catalina Airport"
    )
    
    if graph_fig is not None:
        graph_path = visualizer.save_visualization(graph_fig, "catalina_temperature_graph", "html")
        print(f"Graph saved to: {graph_path}")
    else:
        print("Note: Temperature graph not generated (requires additional data)")
    
    # Generate report
    print("Generating analysis report...")
    report_path = visualizer.generate_analysis_report(
        spatial_results,
        anomaly_results,
        station_usaf="722920"
    )
    print(f"Report saved to: {report_path}")
    print()
    
    print("=== Test Complete ===")
    print("Visualization outputs:")
    print(f"  - Map: {map_path}")
    if graph_fig is not None:
        print(f"  - Graph: {graph_path}")
    print(f"  - Report: {report_path}")
    
    return True


if __name__ == "__main__":
    try:
        success = test_anomaly_prediction_and_visualization()
        if not success:
            print("\nSome tests failed. Please check the error messages above.")
            sys.exit(1)
        else:
            print("\nAll modules tested successfully!")
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
