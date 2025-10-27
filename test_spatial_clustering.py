"""
Test Script for Spatial Clustering Module

This script tests the spatial clustering functionality with Catalina Airport
to validate the implementation before integrating with the full tool.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.spatial_clustering import SpatialClusteringAnalyzer


def test_spatial_clustering():
    """Test the spatial clustering analyzer with Catalina Airport."""
    
    print("=== Spatial Weather Prediction Tool - Test Script ===")
    print("Testing spatial clustering with Catalina Airport (USAF: 722920)")
    print()
    
    # Initialize analyzer
    analyzer = SpatialClusteringAnalyzer()
    
    # Test authentication (will prompt for credentials if needed)
    print("Step 1: Testing BigQuery authentication...")
    if not analyzer.authenticate():
        print(" Authentication failed. Please check your credentials.")
        print("   Make sure you have set GOOGLE_APPLICATION_CREDENTIALS environment variable")
        print("   or run 'gcloud auth application-default login'")
        return False
    print(" Authentication successful!")
    print()
    
    # Test stations data loading
    print("Step 2: Loading weather stations data...")
    try:
        stations = analyzer.load_stations_data()
        print(f" Loaded {len(stations)} weather stations")
        print(f"   Sample stations: {stations['name'].head(3).tolist()}")
        
        # Check if Catalina Airport is in the data
        catalina_stations = stations[stations['usaf'] == '722920']
        if len(catalina_stations) > 0:
            print(f"    Catalina Airport found: {catalina_stations['name'].iloc[0]}")
        else:
            print("     Catalina Airport not found in stations data")
            
    except Exception as e:
        print(f" Failed to load stations data: {e}")
        return False
    print()
    
    # Test neighbor finding
    print("Step 3: Finding neighbors for Catalina Airport...")
    try:
        neighbors = analyzer.find_neighbors("722920")
        print(f" Found {len(neighbors)} neighbors for Catalina Airport")
        print(f"   Neighbor USAFs: {neighbors}")
        
        # Get neighbor names for better understanding
        neighbor_info = stations[stations['usaf'].isin(neighbors)][['usaf', 'name', 'lat', 'lon']]
        print("   Neighbor details:")
        for _, row in neighbor_info.iterrows():
            print(f"     {row['usaf']}: {row['name']} ({row['lat']:.2f}, {row['lon']:.2f})")
            
    except Exception as e:
        print(f" Failed to find neighbors: {e}")
        return False
    print()
    
    # Test temperature data loading
    print("Step 4: Loading temperature data for Catalina Airport...")
    try:
        temp_data = analyzer.load_station_temperature_data("722920")
        print(f" Loaded {len(temp_data)} days of temperature data")
        print(f"   Date range: {temp_data['date'].min()} to {temp_data['date'].max()}")
        print(f"   Temperature range: {temp_data['avg_temp'].min():.1f}°F to {temp_data['avg_temp'].max():.1f}°F")
        
    except Exception as e:
        print(f" Failed to load temperature data: {e}")
        return False
    print()
    
    # Test sine curve fitting
    print("Step 5: Testing sine curve fitting...")
    try:
        sine_params = analyzer.sine_fitter.fit_curve(temp_data, 'avg_temp', "722920")
        print(" Sine curve fitting completed!")
        print(f"   Amplitude: {sine_params['amp']:.2f}°F")
        print(f"   Frequency: {sine_params['freq']:.6f}")
        print(f"   Phase Shift: {sine_params['phase_shift']:.1f} days")
        print(f"   Mean Temperature: {sine_params['mean']:.1f}°F")
        print(f"   Mean Absolute Error: {sine_params['mae']:.2f}°F")
        
    except Exception as e:
        print(f" Failed to fit sine curve: {e}")
        return False
    print()
    
    # Test irregularity analysis
    print("Step 6: Running irregularity analysis...")
    try:
        results = analyzer.analyze_station_irregularity("722920")
        print(" Irregularity analysis completed!")
        print(f"   Station: {results['station_name']}")
        print(f"   Location: ({results['station_lat']:.2f}, {results['station_lon']:.2f})")
        print(f"   Neighbors analyzed: {results['neighbor_count']}")
        print(f"   Irregularity Score: {results['irregularity_score']:.2f}")
        print(f"   Is Irregular: {results['is_irregular']}")
        print(f"   Threshold used: {results['threshold_used']} standard deviations")
        
        # Show parameter deviations
        print("   Parameter deviations from neighbor average:")
        for param, deviation in results['deviations'].items():
            is_irregular = results['irregularity_flags'][param]
            status = " IRREGULAR" if is_irregular else " Normal"
            print(f"     {param}: {deviation:.3f} {status}")
            
    except Exception as e:
        print(f" Failed to run irregularity analysis: {e}")
        return False
    print()
    
    # Test spatial autocorrelation
    print("Step 7: Testing spatial autocorrelation...")
    try:
        autocorr_results = analyzer.calculate_spatial_autocorrelation("722920")
        if autocorr_results:
            print(" Spatial autocorrelation analysis completed!")
            print(f"   Target station mean temp: {autocorr_results['target_mean_temp']:.1f}°F")
            print(f"   Neighbor average temp: {autocorr_results['neighbor_avg_temp']:.1f}°F")
            print(f"   Spatial lag: {autocorr_results['spatial_lag']:.1f}°F")
            print(f"   Correlation coefficient: {autocorr_results['correlation_coefficient']:.3f}")
            print(f"   Spatial autocorrelation: {autocorr_results['spatial_autocorrelation']}")
        else:
            print("  Spatial autocorrelation analysis returned no results")
            
    except Exception as e:
        print(f" Failed to run spatial autocorrelation: {e}")
        return False
    print()
    
    print(" All tests passed! Spatial clustering module is working correctly.")
    print()
    print("Summary:")
    print(f"  - Loaded {len(stations)} weather stations")
    print(f"  - Found {len(neighbors)} neighbors for Catalina Airport")
    print(f"  - Analyzed {len(temp_data)} days of temperature data")
    print(f"  - Irregularity score: {results['irregularity_score']:.2f}")
    print(f"  - Station is {'IRREGULAR' if results['is_irregular'] else 'NORMAL'} compared to neighbors")
    
    return True


def main():
    """Main test function."""
    try:
        success = test_spatial_clustering()
        if not success:
            print("\n Some tests failed. Please check the error messages above.")
            print("\nTroubleshooting tips:")
            print("1. Make sure you have set up Google Cloud credentials")
            print("2. Check your internet connection")
            print("3. Verify your BigQuery project has access to NOAA data")
            sys.exit(1)
        else:
            print("\n Ready to proceed with full tool implementation!")
            print("Next steps:")
            print("1. Create anomaly prediction module")
            print("2. Build visualization module")
            print("3. Create ArcGIS Python Toolbox")
            
    except KeyboardInterrupt:
        print("\n\n  Test interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n Unexpected error during testing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
