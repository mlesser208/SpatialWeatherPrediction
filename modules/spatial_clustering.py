"""
Spatial Clustering Module

Implements spatial clustering algorithms focused on regional irregularity detection.
Primary focus: Compare individual stations to their geographic neighbors using
sine curve parameters and temperature patterns.
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import yaml
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from .bigquery_connector import BigQueryConnector
from .sine_curve_fitting import SineCurveFitter


class SpatialClusteringAnalyzer:
    """
    Analyzes spatial patterns in weather station data using multiple clustering approaches.
    Focuses on detecting regional irregularities by comparing stations to their neighbors.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize spatial clustering analyzer with configuration."""
        self.config = self._load_config(config_path)
        self.bq_connector = BigQueryConnector(config_path)
        self.sine_fitter = SineCurveFitter()
        self.stations_data = None
        self.temperature_data = {}
        self.sine_curve_params = {}
        
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
    
    def load_stations_data(self, start_year: int = None, end_year: int = None) -> pd.DataFrame:
        """
        Load weather stations data from BigQuery.
        
        Args:
            start_year: Start year for analysis (uses config default if None)
            end_year: End year for analysis (uses config default if None)
            
        Returns:
            pd.DataFrame: Weather stations with metadata
        """
        start_year = start_year or self.config['spatial_clustering']['time_period']['start_year']
        end_year = end_year or self.config['spatial_clustering']['time_period']['end_year']
        
        print(f"Loading weather stations data for {start_year}-{end_year}...")
        self.stations_data = self.bq_connector.get_weather_stations(start_year, end_year)
        print(f"Loaded {len(self.stations_data)} weather stations")
        return self.stations_data
    
    def calculate_distances(self, stations_df: pd.DataFrame) -> np.ndarray:
        """
        Calculate pairwise distances between all stations using lat/lon coordinates.
        
        Args:
            stations_df: DataFrame with 'lat' and 'lon' columns
            
        Returns:
            np.ndarray: Distance matrix (stations x stations)
        """
        # Extract coordinates
        coords = stations_df[['lat', 'lon']].values
        
        # Calculate pairwise distances using Euclidean distance
        # Note: This is an approximation - for more accuracy, use Haversine distance
        distances = cdist(coords, coords, metric='euclidean')
        
        return distances
    
    def find_neighbors(self, station_usaf: str, neighbor_count: int = None) -> List[str]:
        """
        Find the N closest neighbors to a target station.
        
        Args:
            station_usaf: USAF identifier of target station
            neighbor_count: Number of neighbors to find (uses config default if None)
            
        Returns:
            List[str]: USAF identifiers of closest neighbors
        """
        if self.stations_data is None:
            raise RuntimeError("Stations data not loaded. Call load_stations_data() first.")
        
        neighbor_count = neighbor_count or self.config['spatial_clustering']['neighbor_count']
        
        # Find target station index
        target_idx = self.stations_data[self.stations_data['usaf'] == station_usaf].index
        if len(target_idx) == 0:
            raise ValueError(f"Station {station_usaf} not found in stations data")
        
        target_idx = target_idx[0]
        
        # Calculate distances
        distances = self.calculate_distances(self.stations_data)
        
        # Get distances from target station to all others
        target_distances = distances[target_idx]
        
        # Find closest neighbors (excluding the station itself)
        neighbor_indices = np.argsort(target_distances)[1:neighbor_count+1]  # Skip self (index 0)
        
        # Get USAF identifiers of neighbors
        neighbor_usafs = self.stations_data.iloc[neighbor_indices]['usaf'].tolist()
        
        return neighbor_usafs
    
    def load_station_temperature_data(self, station_usaf: str, 
                                    start_year: int = None, 
                                    end_year: int = None) -> pd.DataFrame:
        """
        Load temperature data for a specific station.
        
        Args:
            station_usaf: USAF station identifier
            start_year: Start year (uses config default if None)
            end_year: End year (uses config default if None)
            
        Returns:
            pd.DataFrame: Daily temperature data
        """
        start_year = start_year or self.config['spatial_clustering']['time_period']['start_year']
        end_year = end_year or self.config['spatial_clustering']['time_period']['end_year']
        
        print(f"Loading temperature data for station {station_usaf} ({start_year}-{end_year})...")
        temp_data = self.bq_connector.get_station_daily_temperature(
            station_usaf, start_year, end_year
        )
        
        # Store in memory for reuse
        self.temperature_data[station_usaf] = temp_data
        return temp_data
    
    def fit_sine_curves_for_stations(self, station_usafs: List[str]) -> Dict[str, Dict]:
        """
        Fit sine curves to temperature data for multiple stations.
        
        Args:
            station_usafs: List of USAF station identifiers
            
        Returns:
            Dict[str, Dict]: Dictionary mapping station USAF to sine curve parameters
        """
        sine_params = {}
        
        for usaf in station_usafs:
            if usaf not in self.temperature_data:
                try:
                    self.load_station_temperature_data(usaf)
                except Exception as e:
                    print(f"Warning: Could not load data for station {usaf}: {e}")
                    continue
            
            temp_data = self.temperature_data[usaf]
            
            if len(temp_data) >= 500:  # Minimum data requirement
                try:
                    params = self.sine_fitter.fit_curve(temp_data, 'avg_temp', usaf)
                    sine_params[usaf] = params
                    print(f"Fitted sine curve for station {usaf}")
                except Exception as e:
                    print(f"Warning: Could not fit sine curve for station {usaf}: {e}")
            else:
                print(f"Warning: Station {usaf} has insufficient data ({len(temp_data)} days)")
        
        return sine_params
    
    def analyze_station_irregularity(self, station_usaf: str, 
                                   irregularity_threshold: float = None) -> Dict:
        """
        Analyze how irregular a station is compared to its neighbors.
        
        Args:
            station_usaf: USAF identifier of target station
            irregularity_threshold: Threshold for flagging irregularities (uses config default if None)
            
        Returns:
            Dict: Analysis results including irregularity flags and neighbor comparisons
        """
        if self.stations_data is None:
            raise RuntimeError("Stations data not loaded. Call load_stations_data() first.")
        
        irregularity_threshold = irregularity_threshold or self.config['spatial_clustering']['irregularity_threshold']
        
        # Find neighbors
        neighbor_usafs = self.find_neighbors(station_usaf)
        print(f"Found {len(neighbor_usafs)} neighbors for station {station_usaf}")
        
        # Load temperature data for target station and neighbors
        all_stations = [station_usaf] + neighbor_usafs
        station_data = {}
        
        for usaf in all_stations:
            if usaf not in self.temperature_data:
                try:
                    station_data[usaf] = self.load_station_temperature_data(usaf)
                except Exception as e:
                    print(f"Warning: Could not load data for station {usaf}: {e}")
                    continue
            else:
                station_data[usaf] = self.temperature_data[usaf]
        
        # Fit sine curves to all stations
        sine_params = self.fit_sine_curves_for_stations(all_stations)
        
        # Calculate neighbor averages
        neighbor_params = {usaf: params for usaf, params in sine_params.items() 
                          if usaf in neighbor_usafs}
        
        if not neighbor_params:
            raise ValueError("No valid neighbor data found")
        
        # Calculate average parameters for neighbors
        neighbor_avg = {}
        for param in ['amp', 'freq', 'phase_shift', 'mean']:
            values = [params[param] for params in neighbor_params.values()]
            neighbor_avg[param] = np.mean(values)
            neighbor_avg[f'{param}_std'] = np.std(values)
        
        # Compare target station to neighbor average
        target_params = sine_params.get(station_usaf)
        if not target_params:
            raise ValueError(f"No valid data found for target station {station_usaf}")
        
        # Calculate deviations
        deviations = {}
        irregularity_flags = {}
        
        for param in ['amp', 'freq', 'phase_shift', 'mean']:
            deviation = target_params[param] - neighbor_avg[param]
            deviations[param] = deviation
            
            # Check if deviation exceeds threshold
            threshold_value = irregularity_threshold * neighbor_avg[f'{param}_std']
            irregularity_flags[param] = abs(deviation) > threshold_value
        
        # Overall irregularity assessment
        irregular_params = sum(irregularity_flags.values())
        total_params = len(irregularity_flags)
        irregularity_score = irregular_params / total_params
        
        # Get station info
        station_info = self.stations_data[self.stations_data['usaf'] == station_usaf].iloc[0]
        
        results = {
            'station_usaf': station_usaf,
            'station_name': station_info['name'],
            'station_lat': station_info['lat'],
            'station_lon': station_info['lon'],
            'neighbor_count': len(neighbor_usafs),
            'neighbor_usafs': neighbor_usafs,
            'target_params': target_params,
            'neighbor_avg': neighbor_avg,
            'deviations': deviations,
            'irregularity_flags': irregularity_flags,
            'irregularity_score': irregularity_score,
            'is_irregular': irregularity_score > 0.5,  # Flag if more than half parameters are irregular
            'threshold_used': irregularity_threshold
        }
        
        return results
    
    def run_test_analysis(self) -> Dict:
        """
        Run analysis on the test station (Catalina Airport) specified in config.
        
        Returns:
            Dict: Analysis results for the test station
        """
        test_station = self.config['spatial_clustering']['test_station']
        print(f"Running test analysis on station {test_station} (Catalina Airport)...")
        
        # Load stations data
        self.load_stations_data()
        
        # Run irregularity analysis
        results = self.analyze_station_irregularity(test_station)
        
        print(f"Analysis complete for {results['station_name']}")
        print(f"Irregularity score: {results['irregularity_score']:.2f}")
        print(f"Is irregular: {results['is_irregular']}")
        
        return results
    
    def get_neighbor_comparison_data(self, station_usaf: str) -> pd.DataFrame:
        """
        Get temperature data for target station and its neighbors for visualization.
        
        Args:
            station_usaf: USAF identifier of target station
            
        Returns:
            pd.DataFrame: Combined temperature data with station identifiers
        """
        # Find neighbors
        neighbor_usafs = self.find_neighbors(station_usaf)
        all_stations = [station_usaf] + neighbor_usafs
        
        # Load temperature data for all stations
        combined_data = []
        
        for usaf in all_stations:
            if usaf not in self.temperature_data:
                try:
                    self.load_station_temperature_data(usaf)
                except Exception as e:
                    print(f"Warning: Could not load data for station {usaf}: {e}")
                    continue
            
            temp_data = self.temperature_data[usaf].copy()
            temp_data['station_usaf'] = usaf
            temp_data['station_name'] = self.stations_data[
                self.stations_data['usaf'] == usaf
            ]['name'].iloc[0]
            
            combined_data.append(temp_data)
        
        if combined_data:
            return pd.concat(combined_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def calculate_spatial_autocorrelation(self, station_usaf: str) -> Dict:
        """
        Calculate spatial autocorrelation metrics for a station and its neighbors.
        
        Args:
            station_usaf: USAF identifier of target station
            
        Returns:
            Dict: Spatial autocorrelation results
        """
        # Get neighbor comparison data
        comparison_data = self.get_neighbor_comparison_data(station_usaf)
        
        if comparison_data.empty:
            return {}
        
        # Calculate mean temperature for each station
        station_means = comparison_data.groupby('station_usaf')['avg_temp'].mean()
        
        # Get target station mean
        target_mean = station_means.get(station_usaf, 0)
        
        # Get neighbor means
        neighbor_usafs = self.find_neighbors(station_usaf)
        neighbor_means = [station_means.get(usaf, 0) for usaf in neighbor_usafs if usaf in station_means]
        
        if not neighbor_means:
            return {}
        
        # Calculate spatial autocorrelation metrics
        neighbor_avg = np.mean(neighbor_means)
        neighbor_std = np.std(neighbor_means)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef([target_mean] + neighbor_means, 
                                 [neighbor_avg] + neighbor_means)[0, 1]
        
        # Calculate spatial lag (difference from neighbor average)
        spatial_lag = target_mean - neighbor_avg
        
        results = {
            'target_mean_temp': target_mean,
            'neighbor_avg_temp': neighbor_avg,
            'neighbor_std_temp': neighbor_std,
            'spatial_lag': spatial_lag,
            'correlation_coefficient': correlation,
            'spatial_autocorrelation': 'positive' if correlation > 0.3 else 'negative' if correlation < -0.3 else 'none'
        }
        
        return results
