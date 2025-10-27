"""
BigQuery Connector Module

Handles authentication and data retrieval from Google BigQuery,
specifically for NOAA GSOD weather data.
"""

import os
import pandas as pd
from google.cloud import bigquery
from google.auth.exceptions import DefaultCredentialsError
import yaml
from typing import List, Dict, Optional, Tuple


class BigQueryConnector:
    """Handles BigQuery connections and data retrieval for weather data."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize BigQuery connector with configuration."""
        self.config = self._load_config(config_path)
        self.client = None
        self.project_id = self.config['project_id']
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    def authenticate(self, credentials_path: Optional[str] = None) -> bool:
        """
        Authenticate with Google Cloud.
        
        Args:
            credentials_path: Path to service account JSON file (optional)
            
        Returns:
            bool: True if authentication successful
        """
        try:
            # Try using credentials_path parameter first
            if credentials_path:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
                print(f"Using credentials from: {credentials_path}")
            # Try default path in project directory
            elif os.path.exists("spatial_weather_key.json"):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.abspath("spatial_weather_key.json")
                print(f"Using credentials from: spatial_weather_key.json")
            
            self.client = bigquery.Client(project=self.project_id)
            # Test connection
            self.client.query("SELECT 1").result()
            return True
            
        except DefaultCredentialsError:
            print("Error: Google Cloud credentials not found.")
            print("Please set up authentication by either:")
            print("1. Running 'gcloud auth application-default login'")
            print("2. Setting GOOGLE_APPLICATION_CREDENTIALS environment variable")
            print("3. Providing credentials_path parameter")
            return False
        except Exception as e:
            print(f"Authentication error: {e}")
            return False
    
    def get_weather_stations(self, start_year: int = 2005, end_year: int = 2025) -> pd.DataFrame:
        """
        Get weather stations with sufficient data coverage.
        
        Ported from notebook Cell 5 - filters stations based on:
        - Started tracking ≤2000
        - Ended tracking ≥2020  
        - ≥90% of max data coverage for specified years
        
        Args:
            start_year: Start year for analysis
            end_year: End year for analysis
            
        Returns:
            pd.DataFrame: Weather stations with metadata
        """
        if not self.client:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        # Use wildcard table for year range (simpler approach)
        query = f"""
        WITH
        NumTempDatesByStation AS (
          SELECT
            daily_weather.stn,
            COUNT(DISTINCT DATE(
              CAST(daily_weather.year AS INT64),
              CAST(daily_weather.mo AS INT64),
              CAST(daily_weather.da AS INT64)
            )) AS num_temp_dates
          FROM `bigquery-public-data.noaa_gsod.gsod*` daily_weather
          WHERE _table_suffix BETWEEN '{start_year}' AND '{end_year}'
            AND daily_weather.temp IS NOT NULL
            AND daily_weather.max IS NOT NULL
            AND daily_weather.min IS NOT NULL
            AND daily_weather.temp != 9999.9
            AND daily_weather.max != 9999.9
            AND daily_weather.min != 9999.9
          GROUP BY daily_weather.stn
        ),
        
        MaxNumTempDates AS (
          SELECT MAX(num_temp_dates) AS max_num_temp_dates
          FROM NumTempDatesByStation
        )
        
        SELECT
          Stations.*,
          NumTempDatesByStation.num_temp_dates
        FROM `bigquery-public-data.noaa_gsod.stations` Stations
        INNER JOIN NumTempDatesByStation ON stations.usaf = NumTempDatesByStation.stn
        CROSS JOIN MaxNumTempDates
        WHERE Stations.usaf != '999999'
          AND Stations.begin <= '20000101'
          AND Stations.end >= '20201231'
          AND NumTempDatesByStation.num_temp_dates >= 
              (0.90 * MaxNumTempDates.max_num_temp_dates)
        ORDER BY stations.usaf
        """
        
        try:
            result = self.client.query(query).result()
            return result.to_arrow().to_pandas()
        except Exception as e:
            print(f"Error querying weather stations: {e}")
            raise
    
    def get_station_daily_temperature(self, station_usaf: str, 
                                    start_year: int = 2005, 
                                    end_year: int = 2025) -> pd.DataFrame:
        """
        Get daily temperature data for a specific station.
        
        Ported from notebook Cell 17 - multi-year temperature data.
        
        Args:
            station_usaf: USAF station identifier
            start_year: Start year for data
            end_year: End year for data
            
        Returns:
            pd.DataFrame: Daily temperature data with date, avg_temp, max_temp, min_temp
        """
        if not self.client:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        query = f"""
        SELECT
          daily_weather.stn AS usaf,
          DATE(
            CAST(daily_weather.year AS INT64),
            CAST(daily_weather.mo AS INT64),
            CAST(daily_weather.da AS INT64)
          ) AS date,
          daily_weather.temp AS avg_temp,
          daily_weather.count_temp AS n_for_avg_temp,
          daily_weather.max AS max_temp,
          daily_weather.flag_max AS max_temp_flag,
          daily_weather.min AS min_temp,
          daily_weather.flag_min AS min_temp_flag
        FROM `bigquery-public-data.noaa_gsod.gsod*` daily_weather
        WHERE daily_weather.stn = '{station_usaf}'
          AND _table_suffix BETWEEN '{start_year}' AND '{end_year}'
          AND daily_weather.temp != 9999.9
          AND daily_weather.max != 9999.9
          AND daily_weather.min != 9999.9
        ORDER BY date DESC
        """
        
        try:
            result = self.client.query(query).result()
            df = result.to_arrow().to_pandas()
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            print(f"Error querying station temperature data: {e}")
            raise
    
    def get_station_info(self, station_usaf: str) -> Optional[Dict]:
        """
        Get metadata for a specific weather station.
        
        Args:
            station_usaf: USAF station identifier
            
        Returns:
            Dict: Station metadata or None if not found
        """
        if not self.client:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        query = f"""
        SELECT *
        FROM `bigquery-public-data.noaa_gsod.stations`
        WHERE usaf = '{station_usaf}'
        """
        
        try:
            result = self.client.query(query).result()
            df = result.to_arrow().to_pandas()
            if len(df) > 0:
                return df.iloc[0].to_dict()
            return None
        except Exception as e:
            print(f"Error querying station info: {e}")
            return None
    
    def save_results_to_bigquery(self, df: pd.DataFrame, 
                               table_name: str, 
                               dataset_id: Optional[str] = None,
                               replace: bool = True) -> bool:
        """
        Save analysis results back to BigQuery.
        
        Args:
            df: DataFrame to save
            table_name: Name of the table
            dataset_id: Dataset ID (uses config default if None)
            replace: Whether to replace existing table
            
        Returns:
            bool: True if successful
        """
        if not self.client:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        dataset_id = dataset_id or self.config['dataset_id']
        
        try:
            # Create dataset if it doesn't exist
            try:
                self.client.get_dataset(dataset_id)
            except:
                dataset = bigquery.Dataset(f"{self.project_id}.{dataset_id}")
                self.client.create_dataset(dataset)
            
            # Configure job
            job_config = bigquery.LoadJobConfig()
            if replace:
                job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
            else:
                job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
            
            # Add timestamp
            df['timestamp'] = pd.Timestamp.now(tz='UTC').ceil(freq='s')
            
            # Load data
            table_ref = self.client.dataset(dataset_id).table(table_name)
            job = self.client.load_table_from_dataframe(df, table_ref, job_config=job_config)
            job.result()
            
            print(f"Results saved to BigQuery table: {self.project_id}.{dataset_id}.{table_name}")
            return True
            
        except Exception as e:
            print(f"Error saving to BigQuery: {e}")
            return False
