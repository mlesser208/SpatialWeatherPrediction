"""
Visualization Module

Generates interactive maps and graphs for spatial weather prediction analysis.
Creates Plotly-based visualizations for both map and temperature graph outputs.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import os
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from .bigquery_connector import BigQueryConnector
from .spatial_clustering import SpatialClusteringAnalyzer
from .anomaly_predictor import AnomalyPredictor


class WeatherVisualizer:
    """
    Generates interactive visualizations for weather prediction analysis.
    Creates maps showing station locations with anomaly classifications
    and graphs showing temperature patterns and predictions.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize visualizer with configuration."""
        self.config = self._load_config(config_path)
        self.output_dir = "outputs"
        self._ensure_output_dir()
        
        # Set default plotly theme
        pio.templates.default = "plotly_white"
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    def _ensure_output_dir(self):
        """Ensure output directory exists."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def create_station_map(self, spatial_results: Dict, 
                          anomaly_results: Dict = None,
                          title: str = "Weather Station Analysis") -> go.Figure:
        """
        Create interactive map showing weather stations with spatial clustering and anomaly results.
        
        Args:
            spatial_results: Results from spatial clustering analysis
            anomaly_results: Results from anomaly prediction analysis (optional)
            title: Map title
            
        Returns:
            go.Figure: Interactive map figure
        """
        print("Creating interactive station map...")
        
        # Get station data
        target_station = spatial_results['station_usaf']
        target_name = spatial_results['station_name']
        target_lat = spatial_results['station_lat']
        target_lon = spatial_results['station_lon']
        
        # Get neighbor data
        neighbor_usafs = spatial_results['neighbor_usafs']
        stations_data = self._get_stations_data([target_station] + neighbor_usafs)
        
        # Create base map
        fig = go.Figure()
        
        # Add neighbor stations
        if len(neighbor_usafs) > 0:
            neighbor_data = stations_data[stations_data['usaf'].isin(neighbor_usafs)]
            
            # Color neighbors based on anomaly results if available
            if anomaly_results and 'symbolization' in anomaly_results:
                colors = []
                sizes = []
                for _, row in neighbor_data.iterrows():
                    usaf = row['usaf']
                    if usaf in anomaly_results['symbolization']:
                        symbol_data = anomaly_results['symbolization'][usaf]
                        colors.append(symbol_data['color'])
                        sizes.append(symbol_data['size'])
                    else:
                        colors.append('lightblue')
                        sizes.append(8)
            else:
                colors = ['lightblue'] * len(neighbor_data)
                sizes = [8] * len(neighbor_data)
            
            # Add neighbor stations
            fig.add_trace(go.Scattermapbox(
                lat=neighbor_data['lat'],
                lon=neighbor_data['lon'],
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=colors,
                    opacity=0.7,
                    symbol='circle'
                ),
                text=neighbor_data['name'],
                hovertemplate='<b>%{text}</b><br>' +
                             'USAF: %{customdata}<br>' +
                             'Lat: %{lat:.2f}<br>' +
                             'Lon: %{lon:.2f}<br>' +
                             '<extra></extra>',
                customdata=neighbor_data['usaf'],
                name='Neighbor Stations',
                showlegend=True
            ))
        
        # Add target station (highlighted)
        target_color = 'red'
        target_size = 15
        
        if anomaly_results and 'symbolization' in anomaly_results:
            if target_station in anomaly_results['symbolization']:
                target_symbol = anomaly_results['symbolization'][target_station]
                target_color = target_symbol['color']
                target_size = target_symbol['size']
        
        fig.add_trace(go.Scattermapbox(
            lat=[target_lat],
            lon=[target_lon],
            mode='markers',
            marker=dict(
                size=target_size,
                color=target_color,
                opacity=1.0,
                symbol='star'
            ),
            text=[target_name],
            hovertemplate='<b>%{text}</b><br>' +
                         'USAF: ' + target_station + '<br>' +
                         'Lat: %{lat:.2f}<br>' +
                         'Lon: %{lon:.2f}<br>' +
                         '<extra></extra>',
            name='Target Station',
            showlegend=True
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=target_lat, lon=target_lon),
                zoom=8
            ),
            height=600,
            margin=dict(l=0, r=0, t=50, b=0),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        print("Station map created successfully")
        return fig
    
    def create_temperature_graph(self, spatial_results: Dict,
                               anomaly_results: Dict = None,
                               title: str = "Temperature Analysis") -> go.Figure:
        """
        Create temperature graph showing target station vs neighbor average with confidence bands.
        
        Args:
            spatial_results: Results from spatial clustering analysis
            anomaly_results: Results from anomaly prediction analysis (optional)
            title: Graph title
            
        Returns:
            go.Figure: Interactive temperature graph
        """
        print("Creating temperature analysis graph...")
        
        target_station = spatial_results['station_usaf']
        target_name = spatial_results['station_name']
        
        # Get temperature data for target station
        target_data = self._get_temperature_data(target_station)
        if target_data is None or len(target_data) == 0:
            print("No temperature data available for target station")
            return None
        
        # Get neighbor data for comparison
        neighbor_usafs = spatial_results['neighbor_usafs']
        neighbor_data = self._get_neighbor_temperature_data(neighbor_usafs)
        
        # Create subplot with secondary y-axis for deviations
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Temperature Patterns', 'Temperature Deviations'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Plot 1: Temperature patterns
        # Target station actual temperatures
        fig.add_trace(go.Scatter(
            x=target_data['date'],
            y=target_data['avg_temp'],
            mode='markers',
            marker=dict(size=4, color='blue', opacity=0.6),
            name=f'{target_name} (Actual)',
            hovertemplate='<b>%{text}</b><br>' +
                         'Date: %{x}<br>' +
                         'Temperature: %{y:.1f}°F<br>' +
                         '<extra></extra>',
            text=[target_name] * len(target_data)
        ), row=1, col=1)
        
        # Target station sine curve fit
        if 'target_params' in spatial_results:
            target_params = spatial_results['target_params']
            predicted_temps = self._predict_temperatures(target_data['date'], target_params)
            
            fig.add_trace(go.Scatter(
                x=target_data['date'],
                y=predicted_temps,
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name=f'{target_name} (Sine Curve Fit)',
                hovertemplate='<b>%{text}</b><br>' +
                             'Date: %{x}<br>' +
                             'Predicted: %{y:.1f}°F<br>' +
                             '<extra></extra>',
                text=[f'{target_name} (Predicted)'] * len(target_data)
            ), row=1, col=1)
        
        # Neighbor average
        if neighbor_data is not None and len(neighbor_data) > 0:
            neighbor_avg = neighbor_data.groupby('date')['avg_temp'].mean().reset_index()
            
            fig.add_trace(go.Scatter(
                x=neighbor_avg['date'],
                y=neighbor_avg['avg_temp'],
                mode='lines',
                line=dict(color='green', width=2),
                name='Neighbor Average',
                hovertemplate='<b>Neighbor Average</b><br>' +
                             'Date: %{x}<br>' +
                             'Temperature: %{y:.1f}°F<br>' +
                             '<extra></extra>'
            ), row=1, col=1)
            
            # Add confidence bands
            neighbor_std = neighbor_data.groupby('date')['avg_temp'].std().reset_index()
            upper_bound = neighbor_avg['avg_temp'] + neighbor_std['avg_temp']
            lower_bound = neighbor_avg['avg_temp'] - neighbor_std['avg_temp']
            
            fig.add_trace(go.Scatter(
                x=neighbor_avg['date'],
                y=upper_bound,
                mode='lines',
                line=dict(color='rgba(0,100,0,0)'),
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=neighbor_avg['date'],
                y=lower_bound,
                mode='lines',
                line=dict(color='rgba(0,100,0,0)'),
                fill='tonexty',
                fillcolor='rgba(0,100,0,0.2)',
                name='Neighbor Confidence Band',
                hoverinfo='skip'
            ), row=1, col=1)
        
        # Plot 2: Temperature deviations
        if anomaly_results and 'deviation_results' in anomaly_results:
            if target_station in anomaly_results['deviation_results']:
                deviation_data = anomaly_results['deviation_results'][target_station]['data']
                
                # Color points based on extreme classification
                colors = []
                for _, row in deviation_data.iterrows():
                    if 'extreme_type' in row:
                        if row['extreme_type'] == 'extreme_hot':
                            colors.append('red')
                        elif row['extreme_type'] == 'extreme_cold':
                            colors.append('blue')
                        else:
                            colors.append('gray')
                    else:
                        # Fallback to deviation-based coloring
                        if row['deviation'] > 0:
                            colors.append('orange')
                        else:
                            colors.append('blue')
                
                fig.add_trace(go.Scatter(
                    x=deviation_data['date'],
                    y=deviation_data['deviation'],
                    mode='markers',
                    marker=dict(size=4, color=colors, opacity=0.7),
                    name='Temperature Deviations',
                    hovertemplate='<b>%{text}</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Deviation: %{y:.1f}°F<br>' +
                                 '<extra></extra>',
                    text=[f'Deviation: {d:.1f}°F' for d in deviation_data['deviation']]
                ), row=2, col=1)
                
                # Add threshold lines
                if 'thresholds' in anomaly_results:
                    threshold_value = anomaly_results['thresholds']['2_std_dev']['threshold']
                    fig.add_hline(y=threshold_value, line_dash="dash", line_color="red", 
                                 annotation_text=f"Extreme Hot Threshold (+{threshold_value:.1f}°F)", row=2, col=1)
                    fig.add_hline(y=-threshold_value, line_dash="dash", line_color="blue", 
                                 annotation_text=f"Extreme Cold Threshold (-{threshold_value:.1f}°F)", row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Temperature (°F)", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Deviation (°F)", row=2, col=1)
        
        print("Temperature graph created successfully")
        return fig
    
    def create_combined_visualization(self, spatial_results: Dict,
                                    anomaly_results: Dict = None,
                                    station_usaf: str = None) -> Tuple[go.Figure, go.Figure]:
        """
        Create both map and graph visualizations for a complete analysis.
        
        Args:
            spatial_results: Results from spatial clustering analysis
            anomaly_results: Results from anomaly prediction analysis (optional)
            station_usaf: USAF identifier for file naming
            
        Returns:
            Tuple[go.Figure, go.Figure]: Map and graph figures
        """
        print("Creating combined visualization...")
        
        target_station = spatial_results['station_usaf']
        target_name = spatial_results['station_name']
        
        # Create map
        map_title = f"Weather Station Analysis - {target_name} ({target_station})"
        map_fig = self.create_station_map(spatial_results, anomaly_results, map_title)
        
        # Create graph
        graph_title = f"Temperature Analysis - {target_name} ({target_station})"
        graph_fig = self.create_temperature_graph(spatial_results, anomaly_results, graph_title)
        
        return map_fig, graph_fig
    
    def save_visualization(self, fig: go.Figure, filename: str, 
                          format: str = 'html') -> str:
        """
        Save visualization to file.
        
        Args:
            fig: Plotly figure to save
            filename: Name of the file (without extension)
            format: Output format ('html', 'png', 'pdf', 'svg')
            
        Returns:
            str: Path to saved file
        """
        if not filename.endswith(f'.{format}'):
            filename = f"{filename}.{format}"
        
        filepath = os.path.join(self.output_dir, filename)
        
        if format == 'html':
            fig.write_html(filepath)
        elif format == 'png':
            fig.write_image(filepath, width=1200, height=800)
        elif format == 'pdf':
            fig.write_image(filepath, format='pdf', width=1200, height=800)
        elif format == 'svg':
            fig.write_image(filepath, format='svg', width=1200, height=800)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Visualization saved to: {filepath}")
        return filepath
    
    def _get_stations_data(self, station_usafs: List[str]) -> pd.DataFrame:
        """Get station metadata for visualization."""
        # This would typically query the BigQuery connector
        # For now, return empty DataFrame
        return pd.DataFrame()
    
    def _get_temperature_data(self, station_usaf: str) -> Optional[pd.DataFrame]:
        """Get temperature data for a station."""
        # This would typically query the BigQuery connector
        # For now, return None
        return None
    
    def _get_neighbor_temperature_data(self, neighbor_usafs: List[str]) -> Optional[pd.DataFrame]:
        """Get temperature data for neighbor stations."""
        # This would typically query the BigQuery connector
        # For now, return None
        return None
    
    def _predict_temperatures(self, dates: pd.Series, params: Dict) -> np.ndarray:
        """Predict temperatures using sine curve parameters."""
        # This would use the sine curve fitting module
        # For now, return zeros
        return np.zeros(len(dates))
    
    def generate_analysis_report(self, spatial_results: Dict,
                               anomaly_results: Dict = None,
                               station_usaf: str = None) -> str:
        """
        Generate a comprehensive analysis report.
        
        Args:
            spatial_results: Results from spatial clustering analysis
            anomaly_results: Results from anomaly prediction analysis (optional)
            station_usaf: USAF identifier for file naming
            
        Returns:
            str: Path to generated report
        """
        print("Generating analysis report...")
        
        target_station = spatial_results['station_usaf']
        target_name = spatial_results['station_name']
        
        # Create HTML report
        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Weather Analysis Report - {target_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #2c3e50; }}
                .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px 20px; padding: 10px; background: #f8f9fa; border-radius: 3px; }}
                .metric-label {{ font-weight: bold; color: #495057; }}
                .metric-value {{ font-size: 1.2em; color: #2c3e50; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Weather Analysis Report</h1>
                <h2>{target_name} (USAF: {target_station})</h2>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h3>Station Information</h3>
                <div class="metric">
                    <div class="metric-label">Station Name:</div>
                    <div class="metric-value">{target_name}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">USAF ID:</div>
                    <div class="metric-value">{target_station}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Latitude:</div>
                    <div class="metric-value">{spatial_results['station_lat']:.4f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Longitude:</div>
                    <div class="metric-value">{spatial_results['station_lon']:.4f}</div>
                </div>
            </div>
            
            <div class="section">
                <h3>Spatial Analysis</h3>
                <div class="metric">
                    <div class="metric-label">Neighbors Analyzed:</div>
                    <div class="metric-value">{spatial_results['neighbor_count']}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Irregularity Score:</div>
                    <div class="metric-value">{spatial_results['irregularity_score']:.2f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Is Irregular:</div>
                    <div class="metric-value">{'Yes' if spatial_results['is_irregular'] else 'No'}</div>
                </div>
            </div>
        """
        
        # Add anomaly analysis section if available
        if anomaly_results:
            report_html += f"""
            <div class="section">
                <h3>Anomaly Analysis</h3>
                <div class="metric">
                    <div class="metric-label">Analysis Period:</div>
                    <div class="metric-value">2005-2024</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Threshold Method:</div>
                    <div class="metric-value">2 Standard Deviations</div>
                </div>
            </div>
            """
        
        report_html += """
        </body>
        </html>
        """
        
        # Save report
        filename = f"analysis_report_{station_usaf or target_station}.html"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        print(f"Analysis report saved to: {filepath}")
        return filepath
