"""
Simple Temperature Deviation Prediction Graph

Creates a clear visualization of temperature deviations and predictions
using the anomaly prediction results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.spatial_clustering import SpatialClusteringAnalyzer
from modules.anomaly_predictor import AnomalyPredictor


def create_deviation_graph():
    """Create a comprehensive temperature deviation prediction graph."""
    
    print("=== Creating Temperature Deviation Prediction Graph ===")
    print()
    
    # Step 1: Run spatial and anomaly analysis
    print("Step 1: Running analyses...")
    spatial_analyzer = SpatialClusteringAnalyzer()
    spatial_analyzer.authenticate()
    spatial_analyzer.load_stations_data()
    
    spatial_results = spatial_analyzer.analyze_station_irregularity("722920")
    print(f"   ✓ Spatial analysis complete")
    
    anomaly_predictor = AnomalyPredictor()
    anomaly_predictor.authenticate()
    
    # Authenticate spatial analyzer's BigQuery connector
    if not anomaly_predictor.spatial_analyzer.bq_connector.client:
        anomaly_predictor.spatial_analyzer.authenticate()
    
    if anomaly_predictor.spatial_analyzer.stations_data is None:
        anomaly_predictor.spatial_analyzer.load_stations_data()
    
    anomaly_results = anomaly_predictor.run_full_analysis("722920", threshold_method='2_std_dev')
    print(f"   ✓ Anomaly prediction complete")
    print()
    
    # Step 2: Get deviation data
    print("Step 2: Extracting deviation data...")
    target_station = "722920"
    if target_station in anomaly_results['deviation_results']:
        deviation_data = anomaly_results['deviation_results'][target_station]['data'].copy()
        print(f"   ✓ Loaded {len(deviation_data)} days of deviation data")
        
        # Get classification results
        classification = anomaly_results['classification_results'][target_station]
        print(f"   ✓ Station classification: {classification['station_classification']}")
        print(f"   ✓ Extreme days: {classification['extreme_days']}/{classification['total_days']} ({classification['extreme_percentage']:.1f}%)")
        print()
        
        # Step 3: Create the graph
        print("Step 3: Creating visualization...")
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Temperature over time
        ax1.plot(deviation_data['date'], deviation_data['avg_temp'], 
                'b-', linewidth=1.5, alpha=0.7, label='Actual Temperature')
        ax1.plot(deviation_data['date'], deviation_data['predicted_temp'], 
                'r--', linewidth=2, label='Predicted Temperature (Sine Curve)')
        
        # Add threshold lines
        threshold = anomaly_results['thresholds']['2_std_dev']['threshold']
        upper_bound = deviation_data['predicted_temp'] + threshold
        lower_bound = deviation_data['predicted_temp'] - threshold
        
        ax1.fill_between(deviation_data['date'], upper_bound, lower_bound, 
                         alpha=0.2, color='gray', label='Normal Range (±8.94°F)')
        
        # Highlight extreme days
        extreme_hot = deviation_data[deviation_data['deviation'] > threshold]
        extreme_cold = deviation_data[deviation_data['deviation'] < -threshold]
        
        if len(extreme_hot) > 0:
            ax1.scatter(extreme_hot['date'], extreme_hot['avg_temp'], 
                       color='red', s=30, zorder=5, label=f'Extreme Hot ({len(extreme_hot)} days)')
        if len(extreme_cold) > 0:
            ax1.scatter(extreme_cold['date'], extreme_cold['avg_temp'], 
                       color='blue', s=30, zorder=5, label=f'Extreme Cold ({len(extreme_cold)} days)')
        
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Temperature (°F)', fontsize=12)
        ax1.set_title('Catalina Airport Temperature Analysis - Actual vs Predicted', 
                      fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Deviations over time
        colors = ['red' if d > threshold else 'blue' if d < -threshold else 'gray' 
                  for d in deviation_data['deviation']]
        
        ax2.scatter(deviation_data['date'], deviation_data['deviation'], 
                   c=colors, s=20, alpha=0.6)
        ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Extreme Hot Threshold (+{threshold:.1f}°F)')
        ax2.axhline(y=-threshold, color='blue', linestyle='--', linewidth=2, 
                   label=f'Extreme Cold Threshold (-{threshold:.1f}°F)')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Deviation from Predicted (°F)', fontsize=12)
        ax2.set_title('Temperature Deviations - Catalina Airport 2024', 
                      fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Add summary text
        stats = classification
        summary_text = (f"Summary: {stats['extreme_percentage']:.1f}% extreme days\n"
                       f"Extreme Hot: {stats['extreme_hot_days']} days, "
                       f"Extreme Cold: {stats['extreme_cold_days']} days\n"
                       f"Classification: {stats['station_classification']}")
        
        fig.text(0.02, 0.02, summary_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save graph
        output_file = "outputs/catalina_temperature_deviation_prediction.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✓ Graph saved to: {output_file}")
        
        # Also save as interactive HTML
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig_html = make_subplots(rows=2, cols=1, 
                                    subplot_titles=('Temperature Analysis - Actual vs Predicted',
                                                   'Temperature Deviations'),
                                    vertical_spacing=0.12)
            
            # Add temperature plot
            fig_html.add_trace(go.Scatter(x=deviation_data['date'], y=deviation_data['avg_temp'],
                                         mode='lines', name='Actual Temperature',
                                         line=dict(color='blue', width=2)), row=1, col=1)
            fig_html.add_trace(go.Scatter(x=deviation_data['date'], y=deviation_data['predicted_temp'],
                                         mode='lines', name='Predicted (Sine Curve)',
                                         line=dict(color='red', dash='dash', width=2)), row=1, col=1)
            
            # Add extreme points
            if len(extreme_hot) > 0:
                fig_html.add_trace(go.Scatter(x=extreme_hot['date'], y=extreme_hot['avg_temp'],
                                             mode='markers', name='Extreme Hot',
                                             marker=dict(color='red', size=8)), row=1, col=1)
            if len(extreme_cold) > 0:
                fig_html.add_trace(go.Scatter(x=extreme_cold['date'], y=extreme_cold['avg_temp'],
                                             mode='markers', name='Extreme Cold',
                                             marker=dict(color='blue', size=8)), row=1, col=1)
            
            # Add deviation plot
            fig_html.add_trace(go.Scatter(x=deviation_data['date'], y=deviation_data['deviation'],
                                         mode='markers', name='Deviations',
                                         marker=dict(color=colors, size=6, opacity=0.6)), row=2, col=1)
            
            # Add threshold lines
            fig_html.add_hline(y=threshold, line_dash="dash", line_color="red",
                              annotation_text=f"+{threshold:.1f}°F", row=2, col=1)
            fig_html.add_hline(y=-threshold, line_dash="dash", line_color="blue",
                              annotation_text=f"-{threshold:.1f}°F", row=2, col=1)
            
            fig_html.update_layout(height=800, title_text="Catalina Airport Temperature Deviation Prediction")
            fig_html.update_xaxes(title_text="Date", row=1, col=1)
            fig_html.update_xaxes(title_text="Date", row=2, col=1)
            fig_html.update_yaxes(title_text="Temperature (°F)", row=1, col=1)
            fig_html.update_yaxes(title_text="Deviation (°F)", row=2, col=1)
            
            output_html = "outputs/catalina_temperature_deviation_prediction.html"
            fig_html.write_html(output_html)
            print(f"   ✓ Interactive graph saved to: {output_html}")
            
        except Exception as e:
            print(f"   ⚠ Could not create interactive version: {e}")
        
        print()
        print("=== Graph Creation Complete ===")
        print(f"Files created:")
        print(f"  - PNG: outputs/catalina_temperature_deviation_prediction.png")
        print(f"  - HTML: outputs/catalina_temperature_deviation_prediction.html")
        print()
        
        return True
    else:
        print("   ✗ No deviation data available")
        return False


if __name__ == "__main__":
    try:
        success = create_deviation_graph()
        if not success:
            print("Graph creation failed. Please check the error messages above.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error creating graph: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
