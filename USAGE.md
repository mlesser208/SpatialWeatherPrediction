# Usage Guide

This guide explains how to use the Spatial Weather Prediction Tool for analyzing weather station data and predicting temperature anomalies.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Usage](#basic-usage)
3. [Advanced Usage](#advanced-usage)
4. [Understanding Outputs](#understanding-outputs)
5. [Examples](#examples)
6. [Best Practices](#best-practices)

## Quick Start

### Running Your First Analysis

```python
from modules.spatial_clustering import SpatialClusteringAnalyzer

# Initialize and authenticate
analyzer = SpatialClusteringAnalyzer()
analyzer.authenticate()

# Run analysis on Catalina Airport
results = analyzer.run_test_analysis()
print(f"Irregularity Score: {results['irregularity_score']:.2f}")
```

### Generating Visualizations

```python
from modules.visualizer import WeatherVisualizer

# Create visualizations
visualizer = WeatherVisualizer()
map_fig, graph_fig = visualizer.create_combined_visualization(results)

# Save outputs
visualizer.save_visualization(map_fig, "station_map", "html")
visualizer.save_visualization(graph_fig, "temperature_graph", "html")
```

## Basic Usage

### 1. Spatial Clustering Analysis

#### Analyze a Specific Station

```python
from modules.spatial_clustering import SpatialClusteringAnalyzer

# Initialize analyzer
analyzer = SpatialClusteringAnalyzer()
analyzer.authenticate()

# Load weather stations data
stations = analyzer.load_stations_data()

# Analyze specific station
results = analyzer.analyze_station_irregularity("722920")  # Catalina Airport

# View results
print(f"Station: {results['station_name']}")
print(f"Neighbors: {results['neighbor_count']}")
print(f"Irregularity Score: {results['irregularity_score']:.2f}")
print(f"Is Irregular: {results['is_irregular']}")
```

#### Find Neighbors for a Station

```python
# Find 10 closest neighbors
neighbors = analyzer.find_neighbors("722920")
print(f"Neighbors: {neighbors}")

# Find custom number of neighbors
neighbors = analyzer.find_neighbors("722920", neighbor_count=5)
```

### 2. Anomaly Prediction

#### Run Full Anomaly Analysis

```python
from modules.anomaly_predictor import AnomalyPredictor

# Initialize predictor
predictor = AnomalyPredictor()
predictor.authenticate()

# Run full analysis
results = predictor.run_full_analysis("722920", threshold_method="2_std_dev")

# View summary
summary = results['analysis_summary']
print(f"Stations Analyzed: {summary['total_stations_analyzed']}")
print(f"Irregular Stations: {summary['irregular_stations']}")
print(f"Irregularity Rate: {summary['irregularity_rate']:.1f}%")
```

#### Customize Analysis Parameters

```python
# Use different threshold method
results = predictor.run_full_analysis(
    "722920", 
    threshold_method="percentile_95"
)

# Analyze different time period
predictor.baseline_years = (2010, 2024)
predictor.recent_year = 2024
results = predictor.run_full_analysis("722920")
```

### 3. Visualization

#### Create Interactive Maps

```python
from modules.visualizer import WeatherVisualizer

# Initialize visualizer
visualizer = WeatherVisualizer()

# Create station map
map_fig = visualizer.create_station_map(
    spatial_results, 
    anomaly_results,
    title="Weather Station Analysis - Catalina Airport"
)

# Save map
visualizer.save_visualization(map_fig, "catalina_map", "html")
```

#### Create Temperature Graphs

```python
# Create temperature graph
graph_fig = visualizer.create_temperature_graph(
    spatial_results,
    anomaly_results,
    title="Temperature Analysis - Catalina Airport"
)

# Save graph
visualizer.save_visualization(graph_fig, "catalina_temperature", "html")
```

#### Generate Analysis Report

```python
# Generate comprehensive report
report_path = visualizer.generate_analysis_report(
    spatial_results,
    anomaly_results,
    station_usaf="722920"
)
print(f"Report saved to: {report_path}")
```

## Advanced Usage

### 1. Custom Configuration

#### Modify Analysis Parameters

```python
# Load custom configuration
analyzer = SpatialClusteringAnalyzer("path/to/custom_config.yaml")

# Override specific parameters
analyzer.config['spatial_clustering']['neighbor_count'] = 15
analyzer.config['spatial_clustering']['irregularity_threshold'] = 1.5
```

#### Custom Date Ranges

```python
# Analyze different time periods
stations = analyzer.load_stations_data(start_year=2010, end_year=2024)
temp_data = analyzer.load_station_temperature_data("722920", 2010, 2024)
```

### 2. Batch Processing

#### Analyze Multiple Stations

```python
# List of stations to analyze
station_list = ["722920", "722921", "722922", "722923", "722924"]

results = {}
for station in station_list:
    try:
        result = analyzer.analyze_station_irregularity(station)
        results[station] = result
        print(f"✅ Analyzed {station}: {result['station_name']}")
    except Exception as e:
        print(f"❌ Failed to analyze {station}: {e}")
```

#### Export Results to BigQuery

```python
# Save results back to BigQuery
analyzer.bq_connector.save_results_to_bigquery(
    results_df, 
    "spatial_analysis_results",
    replace=True
)
```

### 3. Custom Visualizations

#### Create Custom Maps

```python
# Custom map with specific styling
map_fig = visualizer.create_station_map(
    spatial_results,
    anomaly_results,
    title="Custom Weather Analysis"
)

# Update map styling
map_fig.update_layout(
    mapbox_style="satellite-streets",
    height=800,
    margin=dict(l=0, r=0, t=50, b=0)
)

# Save with custom name
visualizer.save_visualization(map_fig, "custom_analysis_map", "png")
```

#### Create Custom Graphs

```python
# Custom temperature graph
graph_fig = visualizer.create_temperature_graph(
    spatial_results,
    anomaly_results,
    title="Custom Temperature Analysis"
)

# Update graph styling
graph_fig.update_layout(
    height=1000,
    showlegend=True,
    legend=dict(x=0.02, y=0.98)
)

# Save multiple formats
visualizer.save_visualization(graph_fig, "custom_graph", "html")
visualizer.save_visualization(graph_fig, "custom_graph", "png")
```

## Understanding Outputs

### 1. Spatial Analysis Results

```python
results = analyzer.analyze_station_irregularity("722920")

# Key metrics
print(f"Station USAF: {results['station_usaf']}")
print(f"Station Name: {results['station_name']}")
print(f"Location: ({results['station_lat']:.4f}, {results['station_lon']:.4f})")
print(f"Neighbors: {results['neighbor_count']}")

# Irregularity analysis
print(f"Irregularity Score: {results['irregularity_score']:.2f}")
print(f"Is Irregular: {results['is_irregular']}")

# Parameter deviations
for param, deviation in results['deviations'].items():
    is_irregular = results['irregularity_flags'][param]
    status = "🚨 IRREGULAR" if is_irregular else "✅ Normal"
    print(f"{param}: {deviation:.3f} {status}")
```

### 2. Anomaly Prediction Results

```python
anomaly_results = predictor.run_full_analysis("722920")

# Classification results
classification = anomaly_results['classification_results']['722920']
print(f"Station Classification: {classification['station_classification']}")
print(f"Extreme Days: {classification['extreme_days']}/{classification['total_days']}")
print(f"Extreme Percentage: {classification['extreme_percentage']:.1f}%")
print(f"Extreme Hot Days: {classification['extreme_hot_days']}")
print(f"Extreme Cold Days: {classification['extreme_cold_days']}")

# Threshold information
thresholds = anomaly_results['thresholds']
print(f"Threshold Method: {classification['threshold_method']}")
print(f"Threshold Value: {classification['threshold_used']:.2f}°F")
```

### 3. Visualization Outputs

#### Map Interpretation

- **Green markers**: Normal stations (within expected range)
- **Orange markers**: Moderately irregular stations
- **Red markers**: Highly irregular stations
- **Star marker**: Target station (highlighted)
- **Circle markers**: Neighbor stations

#### Graph Interpretation

- **Blue dots**: Actual temperature data
- **Red dashed line**: Fitted sinusoidal curve
- **Green line**: Neighbor average temperature
- **Gray band**: Neighbor confidence interval
- **Red/Blue dots (bottom panel)**: Temperature deviations
- **Dashed lines**: Extreme event thresholds

## Examples

### Example 1: Analyze Catalina Airport

```python
from modules.spatial_clustering import SpatialClusteringAnalyzer
from modules.anomaly_predictor import AnomalyPredictor
from modules.visualizer import WeatherVisualizer

# Initialize all components
analyzer = SpatialClusteringAnalyzer()
predictor = AnomalyPredictor()
visualizer = WeatherVisualizer()

# Authenticate
analyzer.authenticate()
predictor.authenticate()

# Run spatial analysis
spatial_results = analyzer.analyze_station_irregularity("722920")

# Run anomaly prediction
anomaly_results = predictor.run_full_analysis("722920")

# Create visualizations
map_fig, graph_fig = visualizer.create_combined_visualization(
    spatial_results, 
    anomaly_results
)

# Save outputs
visualizer.save_visualization(map_fig, "catalina_analysis_map", "html")
visualizer.save_visualization(graph_fig, "catalina_analysis_graph", "html")

# Generate report
report_path = visualizer.generate_analysis_report(
    spatial_results, 
    anomaly_results, 
    "722920"
)

print(f"Analysis complete! Report saved to: {report_path}")
```

### Example 2: Compare Multiple Stations

```python
# Analyze multiple stations
stations = ["722920", "722921", "722922", "722923", "722924"]
results = {}

for station in stations:
    try:
        spatial_result = analyzer.analyze_station_irregularity(station)
        anomaly_result = predictor.run_full_analysis(station)
        
        results[station] = {
            'spatial': spatial_result,
            'anomaly': anomaly_result
        }
        
        print(f"✅ {station}: {spatial_result['station_name']}")
        print(f"   Irregularity: {spatial_result['irregularity_score']:.2f}")
        print(f"   Classification: {anomaly_result['classification_results'][station]['station_classification']}")
        
    except Exception as e:
        print(f"❌ {station}: {e}")

# Find most irregular station
most_irregular = max(results.keys(), 
                    key=lambda x: results[x]['spatial']['irregularity_score'])
print(f"\nMost irregular station: {most_irregular}")
```

### Example 3: Custom Analysis Parameters

```python
# Custom analysis with different parameters
analyzer = SpatialClusteringAnalyzer()

# Modify configuration
analyzer.config['spatial_clustering']['neighbor_count'] = 15
analyzer.config['spatial_clustering']['irregularity_threshold'] = 1.5

# Analyze with custom parameters
results = analyzer.analyze_station_irregularity("722920")

# Use different threshold method
predictor = AnomalyPredictor()
anomaly_results = predictor.run_full_analysis(
    "722920", 
    threshold_method="percentile_95"
)

print(f"Custom analysis complete!")
print(f"Neighbors used: {results['neighbor_count']}")
print(f"Threshold method: {anomaly_results['classification_results']['722920']['threshold_method']}")
```

## Best Practices

### 1. Data Management

- **Use appropriate date ranges**: 20+ years for robust baseline fitting
- **Check data quality**: Ensure stations have sufficient data coverage
- **Validate station IDs**: Verify USAF IDs exist in NOAA dataset
- **Monitor BigQuery costs**: Use on-demand processing to control costs

### 2. Analysis Parameters

- **Neighbor count**: 10 neighbors provides good regional context
- **Irregularity threshold**: 1.0 standard deviations is a good starting point
- **Threshold method**: 2_std_dev works well for most cases
- **Time periods**: Use consistent periods for comparable results

### 3. Visualization

- **Save multiple formats**: HTML for interactivity, PNG for reports
- **Use descriptive titles**: Include station name and analysis type
- **Check output quality**: Verify maps and graphs are readable
- **Organize outputs**: Use consistent naming conventions

### 4. Performance

- **Process stations individually**: Avoid loading too much data at once
- **Use caching**: Store station metadata locally when possible
- **Monitor memory usage**: Large datasets can be memory-intensive
- **Optimize queries**: Use appropriate date ranges and filters

### 5. Error Handling

- **Always check authentication**: Verify BigQuery access before analysis
- **Handle missing data**: Some stations may have insufficient data
- **Validate inputs**: Check station IDs and parameters
- **Log errors**: Keep track of failed analyses for debugging

## Troubleshooting

### Common Issues

1. **Authentication errors**: Check Google Cloud credentials
2. **Data loading failures**: Verify station IDs and date ranges
3. **Memory issues**: Reduce analysis scope or increase system memory
4. **Visualization problems**: Check Plotly installation and dependencies

### Getting Help

- Check the integration tests: `python test_integration.py`
- Review error messages for specific issues
- Consult the troubleshooting section in `SETUP.md`
- Check the GitHub issues page for known problems

## Next Steps

1. **Explore the examples**: Try running the provided example scripts
2. **Customize analysis**: Modify parameters for your specific needs
3. **Create ArcGIS Toolbox**: Follow the toolbox creation guide
4. **Integrate with workflows**: Incorporate the tool into your analysis pipeline
