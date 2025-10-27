# Spatial Weather Prediction Geoprocessing Tool

A comprehensive geoprocessing tool for spatial clustering of weather stations and temperature anomaly prediction using sinusoidal curve fitting. The tool integrates with Google BigQuery to access NOAA GSOD weather data and can be exported as an ArcGIS Python Toolbox.

## Overview

This tool analyzes weather station data to:
- **Detect regional irregularities** by comparing stations to their geographic neighbors
- **Predict temperature anomalies** using historical deviation analysis
- **Visualize spatial patterns** with interactive maps and temperature graphs
- **Export results** as ArcGIS-compatible tools

## Key Features

- **Spatial Clustering**: Compare individual stations to their 10 closest neighbors
- **Sinusoidal Curve Fitting**: Model seasonal temperature patterns
- **Anomaly Detection**: Identify extreme temperature events using statistical thresholds
- **Interactive Visualizations**: Maps and graphs showing spatial patterns and deviations
- **ArcGIS Integration**: Exportable as Python Toolbox (.pyt) for ArcGIS Pro
- **BigQuery Integration**: Direct access to NOAA GSOD weather data

## Architecture

```
SpatialWeatherPrediction/
├── WeatherPredictionToolbox.pyt          # Main ArcGIS Python Toolbox
├── modules/
│   ├── bigquery_connector.py            # BigQuery data retrieval
│   ├── spatial_clustering.py            # Neighbor analysis and clustering
│   ├── sine_curve_fitting.py            # Sinusoidal model fitting
│   ├── anomaly_predictor.py             # Anomaly detection and prediction
│   └── visualizer.py                    # Map and graph generation
├── config/
│   └── config.yaml                       # Configuration settings
├── outputs/                              # Generated maps and graphs
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Google Cloud account with BigQuery access
- (Optional) ArcGIS Pro for toolbox integration

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd SpatialWeatherPrediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Google Cloud authentication**:
   ```bash
   gcloud auth application-default login
   ```
   Or set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to your service account key file.

4. **Configure the tool**:
   - Edit `config/config.yaml` to set your Google Cloud project ID
   - Adjust analysis parameters as needed

## Quick Start

### Basic Usage

```python
from modules.spatial_clustering import SpatialClusteringAnalyzer
from modules.anomaly_predictor import AnomalyPredictor
from modules.visualizer import WeatherVisualizer

# Initialize analyzer
analyzer = SpatialClusteringAnalyzer()

# Authenticate with BigQuery
analyzer.authenticate()

# Run analysis on Catalina Airport
results = analyzer.run_test_analysis()

# Generate visualizations
visualizer = WeatherVisualizer()
map_fig, graph_fig = visualizer.create_combined_visualization(results)
```

### Running Tests

```bash
# Test module integration
python test_integration.py

# Test with real data (requires authentication)
python test_spatial_clustering.py
```

## Methodology

### Spatial Clustering Approach

The tool uses a **neighbor comparison methodology**:

1. **Find Neighbors**: For each target station, identify the 10 geographically closest stations
2. **Fit Sine Curves**: Calculate sinusoidal parameters (amplitude, frequency, phase shift, mean) for all stations
3. **Compare Patterns**: Analyze how the target station's parameters differ from its neighbors
4. **Flag Irregularities**: Identify stations that deviate significantly from regional patterns

### Anomaly Prediction Process

**Step 1: Baseline Fit**
- Fit sinusoidal curves to 20-year historical data (2005-2024)
- Establish "normal" seasonal patterns for each station

**Step 2: Deviation Analysis**
- Compare recent year actual temperatures to fitted baseline
- Calculate statistical deviations and percentiles

**Step 3: Thresholding & Classification**
- Use multiple threshold methods (2 SD, percentiles, IQR)
- Flag extreme events based on historical deviation patterns
- Classify as "Extreme Hot" or "Extreme Cold"

**Step 4: Symbolization**
- Color-code stations: Green (normal), Orange (moderate), Red (extreme)
- Size markers by deviation magnitude

## Configuration

### Key Parameters

```yaml
# Spatial Clustering
spatial_clustering:
  neighbor_count: 10                    # Number of closest neighbors
  irregularity_threshold: 1.0            # Standard deviations for flagging
  time_period:
    start_year: 2005                    # Analysis start year
    end_year: 2025                      # Analysis end year

# Anomaly Detection
anomaly_detection:
  threshold_methods:
    - "2_std_dev"      # ±2 standard deviations
    - "percentile_95"   # Beyond 95th percentile
    - "iqr_method"      # Interquartile range method
```

## Outputs

### Visualizations

- **Interactive Map**: Shows stations with color-coded anomaly levels
- **Temperature Graph**: Compares target station to neighbor average with confidence bands
- **Analysis Report**: HTML report with detailed results and statistics

### Data Products

- **Spatial Analysis Results**: Station classifications and irregularity scores
- **Anomaly Predictions**: Extreme event flags and classifications
- **Sine Curve Parameters**: Fitted seasonal patterns for each station

## Examples

### Analyzing Catalina Airport

```python
# Run full analysis
analyzer = SpatialClusteringAnalyzer()
analyzer.authenticate()
results = analyzer.analyze_station_irregularity("722920")

print(f"Irregularity Score: {results['irregularity_score']:.2f}")
print(f"Is Irregular: {results['is_irregular']}")
```

### Custom Station Analysis

```python
# Analyze any station by USAF ID
results = analyzer.analyze_station_irregularity("YOUR_STATION_USAF")
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Ensure Google Cloud credentials are set up correctly
   - Check that your project has access to NOAA GSOD data

2. **Import Errors**
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

3. **Data Loading Issues**
   - Confirm station USAF ID exists in NOAA dataset
   - Check date range parameters

### Getting Help

- Check the integration tests: `python test_integration.py`
- Review configuration in `config/config.yaml`
- See `SETUP.md` for detailed setup instructions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure everything works
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NOAA for providing the GSOD weather dataset
- Google Cloud for BigQuery infrastructure
- The spatial analysis community for methodological guidance

## Citation

If you use this tool in your research, please cite:

```
Spatial Weather Prediction Geoprocessing Tool
[Your Name/Institution]
[Year]
```

## Changelog

### Version 1.0.0
- Initial release
- Spatial clustering with neighbor comparison
- Sinusoidal curve fitting
- Anomaly prediction with multiple threshold methods
- Interactive visualizations
- ArcGIS Python Toolbox integration
