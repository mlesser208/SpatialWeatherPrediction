"""
WeatherPredictionToolbox.pyt

ArcGIS Pro Python Toolbox for Spatial Weather Prediction Analysis

This toolbox provides geoprocessing tools for:
- Spatial clustering of weather stations
- Temperature anomaly detection
- Regional irregularity analysis
- BigQuery integration for NOAA GSOD data

Author: [Your Name]
Created: 2025
"""

import arcpy
import os
import sys
import json

# Add the modules directory to the path
# Handle cases where __file__ might not be defined
try:
    toolbox_dir = os.path.dirname(__file__)
except NameError:
    # Fallback 
    toolbox_dir = os.getcwd()

modules_dir = os.path.join(toolbox_dir, 'modules')
if os.path.exists(modules_dir) and modules_dir not in sys.path:
    sys.path.insert(0, modules_dir)

# Module imports will be handled in each tool's execute method


class Toolbox(object):
    """
    The toolbox class for Weather Prediction Analysis.
    """
    
    def __init__(self):
        """Define the toolbox."""
        self.label = "Weather Prediction Toolbox"
        self.alias = "WeatherPrediction"
        self.description = """Spatial weather prediction analysis tools for detecting regional irregularities 
        and temperature anomalies using NOAA GSOD data."""
        
        # List of tool classes associated with this toolbox
        self.tools = [
            AuthenticateBigQuery,
            AnalyzeStationIrregularity,
            PredictTemperatureAnomalies,
            GenerateStationMap,
            GenerateTemperatureGraph,
            FullAnalysisWorkflow
        ]


class AuthenticateBigQuery(object):
    """
    Authenticate with Google BigQuery and verify connection to NOAA GSOD data.
    """
    
    def __init__(self):
        """Define the tool."""
        self.label = "Authenticate BigQuery"
        self.description = """Authenticate with Google BigQuery to access NOAA GSOD weather data.
        This must be run before any other tools in the toolbox."""
        self.canRunInBackground = False
    
    def getParameterInfo(self):
        """Define parameter definitions."""
        
        params = []
        
        # Credentials file parameter
        credentials = arcpy.Parameter(
            displayName="Service Account Key File",
            name="credentials_file",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input"
        )
        credentials.description = "Path to Google Cloud service account JSON key file"
        params.append(credentials)
        
        # Project ID parameter
        project_id = arcpy.Parameter(
            displayName="Google Cloud Project ID",
            name="project_id",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        project_id.value = "focused-mote-472706-u8"
        project_id.description = "Your Google Cloud project ID"
        params.append(project_id)
        
        # Test connection parameter
        test_query = arcpy.Parameter(
            displayName="Test Connection Query",
            name="test_query",
            datatype="GPString",
            parameterType="Derived",
            direction="Output"
        )
        test_query.description = "Authentication status message"
        params.append(test_query)
        
        return params
    
    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True
    
    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed."""
        pass
    
    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter."""
        pass
    
    def execute(self, parameters, messages):  # type: ignore  # type: ignore
        """
        Purpose: Authenticate with BigQuery and test connection
        Parameters: credentials_file, project_id
        Return Value: None
        Exceptions: Raises if authentication fails
        """
        
        try:
            # Import here to ensure modules are available
            from modules import bigquery_connector
        except ImportError as e:
            arcpy.AddError(f"Failed to import bigquery_connector: {e}")
            raise
        
        credentials_file = parameters[0].valueAsText
        project_id = parameters[1].valueAsText
        
        try:
            arcpy.AddMessage("Initializing BigQuery connection...")
            
            # Initialize connector
            connector = bigquery_connector.BigQueryConnector()
            
            # Authenticate
            if credentials_file:
                arcpy.AddMessage(f"Using credentials from: {credentials_file}")
                success = connector.authenticate(credentials_file)
            else:
                arcpy.AddMessage("Using default credentials...")
                success = connector.authenticate()
            
            if success:
                arcpy.AddMessage("Authentication successful!")
                arcpy.AddMessage(f"Connected to project: {project_id}")
                parameters[2].value = "✓ Authentication successful"
            else:
                arcpy.AddError("Authentication failed!")
                parameters[2].value = "✗ Authentication failed"
                raise arcpy.ExecuteError("BigQuery authentication failed")
                
        except Exception as e:
            arcpy.AddError(f"Error during authentication: {str(e)}")
            parameters[2].value = f"✗ Error: {str(e)}"
            raise


class AnalyzeStationIrregularity(object):
    """
    Analyze how irregular a weather station is compared to its geographic neighbors.
    """
    
    def __init__(self):
        """Define the tool."""
        self.label = "Analyze Station Irregularity"
        self.description = """Compares a target weather station to its nearest neighbors to detect
        regional irregularities in temperature patterns."""
        self.canRunInBackground = False
    
    def getParameterInfo(self):
        """Define parameter definitions."""
        
        params = []
        
        # Station USAF parameter
        station_usaf = arcpy.Parameter(
            displayName="Weather Station USAF ID",
            name="station_usaf",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        station_usaf.value = "722920"  # Catalina Airport
        station_usaf.description = "USAF identifier of the target weather station"
        params.append(station_usaf)
        
        # Neighbor count
        neighbor_count = arcpy.Parameter(
            displayName="Number of Neighbors",
            name="neighbor_count",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        neighbor_count.value = 10
        neighbor_count.description = "Number of closest neighbors to analyze"
        params.append(neighbor_count)
        
        # Output feature class
        output_fc = arcpy.Parameter(
            displayName="Output Feature Class",
            name="output_fc",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output"
        )
        output_fc.description = "Output feature class containing analysis results"
        params.append(output_fc)
        
        # Output summary table
        output_table = arcpy.Parameter(
            displayName="Output Summary Table",
            name="output_table",
            datatype="DETable",
            parameterType="Required",
            direction="Output"
        )
        output_table.description = "Summary table with irregularity metrics"
        params.append(output_table)
        
        return params
    
    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True
    
    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed."""
        pass
    
    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter."""
        pass
    
    def execute(self, parameters, messages):  # type: ignore
        """
        Purpose: Analyze station irregularity compared to neighbors
        Parameters: station_usaf, neighbor_count, output_fc, output_table
        Return Value: None
        Exceptions: Raises if station not found or analysis fails
        """
        
        station_usaf = parameters[0].valueAsText
        neighbor_count = parameters[1].valueAsText
        output_fc = parameters[2].valueAsText
        output_table = parameters[3].valueAsText
        
        try:
            # Import here to ensure modules are available
            from modules import spatial_clustering
        except ImportError as e:
            arcpy.AddError(f"Failed to import spatial_clustering: {e}")
            raise
        
        try:
            arcpy.AddMessage(f"Analyzing station irregularity for {station_usaf}...")
            
            # Initialize analyzer
            analyzer = spatial_clustering.SpatialClusteringAnalyzer()
            
            # Authenticate
            if not analyzer.authenticate():
                raise arcpy.ExecuteError("Authentication required. Run 'Authenticate BigQuery' first.")
            
            # Load stations data
            arcpy.AddMessage("Loading weather stations data...")
            analyzer.load_stations_data()
            
            # Run analysis
            arcpy.AddMessage(f"Running irregularity analysis...")
            results = analyzer.analyze_station_irregularity(
                station_usaf, 
                int(neighbor_count)
            )
            
            # Create output feature class
            arcpy.AddMessage("Creating output feature class...")
            self._create_output_feature_class(output_fc, results, analyzer)
            
            # Create summary table
            arcpy.AddMessage("Creating summary table...")
            self._create_summary_table(output_table, results)
            
            arcpy.AddMessage("Analysis complete!")
            
        except Exception as e:
            arcpy.AddError(f"Error during analysis: {str(e)}")
            raise
    
    def _create_output_feature_class(self, output_fc, results, analyzer):
        """Create output feature class with results."""
        # Create spatial reference
        sr = arcpy.SpatialReference(4326)  # WGS84
        
        # Create feature class
        arcpy.management.CreateFeatureclass(
            os.path.dirname(output_fc),
            os.path.basename(output_fc),
            "POINT",
            spatial_reference=sr
        )
        
        # Add fields
        arcpy.management.AddField(output_fc, "usaf", "TEXT")
        arcpy.management.AddField(output_fc, "name", "TEXT")
        arcpy.management.AddField(output_fc, "type", "TEXT")
        arcpy.management.AddField(output_fc, "irregularity", "DOUBLE")
        arcpy.management.AddField(output_fc, "is_irregular", "TEXT")
        
        # Insert features for target and neighbors
        cursor = arcpy.da.InsertCursor(output_fc, ["SHAPE@", "usaf", "name", "type", "irregularity", "is_irregular"])
        
        # Target station
        point = arcpy.Point(results['station_lon'], results['station_lat'])
        cursor.insertRow([point, results['station_usaf'], results['station_name'], 
                         'Target', results['irregularity_score'], 
                         'Yes' if results['is_irregular'] else 'No'])
        
        # Neighbors
        stations_data = analyzer.stations_data
        for neighbor_usaf in results['neighbor_usafs']:
            neighbor_info = stations_data[stations_data['usaf'] == neighbor_usaf]
            if len(neighbor_info) > 0:
                row = neighbor_info.iloc[0]
                point = arcpy.Point(row['lon'], row['lat'])
                cursor.insertRow([point, neighbor_usaf, row['name'], 'Neighbor', 0, 'N/A'])
        
        del cursor
    
    def _create_summary_table(self, output_table, results):
        """Create summary table with metrics."""
        arcpy.management.CreateTable(os.path.dirname(output_table), os.path.basename(output_table))
        
        arcpy.management.AddField(output_table, "metric", "TEXT")
        arcpy.management.AddField(output_table, "value", "TEXT")
        
        with arcpy.da.InsertCursor(output_table, ["metric", "value"]) as cursor:
            cursor.insertRow(["Station USAF", results['station_usaf']])
            cursor.insertRow(["Station Name", results['station_name']])
            cursor.insertRow(["Irregularity Score", f"{results['irregularity_score']:.3f}"])
            cursor.insertRow(["Is Irregular", "Yes" if results['is_irregular'] else "No"])
            cursor.insertRow(["Neighbor Count", str(results['neighbor_count'])])


class PredictTemperatureAnomalies(object):
    """
    Predict temperature anomalies using historical deviation analysis.
    """
    
    def __init__(self):
        """Define the tool."""
        self.label = "Predict Temperature Anomalies"
        self.description = """Detects extreme temperature events using statistical analysis
        of temperature deviations from expected seasonal patterns."""
        self.canRunInBackground = False
    
    def getParameterInfo(self):
        """Define parameter definitions."""
        
        params = []
        
        # Station USAF
        station_usaf = arcpy.Parameter(
            displayName="Weather Station USAF ID",
            name="station_usaf",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        station_usaf.value = "722920"
        station_usaf.description = "USAF identifier of the target weather station"
        params.append(station_usaf)
        
        # Threshold method
        threshold_method = arcpy.Parameter(
            displayName="Threshold Method",
            name="threshold_method",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        threshold_method.filter.type = "ValueList"
        threshold_method.filter.list = ["2_std_dev", "percentile_95", "iqr_method"]
        threshold_method.value = "2_std_dev"
        threshold_method.description = "Statistical method for detecting anomalies"
        params.append(threshold_method)
        
        # Output anomaly feature class
        output_fc = arcpy.Parameter(
            displayName="Output Anomaly Feature Class",
            name="output_fc",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output"
        )
        output_fc.description = "Feature class with daily anomaly classifications"
        params.append(output_fc)
        
        # Output summary
        output_summary = arcpy.Parameter(
            displayName="Output Summary Table",
            name="output_summary",
            datatype="DETable",
            parameterType="Required",
            direction="Output"
        )
        output_summary.description = "Summary statistics for anomaly analysis"
        params.append(output_summary)
        
        return params
    
    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True
    
    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed."""
        pass
    
    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter."""
        pass
    
    def execute(self, parameters, messages):  # type: ignore
        """
        Purpose: Predict temperature anomalies for a weather station
        Parameters: station_usaf, threshold_method, output_fc, output_summary
        Return Value: None
        Exceptions: Raises if analysis fails
        """
        
        station_usaf = parameters[0].valueAsText
        threshold_method = parameters[1].valueAsText
        output_fc = parameters[2].valueAsText
        output_summary = parameters[3].valueAsText
        
        try:
            # Import here to ensure modules are available
            from modules import anomaly_predictor
        except ImportError as e:
            arcpy.AddError(f"Failed to import anomaly_predictor: {e}")
            raise
        
        try:
            arcpy.AddMessage(f"Predicting temperature anomalies for {station_usaf}...")
            
            # Initialize predictor
            predictor = anomaly_predictor.AnomalyPredictor()
            
            # Authenticate
            if not predictor.authenticate():
                raise arcpy.ExecuteError("Authentication required. Run 'Authenticate BigQuery' first.")
            
            # Run full analysis
            arcpy.AddMessage("Running anomaly prediction analysis...")
            results = predictor.run_full_analysis(station_usaf, threshold_method)
            
            # Create outputs
            arcpy.AddMessage("Creating output feature class...")
            self._create_anomaly_feature_class(output_fc, results)
            
            arcpy.AddMessage("Creating summary table...")
            self._create_anomaly_summary(output_summary, results)
            
            arcpy.AddMessage("Anomaly prediction complete!")
            
        except Exception as e:
            arcpy.AddError(f"Error during anomaly prediction: {str(e)}")
            raise
    
    def _create_anomaly_feature_class(self, output_fc, results):
        """Create feature class with daily anomaly data."""
        # This would iterate through daily data and create points
        # For now, create summary points for each station analyzed
        sr = arcpy.SpatialReference(4326)
        
        arcpy.management.CreateFeatureclass(
            os.path.dirname(output_fc),
            os.path.basename(output_fc),
            "POINT",
            spatial_reference=sr
        )
        
        # Add fields
        field_defs = [
            ("usaf", "TEXT"),
            ("classification", "TEXT"),
            ("extreme_pct", "DOUBLE"),
            ("extreme_hot", "LONG"),
            ("extreme_cold", "LONG"),
            ("total_days", "LONG")
        ]
        
        for field_name, field_type in field_defs:
            arcpy.management.AddField(output_fc, field_name, field_type)
        
        # Insert features
        cursor = arcpy.da.InsertCursor(output_fc, ["SHAPE@", "usaf", "classification", 
                                                   "extreme_pct", "extreme_hot", 
                                                   "extreme_cold", "total_days"])
        
        for usaf, data in results['classification_results'].items():
            # Get station location from results
            if 'station_lat' in data:
                point = arcpy.Point(data['station_lon'], data['station_lat'])
                cursor.insertRow([
                    point,
                    usaf,
                    data['station_classification'],
                    data['extreme_percentage'],
                    data['extreme_hot_days'],
                    data['extreme_cold_days'],
                    data['total_days']
                ])
        
        del cursor
    
    def _create_anomaly_summary(self, output_summary, results):
        """Create summary table."""
        arcpy.management.CreateTable(os.path.dirname(output_summary), os.path.basename(output_summary))
        
        arcpy.management.AddField(output_summary, "statistic", "TEXT")
        arcpy.management.AddField(output_summary, "value", "TEXT")
        
        summary = results['analysis_summary']
        
        with arcpy.da.InsertCursor(output_summary, ["statistic", "value"]) as cursor:
            cursor.insertRow(["Total Stations", str(summary['total_stations_analyzed'])])
            cursor.insertRow(["Normal Stations", str(summary['normal_stations'])])
            cursor.insertRow(["Irregular Stations", str(summary['irregular_stations'])])
            cursor.insertRow(["Irregularity Rate", f"{summary['irregularity_rate']:.1f}%"])
            cursor.insertRow(["Avg Extreme Days", f"{summary['average_extreme_percentage']:.1f}%"])


class GenerateStationMap(object):
    """
    Generate interactive map visualization of weather stations.
    """
    
    def __init__(self):
        """Define the tool."""
        self.label = "Generate Station Map"
        self.description = """Creates an interactive HTML map showing weather stations with analysis results."""
        self.canRunInBackground = False
    
    def getParameterInfo(self):
        """Define parameter definitions."""
        params = []
        
        # Input results (from previous tools)
        results_json = arcpy.Parameter(
            displayName="Analysis Results (JSON)",
            name="results_json",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        results_json.description = "JSON string with spatial clustering results"
        params.append(results_json)
        
        # Output HTML
        output_html = arcpy.Parameter(
            displayName="Output HTML File",
            name="output_html",
            datatype="DEFile",
            parameterType="Required",
            direction="Output"
        )
        output_html.description = "Interactive map HTML file"
        params.append(output_html)
        
        return params
    
    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True
    
    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed."""
        pass
    
    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter."""
        pass
    
    def execute(self, parameters, messages):  # type: ignore
        """
        Purpose: Generate interactive station map
        Parameters: results_json, output_html
        Return Value: None
        Exceptions: Raises if visualization fails
        """
        
        try:
            # Import here to ensure modules are available
            from modules import visualizer
        except ImportError as e:
            arcpy.AddError(f"Failed to import visualizer: {e}")
            raise
        
        results_json = parameters[0].valueAsText
        output_html = parameters[1].valueAsText
        
        try:
            arcpy.AddMessage("Generating station map...")
            
            # Parse results
            results = json.loads(results_json)
            
            # Initialize visualizer
            vis = visualizer.WeatherVisualizer()
            
            # Create map
            fig = vis.create_station_map(results)
            
            # Save
            vis.save_visualization(fig, output_html, 'html')
            
            arcpy.AddMessage(f"Map saved to: {output_html}")
            
        except Exception as e:
            arcpy.AddError(f"Error generating map: {str(e)}")
            raise


class GenerateTemperatureGraph(object):
    """
    Generate temperature analysis graph visualization.
    """
    
    def __init__(self):
        """Define the tool."""
        self.label = "Generate Temperature Graph"
        self.description = """Creates an interactive temperature analysis graph."""
        self.canRunInBackground = False
    
    def getParameterInfo(self):
        """Define parameter definitions."""
        params = []
        
        results_json = arcpy.Parameter(
            displayName="Analysis Results (JSON)",
            name="results_json",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        results_json.description = "JSON string with analysis results"
        params.append(results_json)
        
        output_html = arcpy.Parameter(
            displayName="Output HTML File",
            name="output_html",
            datatype="DEFile",
            parameterType="Required",
            direction="Output"
        )
        output_html.description = "Temperature graph HTML file"
        params.append(output_html)
        
        return params
    
    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True
    
    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed."""
        pass
    
    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter."""
        pass
    
    def execute(self, parameters, messages):  # type: ignore
        """
        Purpose: Generate temperature graph
        Parameters: results_json, output_html
        Return Value: None
        Exceptions: Raises if visualization fails
        """
        
        try:
            # Import here to ensure modules are available
            from modules import visualizer
        except ImportError as e:
            arcpy.AddError(f"Failed to import visualizer: {e}")
            raise
        
        results_json = parameters[0].valueAsText
        output_html = parameters[1].valueAsText
        
        try:
            arcpy.AddMessage("Generating temperature graph...")
            
            # Parse results
            results = json.loads(results_json)
            
            # Initialize visualizer
            vis = visualizer.WeatherVisualizer()
            
            # Create graph
            fig = vis.create_temperature_graph(results)
            
            # Save
            vis.save_visualization(fig, output_html, 'html')
            
            arcpy.AddMessage(f"Graph saved to: {output_html}")
            
        except Exception as e:
            arcpy.AddError(f"Error generating graph: {str(e)}")
            raise


class FullAnalysisWorkflow(object):
    """
    Complete workflow combining spatial clustering and anomaly prediction.
    """
    
    def __init__(self):
        """Define the tool."""
        self.label = "Full Analysis Workflow"
        self.description = """Runs complete analysis including spatial clustering and anomaly prediction."""
        self.canRunInBackground = False
    
    def getParameterInfo(self):
        """Define parameter definitions."""
        params = []
        
        # Station USAF
        station_usaf = arcpy.Parameter(
            displayName="Weather Station USAF ID",
            name="station_usaf",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        station_usaf.value = "722920"
        params.append(station_usaf)
        
        # Output directory
        output_dir = arcpy.Parameter(
            displayName="Output Directory",
            name="output_dir",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input"
        )
        output_dir.description = "Directory for output files"
        params.append(output_dir)
        
        return params
    
    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True
    
    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed."""
        pass
    
    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter."""
        pass
    
    def execute(self, parameters, messages):  # type: ignore
        """
        Purpose: Run full analysis workflow
        Parameters: station_usaf, output_dir
        Return Value: None
        Exceptions: Raises if workflow fails
        """
        
        station_usaf = parameters[0].valueAsText
        output_dir = parameters[1].valueAsText
        
        try:
            # Import required modules
            from modules import spatial_clustering, anomaly_predictor, visualizer
            
            arcpy.AddMessage("Running full analysis workflow...")
            arcpy.AddMessage(f"Analyzing station: {station_usaf}")
            
            # Initialize components
            analyzer = spatial_clustering.SpatialClusteringAnalyzer()
            predictor = anomaly_predictor.AnomalyPredictor()
            vis = visualizer.WeatherVisualizer()
            
            # Authenticate
            if not analyzer.authenticate():
                raise arcpy.ExecuteError("Authentication required. Run 'Authenticate BigQuery' first.")
            
            # Load stations data
            arcpy.AddMessage("Loading weather stations data...")
            analyzer.load_stations_data()
            
            # Run spatial clustering analysis
            arcpy.AddMessage("Running spatial clustering analysis...")
            spatial_results = analyzer.analyze_station_irregularity(station_usaf)
            
            # Run anomaly prediction
            arcpy.AddMessage("Running anomaly prediction...")
            anomaly_results = predictor.run_full_analysis(station_usaf)
            
            # Create visualizations
            arcpy.AddMessage("Generating visualizations...")
            output_file = os.path.join(output_dir, f"analysis_{station_usaf}.html")
            
            # Generate combined visualization
            fig, _ = vis.create_combined_visualization(spatial_results, anomaly_results, station_usaf)
            vis.save_visualization(fig, output_file, 'html')
            
            arcpy.AddMessage(f"Analysis complete! Output saved to: {output_file}")
            
        except Exception as e:
            arcpy.AddError(f"Error during workflow: {str(e)}")
            raise

