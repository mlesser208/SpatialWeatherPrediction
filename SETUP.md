# Setup Guide

This guide provides detailed instructions for setting up the Spatial Weather Prediction Tool.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Google Cloud Setup](#google-cloud-setup)
3. [Local Environment Setup](#local-environment-setup)
4. [Authentication Configuration](#authentication-configuration)
5. [Testing the Installation](#testing-the-installation)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Operating System**: Windows 10/11, macOS 10.14+, or Linux
- **Python**: Version 3.8 or higher
- **Memory**: At least 4GB RAM (8GB recommended)
- **Storage**: 2GB free space for dependencies and outputs
- **Internet**: Required for BigQuery data access

### Software Requirements

- Python 3.8+
- pip (Python package manager)
- Git (for cloning the repository)
- (Optional) ArcGIS Pro for toolbox integration

## Google Cloud Setup

### Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" → "New Project"
3. Enter project name: `spatial-weather-prediction`
4. Click "Create"
5. Note your Project ID (e.g., `spatial-weather-prediction-123456`)

### Step 2: Enable Required APIs

1. In the Google Cloud Console, go to "APIs & Services" → "Library"
2. Search for and enable these APIs:
   - **BigQuery API**
   - **BigQuery Storage API** (for faster data access)

### Step 3: Set Up BigQuery Dataset

1. Go to [BigQuery Console](https://console.cloud.google.com/bigquery)
2. The NOAA GSOD dataset is already available as a public dataset
3. No additional setup required - the tool will access `bigquery-public-data.noaa_gsod`

### Step 4: Create Service Account (Recommended)

1. Go to "IAM & Admin" → "Service Accounts"
2. Click "Create Service Account"
3. Enter details:
   - **Name**: `spatial-weather-tool`
   - **Description**: `Service account for spatial weather prediction tool`
4. Click "Create and Continue"
5. Grant roles:
   - **BigQuery Data Viewer**
   - **BigQuery Job User**
6. Click "Continue" → "Done"

### Step 5: Create Service Account Key

1. Click on your service account
2. Go to "Keys" tab
3. Click "Add Key" → "Create new key"
4. Choose "JSON" format
5. Click "Create"
6. **Important**: Save the downloaded JSON file securely
7. **Never commit this file to version control**

## Local Environment Setup

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd SpatialWeatherPrediction
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import pandas, numpy, scipy, plotly; print('All dependencies installed successfully')"
```

## Authentication Configuration

### Option 1: Service Account Key (Recommended)

1. Place your service account JSON key file in a secure location
2. Set environment variable:

**Windows (Command Prompt):**
```cmd
set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\your\service-account-key.json
```

**Windows (PowerShell):**
```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\service-account-key.json"
```

**macOS/Linux:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
```

### Option 2: User Authentication

```bash
gcloud auth application-default login
```

### Option 3: Programmatic Authentication

```python
# In your code
from modules.bigquery_connector import BigQueryConnector

connector = BigQueryConnector()
connector.authenticate(credentials_path="path/to/your/key.json")
```

## Testing the Installation

### Step 1: Run Integration Tests

```bash
python test_integration.py
```

Expected output:
```
=== Spatial Weather Prediction Tool - Integration Tests ===
Testing module integration without BigQuery authentication

=== Testing Module Imports ===
✅ BigQueryConnector imported successfully
✅ SineCurveFitter and functions imported successfully
...
Tests passed: 8/8
🎉 All integration tests passed! Modules are working correctly together.
```

### Step 2: Test BigQuery Connection

```bash
python test_spatial_clustering.py
```

Expected output:
```
=== Spatial Weather Prediction Tool - Test Script ===
Testing spatial clustering with Catalina Airport (USAF: 722920)

Step 1: Testing BigQuery authentication...
✅ Authentication successful!
...
🎉 All tests passed! Spatial clustering module is working correctly.
```

### Step 3: Verify Outputs

Check that the `outputs/` directory is created and contains:
- `analysis_report_722920.html` - Analysis report
- Any generated visualizations

## Configuration

### Update Project ID

Edit `config/config.yaml`:

```yaml
# Google Cloud Project Settings
project_id: "your-project-id-here"  # Replace with your actual project ID
```

### Customize Analysis Parameters

```yaml
# Spatial Clustering for Regional Irregularity Detection
spatial_clustering:
  neighbor_count: 10                    # Number of closest neighbors
  irregularity_threshold: 1.0            # Standard deviations for flagging
  time_period:
    start_year: 2005                    # Analysis start year
    end_year: 2025                      # Analysis end year
  test_station: "722920"                # Catalina Airport USAF for testing
```

## Troubleshooting

### Common Issues

#### 1. Authentication Errors

**Error**: `Authentication error: Reauthentication is needed`

**Solution**:
```bash
gcloud auth application-default login
```

#### 2. Import Errors

**Error**: `ModuleNotFoundError: No module named 'scipy'`

**Solution**:
```bash
pip install -r requirements.txt
```

#### 3. BigQuery Permission Errors

**Error**: `Access Denied: BigQuery BigQuery: Permission denied`

**Solution**:
- Ensure your service account has BigQuery Data Viewer role
- Check that the project ID in config.yaml is correct
- Verify the service account key is valid

#### 4. Data Loading Errors

**Error**: `Station 722920 not found in stations data`

**Solution**:
- Check that the station USAF ID exists in NOAA GSOD dataset
- Verify your date range parameters
- Ensure BigQuery API is enabled

#### 5. Memory Issues

**Error**: `MemoryError` or slow performance

**Solution**:
- Reduce the analysis date range
- Process fewer stations at once
- Increase system memory or use a more powerful machine

### Getting Help

1. **Check the logs**: Look for detailed error messages in the console output
2. **Run integration tests**: `python test_integration.py`
3. **Verify configuration**: Check `config/config.yaml` settings
4. **Test BigQuery access**: Try running a simple query in BigQuery Console
5. **Check dependencies**: Ensure all packages are installed correctly

### Support

If you continue to have issues:

1. Check the [GitHub Issues](https://github.com/your-repo/issues) page
2. Review the troubleshooting section in `README.md`
3. Ensure you're following the setup steps exactly
4. Verify your Google Cloud project configuration

## Next Steps

Once setup is complete:

1. **Read the Usage Guide**: See `USAGE.md` for detailed usage instructions
2. **Try the Examples**: Run the provided example scripts
3. **Customize Analysis**: Modify configuration parameters for your needs
4. **Create ArcGIS Toolbox**: Follow the toolbox creation guide

## Security Notes

- **Never commit credentials**: Add `*.json` to `.gitignore`
- **Use service accounts**: Prefer service accounts over user authentication
- **Rotate keys regularly**: Update service account keys periodically
- **Limit permissions**: Only grant necessary BigQuery roles
- **Monitor usage**: Check BigQuery usage to avoid unexpected costs
