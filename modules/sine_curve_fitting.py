"""
Sinusoidal Curve Fitting Module

Ports the sine curve fitting functions from the Colab notebook to standalone module.
Handles fitting sinusoidal curves to temperature data and extracting parameters.
"""

import pandas as pd
import numpy as np
import scipy.optimize
from typing import Dict, Union, Tuple
import warnings
warnings.filterwarnings('ignore')


def sin_function(t: np.ndarray, amp: float, freq: float, phase_shift: float, mean: float) -> np.ndarray:
    """
    Sinusoidal function for temperature modeling.
    
    Ported from notebook Cell 18 - describes the sinusoidal model as function with parameters.
    
    Args:
        t: Time array (days since start)
        amp: Amplitude of the sine wave
        freq: Frequency of the sine wave
        phase_shift: Phase shift parameter
        mean: Mean temperature value
        
    Returns:
        np.ndarray: Predicted temperature values
    """
    return amp * np.sin(freq * 2 * np.pi * (t - phase_shift)) + mean


def fit_sine_curve_to_daily_temp_data(daily_temp_data: pd.DataFrame, 
                                     temp_field_name: str,
                                     return_value: str = 'sine curve fit info') -> Union[pd.DataFrame, Dict]:
    """
    Fit sinusoidal curve to daily temperature data.
    
    Ported from notebook Cell 18 - fits the sinusoidal model to data and returns
    either fit info or daily temp estimates.
    
    Args:
        daily_temp_data: DataFrame with 'date' and temperature columns
        temp_field_name: Name of temperature field to fit (e.g., 'avg_temp')
        return_value: What to return ('sine curve fit info' or 'daily temp data with estimates')
        
    Returns:
        Union[pd.DataFrame, Dict]: Either fit parameters DataFrame or data with estimates
    """
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(daily_temp_data['date']):
        daily_temp_data = daily_temp_data.copy()
        daily_temp_data['date'] = pd.to_datetime(daily_temp_data['date'])
    
    # Calculate days since start
    daily_temp_data = daily_temp_data.copy()
    daily_temp_data['data_since_start'] = (
        daily_temp_data['date'] - min(daily_temp_data['date'])
    ).dt.days
    
    # Starting point for mean is mean of temperature
    guess_mean = daily_temp_data[temp_field_name].mean()
    
    # Starting point for amplitude is half the difference between 1st and 99th percentiles
    guess_amp = (daily_temp_data[temp_field_name].quantile(0.99) - 
                 daily_temp_data[temp_field_name].quantile(0.01)) / 2
    
    # Starting point for frequency is inverse of average number of days in year
    guess_freq = 1 / 365.25
    
    # Starting point for phase shift is +80 days into spring (typically)
    guess_phase_shift = 80
    
    # Use curve fit optimizer with above guesses as starting point
    try:
        sine_curve_fit = scipy.optimize.curve_fit(
            f=sin_function,
            xdata=np.array(daily_temp_data['data_since_start']),
            ydata=np.array(daily_temp_data[temp_field_name]),
            p0=[guess_amp, guess_freq, guess_phase_shift, guess_mean],
            maxfev=10000  # Increase max function evaluations for better convergence
        )
        
        # Extract estimated parameters from curve fit
        est_amp, est_freq, est_phase_shift, est_mean = sine_curve_fit[0]
        
    except Exception as e:
        print(f"Warning: Curve fitting failed for {temp_field_name}: {e}")
        # Use initial guesses if fitting fails
        est_amp, est_freq, est_phase_shift, est_mean = (
            guess_amp, guess_freq, guess_phase_shift, guess_mean
        )
    
    # Use the sine function and parameters to get daily estimates of temperature
    daily_temp_data[f'est_{temp_field_name}'] = sin_function(
        daily_temp_data['data_since_start'],
        est_amp, est_freq, est_phase_shift, est_mean
    )
    
    # Calculate mean absolute error of estimates vs actual temperature
    curve_estimate_mean_abs_err = abs(
        daily_temp_data[f'est_{temp_field_name}'] - daily_temp_data[temp_field_name]
    ).mean()
    
    # Create data frame of the sine curve fit info
    sine_curve_fit_info_df = pd.DataFrame(data=[{
        f'est_amp_{temp_field_name}': est_amp,
        f'est_freq_{temp_field_name}': est_freq,
        f'est_phase_shift_{temp_field_name}': est_phase_shift,
        f'est_mean_{temp_field_name}': est_mean,
        f'est_range_{temp_field_name}': 2 * abs(est_amp),
        f'mae_fitted_{temp_field_name}': curve_estimate_mean_abs_err
    }])
    
    # Return either sine curve fit info or daily temp data with estimates
    if return_value == 'sine curve fit info':
        return sine_curve_fit_info_df
    elif return_value == 'daily temp data with estimates':
        return daily_temp_data
    else:
        raise ValueError("return_value must be 'sine curve fit info' or 'daily temp data with estimates'")


def extract_sine_curve_parameters(fit_info_df: pd.DataFrame, temp_field_name: str) -> Dict[str, float]:
    """
    Extract sine curve parameters from fit info DataFrame.
    
    Args:
        fit_info_df: DataFrame with sine curve fit information
        temp_field_name: Name of temperature field
        
    Returns:
        Dict[str, float]: Dictionary with parameter names and values
    """
    params = {}
    
    # Standard parameters
    param_mappings = {
        'amp': f'est_amp_{temp_field_name}',
        'freq': f'est_freq_{temp_field_name}',
        'phase_shift': f'est_phase_shift_{temp_field_name}',
        'mean': f'est_mean_{temp_field_name}',
        'range': f'est_range_{temp_field_name}',
        'mae': f'mae_fitted_{temp_field_name}'  # Special case - no 'est_' prefix
    }
    
    for param_name, col_name in param_mappings.items():
        if col_name in fit_info_df.columns:
            params[param_name] = fit_info_df[col_name].iloc[0]
    
    return params


def calculate_sine_curve_metrics(actual_temps: np.ndarray, 
                                predicted_temps: np.ndarray) -> Dict[str, float]:
    """
    Calculate additional metrics for sine curve fit quality.
    
    Args:
        actual_temps: Actual temperature values
        predicted_temps: Predicted temperature values from sine curve
        
    Returns:
        Dict[str, float]: Dictionary with various fit metrics
    """
    residuals = actual_temps - predicted_temps
    
    metrics = {
        'mae': np.mean(np.abs(residuals)),  # Mean Absolute Error
        'rmse': np.sqrt(np.mean(residuals**2)),  # Root Mean Square Error
        'r_squared': 1 - (np.sum(residuals**2) / np.sum((actual_temps - np.mean(actual_temps))**2)),
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals)
    }
    
    return metrics


def validate_sine_curve_fit(daily_temp_data: pd.DataFrame, 
                           temp_field_name: str,
                           min_data_points: int = 365) -> bool:
    """
    Validate that data is suitable for sine curve fitting.
    
    Args:
        daily_temp_data: DataFrame with temperature data
        temp_field_name: Name of temperature field
        min_data_points: Minimum number of data points required
        
    Returns:
        bool: True if data is suitable for fitting
    """
    if len(daily_temp_data) < min_data_points:
        return False
    
    if temp_field_name not in daily_temp_data.columns:
        return False
    
    # Check for sufficient non-null values
    non_null_count = daily_temp_data[temp_field_name].notna().sum()
    if non_null_count < min_data_points * 0.8:  # At least 80% non-null
        return False
    
    # Check for reasonable temperature range (not all same value)
    temp_range = daily_temp_data[temp_field_name].max() - daily_temp_data[temp_field_name].min()
    if temp_range < 5:  # At least 5 degree range
        return False
    
    return True


class SineCurveFitter:
    """
    Class-based interface for sine curve fitting operations.
    """
    
    def __init__(self):
        """Initialize the sine curve fitter."""
        self.fitted_curves = {}
    
    def fit_curve(self, daily_temp_data: pd.DataFrame, 
                  temp_field_name: str = 'avg_temp',
                  station_id: str = None) -> Dict[str, float]:
        """
        Fit sine curve to temperature data and store results.
        
        Args:
            daily_temp_data: DataFrame with temperature data
            temp_field_name: Name of temperature field
            station_id: Optional station identifier for storing results
            
        Returns:
            Dict[str, float]: Sine curve parameters
        """
        if not validate_sine_curve_fit(daily_temp_data, temp_field_name):
            raise ValueError(f"Data validation failed for {temp_field_name}")
        
        # Fit the curve
        fit_info = fit_sine_curve_to_daily_temp_data(
            daily_temp_data, temp_field_name, 'sine curve fit info'
        )
        
        # Extract parameters
        params = extract_sine_curve_parameters(fit_info, temp_field_name)
        
        # Store results if station_id provided
        if station_id:
            self.fitted_curves[station_id] = {
                'parameters': params,
                'fit_info': fit_info,
                'temp_field': temp_field_name
            }
        
        return params
    
    def predict_temperatures(self, dates: pd.Series, 
                           parameters: Dict[str, float]) -> np.ndarray:
        """
        Predict temperatures for given dates using fitted parameters.
        
        Args:
            dates: Series of dates
            parameters: Sine curve parameters
            
        Returns:
            np.ndarray: Predicted temperature values
        """
        if not pd.api.types.is_datetime64_any_dtype(dates):
            dates = pd.to_datetime(dates)
        
        # Calculate days since first date
        days_since_start = (dates - dates.min()).dt.days
        
        # Predict using sine function
        predicted = sin_function(
            days_since_start.values,
            parameters['amp'],
            parameters['freq'],
            parameters['phase_shift'],
            parameters['mean']
        )
        
        return predicted
    
    def get_station_parameters(self, station_id: str) -> Dict[str, float]:
        """
        Get stored parameters for a station.
        
        Args:
            station_id: Station identifier
            
        Returns:
            Dict[str, float]: Stored parameters or None if not found
        """
        if station_id in self.fitted_curves:
            return self.fitted_curves[station_id]['parameters']
        return None
