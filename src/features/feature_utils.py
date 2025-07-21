"""
Utility functions for feature engineering operations.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List


def calculate_percentiles(series: pd.Series, percentiles: List[float]) -> Dict[str, float]:
    """
    Calculate multiple percentiles efficiently
    
    Args:
        series: Input data series
        percentiles: List of percentile values to calculate
        
    Returns:
        Dictionary with percentile values
    """
    if series.empty or series.isna().all():
        return {f'p{int(p)}': np.nan for p in percentiles}
    
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return {f'p{int(p)}': np.nan for p in percentiles}
    
    result = {}
    for p in percentiles:
        try:
            result[f'p{int(p)}'] = np.percentile(clean_series, p)
        except:
            result[f'p{int(p)}'] = np.nan
    
    return result


def detect_spikes(series: pd.Series, threshold: float = 2.0) -> np.ndarray:
    """
    Detect spikes using z-score method
    
    Args:
        series: Input data series
        threshold: Z-score threshold for spike detection
        
    Returns:
        Boolean array indicating spike locations
    """
    if series.empty or series.isna().all():
        return np.array([])
    
    clean_series = series.dropna()
    if len(clean_series) < 2:
        return np.array([False] * len(clean_series))
    
    mean_val = clean_series.mean()
    std_val = clean_series.std()
    
    if std_val == 0:
        return np.array([False] * len(clean_series))
    
    z_scores = np.abs((clean_series - mean_val) / std_val)
    return z_scores > threshold


def count_threshold_crossings(series: pd.Series, threshold: float) -> int:
    """
    Count number of times series crosses a threshold
    
    Args:
        series: Input data series
        threshold: Threshold value
        
    Returns:
        Number of threshold crossings
    """
    if series.empty or len(series) < 2:
        return 0
    
    clean_series = series.dropna()
    if len(clean_series) < 2:
        return 0
    
    crossings = 0
    for i in range(1, len(clean_series)):
        prev_val = clean_series.iloc[i-1]
        curr_val = clean_series.iloc[i]
        
        # Check if crossing threshold in either direction
        if ((prev_val <= threshold < curr_val) or 
            (prev_val >= threshold > curr_val)):
            crossings += 1
    
    return crossings


def calculate_trend_slope(series: pd.Series) -> float:
    """
    Calculate linear trend slope using least squares
    
    Args:
        series: Input data series
        
    Returns:
        Slope of linear trend
    """
    if series.empty or len(series) < 2:
        return 0.0
    
    clean_series = series.dropna()
    if len(clean_series) < 2:
        return 0.0
    
    x = np.arange(len(clean_series))
    
    try:
        slope, _, _, _, _ = stats.linregress(x, clean_series)
        return slope if not np.isnan(slope) else 0.0
    except:
        return 0.0


def safe_correlation(x: pd.Series, y: pd.Series) -> float:
    """
    Calculate correlation with NaN handling
    
    Args:
        x: First series
        y: Second series
        
    Returns:
        Correlation coefficient (0 if cannot calculate)
    """
    if x.empty or y.empty or len(x) != len(y):
        return 0.0
    
    # Remove pairs where either value is NaN
    combined = pd.DataFrame({'x': x, 'y': y}).dropna()
    
    if len(combined) < 2:
        return 0.0
    
    x_clean = combined['x']
    y_clean = combined['y']
    
    # Check for constant series
    if x_clean.std() == 0 or y_clean.std() == 0:
        return 0.0
    
    try:
        corr = np.corrcoef(x_clean, y_clean)[0, 1]
        return corr if not np.isnan(corr) else 0.0
    except:
        return 0.0


def calculate_mean_gradient(series: pd.Series) -> float:
    """
    Calculate mean absolute gradient between consecutive points
    
    Args:
        series: Input data series
        
    Returns:
        Mean absolute gradient
    """
    if series.empty or len(series) < 2:
        return 0.0
    
    clean_series = series.dropna()
    if len(clean_series) < 2:
        return 0.0
    
    gradients = np.abs(np.diff(clean_series))
    return np.mean(gradients)


def calculate_stability_metrics(series: pd.Series, 
                              spike_threshold: float = 2.0) -> Dict[str, float]:
    """
    Calculate multiple stability metrics efficiently
    
    Args:
        series: Input data series
        spike_threshold: Threshold for spike detection
        
    Returns:
        Dictionary with stability metrics
    """
    if series.empty or series.isna().all():
        return {
            'spike_count': np.nan,
            'excursion_freq': np.nan,
            'cv': np.nan,
            'range_ratio': np.nan
        }
    
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return {
            'spike_count': np.nan,
            'excursion_freq': np.nan,
            'cv': np.nan,
            'range_ratio': np.nan
        }
    
    mean_val = clean_series.mean()
    std_val = clean_series.std()
    min_val = clean_series.min()
    max_val = clean_series.max()
    
    metrics = {}
    
    # Spike count
    if std_val > 0:
        spikes = detect_spikes(clean_series, spike_threshold)
        metrics['spike_count'] = spikes.sum()
    else:
        metrics['spike_count'] = 0
    
    # Excursion frequency
    if std_val > 0:
        upper_bound = mean_val + std_val
        lower_bound = mean_val - std_val
        crossings = (count_threshold_crossings(clean_series, upper_bound) + 
                    count_threshold_crossings(clean_series, lower_bound))
        metrics['excursion_freq'] = crossings / len(clean_series)
    else:
        metrics['excursion_freq'] = 0
    
    # Coefficient of variation
    if mean_val != 0:
        metrics['cv'] = std_val / abs(mean_val)
    else:
        metrics['cv'] = np.nan if std_val > 0 else 0
    
    # Range ratio
    range_val = max_val - min_val
    if mean_val != 0:
        metrics['range_ratio'] = range_val / abs(mean_val)
    else:
        metrics['range_ratio'] = np.nan if range_val > 0 else 0
    
    return metrics


def validate_sensor_data(df: pd.DataFrame, 
                        sensor_columns: List[str],
                        min_data_points: int = 50) -> Dict[str, bool]:
    """
    Validate sensor data quality
    
    Args:
        df: Input DataFrame
        sensor_columns: List of expected sensor columns
        min_data_points: Minimum required data points
        
    Returns:
        Dictionary with validation results
    """
    validation = {}
    
    # Check basic structure
    validation['has_data'] = not df.empty
    validation['sufficient_length'] = len(df) >= min_data_points
    
    # Check each sensor
    for sensor in sensor_columns:
        if sensor in df.columns:
            sensor_data = df[sensor].dropna()
            validation[f'{sensor}_present'] = True
            validation[f'{sensor}_sufficient_data'] = len(sensor_data) >= min_data_points * 0.5
            validation[f'{sensor}_not_constant'] = sensor_data.std() > 1e-10 if len(sensor_data) > 1 else False
        else:
            validation[f'{sensor}_present'] = False
            validation[f'{sensor}_sufficient_data'] = False
            validation[f'{sensor}_not_constant'] = False
    
    return validation