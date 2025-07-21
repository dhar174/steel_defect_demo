"""
Stream Analytics Engine for advanced real-time analytics on streaming sensor data.

This module provides the StreamAnalyticsEngine class that performs:
- Statistical Process Control (SPC) checks
- Trend detection using linear regression
- Multivariate anomaly detection using IsolationForest
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from scipy import stats
from typing import Dict, List, Optional, Tuple
import warnings


class StreamAnalyticsEngine:
    """Performs advanced real-time analytics on streaming data."""
    
    def __init__(self, config: Dict):
        """
        Initializes the stream analytics engine.
        
        Args:
            config (Dict): The application configuration.
        """
        self.config = config
        self.anomaly_detector = IsolationForest(contamination='auto', random_state=42)
        self.control_limits: Dict[str, Dict[str, float]] = {}
        self._anomaly_detector_fitted = False
        
        # Configuration parameters
        self.spc_sigma_threshold = config.get('spc_sigma_threshold', 3.0)
        self.trend_min_points = config.get('trend_min_points', 10)
        self.trend_significance_level = config.get('trend_significance_level', 0.05)
        
    def update_with_new_data(self, data_buffer: pd.DataFrame) -> Dict:
        """
        Updates the engine with a new data buffer and runs all analytics.
        
        Args:
            data_buffer (pd.DataFrame): The latest sliding window of sensor data.
        
        Returns:
            A dictionary containing the results of the analytics.
        """
        if data_buffer.empty:
            return {
                'spc_violations': {},
                'trends': {},
                'anomalies': [],
                'anomaly_scores': [],
                'summary': {
                    'total_points': 0,
                    'spc_violations_count': 0,
                    'trends_count': 0,
                    'anomalies_count': 0
                }
            }
        
        # Update control limits if needed
        self._update_control_limits(data_buffer)
        
        # Run analytics
        spc_violations = self.check_spc(data_buffer)
        trends = self.detect_trends(data_buffer)
        anomaly_scores = self.detect_anomalies(data_buffer)
        
        # Process anomaly results
        anomalies = []
        if len(anomaly_scores) > 0:
            anomalies = [i for i, score in enumerate(anomaly_scores) if score == -1]
        
        # Create summary
        summary = {
            'total_points': len(data_buffer),
            'spc_violations_count': len(spc_violations),
            'trends_count': len(trends),
            'anomalies_count': len(anomalies)
        }
        
        return {
            'spc_violations': spc_violations,
            'trends': trends,
            'anomalies': anomalies,
            'anomaly_scores': anomaly_scores.tolist() if anomaly_scores.size > 0 else [],
            'summary': summary
        }
    
    def check_spc(self, data_buffer: pd.DataFrame) -> Dict[str, str]:
        """
        Performs statistical process control checks (e.g., Western Electric rules).
        
        Args:
            data_buffer (pd.DataFrame): The data buffer to analyze.
        
        Returns:
            A dictionary of signals that are out of control.
        """
        violations = {}
        
        # Get numeric columns only
        numeric_columns = data_buffer.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            series = data_buffer[column].dropna()
            if len(series) == 0:
                continue
                
            # Get or calculate control limits for this column
            if column not in self.control_limits:
                self._calculate_control_limits(column, series)
            
            limits = self.control_limits[column]
            
            # Check for points outside control limits (Rule 1: Beyond 3 sigma)
            out_of_control_points = []
            
            for i, value in enumerate(series):
                if value > limits['upper_control_limit'] or value < limits['lower_control_limit']:
                    out_of_control_points.append(i)
            
            if out_of_control_points:
                violations[column] = f"Points {out_of_control_points} exceed {self.spc_sigma_threshold}-sigma control limits"
            
            # Additional SPC rule: 2 out of 3 consecutive points beyond 2 sigma
            two_sigma_upper = limits['mean'] + 2 * limits['std']
            two_sigma_lower = limits['mean'] - 2 * limits['std']
            
            consecutive_beyond_2sigma = 0
            for i in range(len(series) - 2):
                beyond_2sigma_count = 0
                for j in range(3):
                    value = series.iloc[i + j]
                    if value > two_sigma_upper or value < two_sigma_lower:
                        beyond_2sigma_count += 1
                
                if beyond_2sigma_count >= 2:
                    if column not in violations:
                        violations[column] = [f"2 out of 3 consecutive points beyond 2-sigma starting at index {i}"]
                    else:
                        violations[column].append(f"2 out of 3 consecutive points beyond 2-sigma starting at index {i}")
        
        return {column: "; ".join(messages) for column, messages in violations.items()}
    
    def detect_trends(self, data_buffer: pd.DataFrame) -> Dict[str, str]:
        """
        Detects gradual trends or drifts in sensor readings.
        
        Args:
            data_buffer (pd.DataFrame): The data buffer to analyze.
        
        Returns:
            A dictionary of sensors exhibiting significant trends.
        """
        trends = {}
        
        # Get numeric columns only
        numeric_columns = data_buffer.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            series = data_buffer[column].dropna()
            if len(series) < self.trend_min_points:
                continue
            
            # Create time index for regression
            X = np.arange(len(series)).reshape(-1, 1)
            y = series.values
            
            # Fit linear regression
            try:
                reg = LinearRegression()
                reg.fit(X, y)
                slope = reg.coef_[0]
                
                # Calculate p-value for slope significance
                y_pred = reg.predict(X)
                residuals = y - y_pred
                mse = np.mean(residuals**2)
                
                # Calculate standard error of slope
                n = len(X)
                x_mean = np.mean(X)
                ss_x = np.sum((X.flatten() - x_mean)**2)
                
                if ss_x > 0 and mse > 0:
                    se_slope = np.sqrt(mse / ss_x)
                    
                    # t-statistic and p-value
                    if se_slope > 0:
                        t_stat = slope / se_slope
                        df = n - 2
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                        
                        # Check if trend is significant
                        if p_value < self.trend_significance_level:
                            direction = "increasing" if slope > 0 else "decreasing"
                            trends[column] = f"Significant {direction} trend (slope={slope:.6f}, p={p_value:.4f})"
                            
            except Exception as e:
                # Skip this column if regression fails
                warnings.warn(f"Trend detection failed for column {column}: {e}")
                continue
        
        return trends
    
    def detect_anomalies(self, data_buffer: pd.DataFrame) -> np.ndarray:
        """
        Uses an Isolation Forest to detect multivariate anomalies.
        
        Args:
            data_buffer (pd.DataFrame): The data buffer to analyze.
        
        Returns:
            An array of anomaly scores for the data points in the buffer.
        """
        # Get numeric columns only
        numeric_data = data_buffer.select_dtypes(include=[np.number])
        
        if numeric_data.empty or len(numeric_data.columns) == 0:
            return np.array([])
        
        # Remove rows with any NaN values
        clean_data = numeric_data.dropna()
        
        if len(clean_data) == 0:
            return np.array([])
        
        # Fit the model if not already fitted or if we have enough new data
        if not self._anomaly_detector_fitted and len(clean_data) >= 10:
            try:
                self.anomaly_detector.fit(clean_data)
                self._anomaly_detector_fitted = True
            except Exception as e:
                warnings.warn(f"Failed to fit anomaly detector: {e}")
                return np.array([])
        
        # Predict anomalies if model is fitted
        if self._anomaly_detector_fitted:
            try:
                predictions = self.anomaly_detector.predict(clean_data)
                return predictions
            except Exception as e:
                warnings.warn(f"Failed to predict anomalies: {e}")
                return np.array([])
        else:
            # Not enough data to fit model yet
            return np.array([])
    
    def _update_control_limits(self, data_buffer: pd.DataFrame):
        """Update control limits based on current data buffer."""
        numeric_columns = data_buffer.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            series = data_buffer[column].dropna()
            if len(series) > 0:
                self._calculate_control_limits(column, series)
    
    def _calculate_control_limits(self, column: str, series: pd.Series):
        """Calculate control limits for a given column."""
        mean_val = series.mean()
        std_val = series.std()
        
        if std_val < 1e-6:  # Handle zero or near-zero standard deviation
            self.control_limits[column] = {
                'mean': mean_val,
                'std': std_val,
                'upper_control_limit': mean_val,  # No variability, limits equal to mean
                'lower_control_limit': mean_val
            }
        else:
            self.control_limits[column] = {
                'mean': mean_val,
                'std': std_val,
                'upper_control_limit': mean_val + self.spc_sigma_threshold * std_val,
                'lower_control_limit': mean_val - self.spc_sigma_threshold * std_val
            }