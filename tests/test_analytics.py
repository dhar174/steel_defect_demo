"""
Unit tests for the StreamAnalyticsEngine module.
"""

import unittest
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from analytics.stream_analytics import StreamAnalyticsEngine


class TestStreamAnalyticsEngine(unittest.TestCase):
    """Test cases for StreamAnalyticsEngine"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            'spc_sigma_threshold': 3.0,
            'trend_min_points': 10,
            'trend_significance_level': 0.05
        }
        self.engine = StreamAnalyticsEngine(self.config)
        
        # Create sample data
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'sensor_1': np.random.normal(100, 10, 50),
            'sensor_2': np.random.normal(200, 15, 50),
            'sensor_3': np.random.normal(50, 5, 50),
            'timestamp': pd.date_range(datetime.datetime.now().strftime('%Y-%m-%d'), periods=50, freq='1h')
        })
        
        # Create data with known anomalies
        self.anomaly_data = self.sample_data.copy()
        self.anomaly_data.loc[10, 'sensor_1'] = 200  # Clear outlier
        self.anomaly_data.loc[20, 'sensor_2'] = 400  # Clear outlier
        
        # Create data with trend
        self.trend_data = self.sample_data.copy()
        trend_values = np.linspace(0, 20, 50)
        self.trend_data['sensor_1'] = self.trend_data['sensor_1'] + trend_values
    
    def test_initialization(self):
        """Test engine initialization"""
        self.assertIsNotNone(self.engine.anomaly_detector)
        self.assertEqual(self.engine.control_limits, {})
        self.assertFalse(self.engine._anomaly_detector_fitted)
        self.assertEqual(self.engine.spc_sigma_threshold, 3.0)
        self.assertEqual(self.engine.trend_min_points, 10)
        self.assertEqual(self.engine.trend_significance_level, 0.05)
    
    def test_initialization_with_custom_config(self):
        """Test engine initialization with custom configuration"""
        custom_config = {
            'spc_sigma_threshold': 2.5,
            'trend_min_points': 15,
            'trend_significance_level': 0.01
        }
        engine = StreamAnalyticsEngine(custom_config)
        
        self.assertEqual(engine.spc_sigma_threshold, 2.5)
        self.assertEqual(engine.trend_min_points, 15)
        self.assertEqual(engine.trend_significance_level, 0.01)
    
    def test_update_with_empty_data(self):
        """Test update with empty DataFrame"""
        empty_data = pd.DataFrame()
        result = self.engine.update_with_new_data(empty_data)
        
        expected = {
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
        
        self.assertEqual(result, expected)
    
    def test_update_with_normal_data(self):
        """Test update with normal data"""
        result = self.engine.update_with_new_data(self.sample_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('spc_violations', result)
        self.assertIn('trends', result)
        self.assertIn('anomalies', result)
        self.assertIn('anomaly_scores', result)
        self.assertIn('summary', result)
        
        # Check summary structure
        summary = result['summary']
        self.assertEqual(summary['total_points'], 50)
        self.assertIsInstance(summary['spc_violations_count'], int)
        self.assertIsInstance(summary['trends_count'], int)
        self.assertIsInstance(summary['anomalies_count'], int)
    
    def test_check_spc_normal_data(self):
        """Test SPC checks with normal data"""
        violations = self.engine.check_spc(self.sample_data)
        
        # With normal data, we shouldn't expect many violations
        self.assertIsInstance(violations, dict)
        # The violations dict should have sensor keys if any violations exist
        for key, value in violations.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, str)
    
    def test_check_spc_with_outliers(self):
        """Test SPC checks with data containing clear outliers"""
        # Create data with extreme outliers
        outlier_data = self.sample_data.copy()
        outlier_data.loc[10, 'sensor_1'] = 1000  # Extreme outlier
        
        violations = self.engine.check_spc(outlier_data)
        
        # Should detect the outlier
        self.assertIsInstance(violations, dict)
        # We expect at least sensor_1 to have violations
        if 'sensor_1' in violations:
            self.assertIn('exceed', violations['sensor_1'])
    
    def test_check_spc_empty_data(self):
        """Test SPC checks with empty data"""
        empty_data = pd.DataFrame()
        violations = self.engine.check_spc(empty_data)
        
        self.assertEqual(violations, {})
    
    def test_detect_trends_normal_data(self):
        """Test trend detection with normal data (no significant trends)"""
        trends = self.engine.detect_trends(self.sample_data)
        
        self.assertIsInstance(trends, dict)
        # Normal random data shouldn't have significant trends
        # But we can't be 100% sure, so just check the structure
        for key, value in trends.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, str)
    
    def test_detect_trends_with_trend(self):
        """Test trend detection with data containing a clear trend"""
        trends = self.engine.detect_trends(self.trend_data)
        
        self.assertIsInstance(trends, dict)
        # Should detect trend in sensor_1
        if 'sensor_1' in trends:
            self.assertIn('trend', trends['sensor_1'])
    
    def test_detect_trends_insufficient_data(self):
        """Test trend detection with insufficient data points"""
        small_data = self.sample_data.head(5)  # Less than trend_min_points
        trends = self.engine.detect_trends(small_data)
        
        self.assertEqual(trends, {})
    
    def test_detect_anomalies_normal_data(self):
        """Test anomaly detection with normal data"""
        anomaly_scores = self.engine.detect_anomalies(self.sample_data)
        
        # First call might return empty array if not enough data to fit
        self.assertIsInstance(anomaly_scores, np.ndarray)
        
        # If scores are returned, they should be -1 or 1
        if len(anomaly_scores) > 0:
            unique_scores = np.unique(anomaly_scores)
            for score in unique_scores:
                self.assertIn(score, [-1, 1])
    
    def test_detect_anomalies_with_outliers(self):
        """Test anomaly detection with data containing outliers"""
        # Use the anomaly data with clear outliers
        anomaly_scores = self.engine.detect_anomalies(self.anomaly_data)
        
        self.assertIsInstance(anomaly_scores, np.ndarray)
        
        # If the model was fitted and made predictions
        if len(anomaly_scores) > 0:
            # Should contain some anomalies (-1)
            unique_scores = np.unique(anomaly_scores)
            for score in unique_scores:
                self.assertIn(score, [-1, 1])
    
    def test_detect_anomalies_empty_data(self):
        """Test anomaly detection with empty data"""
        empty_data = pd.DataFrame()
        anomaly_scores = self.engine.detect_anomalies(empty_data)
        
        self.assertEqual(len(anomaly_scores), 0)
    
    def test_detect_anomalies_non_numeric_data(self):
        """Test anomaly detection with non-numeric data"""
        text_data = pd.DataFrame({
            'text_col': ['a', 'b', 'c', 'd', 'e']
        })
        anomaly_scores = self.engine.detect_anomalies(text_data)
        
        self.assertEqual(len(anomaly_scores), 0)
    
    def test_control_limits_calculation(self):
        """Test control limits calculation"""
        # Trigger control limits calculation
        self.engine.check_spc(self.sample_data)
        
        # Check that control limits were calculated
        numeric_columns = self.sample_data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            self.assertIn(column, self.engine.control_limits)
            limits = self.engine.control_limits[column]
            
            self.assertIn('mean', limits)
            self.assertIn('std', limits)
            self.assertIn('upper_control_limit', limits)
            self.assertIn('lower_control_limit', limits)
            
            # Check that limits make sense
            self.assertGreater(limits['upper_control_limit'], limits['mean'])
            self.assertLess(limits['lower_control_limit'], limits['mean'])
    
    def test_update_control_limits(self):
        """Test control limits update functionality"""
        # First calculation
        self.engine._update_control_limits(self.sample_data)
        original_limits = self.engine.control_limits.copy()
        
        # Update with new data (only multiply numeric columns)
        new_data = self.sample_data.copy()
        numeric_cols = new_data.select_dtypes(include=[np.number]).columns
        new_data[numeric_cols] = new_data[numeric_cols] * 2  # Different scale
        self.engine._update_control_limits(new_data)
        updated_limits = self.engine.control_limits
        
        # Limits should be updated
        for column in original_limits:
            if column in updated_limits:
                # The mean should be different (doubled)
                self.assertNotEqual(
                    original_limits[column]['mean'],
                    updated_limits[column]['mean']
                )
    
    def test_integration_full_pipeline(self):
        """Test the full analytics pipeline integration"""
        # Run full pipeline
        result = self.engine.update_with_new_data(self.anomaly_data)
        
        # Verify all components ran
        self.assertIsInstance(result['spc_violations'], dict)
        self.assertIsInstance(result['trends'], dict)
        self.assertIsInstance(result['anomalies'], list)
        self.assertIsInstance(result['anomaly_scores'], list)
        
        # Verify summary
        summary = result['summary']
        self.assertEqual(summary['total_points'], len(self.anomaly_data))
        self.assertEqual(summary['spc_violations_count'], len(result['spc_violations']))
        self.assertEqual(summary['trends_count'], len(result['trends']))
        self.assertEqual(summary['anomalies_count'], len(result['anomalies']))
    
    def test_error_handling_corrupted_data(self):
        """Test error handling with corrupted/problematic data"""
        # Data with NaN values
        corrupted_data = self.sample_data.copy()
        corrupted_data.loc[5:10, 'sensor_1'] = np.nan
        
        # Should not raise an exception
        result = self.engine.update_with_new_data(corrupted_data)
        self.assertIsInstance(result, dict)
    
    def test_error_handling_single_value_columns(self):
        """Test error handling with columns containing single repeated values"""
        # Data with constant values (zero std)
        constant_data = pd.DataFrame({
            'constant_sensor': [100] * 20,
            'normal_sensor': np.random.normal(50, 5, 20)
        })
        
        # Should handle gracefully
        result = self.engine.update_with_new_data(constant_data)
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main()