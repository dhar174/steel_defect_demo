import pytest
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import time
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from monitoring.real_time_monitor import RealTimeMonitor


class TestRealTimeMonitor:
    """Test suite for RealTimeMonitor class"""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Test configuration matching the expected structure
        self.test_config = {
            'monitoring': {
                'metrics_logging': True,
                'performance_tracking': True,
                'data_drift_detection': True
            },
            'inference': {
                'thresholds': {
                    'defect_probability': 0.5,
                    'high_risk_threshold': 0.7,
                    'alert_threshold': 0.8
                }
            }
        }
        
        # Sample prediction results for testing
        self.sample_prediction_low_risk = {
            'ensemble_prediction': 0.3,
            'confidence': 0.8,
            'latency': {'total_time': 0.05}
        }
        
        self.sample_prediction_high_risk = {
            'ensemble_prediction': 0.75,
            'confidence': 0.9,
            'latency': {'total_time': 0.08}
        }
        
        self.sample_prediction_critical = {
            'ensemble_prediction': 0.85,
            'confidence': 0.95,
            'latency': {'total_time': 0.12}
        }
        
        # Sample data buffer for testing
        self.sample_data_buffer = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1s'),
            'temperature': np.random.normal(1520, 10, 100),
            'pressure': np.random.normal(100, 5, 100),
            'flow_rate': np.random.normal(200, 15, 100),
            'mold_temperature': np.random.normal(1520, 10, 100),
            'cooling_water_flow': np.random.normal(200, 15, 100)
        })

    def test_initialization(self):
        """Test RealTimeMonitor initialization."""
        monitor = RealTimeMonitor(self.test_config)
        
        # Check configuration is stored correctly
        assert monitor.config == self.test_config['monitoring']
        assert monitor.thresholds == self.test_config['inference']['thresholds']
        
        # Check data structures are initialized
        assert len(monitor.prediction_history) == 0
        assert len(monitor.latency_history) == 0
        assert monitor.prediction_history.maxlen == 1000
        assert monitor.latency_history.maxlen == 1000
        
        # Check logger is set up
        assert monitor.logger is not None
        assert monitor.logger.name == 'monitoring.real_time_monitor'

    def test_track_prediction_low_risk(self):
        """Test tracking low-risk predictions."""
        monitor = RealTimeMonitor(self.test_config)
        
        with patch.object(monitor.logger, 'warning') as mock_warning, \
             patch.object(monitor.logger, 'error') as mock_error:
            
            monitor.track_prediction(self.sample_prediction_low_risk)
            
            # Check prediction is stored
            assert len(monitor.prediction_history) == 1
            assert len(monitor.latency_history) == 1
            
            # Check prediction data
            pred_data = monitor.prediction_history[0]
            assert pred_data['prediction'] == 0.3
            assert pred_data['confidence'] == 0.8
            assert 'timestamp' in pred_data
            
            # Check latency data
            latency_data = monitor.latency_history[0]
            assert latency_data['latency_ms'] == 50.0  # 0.05 * 1000
            assert 'timestamp' in latency_data
            
            # No alerts should be triggered
            mock_warning.assert_not_called()
            mock_error.assert_not_called()

    def test_track_prediction_high_risk(self):
        """Test tracking high-risk predictions triggers warning."""
        monitor = RealTimeMonitor(self.test_config)
        
        with patch.object(monitor.logger, 'warning') as mock_warning, \
             patch.object(monitor.logger, 'error') as mock_error:
            
            monitor.track_prediction(self.sample_prediction_high_risk)
            
            # Check prediction is stored
            assert len(monitor.prediction_history) == 1
            assert monitor.prediction_history[0]['prediction'] == 0.75
            
            # Warning should be triggered, but not error
            mock_warning.assert_called_once()
            mock_error.assert_not_called()
            
            # Check warning message content
            warning_call = mock_warning.call_args[0][0]
            assert "HIGH RISK" in warning_call
            assert "0.7500" in warning_call

    def test_track_prediction_critical_alert(self):
        """Test tracking critical predictions triggers error alert."""
        monitor = RealTimeMonitor(self.test_config)
        
        with patch.object(monitor.logger, 'warning') as mock_warning, \
             patch.object(monitor.logger, 'error') as mock_error:
            
            monitor.track_prediction(self.sample_prediction_critical)
            
            # Check prediction is stored
            assert len(monitor.prediction_history) == 1
            assert monitor.prediction_history[0]['prediction'] == 0.85
            
            # Error should be triggered (critical takes precedence)
            mock_error.assert_called_once()
            
            # Check error message content
            error_call = mock_error.call_args[0][0]
            assert "CRITICAL ALERT" in error_call
            assert "0.8500" in error_call

    def test_track_prediction_legacy_format(self):
        """Test tracking predictions with legacy format (direct 'prediction' key)."""
        monitor = RealTimeMonitor(self.test_config)
        
        legacy_prediction = {
            'prediction': 0.6,
            'confidence': 0.85,
            'latency': 0.07  # Direct float value
        }
        
        monitor.track_prediction(legacy_prediction)
        
        # Check prediction is stored correctly
        assert len(monitor.prediction_history) == 1
        assert monitor.prediction_history[0]['prediction'] == 0.6
        
        # Check latency is handled correctly
        assert len(monitor.latency_history) == 1
        assert monitor.latency_history[0]['latency_ms'] == 70.0

    def test_check_data_quality_clean_data(self):
        """Test data quality check with clean data."""
        monitor = RealTimeMonitor(self.test_config)
        
        issues = monitor.check_data_quality(self.sample_data_buffer)
        
        # Clean data should have no issues
        assert issues == []

    def test_check_data_quality_empty_data(self):
        """Test data quality check with empty data."""
        monitor = RealTimeMonitor(self.test_config)
        
        # Test with None
        issues = monitor.check_data_quality(None)
        assert len(issues) == 1
        assert "empty or None" in issues[0]
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        issues = monitor.check_data_quality(empty_df)
        assert len(issues) == 1
        assert "empty or None" in issues[0]

    def test_check_data_quality_nan_values(self):
        """Test data quality check with NaN values."""
        monitor = RealTimeMonitor(self.test_config)
        
        # Create data with NaN values
        data_with_nan = self.sample_data_buffer.copy()
        data_with_nan.loc[0:5, 'temperature'] = np.nan
        data_with_nan.loc[10:15, 'pressure'] = np.nan
        
        issues = monitor.check_data_quality(data_with_nan)
        
        # Should detect NaN issues
        assert len(issues) == 1
        assert "NaN values detected" in issues[0]
        assert "temperature" in issues[0]
        assert "pressure" in issues[0]

    def test_check_data_quality_temperature_range(self):
        """Test data quality check with temperature out of range."""
        monitor = RealTimeMonitor(self.test_config)
        
        # Create data with invalid temperatures
        data_with_bad_temp = self.sample_data_buffer.copy()
        data_with_bad_temp.loc[0:2, 'temperature'] = -100  # Too low
        data_with_bad_temp.loc[5:7, 'mold_temperature'] = 3000  # Too high
        
        issues = monitor.check_data_quality(data_with_bad_temp)
        
        # Should detect temperature range issues
        temp_issues = [issue for issue in issues if "Temperature values out of range" in issue]
        assert len(temp_issues) == 2  # One for each temperature column

    def test_check_data_quality_negative_values(self):
        """Test data quality check with negative values for positive-only fields."""
        monitor = RealTimeMonitor(self.test_config)
        
        # Create data with negative values
        data_with_negatives = self.sample_data_buffer.copy()
        data_with_negatives.loc[0:2, 'pressure'] = -50
        data_with_negatives.loc[5:7, 'flow_rate'] = -20
        data_with_negatives.loc[10:12, 'cooling_water_flow'] = -30
        
        issues = monitor.check_data_quality(data_with_negatives)
        
        # Should detect negative value issues
        negative_issues = [issue for issue in issues if "Negative" in issue]
        assert len(negative_issues) == 3  # One for each problematic column

    def test_get_system_performance_metrics_empty(self):
        """Test performance metrics with no data."""
        monitor = RealTimeMonitor(self.test_config)
        
        metrics = monitor.get_system_performance_metrics()
        
        # Should return default values
        assert metrics['avg_latency_ms'] == 0.0
        assert metrics['throughput_preds_per_sec'] == 0.0
        assert metrics['total_predictions'] == 0
        assert metrics['high_risk_predictions'] == 0

    def test_get_system_performance_metrics_with_data(self):
        """Test performance metrics calculation with data."""
        monitor = RealTimeMonitor(self.test_config)
        
        # Add multiple predictions with some delay to simulate time passage
        predictions = [
            self.sample_prediction_low_risk,
            self.sample_prediction_high_risk,
            self.sample_prediction_critical,
            self.sample_prediction_low_risk
        ]
        
        for i, pred in enumerate(predictions):
            monitor.track_prediction(pred)
            if i < len(predictions) - 1:
                time.sleep(0.01)  # Small delay between predictions
        
        metrics = monitor.get_system_performance_metrics()
        
        # Check basic counts
        assert metrics['total_predictions'] == 4
        assert metrics['high_risk_predictions'] == 2  # high_risk and critical
        
        # Check latency calculation
        assert metrics['avg_latency_ms'] > 0
        expected_avg_latency = (50 + 80 + 120 + 50) / 4  # From our test data
        assert abs(metrics['avg_latency_ms'] - expected_avg_latency) < 1.0
        
        # Check throughput is calculated (should be > 0 due to time passage)
        assert metrics['throughput_preds_per_sec'] > 0

    def test_maxlen_behavior(self):
        """Test that deques respect maxlen=1000."""
        monitor = RealTimeMonitor(self.test_config)
        
        # Add more than 1000 predictions
        for i in range(1100):
            prediction = {
                'prediction': 0.5,
                'confidence': 0.8,
                'latency': 0.05
            }
            monitor.track_prediction(prediction)
        
        # Should only keep last 1000
        assert len(monitor.prediction_history) == 1000
        assert len(monitor.latency_history) == 1000

    def test_generate_alert_threshold_boundaries(self):
        """Test alert generation at threshold boundaries."""
        monitor = RealTimeMonitor(self.test_config)
        
        with patch.object(monitor.logger, 'warning') as mock_warning, \
             patch.object(monitor.logger, 'error') as mock_error:
            
            # Test exactly at high_risk_threshold (0.7)
            boundary_high_risk = {
                'ensemble_prediction': 0.7,
                'confidence': 0.8,
                'latency': {'total_time': 0.05}
            }
            monitor.track_prediction(boundary_high_risk)
            mock_warning.assert_called_once()
            mock_error.assert_not_called()
            
            # Reset mocks
            mock_warning.reset_mock()
            mock_error.reset_mock()
            
            # Test exactly at alert_threshold (0.8)
            boundary_critical = {
                'ensemble_prediction': 0.8,
                'confidence': 0.8,
                'latency': {'total_time': 0.05}
            }
            monitor.track_prediction(boundary_critical)
            mock_error.assert_called_once()

    def test_config_validation(self):
        """Test monitor handles missing config sections gracefully."""
        # Test with minimal config
        minimal_config = {
            'monitoring': {},
            'inference': {
                'thresholds': {}
            }
        }
        
        monitor = RealTimeMonitor(minimal_config)
        
        # Should not crash and use defaults
        assert monitor.config == {}
        assert monitor.thresholds == {}
        
        # Should handle missing thresholds gracefully
        prediction = {
            'ensemble_prediction': 0.9,
            'confidence': 0.8,
            'latency': {'total_time': 0.05}
        }
        
        # Should not crash when thresholds are missing
        with patch.object(monitor.logger, 'warning') as mock_warning, \
             patch.object(monitor.logger, 'error') as mock_error:
            monitor.track_prediction(prediction)
            # With default threshold of 0.8, this should trigger critical alert
            mock_error.assert_called_once()
            
        # Test with high risk level (between 0.7 and 0.8)
        high_risk_prediction = {
            'ensemble_prediction': 0.75,
            'confidence': 0.8,
            'latency': {'total_time': 0.05}
        }
        
        with patch.object(monitor.logger, 'warning') as mock_warning:
            monitor.track_prediction(high_risk_prediction)
            # This should trigger warning
            mock_warning.assert_called_once()

    def test_integration_with_real_config(self):
        """Test monitor works with actual config file structure."""
        # Load actual config structure similar to inference_config.yaml
        real_config = {
            'inference': {
                'model_types': ['baseline', 'lstm'],
                'real_time_simulation': {
                    'playback_speed_multiplier': 10,
                    'update_interval_seconds': 30,
                    'buffer_size_seconds': 300
                },
                'thresholds': {
                    'defect_probability': 0.5,
                    'high_risk_threshold': 0.7,
                    'alert_threshold': 0.8
                }
            },
            'monitoring': {
                'metrics_logging': True,
                'performance_tracking': True,
                'data_drift_detection': True
            }
        }
        
        monitor = RealTimeMonitor(real_config)
        
        # Should initialize correctly
        assert monitor.thresholds['high_risk_threshold'] == 0.7
        assert monitor.thresholds['alert_threshold'] == 0.8
        assert monitor.config['metrics_logging'] is True
        
        # Should work with real prediction data
        prediction = {
            'ensemble_prediction': 0.75,
            'baseline_prediction': 0.7,
            'lstm_prediction': 0.8,
            'confidence': 0.85,
            'latency': {
                'total_time': 0.125,
                'baseline_time': 0.05,
                'lstm_time': 0.075
            }
        }
        
        with patch.object(monitor.logger, 'warning') as mock_warning:
            monitor.track_prediction(prediction)
            mock_warning.assert_called_once()

    def test_prediction_pipeline_integration(self):
        """Test RealTimeMonitor integration with PredictionPipeline."""
        # Import here to avoid circular imports
        from inference.prediction_pipeline import PredictionPipeline
        
        # Test config with monitoring section
        config_with_monitoring = {
            'inference': {
                'model_types': ['baseline', 'lstm'],
                'real_time_simulation': {
                    'playback_speed_multiplier': 10,
                    'update_interval_seconds': 30,
                    'buffer_size_seconds': 300
                },
                'thresholds': {
                    'defect_probability': 0.5,
                    'high_risk_threshold': 0.7,
                    'alert_threshold': 0.8
                }
            },
            'monitoring': {
                'metrics_logging': True,
                'performance_tracking': True
            }
        }
        
        # Create pipeline with monitor integration
        pipeline = PredictionPipeline(
            config=config_with_monitoring,
            cast_files=['/tmp/test_cast.csv']
        )
        
        # Verify monitor is initialized and functioning
        assert pipeline.monitor is not None
        assert hasattr(pipeline.monitor, 'track_prediction')
        assert hasattr(pipeline.monitor, 'check_data_quality')
        assert hasattr(pipeline.monitor, 'get_system_performance_metrics')
        assert pipeline.monitor.thresholds['high_risk_threshold'] == 0.7
        
        # Test that monitor can track predictions
        test_prediction = {
            'ensemble_prediction': 0.6,
            'confidence': 0.8,
            'latency': {'total_time': 0.05}
        }
        
        pipeline.monitor.track_prediction(test_prediction)
        assert len(pipeline.monitor.prediction_history) == 1
        assert pipeline.monitor.prediction_history[0]['prediction'] == 0.6
        
        # Test that monitor can check data quality
        test_data = pd.DataFrame({
            'temperature': [1500, 1520, 1510],
            'pressure': [100, 105, 102]
        })
        
        issues = pipeline.monitor.check_data_quality(test_data)
        assert isinstance(issues, list)