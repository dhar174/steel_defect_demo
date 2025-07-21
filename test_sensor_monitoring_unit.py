"""
Unit tests for the SensorMonitoringComponent.

This module provides comprehensive tests for all functionality of the
sensor monitoring component including data handling, plotting, health
monitoring, and anomaly detection.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from visualization.components.sensor_monitoring import SensorMonitoringComponent


class TestSensorMonitoringComponent(unittest.TestCase):
    """Test cases for SensorMonitoringComponent."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.component = SensorMonitoringComponent(
            component_id="test-component",
            buffer_size=100,
            update_interval=1000
        )
        
        # Sample test data
        self.test_data = {
            'casting_speed': [1.1, 1.2, 1.15, 1.3, 1.25],
            'mold_temperature': [1530, 1535, 1540, 1545, 1550],
            'mold_level': [150, 151, 149, 152, 148]
        }
        
        self.test_timestamps = [
            datetime.now() - timedelta(seconds=20),
            datetime.now() - timedelta(seconds=15),
            datetime.now() - timedelta(seconds=10),
            datetime.now() - timedelta(seconds=5),
            datetime.now()
        ]
    
    def test_component_initialization(self):
        """Test proper component initialization."""
        self.assertEqual(self.component.component_id, "test-component")
        self.assertEqual(self.component.buffer_size, 100)
        self.assertEqual(self.component.update_interval, 1000)
        self.assertEqual(len(self.component.sensors), 6)
        self.assertIn('casting_speed', self.component.sensors)
        self.assertIn('mold_temperature', self.component.sensors)
        
        # Check buffers are initialized
        for sensor in self.component.sensors:
            self.assertIn(sensor, self.component.data_buffers)
            self.assertEqual(len(self.component.data_buffers[sensor]), 0)
    
    def test_add_data_point(self):
        """Test adding data points to buffers."""
        test_point = {'casting_speed': 1.2, 'mold_temperature': 1530}
        timestamp = datetime.now()
        
        self.component.add_data_point(test_point, timestamp)
        
        # Check data was added
        self.assertEqual(len(self.component.timestamp_buffer), 1)
        self.assertEqual(self.component.data_buffers['casting_speed'][-1], 1.2)
        self.assertEqual(self.component.data_buffers['mold_temperature'][-1], 1530)
        self.assertEqual(self.component.last_update_times['casting_speed'], timestamp)
    
    def test_rolling_window_buffer(self):
        """Test rolling window buffer functionality."""
        # Add more data than buffer size
        for i in range(150):  # More than buffer_size of 100
            test_point = {'casting_speed': i}
            self.component.add_data_point(test_point)
        
        # Check buffer doesn't exceed max size
        self.assertEqual(len(self.component.timestamp_buffer), 100)
        self.assertEqual(len(self.component.data_buffers['casting_speed']), 100)
        
        # Check latest data is preserved
        self.assertEqual(self.component.data_buffers['casting_speed'][-1], 149)
    
    def test_get_current_data(self):
        """Test retrieving current data from buffers."""
        # Add test data
        for i, timestamp in enumerate(self.test_timestamps):
            test_point = {sensor: data[i] for sensor, data in self.test_data.items()}
            self.component.add_data_point(test_point, timestamp)
        
        data, timestamps = self.component.get_current_data()
        
        # Check data integrity
        self.assertEqual(len(timestamps), 5)
        self.assertEqual(len(data['casting_speed']), 5)
        self.assertEqual(data['casting_speed'][-1], 1.25)
        self.assertEqual(data['mold_temperature'][-1], 1550)
    
    def test_mock_data_generation(self):
        """Test mock data generation."""
        mock_data = self.component.generate_mock_data_point()
        
        # Check all sensors have data
        for sensor in self.component.sensors:
            self.assertIn(sensor, mock_data)
            self.assertIsInstance(mock_data[sensor], (int, float, np.number))
        
        # Check realistic ranges
        self.assertGreater(mock_data['casting_speed'], 0.5)
        self.assertLess(mock_data['casting_speed'], 2.0)
        self.assertGreater(mock_data['mold_temperature'], 1400)
        self.assertLess(mock_data['mold_temperature'], 1700)
    
    def test_anomaly_detection_threshold(self):
        """Test threshold-based anomaly detection."""
        # Create data with known anomalies
        sensor_data = [1.0, 1.1, 1.2, 2.0, 1.15]  # 2.0 is above max threshold of 1.5
        
        anomalies = self.component._detect_anomalies(sensor_data, 'casting_speed')
        
        # Should detect the anomaly at index 3
        self.assertIn(3, anomalies)
    
    def test_anomaly_detection_statistical(self):
        """Test statistical anomaly detection."""
        # Create data with statistical outlier
        sensor_data = [100] * 20 + [200]  # Last value is statistical outlier
        
        anomalies = self.component._detect_anomalies(sensor_data, 'mold_temperature')
        
        # Should detect the outlier
        self.assertTrue(len(anomalies) > 0)
        self.assertIn(20, anomalies)  # Index of outlier
    
    def test_sensor_health_update(self):
        """Test sensor health status updates."""
        # Add some test data
        for i, timestamp in enumerate(self.test_timestamps):
            test_point = {sensor: data[i] for sensor, data in self.test_data.items()}
            self.component.add_data_point(test_point, timestamp)
        
        data, timestamps = self.component.get_current_data()
        health_status = self.component.update_sensor_health(data, timestamps)
        
        # Check health status is calculated
        for sensor in self.component.sensors[:3]:  # Only check sensors with data
            self.assertIn(sensor, health_status)
            self.assertIn(health_status[sensor], ['good', 'warning', 'critical', 'offline', 'stale'])
    
    def test_sensor_health_offline_detection(self):
        """Test detection of offline sensors."""
        # Test with empty data
        health_status = self.component.update_sensor_health({}, [])
        
        # All sensors should be offline
        for sensor in self.component.sensors:
            self.assertEqual(health_status[sensor], 'offline')
    
    def test_sensor_health_stale_detection(self):
        """Test detection of stale sensor data."""
        # Add old data
        old_timestamp = datetime.now() - timedelta(minutes=2)
        data = {'casting_speed': [1.2]}
        timestamps = [old_timestamp]
        
        health_status = self.component.update_sensor_health(data, timestamps)
        
        # Should detect stale data
        self.assertEqual(health_status['casting_speed'], 'stale')
    
    def test_create_layout(self):
        """Test layout creation."""
        layout = self.component.create_layout()
        
        # Check layout structure
        self.assertIsNotNone(layout)
        self.assertEqual(layout.id, f'{self.component.component_id}-container')
    
    def test_create_multi_sensor_plot(self):
        """Test multi-sensor plot creation."""
        # Add test data
        for i, timestamp in enumerate(self.test_timestamps):
            test_point = {sensor: data[i] for sensor, data in self.test_data.items()}
            self.component.add_data_point(test_point, timestamp)
        
        data, timestamps = self.component.get_current_data()
        config = {
            'time_range_minutes': 30,
            'auto_scale': True,
            'show_thresholds': True,
            'show_anomalies': True
        }
        
        fig = self.component.create_multi_sensor_plot(data, timestamps, config)
        
        # Check plot creation
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.data), 0)  # Should have traces
    
    def test_create_detail_plots(self):
        """Test detail plots creation."""
        # Add test data
        for i, timestamp in enumerate(self.test_timestamps):
            test_point = {sensor: data[i] for sensor, data in self.test_data.items()}
            self.component.add_data_point(test_point, timestamp)
        
        data, timestamps = self.component.get_current_data()
        config = {'auto_scale': True, 'show_thresholds': True}
        
        fig = self.component.create_detail_plots(data, timestamps, config)
        
        # Check detail plots creation
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.data), 0)
    
    def test_create_health_indicators(self):
        """Test health indicators creation."""
        health_status = {
            'casting_speed': 'good',
            'mold_temperature': 'warning',
            'mold_level': 'critical'
        }
        
        indicators = self.component.create_health_indicators(health_status)
        
        # Check indicators creation
        self.assertIsNotNone(indicators)
    
    def test_configuration_defaults(self):
        """Test default configuration values."""
        self.assertEqual(self.component.time_range_minutes, 30)
        self.assertTrue(self.component.auto_scale)
        self.assertTrue(self.component.show_thresholds)
        self.assertTrue(self.component.show_anomalies)
    
    def test_thresholds_configuration(self):
        """Test sensor thresholds configuration."""
        # Check default thresholds exist
        self.assertIn('casting_speed', self.component.thresholds)
        self.assertIn('mold_temperature', self.component.thresholds)
        
        # Check threshold structure
        speed_thresholds = self.component.thresholds['casting_speed']
        self.assertIn('min', speed_thresholds)
        self.assertIn('max', speed_thresholds)
        self.assertIn('warning_min', speed_thresholds)
        self.assertIn('warning_max', speed_thresholds)
    
    def test_data_with_none_values(self):
        """Test handling of None values in sensor data."""
        # Add data with missing sensors
        test_point = {'casting_speed': 1.2}  # Missing other sensors
        self.component.add_data_point(test_point)
        
        data, timestamps = self.component.get_current_data()
        
        # Check data handling
        self.assertEqual(len(data['casting_speed']), 1)
        self.assertEqual(len(data['mold_temperature']), 0)  # Should filter out None


class TestSensorMonitoringIntegration(unittest.TestCase):
    """Integration tests for sensor monitoring component."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.component = SensorMonitoringComponent()
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from data input to visualization."""
        # Step 1: Generate and add data
        for i in range(20):
            mock_data = self.component.generate_mock_data_point()
            timestamp = datetime.now() - timedelta(minutes=20-i)
            self.component.add_data_point(mock_data, timestamp)
        
        # Step 2: Get current data
        data, timestamps = self.component.get_current_data()
        self.assertEqual(len(timestamps), 20)
        
        # Step 3: Update health status
        health_status = self.component.update_sensor_health(data, timestamps)
        self.assertEqual(len(health_status), len(self.component.sensors))
        
        # Step 4: Create visualizations
        config = {
            'time_range_minutes': 30,
            'auto_scale': True,
            'show_thresholds': True,
            'show_anomalies': True
        }
        
        main_plot = self.component.create_multi_sensor_plot(data, timestamps, config)
        detail_plots = self.component.create_detail_plots(data, timestamps, config)
        health_indicators = self.component.create_health_indicators(health_status)
        
        # Verify all components work together
        self.assertIsNotNone(main_plot)
        self.assertIsNotNone(detail_plots)
        self.assertIsNotNone(health_indicators)
        self.assertGreater(len(main_plot.data), 0)
        self.assertGreater(len(detail_plots.data), 0)
    
    def test_performance_with_large_dataset(self):
        """Test performance with large amounts of data."""
        import time
        
        # Add large amount of data
        start_time = time.time()
        for i in range(1000):
            mock_data = self.component.generate_mock_data_point()
            self.component.add_data_point(mock_data)
        
        data, timestamps = self.component.get_current_data()
        
        # Should complete quickly due to rolling window
        elapsed_time = time.time() - start_time
        self.assertLess(elapsed_time, 2.0)  # Should take less than 2 seconds
        
        # Buffer should not exceed limit
        self.assertLessEqual(len(timestamps), self.component.buffer_size)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)