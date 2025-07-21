"""
Unit tests for the Alert Management Interface.

This module tests the core functionality of the AlertManagementComponent
to ensure all features work as expected.
"""

import unittest
import os
import sys
from datetime import datetime, timedelta
import tempfile
import yaml

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.visualization.components.alert_management import AlertManagementComponent


class TestAlertManagementComponent(unittest.TestCase):
    """Test cases for AlertManagementComponent."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        config_data = {
            'inference': {
                'thresholds': {
                    'defect_probability': 0.5,
                    'high_risk_threshold': 0.7,
                    'alert_threshold': 0.8
                }
            },
            'monitoring': {
                'alerts': {
                    'alert_suppression_minutes': 5
                }
            }
        }
        yaml.dump(config_data, self.temp_config, default_flow_style=False)
        self.temp_config.close()
        
        # Initialize component with test config and disable sample data
        self.component = AlertManagementComponent(
            component_id="test-alert-mgmt",
            config_file=self.temp_config.name,
            max_alerts=100,
            initialize_sample_data=False
        )
        
        # Reset alert counter for consistent ID generation
        self.component._alert_counter = 0
        self.component._update_performance_metrics()
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_config.name)
    
    def test_component_initialization(self):
        """Test component initialization."""
        self.assertEqual(self.component.component_id, "test-alert-mgmt")
        self.assertEqual(self.component.max_alerts, 100)
        self.assertIsInstance(self.component.config, dict)
        self.assertIn('inference', self.component.config)
        self.assertIn('monitoring', self.component.config)
    
    def test_add_alert(self):
        """Test adding new alerts."""
        # Add a test alert
        alert_id = self.component.add_alert('High', 'Test alert description', 'Test System')
        
        self.assertIsNotNone(alert_id)
        self.assertTrue(alert_id.startswith('ALT-'))
        self.assertEqual(len(self.component.alerts_buffer), 1)
        
        # Check alert properties
        alert = list(self.component.alerts_buffer)[0]
        self.assertEqual(alert['id'], alert_id)
        self.assertEqual(alert['severity'], 'High')
        self.assertEqual(alert['description'], 'Test alert description')
        self.assertEqual(alert['status'], 'New')
        self.assertEqual(alert['source'], 'Test System')
        self.assertIsNone(alert['acknowledged_at'])
        self.assertIsNone(alert['resolved_at'])
    
    def test_acknowledge_alert(self):
        """Test alert acknowledgment functionality."""
        # Add an alert first
        alert_id = self.component.add_alert('Medium', 'Test alert for acknowledgment')
        
        # Acknowledge the alert
        result = self.component.acknowledge_alert(alert_id, 'test_user')
        self.assertTrue(result)
        
        # Check alert status
        alert = list(self.component.alerts_buffer)[0]
        self.assertEqual(alert['status'], 'Acknowledged')
        self.assertEqual(alert['acknowledged_by'], 'test_user')
        self.assertIsNotNone(alert['acknowledged_at'])
        self.assertIsNone(alert['resolved_at'])
        
        # Try to acknowledge again (should fail)
        result = self.component.acknowledge_alert(alert_id, 'another_user')
        self.assertFalse(result)
    
    def test_resolve_alert(self):
        """Test alert resolution functionality."""
        # Add an alert first
        alert_id = self.component.add_alert('Critical', 'Test alert for resolution')
        
        # Resolve the alert directly (should auto-acknowledge)
        result = self.component.resolve_alert(alert_id, 'test_user')
        self.assertTrue(result)
        
        # Check alert status
        alert = list(self.component.alerts_buffer)[0]
        self.assertEqual(alert['status'], 'Resolved')
        self.assertEqual(alert['resolved_by'], 'test_user')
        self.assertEqual(alert['acknowledged_by'], 'test_user')  # Auto-acknowledged
        self.assertIsNotNone(alert['acknowledged_at'])
        self.assertIsNotNone(alert['resolved_at'])
        
        # Try to resolve again (should fail)
        result = self.component.resolve_alert(alert_id, 'another_user')
        self.assertFalse(result)
    
    def test_acknowledge_then_resolve(self):
        """Test acknowledge followed by resolve workflow."""
        # Add an alert
        alert_id = self.component.add_alert('High', 'Test workflow alert')
        
        # Acknowledge first
        ack_result = self.component.acknowledge_alert(alert_id, 'ack_user')
        self.assertTrue(ack_result)
        
        # Then resolve
        resolve_result = self.component.resolve_alert(alert_id, 'resolve_user')
        self.assertTrue(resolve_result)
        
        # Check final status
        alert = list(self.component.alerts_buffer)[0]
        self.assertEqual(alert['status'], 'Resolved')
        self.assertEqual(alert['acknowledged_by'], 'ack_user')
        self.assertEqual(alert['resolved_by'], 'resolve_user')
    
    def test_get_alerts_data(self):
        """Test retrieving alerts data for DataTable."""
        # Add multiple alerts
        self.component.add_alert('Low', 'Alert 1')
        self.component.add_alert('High', 'Alert 2')
        self.component.add_alert('Critical', 'Alert 3')
        
        # Get alerts data
        alerts_data = self.component.get_alerts_data()
        
        self.assertEqual(len(alerts_data), 3)
        
        # Check data structure
        for alert_data in alerts_data:
            self.assertIn('id', alert_data)
            self.assertIn('timestamp', alert_data)
            self.assertIn('severity', alert_data)
            self.assertIn('description', alert_data)
            self.assertIn('status', alert_data)
            self.assertIn('source', alert_data)
            self.assertIn('select', alert_data)
    
    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation."""
        # Start with empty metrics
        self.assertEqual(self.component.performance_metrics['total_alerts'], 0)
        self.assertEqual(self.component.performance_metrics['active_alerts'], 0)
        
        # Add alerts with different statuses
        alert1_id = self.component.add_alert('High', 'Active alert')
        alert2_id = self.component.add_alert('Medium', 'To be acknowledged')
        alert3_id = self.component.add_alert('Low', 'To be resolved')
        
        # Acknowledge one alert
        self.component.acknowledge_alert(alert2_id, 'user1')
        
        # Resolve one alert
        self.component.resolve_alert(alert3_id, 'user2')
        
        # Check metrics
        metrics = self.component.performance_metrics
        self.assertEqual(metrics['total_alerts'], 3)
        self.assertEqual(metrics['active_alerts'], 1)  # alert1
        self.assertEqual(metrics['acknowledged_alerts'], 1)  # alert2
        self.assertEqual(metrics['resolved_alerts'], 1)  # alert3
        self.assertGreater(metrics['mean_time_to_acknowledge'], 0)
        self.assertGreater(metrics['mean_time_to_resolve'], 0)
    
    def test_layout_creation(self):
        """Test layout creation."""
        layout = self.component.create_layout()
        self.assertIsNotNone(layout)
        # Layout should be a Div containing all the interface elements
        self.assertEqual(layout.id, f'{self.component.component_id}-container')
    
    def test_chart_creation(self):
        """Test chart creation methods."""
        # Add some test data
        self.component.add_alert('High', 'Chart test alert 1')
        self.component.add_alert('Medium', 'Chart test alert 2')
        self.component.add_alert('Critical', 'Chart test alert 3')
        
        # Test frequency chart
        freq_chart = self.component.create_frequency_chart()
        self.assertIsNotNone(freq_chart)
        
        # Test severity distribution chart
        severity_chart = self.component.create_severity_distribution()
        self.assertIsNotNone(severity_chart)
    
    def test_config_update(self):
        """Test configuration update functionality."""
        new_config = {
            'inference': {
                'thresholds': {
                    'defect_probability': 0.6,
                    'high_risk_threshold': 0.75,
                    'alert_threshold': 0.85
                }
            }
        }
        
        result = self.component.update_config_file(new_config)
        self.assertTrue(result)
        
        # Check that config was updated
        self.assertEqual(
            self.component.config['inference']['thresholds']['defect_probability'], 
            0.6
        )
    
    def test_alert_buffer_limit(self):
        """Test that alert buffer respects max_alerts limit."""
        # Set a small limit for testing
        self.component.alerts_buffer = type(self.component.alerts_buffer)(maxlen=3)
        
        # Add more alerts than the limit
        for i in range(5):
            self.component.add_alert('Medium', f'Alert {i}')
        
        # Should only have 3 alerts (the limit)
        self.assertEqual(len(self.component.alerts_buffer), 3)
        
        # Should have the latest 3 alerts
        alert_ids = [alert['id'] for alert in self.component.alerts_buffer]
        self.assertIn('ALT-0002', alert_ids)  # Latest alerts
        self.assertIn('ALT-0003', alert_ids)
        self.assertIn('ALT-0004', alert_ids)
        self.assertNotIn('ALT-0000', alert_ids)  # Oldest should be removed
        self.assertNotIn('ALT-0001', alert_ids)
    
    def test_invalid_operations(self):
        """Test invalid operations and edge cases."""
        # Try to acknowledge non-existent alert
        result = self.component.acknowledge_alert('INVALID-ID', 'user')
        self.assertFalse(result)
        
        # Try to resolve non-existent alert
        result = self.component.resolve_alert('INVALID-ID', 'user')
        self.assertFalse(result)
        
        # Test with empty buffer
        alerts_data = self.component.get_alerts_data()
        self.assertEqual(len(alerts_data), 0)
        
        # Charts should handle empty data gracefully
        freq_chart = self.component.create_frequency_chart()
        severity_chart = self.component.create_severity_distribution()
        self.assertIsNotNone(freq_chart)
        self.assertIsNotNone(severity_chart)


class TestAlertManagementIntegration(unittest.TestCase):
    """Integration tests for AlertManagementComponent."""
    
    def test_full_workflow(self):
        """Test a complete alert management workflow."""
        component = AlertManagementComponent(
            component_id="integration-test",
            initialize_sample_data=False
        )
        
        # Reset for clean testing
        component._alert_counter = 0
        component._update_performance_metrics()
        
        # Simulate a typical workflow
        # 1. System generates alerts
        critical_alert = component.add_alert('Critical', 'Severe defect detected', 'LSTM Model')
        high_alert = component.add_alert('High', 'High probability defect', 'Ensemble')
        medium_alert = component.add_alert('Medium', 'Moderate risk detected', 'Baseline Model')
        
        # 2. Operator reviews alerts
        alerts_data = component.get_alerts_data()
        self.assertEqual(len(alerts_data), 3)
        
        # 3. Operator acknowledges critical alert first
        component.acknowledge_alert(critical_alert, 'operator1')
        
        # 4. Operator resolves medium alert directly
        component.resolve_alert(medium_alert, 'operator1')
        
        # 5. Another operator resolves the critical alert
        component.resolve_alert(critical_alert, 'operator2')
        
        # 6. Check final state
        metrics = component.performance_metrics
        self.assertEqual(metrics['total_alerts'], 3)
        self.assertEqual(metrics['active_alerts'], 1)  # high_alert still new
        self.assertEqual(metrics['resolved_alerts'], 2)
        
        # 7. Generate charts for analysis
        freq_chart = component.create_frequency_chart()
        severity_chart = component.create_severity_distribution()
        
        self.assertIsNotNone(freq_chart)
        self.assertIsNotNone(severity_chart)
        
        print("Full workflow integration test completed successfully!")


def run_tests():
    """Run all tests."""
    print("Running Alert Management Component Tests...")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test cases
    test_suite.addTest(unittest.makeSuite(TestAlertManagementComponent))
    test_suite.addTest(unittest.makeSuite(TestAlertManagementIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)