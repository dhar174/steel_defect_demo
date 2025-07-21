import unittest
import sys
import tempfile
import shutil
import os
from pathlib import Path
import yaml
from unittest.mock import patch

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.data_generator import SteelCastingDataGenerator
from utils.metrics import MetricsCalculator


class TestConfigurableMagicNumbers(unittest.TestCase):
    """Test cases for configurable magic numbers in validation sampling logic"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = Path.cwd()
        
    def tearDown(self):
        """Clean up test environment"""
        # Change back to original directory
        os.chdir(self.original_cwd)
        # Clean up temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
    def test_progress_reporting_frequency_configurable(self):
        """Test that progress reporting frequency can be configured"""
        # Test config with custom progress reporting frequency
        test_config = {
            'data_generation': {
                'num_casts': 25,  # Small number for testing
                'cast_duration_minutes': 1,
                'sampling_rate_hz': 1,
                'random_seed': 42,
                'progress_reporting_frequency': 5,  # Custom frequency
                'sensors': {
                    'casting_speed': {'base_value': 1.2, 'noise_std': 0.05, 'min_value': 0.8, 'max_value': 1.8},
                    'mold_temperature': {'base_value': 1520, 'noise_std': 10, 'min_value': 1480, 'max_value': 1580},
                    'mold_level': {'base_value': 150, 'noise_std': 5, 'min_value': 120, 'max_value': 180},
                    'cooling_water_flow': {'base_value': 200, 'noise_std': 15, 'min_value': 150, 'max_value': 250},
                    'superheat': {'base_value': 25, 'noise_std': 3, 'min_value': 15, 'max_value': 40},
                    'mold_level_normal_range': [130, 170]
                },
                'defect_simulation': {
                    'defect_probability': 0.15,
                    'max_defect_probability': 0.8,
                    'trigger_probability_factor': 0.2,
                    'defect_triggers': {
                        'prolonged_mold_level_deviation': 30,
                        'rapid_temperature_drop': 50,
                        'high_speed_with_low_superheat': True
                    }
                },
                'output': {
                    'raw_data_format': 'parquet',
                    'metadata_format': 'json',
                    'train_test_split': 0.8
                }
            }
        }
        
        # Create config file
        config_path = Path(self.test_dir) / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Change to test directory
        os.chdir(self.test_dir)
        
        # Create generator
        generator = SteelCastingDataGenerator(str(config_path))
        
        # Verify that the progress frequency is read correctly
        self.assertEqual(generator.data_config.get('progress_reporting_frequency', 100), 5)
        
        # Mock print to capture progress messages
        with patch('builtins.print') as mock_print:
            generator.generate_dataset()
            
            # Check that progress was reported at the correct intervals (every 5 casts)
            print_calls = [call for call in mock_print.call_args_list if 'Generated' in str(call)]
            
            # Should have progress messages at casts 5, 10, 15, 20, 25
            expected_calls = 5  # 25 casts / 5 frequency = 5 calls
            self.assertEqual(len(print_calls), expected_calls)
    
    def test_progress_reporting_frequency_default_fallback(self):
        """Test that progress reporting uses default value when not configured"""
        # Test config without progress reporting frequency
        test_config = {
            'data_generation': {
                'num_casts': 5,
                'cast_duration_minutes': 1,
                'sampling_rate_hz': 1,
                'random_seed': 42,
                # progress_reporting_frequency intentionally omitted
                'sensors': {
                    'casting_speed': {'base_value': 1.2, 'noise_std': 0.05, 'min_value': 0.8, 'max_value': 1.8},
                    'mold_temperature': {'base_value': 1520, 'noise_std': 10, 'min_value': 1480, 'max_value': 1580},
                    'mold_level': {'base_value': 150, 'noise_std': 5, 'min_value': 120, 'max_value': 180},
                    'cooling_water_flow': {'base_value': 200, 'noise_std': 15, 'min_value': 150, 'max_value': 250},
                    'superheat': {'base_value': 25, 'noise_std': 3, 'min_value': 15, 'max_value': 40},
                    'mold_level_normal_range': [130, 170]
                },
                'defect_simulation': {
                    'defect_probability': 0.15,
                    'max_defect_probability': 0.8,
                    'trigger_probability_factor': 0.2,
                    'defect_triggers': {
                        'prolonged_mold_level_deviation': 30,
                        'rapid_temperature_drop': 50,
                        'high_speed_with_low_superheat': True
                    }
                },
                'output': {
                    'raw_data_format': 'parquet',
                    'metadata_format': 'json',
                    'train_test_split': 0.8
                }
            }
        }
        
        # Create config file
        config_path = Path(self.test_dir) / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Create generator
        generator = SteelCastingDataGenerator(str(config_path))
        
        # Verify that the default value (100) is used when not configured
        self.assertEqual(generator.data_config.get('progress_reporting_frequency', 100), 100)
    
    def test_metrics_cost_defaults_configurable(self):
        """Test that metrics cost defaults can be configured"""
        # Test config with custom cost defaults
        test_config = {
            'evaluation': {
                'cost_sensitive_defaults': {
                    'false_positive_cost': 2.5,
                    'false_negative_cost': 15.0
                }
            }
        }
        
        # Create metrics calculator with config
        calculator = MetricsCalculator(config=test_config)
        
        # Verify that the configured values are used
        self.assertEqual(calculator.default_false_positive_cost, 2.5)
        self.assertEqual(calculator.default_false_negative_cost, 15.0)
    
    def test_metrics_cost_defaults_fallback(self):
        """Test that metrics cost defaults fallback to hardcoded values when not configured"""
        # Create metrics calculator without config
        calculator = MetricsCalculator()
        
        # Verify that the hardcoded defaults are used
        self.assertEqual(calculator.default_false_positive_cost, 1.0)
        self.assertEqual(calculator.default_false_negative_cost, 10.0)
    
    def test_metrics_cost_defaults_partial_config(self):
        """Test that metrics cost defaults handle partial configuration"""
        # Test config with only one cost configured
        test_config = {
            'evaluation': {
                'cost_sensitive_defaults': {
                    'false_positive_cost': 3.0
                    # false_negative_cost intentionally omitted
                }
            }
        }
        
        # Create metrics calculator with partial config
        calculator = MetricsCalculator(config=test_config)
        
        # Verify that configured value is used and missing value falls back to default
        self.assertEqual(calculator.default_false_positive_cost, 3.0)
        self.assertEqual(calculator.default_false_negative_cost, 10.0)  # fallback default


if __name__ == '__main__':
    unittest.main()