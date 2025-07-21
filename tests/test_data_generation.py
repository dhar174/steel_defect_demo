import unittest
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import json

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.data_generator import SteelCastingDataGenerator


class TestSteelCastingDataGenerator(unittest.TestCase):
    """Test cases for SteelCastingDataGenerator"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = Path.cwd()
        
        # Create test configuration
        self.test_config = {
            'data_generation': {
                'num_casts': 5,  # Small number for testing
                'cast_duration_minutes': 2,  # Short duration for testing
                'sampling_rate_hz': 1,
                'random_seed': 42,
                'sensors': {
                    'casting_speed': {
                        'base_value': 1.2,
                        'noise_std': 0.05,
                        'min_value': 0.8,
                        'max_value': 1.8
                    },
                    'mold_temperature': {
                        'base_value': 1520,
                        'noise_std': 10,
                        'min_value': 1480,
                        'max_value': 1580
                    },
                    'mold_level': {
                        'base_value': 150,
                        'noise_std': 5,
                        'min_value': 120,
                        'max_value': 180
                    },
                    'cooling_water_flow': {
                        'base_value': 200,
                        'noise_std': 15,
                        'min_value': 150,
                        'max_value': 250
                    },
                    'superheat': {
                        'base_value': 25,
                        'noise_std': 3,
                        'min_value': 15,
                        'max_value': 40
                    },
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
        
        # Save test config to temporary file
        self.config_path = Path(self.test_dir) / 'test_config.yaml'
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
        
        # Change to test directory
        import os
        os.chdir(self.test_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        import os
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def test_configuration_loading(self):
        """Test that configuration is loaded correctly"""
        generator = SteelCastingDataGenerator(str(self.config_path))
        
        self.assertEqual(generator.data_config['num_casts'], 5)
        self.assertEqual(generator.data_config['cast_duration_minutes'], 2)
        self.assertEqual(generator.data_config['random_seed'], 42)
        self.assertEqual(len(generator.sensor_config), 6)  # 5 sensors + mold_level_normal_range
    
    def test_sensor_value_generation(self):
        """Test that sensor values are within specified ranges"""
        generator = SteelCastingDataGenerator(str(self.config_path))
        
        # Test each sensor (excluding mold_level_normal_range which is not a sensor)
        sensor_names = ['casting_speed', 'mold_temperature', 'mold_level', 'cooling_water_flow', 'superheat']
        for sensor_name in sensor_names:
            sensor_cfg = generator.sensor_config[sensor_name]
            values = generator._generate_sensor_values(100, sensor_name)
            
            # Check that all values are within range
            self.assertTrue(np.all(values >= sensor_cfg['min_value']), 
                          f"{sensor_name} values below minimum")
            self.assertTrue(np.all(values <= sensor_cfg['max_value']), 
                          f"{sensor_name} values above maximum")
            
            # Check that values are reasonably close to base value
            mean_value = np.mean(values)
            self.assertLess(abs(mean_value - sensor_cfg['base_value']), 
                          sensor_cfg['noise_std'] * 3, 
                          f"{sensor_name} mean too far from base value")
    
    def test_cast_sequence_generation(self):
        """Test generation of a single cast sequence"""
        generator = SteelCastingDataGenerator(str(self.config_path))
        
        df, metadata = generator.generate_cast_sequence('test_cast_001')
        
        # Check DataFrame structure
        expected_samples = 2 * 60 * 1  # 2 minutes * 60 seconds * 1 Hz
        self.assertEqual(len(df), expected_samples)
        
        # Check columns
        expected_columns = ['casting_speed', 'mold_temperature', 'mold_level', 
                          'cooling_water_flow', 'superheat']
        self.assertEqual(list(df.columns), expected_columns)
        
        # Check index is datetime
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        
        # Check metadata structure
        self.assertEqual(metadata['cast_id'], 'test_cast_001')
        self.assertIn('defect_label', metadata)
        self.assertIn('process_summary', metadata)
        self.assertIsInstance(metadata['defect_label'], int)
        self.assertIn(metadata['defect_label'], [0, 1])
    
    def test_defect_labeling_logic(self):
        """Test that defect labeling works correctly"""
        generator = SteelCastingDataGenerator(str(self.config_path))
        
        # Generate multiple casts and check defect distribution
        defect_count = 0
        num_test_casts = 20
        
        for i in range(num_test_casts):
            df, metadata = generator.generate_cast_sequence(f'test_cast_{i:03d}')
            defect_count += metadata['defect_label']
        
        # Defect rate should be reasonable (not 0% or 100%)
        defect_rate = defect_count / num_test_casts
        self.assertGreater(defect_rate, 0.0)
        self.assertLess(defect_rate, 1.0)
    
    def test_reproducibility(self):
        """Test that results are reproducible with same random seed"""
        generator1 = SteelCastingDataGenerator(str(self.config_path))
        generator2 = SteelCastingDataGenerator(str(self.config_path))
        
        df1, metadata1 = generator1.generate_cast_sequence('test_cast')
        df2, metadata2 = generator2.generate_cast_sequence('test_cast')
        
        # DataFrames should be identical
        pd.testing.assert_frame_equal(df1, df2)
        
        # Defect labels should be the same
        self.assertEqual(metadata1['defect_label'], metadata2['defect_label'])
    
    def test_output_file_formats(self):
        """Test that output files are created with correct formats"""
        generator = SteelCastingDataGenerator(str(self.config_path))
        
        # Generate a small dataset
        generator.generate_dataset()
        
        # Check that parquet files were created
        parquet_files = list(Path('data/raw').glob('*.parquet'))
        self.assertEqual(len(parquet_files), 5)  # 5 test casts
        
        # Check that JSON files were created
        self.assertTrue(Path('data/synthetic/dataset_metadata.json').exists())
        self.assertTrue(Path('data/synthetic/generation_summary.json').exists())
        
        # Validate JSON file contents
        with open('data/synthetic/dataset_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.assertIn('dataset_info', metadata)
        self.assertIn('cast_metadata', metadata)
        self.assertEqual(metadata['dataset_info']['total_casts'], 5)


if __name__ == '__main__':
    unittest.main()
