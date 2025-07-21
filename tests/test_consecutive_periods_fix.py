"""
Test to verify the consecutive periods detection logic is fixed.
This test focuses on the trigger detection, not the probabilistic defect labeling.
"""
import unittest
import pandas as pd
import numpy as np
import tempfile
import yaml
import sys
from pathlib import Path

# Add src to Python path for import
sys.path.append(str(Path(__file__).parent.parent))

class TestConsecutivePeriodsDetectionLogic(unittest.TestCase):
    """Test the consecutive periods detection logic specifically"""
    
    def setUp(self):
        """Set up test environment"""
        # Import here to avoid import issues
        from src.data.data_generator import SteelCastingDataGenerator
        self.SteelCastingDataGenerator = SteelCastingDataGenerator
        
        # Create minimal test configuration
        self.test_dir = tempfile.mkdtemp()
        self.test_config = {
            'data_generation': {
                'random_seed': 42,  # Required field
                'sensors': {
                    'mold_level_normal_range': [130, 170],
                },
                'defect_simulation': {
                    'defect_triggers': {
                        'prolonged_mold_level_deviation': 5,
                        'rapid_temperature_drop': 50,
                        'high_speed_with_low_superheat': True,
                        'high_speed_threshold': 1.5,
                        'low_superheat_threshold': 20
                    },
                    'defect_probability': 0.1,
                    'max_defect_probability': 0.8,
                    'trigger_probability_factor': 0.2
                }
            }
        }
        
        # Save test config
        self.config_path = Path(self.test_dir) / 'test_config.yaml'
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
    
    def test_consecutive_periods_trigger_detection(self):
        """Test that consecutive periods are correctly detected in triggers"""
        generator = self.SteelCastingDataGenerator(str(self.config_path))
        
        # Test data with known consecutive periods
        # Normal range is [130, 170], threshold is 5
        test_data = pd.DataFrame({
            'mold_level': [
                140, 150, 160,  # Normal (3 points)
                120, 110, 100, 105, 115,  # Outside for 5 consecutive points
                140, 150,  # Normal (2 points)  
                180, 190, 200, 175, 185, 195, 180, 190,  # Outside for 8 consecutive points
                150, 160  # Normal (2 points)
            ],
            'casting_speed': [1.2] * 20,
            'mold_temperature': [1520] * 20,
            'superheat': [35] * 20
        })
        
        # Call the trigger detection method
        has_defect, triggers = generator._detect_defect_triggers(test_data)
        
        # Check that the prolonged_mold_level_deviation trigger is detected
        # (regardless of the probabilistic has_defect outcome)
        self.assertIn('prolonged_mold_level_deviation', triggers)
    
    def test_consecutive_periods_below_threshold_no_trigger(self):
        """Test that periods below threshold don't trigger detection"""
        generator = self.SteelCastingDataGenerator(str(self.config_path))
        
        # Test data with consecutive periods below threshold (< 5)
        test_data = pd.DataFrame({
            'mold_level': [
                140, 150, 160,  # Normal (3 points)
                120, 110, 100,  # Outside for 3 consecutive points (< 5)
                140, 150,  # Normal (2 points)  
                180, 190, 175, 185,  # Outside for 4 consecutive points (< 5)
                150, 160  # Normal (2 points)
            ],
            'casting_speed': [1.2] * 14,
            'mold_temperature': [1520] * 14,
            'superheat': [35] * 14
        })
        
        # Call the trigger detection method
        has_defect, triggers = generator._detect_defect_triggers(test_data)
        
        # Check that prolonged_mold_level_deviation trigger is NOT detected
        self.assertNotIn('prolonged_mold_level_deviation', triggers)
    
    def test_consecutive_periods_exactly_at_threshold(self):
        """Test edge case where consecutive period is exactly at threshold"""
        generator = self.SteelCastingDataGenerator(str(self.config_path))
        
        # Test data with exactly 5 consecutive points outside range
        test_data = pd.DataFrame({
            'mold_level': [
                150, 150,  # Normal (2 points)
                100, 100, 100, 100, 100,  # Outside for exactly 5 consecutive points
                150, 150  # Normal (2 points)
            ],
            'casting_speed': [1.2] * 9,
            'mold_temperature': [1520] * 9,
            'superheat': [35] * 9
        })
        
        # Call the trigger detection method
        has_defect, triggers = generator._detect_defect_triggers(test_data)
        
        # Check that the trigger IS detected (5 >= 5)
        self.assertIn('prolonged_mold_level_deviation', triggers)
    
    def test_all_data_within_normal_range(self):
        """Test that no trigger is detected when all data is within normal range"""
        generator = self.SteelCastingDataGenerator(str(self.config_path))
        
        # Test data all within normal range [130, 170]
        test_data = pd.DataFrame({
            'mold_level': [150] * 10,  # All within range
            'casting_speed': [1.2] * 10,
            'mold_temperature': [1520] * 10,
            'superheat': [35] * 10
        })
        
        # Call the trigger detection method
        has_defect, triggers = generator._detect_defect_triggers(test_data)
        
        # Check that prolonged_mold_level_deviation trigger is NOT detected
        self.assertNotIn('prolonged_mold_level_deviation', triggers)

if __name__ == "__main__":
    unittest.main()