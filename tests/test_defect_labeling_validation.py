"""Tests for defect labeling validation functionality"""

import unittest
import sys
import tempfile
import shutil
import os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import json

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils.defect_labeling_validator import DefectLabelingValidator
from data.data_generator import SteelCastingDataGenerator


class TestDefectLabelingValidator(unittest.TestCase):
    """Test cases for DefectLabelingValidator"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = Path.cwd()
        
        # Create test configuration
        self.test_config = {
            'data_generation': {
                'num_casts': 10,
                'cast_duration_minutes': 2,
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
                    'defect_probability': 0.3,  # Higher for testing
                    'max_defect_probability': 0.8,
                    'trigger_probability_factor': 0.4,
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
        
        # Initialize validator
        self.validator = DefectLabelingValidator(self.test_config)
        
        # Generate test data
        self._generate_test_data()
        
        # Change to test directory
        import os
        os.chdir(self.test_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        import os
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def _generate_test_data(self):
        """Generate test dataset"""
        config_path = Path(self.test_dir) / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.test_config, f)
        
        os.chdir(self.test_dir)
        generator = SteelCastingDataGenerator(str(config_path))
        generator.generate_dataset()
        
        # Load generated metadata
        with open('data/synthetic/dataset_metadata.json', 'r') as f:
            self.dataset_info = json.load(f)
        
        self.cast_metadata_list = self.dataset_info['cast_metadata']
    
    def test_validator_initialization(self):
        """Test validator initialization"""
        self.assertIsInstance(self.validator, DefectLabelingValidator)
        self.assertIn('mold_level_critical_deviation', self.validator.domain_rules)
        self.assertIn('normal_operation_ranges', self.validator.domain_rules)
    
    def test_analyze_label_distribution(self):
        """Test label distribution analysis"""
        result = self.validator.analyze_label_distribution(self.cast_metadata_list)
        
        # Check result structure
        self.assertEqual(result['analysis_type'], 'label_distribution')
        self.assertIn('dataset_summary', result)
        self.assertIn('trigger_analysis', result)
        self.assertIn('grade_analysis', result)
        self.assertIn('statistical_tests', result)
        
        # Check dataset summary
        dataset_summary = result['dataset_summary']
        self.assertEqual(dataset_summary['total_casts'], 10)
        self.assertIsInstance(dataset_summary['defect_rate'], float)
        self.assertGreaterEqual(dataset_summary['defect_rate'], 0.0)
        self.assertLessEqual(dataset_summary['defect_rate'], 1.0)
        
        # Check trigger analysis
        trigger_analysis = result['trigger_analysis']
        self.assertIn('individual_triggers', trigger_analysis)
        self.assertIn('trigger_coverage', trigger_analysis)
        self.assertIsInstance(trigger_analysis['trigger_coverage'], float)
    
    def test_validate_domain_knowledge_alignment(self):
        """Test domain knowledge validation"""
        # Get a test cast
        test_cast = self.cast_metadata_list[0]
        
        # Load corresponding time series data
        cast_number = int(test_cast['cast_id'].split('_')[-1])
        df = pd.read_parquet(f'data/raw/cast_timeseries_{cast_number:04d}.parquet')
        
        result = self.validator.validate_domain_knowledge_alignment(df, test_cast)
        
        # Check result structure
        self.assertEqual(result['validation_type'], 'domain_knowledge')
        self.assertEqual(result['cast_id'], test_cast['cast_id'])
        self.assertIn('domain_checks', result)
        self.assertIn('severity_assessment', result)
        self.assertIn('passed', result)
        self.assertIsInstance(result['passed'], bool)
    
    def test_identify_edge_cases(self):
        """Test edge case identification"""
        # Get a test cast
        test_cast = self.cast_metadata_list[0]
        
        # Load corresponding time series data
        cast_number = int(test_cast['cast_id'].split('_')[-1])
        df = pd.read_parquet(f'data/raw/cast_timeseries_{cast_number:04d}.parquet')
        
        result = self.validator.identify_edge_cases(df, test_cast)
        
        # Check result structure
        self.assertEqual(result['analysis_type'], 'edge_case_detection')
        self.assertEqual(result['cast_id'], test_cast['cast_id'])
        self.assertIn('edge_case_flags', result)
        self.assertIn('borderline_conditions', result)
        self.assertIn('uncertainty_score', result)
        self.assertIn('requires_expert_review', result)
        
        # Check uncertainty score is valid
        self.assertIsInstance(result['uncertainty_score'], float)
        self.assertGreaterEqual(result['uncertainty_score'], 0.0)
        self.assertLessEqual(result['uncertainty_score'], 1.0)
    
    def test_domain_trigger_validation(self):
        """Test validation of specific domain triggers"""
        # Create test data with known defect conditions
        timestamps = pd.date_range('2023-01-01', periods=120, freq='1s')
        
        # Test mold level deviation
        test_data = {
            'casting_speed': np.full(120, 1.2),
            'mold_temperature': np.full(120, 1520),
            'mold_level': np.full(120, 150),
            'cooling_water_flow': np.full(120, 200),
            'superheat': np.full(120, 25)
        }
        
        # Create prolonged mold level deviation (40 seconds outside range)
        test_data['mold_level'][30:70] = 120  # 40 seconds below normal range
        
        df = pd.DataFrame(test_data, index=timestamps)
        
        # Validate mold level domain logic
        result = self.validator._validate_mold_level_domain_logic(df)
        
        self.assertEqual(result['trigger_type'], 'mold_level_deviation')
        self.assertGreaterEqual(result['max_consecutive_seconds'], 30)
        self.assertIn('domain_assessment', result)
        self.assertIn('justification', result)
    
    def test_borderline_condition_detection(self):
        """Test detection of borderline conditions"""
        # Create test data with borderline conditions
        timestamps = pd.date_range('2023-01-01', periods=120, freq='1s')
        
        test_data = {
            'casting_speed': np.full(120, 1.45),  # Near 1.5 threshold
            'mold_temperature': np.full(120, 1520),
            'mold_level': np.full(120, 132),  # Near 130 boundary
            'cooling_water_flow': np.full(120, 200),
            'superheat': np.full(120, 19)  # Near 20 threshold
        }
        
        df = pd.DataFrame(test_data, index=timestamps)
        
        result = self.validator._detect_borderline_conditions(df)
        
        self.assertIn('near_threshold_conditions', result)
        self.assertIn('near_threshold_count', result)
        self.assertGreaterEqual(result['near_threshold_count'], 0)
    
    def test_conflicting_signals_detection(self):
        """Test detection of conflicting signals"""
        # Create test data with conflicting signals
        timestamps = pd.date_range('2023-01-01', periods=120, freq='1s')
        
        test_data = {
            'casting_speed': np.full(120, 0.9),  # Low speed
            'mold_temperature': np.full(120, 1560),  # High temperature (conflicting)
            'mold_level': np.full(120, 150),
            'cooling_water_flow': np.full(120, 200),
            'superheat': np.full(120, 35)  # High superheat
        }
        
        # Add high cooling with high superheat (conflicting)
        test_data['cooling_water_flow'][:60] = 240  # High cooling
        
        df = pd.DataFrame(test_data, index=timestamps)
        
        conflicts = self.validator._detect_conflicting_signals(df)
        
        self.assertIsInstance(conflicts, list)
        # Should detect at least the low speed + high temperature conflict
        self.assertGreater(len(conflicts), 0)
    
    def test_statistical_analysis(self):
        """Test statistical analysis functionality"""
        # Separate defect and good casts
        defect_casts = [cast for cast in self.cast_metadata_list if cast.get('defect_label', 0)]
        good_casts = [cast for cast in self.cast_metadata_list if not cast.get('defect_label', 0)]
        
        if defect_casts and good_casts:
            result = self.validator._perform_statistical_tests(defect_casts, good_casts)
            
            self.assertTrue(result['sufficient_data'])
            self.assertIn('casting_speed', result)
            self.assertIn('mold_temperature', result)
            
            # Check that means are calculated
            self.assertIsInstance(result['casting_speed']['defect_mean'], float)
            self.assertIsInstance(result['casting_speed']['good_mean'], float)
    
    def test_trigger_effectiveness_analysis(self):
        """Test trigger effectiveness analysis"""
        # Create mock label distribution with trigger analysis
        label_distribution = {
            'trigger_analysis': {
                'individual_triggers': {
                    'prolonged_mold_level_deviation': 3,
                    'rapid_temperature_drop': 2,
                    'high_speed_with_low_superheat': 1
                },
                'trigger_coverage': 0.8
            }
        }
        
        # Create mock domain validations
        domain_validations = [
            {'triggers': ['prolonged_mold_level_deviation'], 'passed': True},
            {'triggers': ['rapid_temperature_drop'], 'passed': True},
            {'triggers': ['prolonged_mold_level_deviation'], 'passed': False},
            {'triggers': ['high_speed_with_low_superheat'], 'passed': True}
        ]
        
        result = self.validator._analyze_trigger_effectiveness(label_distribution, domain_validations)
        
        self.assertIn('trigger_frequency', result)
        self.assertIn('trigger_reliability', result)
        self.assertIn('coverage', result)
        
        # Check reliability calculation
        self.assertIn('prolonged_mold_level_deviation', result['trigger_reliability'])
        # Should be 0.5 (1 passed out of 2 occurrences)
        self.assertEqual(result['trigger_reliability']['prolonged_mold_level_deviation'], 0.5)
    
    def test_expert_review_documentation_generation(self):
        """Test expert review documentation generation"""
        # Create minimal test data
        label_distribution = {
            'dataset_summary': {
                'total_casts': 10,
                'defect_rate': 0.3,
                'target_defect_rate': 0.3
            },
            'trigger_analysis': {
                'individual_triggers': {'prolonged_mold_level_deviation': 2}
            },
            'statistical_tests': {'sufficient_data': True},
            'recommendations': ['Test recommendation']
        }
        
        domain_validations = [
            {'cast_id': 'test_001', 'passed': True},
            {'cast_id': 'test_002', 'passed': False, 'issues': ['Test issue']}
        ]
        
        edge_cases = [
            {'cast_id': 'test_001', 'requires_expert_review': False, 'uncertainty_score': 0.3},
            {'cast_id': 'test_002', 'requires_expert_review': True, 'uncertainty_score': 0.8}
        ]
        
        output_path = Path(self.test_dir) / 'test_expert_review.json'
        
        result = self.validator.generate_expert_review_documentation(
            label_distribution, domain_validations, edge_cases, output_path
        )
        
        # Check result structure
        self.assertEqual(result['document_type'], 'expert_review_documentation')
        self.assertIn('executive_summary', result)
        self.assertIn('detailed_findings', result)
        self.assertIn('recommendations', result)
        self.assertIn('appendices', result)
        
        # Check file was created
        self.assertTrue(output_path.exists())
        
        # Verify file content
        with open(output_path, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data['document_type'], 'expert_review_documentation')
    
    def test_severity_assessment(self):
        """Test defect severity assessment"""
        # Create test data with varying severity conditions
        timestamps = pd.date_range('2023-01-01', periods=120, freq='1s')
        
        # Mild severity test
        test_data_mild = {
            'casting_speed': np.full(120, 1.2),
            'mold_temperature': np.full(120, 1520),
            'mold_level': np.full(120, 150),
            'cooling_water_flow': np.full(120, 200),
            'superheat': np.full(120, 25)
        }
        
        # Create mild mold level deviation (35 seconds)
        test_data_mild['mold_level'][30:65] = 125
        
        df_mild = pd.DataFrame(test_data_mild, index=timestamps)
        
        result_mild = self.validator._assess_defect_severity(df_mild, ['prolonged_mold_level_deviation'])
        
        self.assertIn('overall_severity', result_mild)
        self.assertIn('severity_level', result_mild)
        self.assertIsInstance(result_mild['overall_severity'], float)
        self.assertIn(result_mild['severity_level'], ['Minimal', 'Low', 'Moderate', 'High', 'Critical'])
    
    def test_helper_methods(self):
        """Test helper methods"""
        # Test _get_max_consecutive_true
        test_series = pd.Series([False, True, True, True, False, True, True, False])
        max_consecutive = self.validator._get_max_consecutive_true(test_series)
        self.assertEqual(max_consecutive, 3)
        
        # Test with no True values
        test_series_false = pd.Series([False, False, False])
        max_consecutive_false = self.validator._get_max_consecutive_true(test_series_false)
        self.assertEqual(max_consecutive_false, 0)
        
        # Test severity categorization
        self.assertEqual(self.validator._categorize_severity(0.1), 'Minimal')
        self.assertEqual(self.validator._categorize_severity(0.3), 'Low')
        self.assertEqual(self.validator._categorize_severity(0.5), 'Moderate')
        self.assertEqual(self.validator._categorize_severity(0.7), 'High')
        self.assertEqual(self.validator._categorize_severity(0.9), 'Critical')


class TestDefectLabelingValidatorIntegration(unittest.TestCase):
    """Integration tests for defect labeling validation"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = Path.cwd()
        
        # Minimal config for integration tests
        self.test_config = {
            'data_generation': {
                'num_casts': 5,
                'cast_duration_minutes': 1,
                'sampling_rate_hz': 1,
                'random_seed': 42,
                'sensors': {
                    'casting_speed': {'base_value': 1.2, 'noise_std': 0.05, 'min_value': 0.8, 'max_value': 1.8},
                    'mold_temperature': {'base_value': 1520, 'noise_std': 10, 'min_value': 1480, 'max_value': 1580},
                    'mold_level': {'base_value': 150, 'noise_std': 5, 'min_value': 120, 'max_value': 180},
                    'cooling_water_flow': {'base_value': 200, 'noise_std': 15, 'min_value': 150, 'max_value': 250},
                    'superheat': {'base_value': 25, 'noise_std': 3, 'min_value': 15, 'max_value': 40},
                    'mold_level_normal_range': [130, 170]
                },
                'defect_simulation': {
                    'defect_probability': 0.4,
                    'max_defect_probability': 0.8,
                    'trigger_probability_factor': 0.3,
                    'defect_triggers': {
                        'prolonged_mold_level_deviation': 20,
                        'rapid_temperature_drop': 30,
                        'high_speed_with_low_superheat': True
                    }
                },
                'output': {'raw_data_format': 'parquet', 'metadata_format': 'json', 'train_test_split': 0.8}
            }
        }
        
        self.validator = DefectLabelingValidator(self.test_config)
        
        import os
        os.chdir(self.test_dir)
    
    def tearDown(self):
        """Clean up integration test environment"""
        import os
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def test_end_to_end_validation_workflow(self):
        """Test complete validation workflow"""
        # Generate test data
        config_path = Path(self.test_dir) / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.test_config, f)
        
        generator = SteelCastingDataGenerator(str(config_path))
        generator.generate_dataset()
        
        # Load metadata
        with open('data/synthetic/dataset_metadata.json', 'r') as f:
            dataset_info = json.load(f)
        
        cast_metadata_list = dataset_info['cast_metadata']
        
        # Run full validation workflow
        
        # 1. Label distribution analysis
        label_distribution = self.validator.analyze_label_distribution(cast_metadata_list)
        self.assertIn('dataset_summary', label_distribution)
        
        # 2. Domain knowledge validation (sample)
        domain_validations = []
        for cast_metadata in cast_metadata_list[:3]:  # Test first 3 casts
            cast_number = int(cast_metadata['cast_id'].split('_')[-1])
            df = pd.read_parquet(f'data/raw/cast_timeseries_{cast_number:04d}.parquet')
            
            domain_result = self.validator.validate_domain_knowledge_alignment(df, cast_metadata)
            domain_validations.append(domain_result)
        
        self.assertEqual(len(domain_validations), 3)
        
        # 3. Edge case detection
        edge_cases = []
        for cast_metadata in cast_metadata_list[:3]:
            cast_number = int(cast_metadata['cast_id'].split('_')[-1])
            df = pd.read_parquet(f'data/raw/cast_timeseries_{cast_number:04d}.parquet')
            
            edge_case_result = self.validator.identify_edge_cases(df, cast_metadata)
            edge_cases.append(edge_case_result)
        
        self.assertEqual(len(edge_cases), 3)
        
        # 4. Expert review documentation
        output_path = Path(self.test_dir) / 'expert_review.json'
        expert_review = self.validator.generate_expert_review_documentation(
            label_distribution, domain_validations, edge_cases, output_path
        )
        
        self.assertTrue(output_path.exists())
        self.assertIn('executive_summary', expert_review)
        
        # Verify all components work together
        self.assertIsInstance(expert_review['executive_summary']['overall_assessment'], str)
        self.assertIn('key_findings', expert_review['executive_summary'])


if __name__ == '__main__':
    unittest.main()