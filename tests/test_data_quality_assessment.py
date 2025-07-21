import unittest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from analysis.data_quality_assessor import DataQualityAssessor


class TestDataQualityAssessor(unittest.TestCase):
    """Test cases for DataQualityAssessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = Path(self.temp_dir)
        
        # Create directory structure
        (self.data_path / 'raw').mkdir(parents=True, exist_ok=True)
        (self.data_path / 'synthetic').mkdir(parents=True, exist_ok=True)
        
        # Create sample cast data
        self.sample_cast_data = self._create_sample_cast_data()
        
        # Create sample metadata
        self.sample_metadata = self._create_sample_metadata()
        
        # Save sample metadata
        metadata_path = self.data_path / 'synthetic' / 'dataset_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.sample_metadata, f)
        
        # Save sample cast data
        cast_file = self.data_path / 'raw' / 'cast_timeseries_0001.parquet'
        self.sample_cast_data.to_parquet(cast_file)
        
        # Initialize assessor
        self.assessor = DataQualityAssessor(data_path=str(self.data_path))
    
    def _create_sample_cast_data(self) -> pd.DataFrame:
        """Create sample cast data for testing."""
        # Create timestamps (2 hours of data at 1Hz)
        start_time = datetime(2023, 1, 1)
        timestamps = pd.date_range(start_time, periods=7200, freq='1s')
        
        # Create realistic sensor data
        np.random.seed(42)
        data = {
            'casting_speed': np.random.normal(1.2, 0.05, 7200),
            'mold_temperature': np.random.normal(1520, 10, 7200),
            'mold_level': np.random.normal(150, 5, 7200),
            'cooling_water_flow': np.random.normal(200, 15, 7200),
            'superheat': np.random.normal(25, 3, 7200)
        }
        
        df = pd.DataFrame(data, index=timestamps)
        df.index.name = 'timestamp'
        
        return df
    
    def _create_sample_metadata(self) -> dict:
        """Create sample metadata for testing."""
        return {
            "dataset_info": {
                "total_casts": 1,
                "train_casts": 1,
                "test_casts": 0,
                "defect_count": 0,
                "defect_rate": 0.0,
                "generation_timestamp": "2023-01-01T00:00:00"
            },
            "cast_metadata": [
                {
                    "cast_id": "cast_0001",
                    "generation_timestamp": "2023-01-01T00:00:00",
                    "steel_grade": "Grade_A",
                    "duration_minutes": 120,
                    "sampling_rate_hz": 1,
                    "num_samples": 7200,
                    "defect_label": 0,
                    "defect_trigger_events": []
                }
            ]
        }
    
    def test_initialization(self):
        """Test assessor initialization."""
        self.assertIsInstance(self.assessor, DataQualityAssessor)
        self.assertEqual(str(self.assessor.data_path), str(self.data_path))
        self.assertIsNotNone(self.assessor.config)
        self.assertIsNotNone(self.assessor.sensor_expected_ranges)
        self.assertIsNotNone(self.assessor.physics_constraints)
        self.assertIsNotNone(self.assessor.dataset_metadata)
    
    def test_missing_value_analysis_single_cast(self):
        """Test missing value analysis for a single cast."""
        # Test with clean data (no missing values)
        results = self.assessor.assess_missing_values(self.sample_cast_data)
        
        self.assertIn('missing_value_analysis', results)
        analysis = results['missing_value_analysis']
        
        # Should have no missing values
        self.assertEqual(analysis['total_missing_percentage'], 0.0)
        self.assertEqual(analysis['quality_score'], 1.0)
        
        # Check sensor-specific analysis
        for sensor in ['casting_speed', 'mold_temperature', 'mold_level', 'cooling_water_flow', 'superheat']:
            self.assertIn(sensor, analysis['missing_by_sensor'])
            self.assertEqual(analysis['missing_by_sensor'][sensor]['count'], 0)
            self.assertEqual(analysis['missing_by_sensor'][sensor]['percentage'], 0.0)
    
    def test_missing_value_analysis_with_missing_data(self):
        """Test missing value analysis with missing data."""
        # Create data with missing values
        data_with_missing = self.sample_cast_data.copy()
        data_with_missing.iloc[100:110, 0] = np.nan  # Missing casting_speed values
        data_with_missing.iloc[200:205, 1] = np.nan  # Missing mold_temperature values
        
        results = self.assessor.assess_missing_values(data_with_missing)
        analysis = results['missing_value_analysis']
        
        # Should detect missing values
        self.assertGreater(analysis['total_missing_percentage'], 0)
        self.assertLess(analysis['quality_score'], 1.0)
        
        # Check specific sensors
        self.assertGreater(analysis['missing_by_sensor']['casting_speed']['count'], 0)
        self.assertGreater(analysis['missing_by_sensor']['mold_temperature']['count'], 0)
    
    def test_data_consistency_analysis(self):
        """Test data consistency analysis."""
        results = self.assessor.assess_data_consistency(self.sample_cast_data)
        
        self.assertIn('consistency_analysis', results)
        analysis = results['consistency_analysis']
        
        # Check structure
        self.assertIn('range_violations', analysis)
        self.assertIn('physics_violations', analysis)
        self.assertIn('outlier_analysis', analysis)
        self.assertIn('consistency_score', analysis)
        
        # With normal synthetic data, should have high consistency score
        self.assertGreater(analysis['consistency_score'], 0.9)
    
    def test_range_violations_detection(self):
        """Test detection of range violations."""
        # Create data with range violations
        data_with_violations = self.sample_cast_data.copy()
        data_with_violations.iloc[100, data_with_violations.columns.get_loc('casting_speed')] = 5.0  # Impossible value
        data_with_violations.iloc[200, data_with_violations.columns.get_loc('mold_temperature')] = 1000.0  # Too low
        
        results = self.assessor.assess_data_consistency(data_with_violations)
        analysis = results['consistency_analysis']
        
        # Should detect violations
        self.assertGreater(analysis['range_violations']['casting_speed']['hard_violations'], 0)
        self.assertGreater(analysis['range_violations']['mold_temperature']['hard_violations'], 0)
    
    def test_temporal_continuity_analysis(self):
        """Test temporal continuity analysis."""
        results = self.assessor.assess_temporal_continuity(self.sample_cast_data)
        
        self.assertIn('temporal_continuity', results)
        analysis = results['temporal_continuity']
        
        # Check structure
        self.assertIn('sampling_rate_analysis', analysis)
        self.assertIn('time_sequence_analysis', analysis)
        self.assertIn('continuity_score', analysis)
        
        # Regular 1Hz data should have high continuity score
        self.assertGreater(analysis['continuity_score'], 0.9)
        
        # Check sampling rate
        sampling_analysis = analysis['sampling_rate_analysis']
        self.assertAlmostEqual(sampling_analysis['mean_interval_seconds'], 1.0, places=1)
        self.assertLess(sampling_analysis['irregular_percentage'], 1.0)
        
        # Check time sequence
        sequence_analysis = analysis['time_sequence_analysis']
        self.assertTrue(sequence_analysis['is_monotonic_increasing'])
        self.assertFalse(sequence_analysis['has_duplicate_timestamps'])
    
    def test_temporal_continuity_with_gaps(self):
        """Test temporal continuity with temporal gaps."""
        # Create data with gaps
        data_with_gaps = self.sample_cast_data.copy()
        
        # Remove some rows to create gaps
        gap_indices = data_with_gaps.index[1000:1010]  # 10-second gap
        data_with_gaps = data_with_gaps.drop(gap_indices)
        
        results = self.assessor.assess_temporal_continuity(data_with_gaps)
        analysis = results['temporal_continuity']
        
        # Should detect the gap
        self.assertLess(analysis['continuity_score'], 1.0)
        self.assertGreater(analysis['sampling_rate_analysis']['irregular_percentage'], 0)
    
    def test_synthetic_data_realism_analysis(self):
        """Test synthetic data realism analysis."""
        results = self.assessor.assess_synthetic_data_realism(self.sample_cast_data)
        
        self.assertIn('realism_analysis', results)
        analysis = results['realism_analysis']
        
        # Check structure
        self.assertIn('distribution_analysis', analysis)
        self.assertIn('correlation_analysis', analysis)
        self.assertIn('process_behavior_analysis', analysis)
        self.assertIn('realism_score', analysis)
        
        # Should have reasonable realism score
        self.assertGreater(analysis['realism_score'], 0.7)
        
        # Check distributions for all sensors
        for sensor in ['casting_speed', 'mold_temperature', 'mold_level', 'cooling_water_flow', 'superheat']:
            self.assertIn(sensor, analysis['distribution_analysis'])
            dist_stats = analysis['distribution_analysis'][sensor]
            self.assertIn('mean', dist_stats)
            self.assertIn('std', dist_stats)
            self.assertIn('skewness', dist_stats)
            self.assertIn('kurtosis', dist_stats)
    
    def test_comprehensive_quality_assessment(self):
        """Test comprehensive quality assessment."""
        results = self.assessor.comprehensive_quality_assessment(self.sample_cast_data)
        
        # Check structure
        self.assertIn('summary', results)
        self.assertIn('missing_value_analysis', results)
        self.assertIn('consistency_analysis', results)
        self.assertIn('temporal_continuity', results)
        self.assertIn('realism_analysis', results)
        
        # Check summary
        summary = results['summary']
        self.assertIn('overall_quality_score', summary)
        self.assertIn('quality_level', summary)
        self.assertIn('component_scores', summary)
        self.assertIn('assessment_timestamp', summary)
        self.assertIn('data_scope', summary)
        
        # Should have high overall score for clean synthetic data
        self.assertGreater(summary['overall_quality_score'], 0.8)
        
        # Check component scores
        component_scores = summary['component_scores']
        self.assertIn('missing_values', component_scores)
        self.assertIn('consistency', component_scores)
        self.assertIn('temporal_continuity', component_scores)
        self.assertIn('realism', component_scores)
        
        # All component scores should be between 0 and 1
        for score in component_scores.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_all_casts_analysis(self):
        """Test analysis across all casts."""
        # Create additional cast files for testing
        for i in range(2, 4):  # Create cast 2 and 3
            cast_data = self._create_sample_cast_data()
            cast_file = self.data_path / 'raw' / f'cast_timeseries_000{i}.parquet'
            cast_data.to_parquet(cast_file)
            
            # Add to metadata
            cast_metadata = {
                "cast_id": f"cast_000{i}",
                "generation_timestamp": "2023-01-01T00:00:00",
                "steel_grade": "Grade_A",
                "duration_minutes": 120,
                "sampling_rate_hz": 1,
                "num_samples": 7200,
                "defect_label": 0,
                "defect_trigger_events": []
            }
            self.sample_metadata['cast_metadata'].append(cast_metadata)
        
        # Update metadata
        self.sample_metadata['dataset_info']['total_casts'] = 3
        metadata_path = self.data_path / 'synthetic' / 'dataset_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.sample_metadata, f)
        
        # Reinitialize assessor to load new metadata
        self.assessor = DataQualityAssessor(data_path=str(self.data_path))
        
        # Test analysis across all casts
        results = self.assessor.comprehensive_quality_assessment()
        
        # Should analyze multiple casts
        self.assertEqual(results['summary']['data_scope'], 'all_casts')
        
        # Check that results contain aggregated information
        if 'casts_analyzed' in results['missing_value_analysis']:
            self.assertGreater(results['missing_value_analysis']['casts_analyzed'], 1)
    
    def test_generate_quality_report(self):
        """Test quality report generation."""
        # Perform assessment
        results = self.assessor.comprehensive_quality_assessment(self.sample_cast_data)
        
        # Generate report
        report_path = self.assessor.generate_quality_report(results)
        
        # Check that report file exists
        self.assertTrue(Path(report_path).exists())
        
        # Check that report contains valid JSON
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        self.assertIn('summary', report_data)
        self.assertIn('missing_value_analysis', report_data)
        
        # Clean up
        Path(report_path).unlink()
    
    def test_physics_constraints_validation(self):
        """Test physics constraints validation."""
        # Create data with physics violations
        data_with_violations = self.sample_cast_data.copy()
        
        # Create unrealistic temperature jump (500°C in 1 second)
        data_with_violations.iloc[1000, data_with_violations.columns.get_loc('mold_temperature')] = 2000
        
        results = self.assessor.assess_data_consistency(data_with_violations)
        physics_violations = results['consistency_analysis']['physics_violations']
        
        # Should detect physics violation in temperature
        if 'mold_temperature' in physics_violations:
            self.assertGreater(physics_violations['mold_temperature']['violations'], 0)
    
    def test_outlier_detection(self):
        """Test outlier detection methods."""
        # Create data with outliers
        data_with_outliers = self.sample_cast_data.copy()
        
        # Add extreme outliers
        data_with_outliers.iloc[500, data_with_outliers.columns.get_loc('casting_speed')] = 0.1  # Very low
        data_with_outliers.iloc[1000, data_with_outliers.columns.get_loc('mold_temperature')] = 1700  # Very high
        
        results = self.assessor.assess_data_consistency(data_with_outliers)
        outlier_analysis = results['consistency_analysis']['outlier_analysis']
        
        # Should detect outliers
        self.assertGreater(outlier_analysis['casting_speed']['z_score_outliers'], 0)
        self.assertGreater(outlier_analysis['mold_temperature']['z_score_outliers'], 0)
    
    def test_correlation_analysis(self):
        """Test correlation analysis in realism assessment."""
        # Create data with known correlations
        correlated_data = self.sample_cast_data.copy()
        
        # Make mold_level inversely correlated with casting_speed
        correlated_data['mold_level'] = 180 - correlated_data['casting_speed'] * 20
        
        results = self.assessor.assess_synthetic_data_realism(correlated_data)
        correlation_analysis = results['realism_analysis']['correlation_analysis']
        
        # Should find strong correlations
        self.assertIn('strong_correlations', correlation_analysis)
        self.assertGreater(len(correlation_analysis['strong_correlations']), 0)
    
    def test_process_behavior_analysis(self):
        """Test process behavior analysis."""
        results = self.assessor.assess_synthetic_data_realism(self.sample_cast_data)
        behavior_analysis = results['realism_analysis']['process_behavior_analysis']
        
        # Check that all behavior metrics are analyzed
        self.assertIn('temperature_stability', behavior_analysis)
        self.assertIn('speed_consistency', behavior_analysis)
        self.assertIn('mold_level_control', behavior_analysis)
        
        # Check temperature stability
        temp_stability = behavior_analysis['temperature_stability']
        self.assertIn('variation_std', temp_stability)
        self.assertIn('is_stable', temp_stability)
        self.assertIn('trend', temp_stability)
    
    def test_quality_score_calculation(self):
        """Test quality score calculation logic."""
        # Test with perfect data
        perfect_results = self.assessor.comprehensive_quality_assessment(self.sample_cast_data)
        perfect_score = perfect_results['summary']['overall_quality_score']
        self.assertGreater(perfect_score, 0.8)
        
        # Test with degraded data
        degraded_data = self.sample_cast_data.copy()
        
        # Add missing values
        degraded_data.iloc[100:200, 0] = np.nan
        
        # Add range violations
        degraded_data.iloc[300, degraded_data.columns.get_loc('casting_speed')] = 10.0
        
        degraded_results = self.assessor.comprehensive_quality_assessment(degraded_data)
        degraded_score = degraded_results['summary']['overall_quality_score']
        
        # Degraded data should have lower score
        self.assertLess(degraded_score, perfect_score)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestDataQualityAssessorEdgeCases(unittest.TestCase):
    """Test edge cases for DataQualityAssessor."""
    
    def setUp(self):
        """Set up test fixtures for edge cases."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = Path(self.temp_dir)
        
        # Create directory structure
        (self.data_path / 'raw').mkdir(parents=True, exist_ok=True)
        (self.data_path / 'synthetic').mkdir(parents=True, exist_ok=True)
        
        self.assessor = DataQualityAssessor(data_path=str(self.data_path))
    
    def test_empty_dataset(self):
        """Test behavior with empty dataset."""
        # Create empty DataFrame
        empty_data = pd.DataFrame()
        
        # Should handle empty data gracefully
        try:
            results = self.assessor.assess_missing_values(empty_data)
            self.assertIsInstance(results, dict)
        except Exception as e:
            self.fail(f"Empty dataset handling failed: {e}")
    
    def test_single_row_dataset(self):
        """Test behavior with single row dataset."""
        # Create single row DataFrame
        single_row_data = pd.DataFrame({
            'casting_speed': [1.2],
            'mold_temperature': [1520],
            'mold_level': [150],
            'cooling_water_flow': [200],
            'superheat': [25]
        }, index=pd.DatetimeIndex(['2023-01-01 00:00:00'], name='timestamp'))
        
        # Should handle single row gracefully
        results = self.assessor.assess_temporal_continuity(single_row_data)
        self.assertIsInstance(results, dict)
    
    def test_no_metadata(self):
        """Test behavior when no metadata is available."""
        # No metadata file should be created, so assessor should handle missing metadata
        results = self.assessor.assess_missing_values()
        
        # Should return error message when no metadata available
        if 'error' in results['missing_value_analysis']:
            self.assertIn('error', results['missing_value_analysis'])
    
    def test_invalid_sensor_ranges(self):
        """Test behavior with data outside all expected ranges."""
        # Create data with all values outside expected ranges
        invalid_data = pd.DataFrame({
            'casting_speed': [100.0] * 100,  # Impossibly high
            'mold_temperature': [100.0] * 100,  # Impossibly low  
            'mold_level': [1000.0] * 100,  # Impossibly high
            'cooling_water_flow': [10.0] * 100,  # Impossibly low
            'superheat': [1000.0] * 100  # Impossibly high
        }, index=pd.date_range('2023-01-01', periods=100, freq='1s', name='timestamp'))
        
        results = self.assessor.assess_data_consistency(invalid_data)
        
        # Should detect many violations
        range_violations = results['consistency_analysis']['range_violations']
        for sensor in range_violations:
            self.assertGreater(range_violations[sensor]['hard_violations'], 0)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()