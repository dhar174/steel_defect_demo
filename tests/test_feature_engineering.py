import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from features.feature_engineer import CastingFeatureEngineer
from features.feature_extractor import SequenceFeatureExtractor
from features.feature_validation import FeatureValidator
from features.feature_utils import calculate_percentiles, detect_spikes, safe_correlation

class TestFeatureEngineering:
    """Test suite for feature engineering components"""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create sample time series data with proper sensor columns
        np.random.seed(42)
        self.sample_cast_data = pd.DataFrame({
            'temperature': np.random.normal(1520, 10, 100),
            'pressure': np.random.normal(150, 5, 100),
            'flow_rate': np.random.normal(200, 15, 100),
            'vibration': np.random.normal(1.2, 0.05, 100),
            'power_consumption': np.random.normal(25, 3, 100)
        })
        
        # Create data with some issues for testing edge cases
        self.problematic_data = pd.DataFrame({
            'temperature': [1520] * 50 + [np.nan] * 50,  # Constant then missing
            'pressure': np.random.normal(150, 5, 100),
            'flow_rate': np.random.normal(200, 15, 100),
            'vibration': np.random.normal(1.2, 0.05, 100),
            'power_consumption': np.random.normal(25, 3, 100)
        })
        
        self.feature_engineer = CastingFeatureEngineer()
        self.sequence_extractor = SequenceFeatureExtractor(sequence_length=50)
        self.validator = FeatureValidator()
    
    def test_statistical_feature_extraction(self):
        """Test statistical feature extraction."""
        features = self.feature_engineer.extract_statistical_features(self.sample_cast_data)
        
        # Should have 9 stats × 5 sensors = 45 features (mean, std, min, max, median + 4 percentiles)
        expected_features = 9 * 5
        assert len(features.columns) == expected_features
        
        # Check specific features exist
        assert 'temperature_mean' in features.columns
        assert 'temperature_std' in features.columns
        assert 'temperature_p90' in features.columns
        
        # Check values are reasonable
        temp_mean = features['temperature_mean'].iloc[0]
        assert 1500 < temp_mean < 1540  # Should be around 1520
        
        # Test with missing data
        features_prob = self.feature_engineer.extract_statistical_features(self.problematic_data)
        assert not features_prob.empty
        assert features_prob['temperature_std'].iloc[0] == 0.0  # Constant data should have std=0
    
    def test_stability_feature_extraction(self):
        """Test stability feature extraction."""
        features = self.feature_engineer.extract_stability_features(self.sample_cast_data)
        
        # Should have 4 metrics × 5 sensors = 20 features
        expected_features = 4 * 5
        assert len(features.columns) == expected_features
        
        # Check specific features exist
        assert 'temperature_spike_count' in features.columns
        assert 'temperature_cv' in features.columns
        
        # Check values are reasonable
        spike_count = features['temperature_spike_count'].iloc[0]
        assert spike_count >= 0
        
        cv = features['temperature_cv'].iloc[0]
        assert cv > 0  # Should have some variation
    
    def test_duration_feature_extraction(self):
        """Test duration-based feature extraction."""
        features = self.feature_engineer.extract_duration_features(self.sample_cast_data)
        
        # Should have 3 metrics × 5 sensors = 15 features
        expected_features = 3 * 5
        assert len(features.columns) == expected_features
        
        # Check specific features exist
        assert 'temperature_time_extremes' in features.columns
        assert 'temperature_threshold_crossings' in features.columns
        
        # Check values are reasonable
        time_extremes = features['temperature_time_extremes'].iloc[0]
        assert 0 <= time_extremes <= 100  # Should be percentage
    
    def test_interaction_feature_creation(self):
        """Test interaction feature creation."""
        features = self.feature_engineer.extract_interaction_features(self.sample_cast_data)
        
        # Should have 10 interaction features
        expected_features = 10
        assert len(features.columns) == expected_features
        
        # Check specific features exist
        assert 'temp_pressure_corr' in features.columns
        assert 'temp_flow_ratio' in features.columns
        assert 'overall_sensor_corr' in features.columns
        
        # Check correlation values are reasonable
        corr = features['temp_pressure_corr'].iloc[0]
        assert -1 <= corr <= 1
    
    def test_temporal_feature_extraction(self):
        """Test temporal feature extraction."""
        features = self.feature_engineer.extract_temporal_features(self.sample_cast_data)
        
        # Should have 2 metrics × 5 sensors = 10 features
        expected_features = 2 * 5
        assert len(features.columns) == expected_features
        
        # Check specific features exist
        assert 'temperature_trend' in features.columns
        assert 'temperature_gradient' in features.columns
        
        # Check gradient is positive (absolute differences)
        gradient = features['temperature_gradient'].iloc[0]
        assert gradient >= 0
    
    def test_complete_feature_engineering(self):
        """Test complete feature engineering pipeline."""
        features = self.feature_engineer.engineer_features(self.sample_cast_data, cast_id='test_001')
        
        # Should have 100 features plus cast_id (45+20+15+10+10 = 100)
        assert len(features.columns) >= 100
        assert 'cast_id' in features.columns
        assert features['cast_id'].iloc[0] == 'test_001'
        
        # Check no infinite values
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        assert not np.isinf(features[numeric_cols]).any().any()
    
    def test_batch_processing(self):
        """Test batch processing of multiple casts."""
        data_dict = {
            'cast_001': self.sample_cast_data,
            'cast_002': self.sample_cast_data.copy(),
            'cast_003': self.problematic_data
        }
        
        features = self.feature_engineer.engineer_features_batch(data_dict, n_jobs=1)
        
        # Should have 3 rows (one per cast)
        assert len(features) == 3
        assert 'cast_id' in features.columns
        assert set(features['cast_id']) == {'cast_001', 'cast_002', 'cast_003'}
    
    def test_feature_scaling(self):
        """Test feature scaling functionality."""
        features = self.feature_engineer.engineer_features(self.sample_cast_data)
        
        # Fit scaler
        self.feature_engineer.fit_scaler(features)
        
        # Scale features
        scaled_features = self.feature_engineer.scale_features(features)
        
        # Check that features are scaled (approximately zero mean, unit variance)
        numeric_cols = scaled_features.select_dtypes(include=[np.number]).columns
        if 'cast_id' in numeric_cols:
            numeric_cols = numeric_cols.drop('cast_id')
        
        means = scaled_features[numeric_cols].mean()
        assert np.allclose(means, 0, atol=1e-6)
    
    def test_missing_data_handling(self):
        """Test handling of missing data in features."""
        # Create data with missing sensors
        incomplete_data = self.sample_cast_data.copy()
        incomplete_data = incomplete_data.drop(['vibration'], axis=1)
        
        features = self.feature_engineer.engineer_features(incomplete_data)
        
        # Should still extract features for available sensors
        assert not features.empty
        
        # Missing sensor features should be NaN
        assert np.isnan(features['vibration_mean'].iloc[0])
    
    def test_feature_validation(self):
        """Test feature validation functionality."""
        # Test input validation
        input_validation = self.validator.validate_input_data(self.sample_cast_data)
        assert input_validation['is_valid']
        assert input_validation['has_data']
        assert input_validation['sufficient_length']
        
        # Test feature validation
        features = self.feature_engineer.engineer_features(self.sample_cast_data)
        feature_validation = self.validator.validate_features(features)
        assert feature_validation['is_valid']
        assert feature_validation['has_features']
    
    def test_utility_functions(self):
        """Test utility functions."""
        test_series = pd.Series([1, 2, 3, 4, 5, 100])  # Last value is spike
        
        # Test percentiles
        percentiles = calculate_percentiles(test_series, [25, 75])
        assert 'p25' in percentiles
        assert percentiles['p25'] < percentiles['p75']
        
        # Test spike detection
        spikes = detect_spikes(test_series, threshold=2.0)
        assert spikes.iloc[-1]  # Last value should be detected as spike
        
        # Test correlation
        series1 = pd.Series([1, 2, 3, 4, 5])
        series2 = pd.Series([2, 4, 6, 8, 10])  # Perfect correlation
        corr = safe_correlation(series1, series2)
        assert np.isclose(corr, 1.0)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        features = self.feature_engineer.engineer_features(empty_df)
        assert features.empty
        
        # Single row DataFrame
        single_row = self.sample_cast_data.iloc[:1]
        features = self.feature_engineer.engineer_features(single_row)
        assert not features.empty  # Should handle gracefully
        
        # All NaN data
        nan_data = pd.DataFrame({
            'temperature': [np.nan] * 10,
            'pressure': [np.nan] * 10,
            'flow_rate': [np.nan] * 10,
            'vibration': [np.nan] * 10,
            'power_consumption': [np.nan] * 10
        })
        features = self.feature_engineer.engineer_features(nan_data)
        assert not features.empty  # Should extract NaN features
    
    def test_performance_benchmark(self):
        """Test processing speed meets basic requirements."""
        import time
        
        # Create larger dataset
        large_data = pd.DataFrame({
            'temperature': np.random.normal(1520, 10, 1000),
            'pressure': np.random.normal(150, 5, 1000),
            'flow_rate': np.random.normal(200, 15, 1000),
            'vibration': np.random.normal(1.2, 0.05, 1000),
            'power_consumption': np.random.normal(25, 3, 1000)
        })
        
        # Test single cast processing
        start_time = time.time()
        features = self.feature_engineer.engineer_features(large_data)
        single_time = time.time() - start_time
        
        # Should process single cast quickly (< 1 second)
        assert single_time < 1.0
        assert not features.empty