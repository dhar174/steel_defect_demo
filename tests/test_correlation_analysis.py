"""
Tests for correlation analysis functionality.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.correlation_analyzer import SensorCorrelationAnalyzer
from src.visualization.plotting_utils import PlottingUtils
import tempfile
import json


@pytest.fixture
def sample_sensor_data():
    """Create sample sensor data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create correlated data
    time_index = pd.date_range('2023-01-01', periods=n_samples, freq='1S')
    
    # Base signals with some correlation structure
    base_signal = np.sin(np.linspace(0, 4*np.pi, n_samples)) + np.random.normal(0, 0.1, n_samples)
    
    data = pd.DataFrame({
        'casting_speed': 1.2 + 0.1 * base_signal + np.random.normal(0, 0.05, n_samples),
        'mold_temperature': 1520 + 10 * base_signal + np.random.normal(0, 5, n_samples),
        'mold_level': 150 + 5 * np.random.normal(0, 1, n_samples),
        'cooling_water_flow': 200 + 15 * np.random.normal(0, 1, n_samples),
        'superheat': 25 + 3 * (-base_signal) + np.random.normal(0, 2, n_samples)  # Negatively correlated
    }, index=time_index)
    
    return data


@pytest.fixture
def sample_cast_data_list(sample_sensor_data):
    """Create sample cast data list with metadata."""
    cast_data_list = []
    
    # Create 5 good casts and 3 defective casts
    for i in range(8):
        # Slightly modify the base data for each cast
        cast_data = sample_sensor_data.copy()
        cast_data += np.random.normal(0, 0.01, cast_data.shape)
        
        metadata = {
            'cast_id': f'test_cast_{i:03d}',
            'defect_label': 1 if i >= 5 else 0,  # Last 3 are defective
            'steel_grade': 'Grade_A',
            'duration_minutes': 120,
            'sampling_rate_hz': 1
        }
        
        cast_data_list.append((cast_data, metadata))
    
    return cast_data_list


class TestSensorCorrelationAnalyzer:
    """Test the SensorCorrelationAnalyzer class."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = SensorCorrelationAnalyzer()
        assert len(analyzer.sensor_columns) == 5
        assert 'casting_speed' in analyzer.sensor_columns
        
        # Test custom sensor columns
        custom_sensors = ['sensor_a', 'sensor_b']
        analyzer_custom = SensorCorrelationAnalyzer(custom_sensors)
        assert analyzer_custom.sensor_columns == custom_sensors
    
    def test_cross_sensor_correlations(self, sample_sensor_data):
        """Test cross-sensor correlation computation."""
        analyzer = SensorCorrelationAnalyzer()
        
        # Test Pearson correlation
        corr_matrix = analyzer.compute_cross_sensor_correlations(sample_sensor_data, method='pearson')
        
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (5, 5)
        assert np.all(np.diag(corr_matrix) == 1.0)  # Diagonal should be 1
        assert np.allclose(corr_matrix, corr_matrix.T)  # Should be symmetric
        
        # Test other correlation methods
        spearman_corr = analyzer.compute_cross_sensor_correlations(sample_sensor_data, method='spearman')
        assert isinstance(spearman_corr, pd.DataFrame)
        
        # Test invalid method
        with pytest.raises(ValueError):
            analyzer.compute_cross_sensor_correlations(sample_sensor_data, method='invalid')
    
    def test_time_lagged_correlations(self, sample_sensor_data):
        """Test time-lagged correlation computation."""
        analyzer = SensorCorrelationAnalyzer()
        
        # Test with target sensor
        lagged_corr = analyzer.compute_time_lagged_correlations(
            sample_sensor_data, 
            max_lag=10, 
            target_sensor='casting_speed'
        )
        
        assert isinstance(lagged_corr, dict)
        assert len(lagged_corr) == 4  # 4 other sensors
        
        for key, lag_df in lagged_corr.items():
            assert isinstance(lag_df, pd.DataFrame)
            assert 'lag' in lag_df.columns
            assert 'correlation' in lag_df.columns
            assert len(lag_df) == 21  # -10 to +10 inclusive
        
        # Test without target sensor (all pairs)
        all_pairs_corr = analyzer.compute_time_lagged_correlations(
            sample_sensor_data, 
            max_lag=5
        )
        
        assert isinstance(all_pairs_corr, dict)
        assert len(all_pairs_corr) == 10  # C(5,2) = 10 pairs
    
    def test_defect_specific_correlations(self, sample_cast_data_list):
        """Test defect-specific correlation analysis."""
        analyzer = SensorCorrelationAnalyzer()
        
        defect_analysis = analyzer.compute_defect_specific_correlations(sample_cast_data_list)
        
        assert isinstance(defect_analysis, dict)
        assert 'good_casts' in defect_analysis
        assert 'defect_casts' in defect_analysis
        assert 'difference' in defect_analysis
        
        # Check that matrices have correct shape
        for key in ['good_casts', 'defect_casts', 'difference']:
            matrix = defect_analysis[key]
            assert isinstance(matrix, pd.DataFrame)
            assert matrix.shape == (5, 5)
    
    def test_predictive_sensor_combinations(self, sample_cast_data_list):
        """Test identification of predictive sensor combinations."""
        analyzer = SensorCorrelationAnalyzer()
        
        importance_df = analyzer.identify_predictive_sensor_combinations(
            sample_cast_data_list, 
            top_k=5
        )
        
        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == 5
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        
        # Check that importance values are sorted in descending order
        importances = importance_df['importance'].values
        assert np.all(importances[:-1] >= importances[1:])
    
    def test_rolling_correlations(self, sample_sensor_data):
        """Test rolling correlation computation."""
        analyzer = SensorCorrelationAnalyzer()
        
        # Test specific sensor pair
        rolling_corr = analyzer.compute_rolling_correlations(
            sample_sensor_data,
            window_size=100,
            sensor_pair=('casting_speed', 'mold_temperature')
        )
        
        assert isinstance(rolling_corr, pd.DataFrame)
        assert len(rolling_corr) == len(sample_sensor_data)
        assert 'rolling_corr_casting_speed_mold_temperature' in rolling_corr.columns
        
        # Test all pairs
        all_rolling_corr = analyzer.compute_rolling_correlations(
            sample_sensor_data,
            window_size=100
        )
        
        assert isinstance(all_rolling_corr, pd.DataFrame)
        assert len(all_rolling_corr) == len(sample_sensor_data)
        assert len(all_rolling_corr.columns) == 10  # C(5,2) = 10 pairs
    
    def test_export_correlation_analysis(self, sample_cast_data_list):
        """Test exporting correlation analysis to JSON."""
        analyzer = SensorCorrelationAnalyzer()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            analyzer.export_correlation_analysis(sample_cast_data_list, tmp_file.name)
            
            # Read back and verify
            with open(tmp_file.name, 'r') as f:
                results = json.load(f)
            
            assert 'analysis_metadata' in results
            assert 'cross_sensor_correlations' in results
            assert 'defect_specific_correlations' in results
            assert 'predictive_combinations' in results
            
            # Verify structure
            metadata = results['analysis_metadata']
            assert metadata['num_casts_analyzed'] == 8
            assert len(metadata['sensor_columns']) == 5


class TestPlottingUtilsCorrelation:
    """Test correlation plotting functionality."""
    
    def test_plot_correlation_heatmap(self, sample_sensor_data):
        """Test correlation heatmap plotting."""
        plotting_utils = PlottingUtils()
        
        # Test with data
        fig = plotting_utils.plot_correlation_heatmap(sample_sensor_data)
        assert fig is not None
        
        # Test with pre-computed correlation matrix
        corr_matrix = sample_sensor_data.corr()
        fig_precomputed = plotting_utils.plot_correlation_heatmap(
            sample_sensor_data, 
            correlation_matrix=corr_matrix
        )
        assert fig_precomputed is not None
    
    def test_plot_defect_correlation_comparison(self, sample_cast_data_list):
        """Test defect correlation comparison plotting."""
        plotting_utils = PlottingUtils()
        analyzer = SensorCorrelationAnalyzer()
        
        # Get correlation data
        defect_analysis = analyzer.compute_defect_specific_correlations(sample_cast_data_list)
        
        fig = plotting_utils.plot_defect_correlation_comparison(
            defect_analysis['good_casts'],
            defect_analysis['defect_casts'],
            defect_analysis['difference']
        )
        
        assert fig is not None
        
        # Test without difference matrix
        fig_no_diff = plotting_utils.plot_defect_correlation_comparison(
            defect_analysis['good_casts'],
            defect_analysis['defect_casts']
        )
        
        assert fig_no_diff is not None
    
    def test_plot_time_lagged_correlations(self, sample_sensor_data):
        """Test time-lagged correlation plotting."""
        plotting_utils = PlottingUtils()
        analyzer = SensorCorrelationAnalyzer()
        
        # Get lagged correlation data
        lagged_corr = analyzer.compute_time_lagged_correlations(
            sample_sensor_data, 
            max_lag=10,
            target_sensor='casting_speed'
        )
        
        # Test plotting all pairs
        fig_all = plotting_utils.plot_time_lagged_correlations(lagged_corr)
        assert fig_all is not None
        
        # Test plotting specific pair
        sensor_pair = list(lagged_corr.keys())[0]
        fig_specific = plotting_utils.plot_time_lagged_correlations(
            lagged_corr, 
            sensor_pair=sensor_pair
        )
        assert fig_specific is not None
    
    def test_plot_feature_importance_ranking(self, sample_cast_data_list):
        """Test feature importance ranking plotting."""
        plotting_utils = PlottingUtils()
        analyzer = SensorCorrelationAnalyzer()
        
        # Get importance data
        importance_df = analyzer.identify_predictive_sensor_combinations(
            sample_cast_data_list,
            top_k=10
        )
        
        fig = plotting_utils.plot_feature_importance_ranking(importance_df)
        assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__])