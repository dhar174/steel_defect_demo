import unittest
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import json

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from analysis.statistical_analyzer import StatisticalAnalyzer
from visualization.plotting_utils import PlottingUtils


class TestStatisticalAnalyzer(unittest.TestCase):
    """Test cases for StatisticalAnalyzer"""
    
    def setUp(self):
        """Set up test environment with sample data"""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = Path.cwd()
        
        # Create test directory structure
        (Path(self.test_dir) / 'data' / 'raw').mkdir(parents=True)
        (Path(self.test_dir) / 'data' / 'synthetic').mkdir(parents=True)
        
        # Generate sample dataset metadata
        self.sample_metadata = {
            'dataset_info': {
                'total_casts': 20,
                'defect_rate': 0.3
            },
            'cast_metadata': []
        }
        
        # Create sample cast data
        np.random.seed(42)
        for i in range(20):
            cast_id = f"cast_{i+1:04d}"
            
            # Generate sample time series data
            duration = 120  # 2 minutes
            timestamps = pd.date_range('2023-01-01', periods=duration, freq='1s')
            
            # Different patterns for good vs defect casts
            is_defect = i < 6  # First 6 are defects
            
            if is_defect:
                # Defect casts have more extreme values
                casting_speed = np.random.normal(1.4, 0.1, duration)
                mold_temperature = np.random.normal(1540, 20, duration)
                mold_level = np.random.normal(160, 10, duration)
            else:
                # Good casts have normal values
                casting_speed = np.random.normal(1.2, 0.05, duration)
                mold_temperature = np.random.normal(1520, 10, duration)
                mold_level = np.random.normal(150, 5, duration)
            
            cooling_water_flow = np.random.normal(200, 15, duration)
            superheat = np.random.normal(25, 3, duration)
            
            # Create DataFrame
            df = pd.DataFrame({
                'casting_speed': casting_speed,
                'mold_temperature': mold_temperature,
                'mold_level': mold_level,
                'cooling_water_flow': cooling_water_flow,
                'superheat': superheat
            }, index=timestamps)
            
            # Save parquet file
            parquet_path = Path(self.test_dir) / 'data' / 'raw' / f'cast_timeseries_{i+1:04d}.parquet'
            df.to_parquet(parquet_path)
            
            # Add to metadata
            self.sample_metadata['cast_metadata'].append({
                'cast_id': cast_id,
                'defect_label': int(is_defect),
                'steel_grade': 'Grade_A'
            })
        
        # Save metadata
        metadata_path = Path(self.test_dir) / 'data' / 'synthetic' / 'dataset_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.sample_metadata, f)
        
        # Change to test directory
        import os
        os.chdir(self.test_dir)
        
        # Initialize analyzer
        self.analyzer = StatisticalAnalyzer(data_path='data')
    
    def tearDown(self):
        """Clean up test environment"""
        import os
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test analyzer initialization"""
        self.assertIsNotNone(self.analyzer.dataset_metadata)
        self.assertEqual(len(self.analyzer.dataset_metadata['cast_metadata']), 20)
    
    def test_load_cast_data(self):
        """Test loading and processing cast data"""
        features_df, metadata_df = self.analyzer.load_cast_data()
        
        # Check data structure
        self.assertEqual(len(features_df), 20)
        self.assertEqual(len(metadata_df), 20)
        
        # Check that we have aggregated features
        self.assertIn('casting_speed_mean', features_df.columns)
        self.assertIn('mold_temperature_std', features_df.columns)
        self.assertIn('defect_label', features_df.columns)
        
        # Check defect distribution
        defect_count = (features_df['defect_label'] == 1).sum()
        self.assertEqual(defect_count, 6)  # We created 6 defects
    
    def test_analyze_sensor_distributions(self):
        """Test sensor distribution analysis"""
        features_df, _ = self.analyzer.load_cast_data()
        results = self.analyzer.analyze_sensor_distributions(features_df)
        
        # Check that we have results for all sensors
        expected_sensors = ['casting_speed', 'mold_temperature', 'mold_level', 
                          'cooling_water_flow', 'superheat']
        for sensor in expected_sensors:
            self.assertIn(sensor, results)
            
        # Check that each sensor has statistical measures
        for sensor in expected_sensors:
            sensor_stats = results[sensor]
            for feature_name in sensor_stats:
                stats_dict = sensor_stats[feature_name]
                self.assertIn('mean', stats_dict)
                self.assertIn('std', stats_dict)
                self.assertIn('normality_test', stats_dict)
    
    def test_analyze_defect_stratification(self):
        """Test defect class stratification analysis"""
        features_df, _ = self.analyzer.load_cast_data()
        results = self.analyzer.analyze_defect_stratification(features_df)
        
        # Check sample sizes
        self.assertEqual(results['sample_sizes']['good'], 14)
        self.assertEqual(results['sample_sizes']['defect'], 6)
        self.assertEqual(results['sample_sizes']['total'], 20)
        
        # Check that we have comparisons for sensors
        self.assertIn('sensors', results)
        for sensor in results['sensors']:
            sensor_results = results['sensors'][sensor]
            for feature in sensor_results:
                comparison = sensor_results[feature]
                self.assertIn('good_stats', comparison)
                self.assertIn('defect_stats', comparison)
                self.assertIn('statistical_tests', comparison)
    
    def test_detect_outliers(self):
        """Test outlier detection"""
        features_df, _ = self.analyzer.load_cast_data()
        
        # Test IQR method
        results_iqr = self.analyzer.detect_outliers(features_df, method='iqr')
        self.assertEqual(results_iqr['method'], 'iqr')
        self.assertIn('sensors', results_iqr)
        
        # Test Z-score method
        results_zscore = self.analyzer.detect_outliers(features_df, method='zscore')
        self.assertEqual(results_zscore['method'], 'zscore')
        
        # Check that each sensor has outlier information
        for sensor in results_iqr['sensors']:
            sensor_outliers = results_iqr['sensors'][sensor]
            for feature in sensor_outliers:
                outlier_info = sensor_outliers[feature]
                self.assertIn('outlier_count', outlier_info)
                self.assertIn('outlier_percentage', outlier_info)
                self.assertIn('outlier_indices', outlier_info)
    
    def test_perform_ks_tests(self):
        """Test Kolmogorov-Smirnov tests"""
        features_df, _ = self.analyzer.load_cast_data()
        results = self.analyzer.perform_ks_tests(features_df)
        
        self.assertEqual(results['reference_distribution'], 'normal')
        self.assertIn('sensors', results)
        
        # Check that each sensor has KS test results
        for sensor in results['sensors']:
            sensor_results = results['sensors'][sensor]
            for feature in sensor_results:
                ks_result = sensor_results[feature]
                self.assertIn('ks_statistic', ks_result)
                self.assertIn('p_value', ks_result)
                self.assertIn('is_significant', ks_result)
    
    def test_generate_summary_report(self):
        """Test comprehensive summary report generation"""
        features_df, _ = self.analyzer.load_cast_data()
        report = self.analyzer.generate_summary_report(features_df)
        
        # Check report structure
        self.assertIn('dataset_overview', report)
        self.assertIn('sensor_distributions', report)
        self.assertIn('defect_stratification', report)
        self.assertIn('outlier_detection', report)
        self.assertIn('ks_tests', report)
        
        # Check dataset overview
        overview = report['dataset_overview']
        self.assertEqual(overview['total_casts'], 20)
        self.assertAlmostEqual(overview['defect_rate'], 0.3, places=2)


class TestPlottingUtilsStatistical(unittest.TestCase):
    """Test cases for statistical plotting utilities"""
    
    def setUp(self):
        """Set up test environment with sample data"""
        self.plotter = PlottingUtils()
        
        # Create sample features DataFrame
        np.random.seed(42)
        n_samples = 50
        
        # Create features with different patterns for good/defect
        good_indices = np.random.choice(n_samples, size=35, replace=False)
        defect_labels = np.zeros(n_samples)
        defect_labels[~np.isin(np.arange(n_samples), good_indices)] = 1
        
        self.sample_data = pd.DataFrame({
            'cast_id': [f'cast_{i:04d}' for i in range(n_samples)],
            'defect_label': defect_labels,
            'casting_speed_mean': np.random.normal(1.2, 0.1, n_samples),
            'casting_speed_std': np.random.normal(0.05, 0.01, n_samples),
            'mold_temperature_mean': np.random.normal(1520, 20, n_samples),
            'mold_temperature_std': np.random.normal(10, 2, n_samples),
        })
        
        # Make defect cases have more extreme values
        defect_mask = self.sample_data['defect_label'] == 1
        self.sample_data.loc[defect_mask, 'casting_speed_mean'] += 0.2
        self.sample_data.loc[defect_mask, 'mold_temperature_mean'] += 30
    
    def test_plot_defect_distribution(self):
        """Test defect distribution plotting"""
        labels = self.sample_data['defect_label'].values
        fig = self.plotter.plot_defect_distribution(labels)
        
        # Check that figure is created
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 1)  # One bar chart
        
        # Check data
        bar_data = fig.data[0]
        self.assertEqual(len(bar_data.x), 2)  # Good and Defect
        self.assertEqual(bar_data.x[0], 'Good')
        self.assertEqual(bar_data.x[1], 'Defect')
    
    def test_plot_sensor_histograms(self):
        """Test sensor histogram plotting"""
        fig = self.plotter.plot_sensor_histograms(
            self.sample_data, 
            'casting_speed'
        )
        
        # Check that figure is created
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 2)  # Good and Defect histograms
        
        # Check trace names
        trace_names = [trace.name for trace in fig.data]
        self.assertIn('Good', trace_names)
        self.assertIn('Defect', trace_names)
    
    def test_plot_sensor_boxplots(self):
        """Test sensor box plot creation"""
        fig = self.plotter.plot_sensor_boxplots(
            self.sample_data,
            ['casting_speed', 'mold_temperature']
        )
        
        # Check that figure is created
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 4)  # 2 sensors × 2 classes
    
    def test_plot_correlation_heatmap(self):
        """Test correlation heatmap plotting"""
        # Select numeric columns only
        numeric_data = self.sample_data.select_dtypes(include=[np.number])
        fig = self.plotter.plot_correlation_heatmap(numeric_data)
        
        # Check that figure is created
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 1)  # One heatmap
        
        # Check that it's a heatmap
        self.assertEqual(fig.data[0].type, 'heatmap')


if __name__ == '__main__':
    unittest.main()