import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
from scipy import stats
from scipy.stats import kstest, ks_2samp
import warnings


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for steel casting sensor data.
    
    Provides functionality for:
    - Sensor value distribution analysis (histograms, box plots)
    - Defect class stratification (good vs defect comparisons)
    - Outlier detection using statistical methods
    - Kolmogorov-Smirnov tests for distribution comparisons
    """
    
    def __init__(self, data_path: str = "data", config: Optional[Dict] = None):
        """
        Initialize the statistical analyzer.
        
        Args:
            data_path (str): Path to the data directory
            config (Dict, optional): Configuration parameters
        """
        self.data_path = Path(data_path)
        self.config = config or {}
        
        # Default outlier detection parameters
        self.outlier_methods = {
            'iqr': {'factor': 1.5},
            'zscore': {'threshold': 3.0},
            'modified_zscore': {'threshold': 3.5}
        }
        
        self._load_dataset_metadata()
    
    def _load_dataset_metadata(self) -> None:
        """Load dataset metadata if available."""
        metadata_path = self.data_path / "synthetic" / "dataset_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.dataset_metadata = json.load(f)
        else:
            self.dataset_metadata = None
    
    def load_cast_data(self, cast_ids: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load cast data with metadata.
        
        Args:
            cast_ids: List of specific cast IDs to load. If None, loads all available.
            
        Returns:
            Tuple of (aggregated_features_df, metadata_df)
        """
        if self.dataset_metadata is None:
            raise ValueError("Dataset metadata not found. Ensure data has been generated.")
        
        cast_metadata = self.dataset_metadata['cast_metadata']
        
        if cast_ids:
            cast_metadata = [m for m in cast_metadata if m['cast_id'] in cast_ids]
        
        # Load time series data and compute aggregated features
        aggregated_features = []
        metadata_records = []
        
        for cast_meta in cast_metadata:
            cast_id = cast_meta['cast_id']
            
            # Load time series data
            cast_num = int(cast_id.split('_')[1])
            parquet_path = self.data_path / "raw" / f"cast_timeseries_{cast_num:04d}.parquet"
            
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                
                # Compute aggregated features for this cast
                features = self._compute_aggregated_features(df, cast_meta)
                aggregated_features.append(features)
                metadata_records.append(cast_meta)
        
        features_df = pd.DataFrame(aggregated_features)
        metadata_df = pd.DataFrame(metadata_records)
        
        return features_df, metadata_df
    
    def _compute_aggregated_features(self, timeseries_df: pd.DataFrame, metadata: Dict) -> Dict:
        """
        Compute aggregated statistical features from time series data.
        
        Args:
            timeseries_df: Time series data for a single cast
            metadata: Cast metadata
            
        Returns:
            Dictionary of aggregated features
        """
        features = {'cast_id': metadata['cast_id'], 'defect_label': metadata['defect_label']}
        
        # Compute statistics for each sensor
        for sensor in timeseries_df.columns:
            sensor_data = timeseries_df[sensor]
            
            features.update({
                f'{sensor}_mean': sensor_data.mean(),
                f'{sensor}_std': sensor_data.std(),
                f'{sensor}_min': sensor_data.min(),
                f'{sensor}_max': sensor_data.max(),
                f'{sensor}_median': sensor_data.median(),
                f'{sensor}_q25': sensor_data.quantile(0.25),
                f'{sensor}_q75': sensor_data.quantile(0.75),
                f'{sensor}_skew': sensor_data.skew(),
                f'{sensor}_kurtosis': sensor_data.kurtosis(),
                f'{sensor}_range': sensor_data.max() - sensor_data.min()
            })
        
        return features
    
    def _extract_sensor_names(self, features_df: pd.DataFrame) -> List[str]:
        """Extract unique sensor names from feature column names."""
        sensor_names = set()
        excluded_cols = {'cast_id', 'defect_label'}
        
        for col in features_df.columns:
            if col not in excluded_cols:
                # Find the sensor name by looking for the last underscore and stat name
                parts = col.split('_')
                if len(parts) >= 2:
                    stat_names = {'mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 
                                'skew', 'kurtosis', 'range'}
                    if parts[-1] in stat_names:
                        # Rejoin all parts except the last one
                        sensor_name = '_'.join(parts[:-1])
                        sensor_names.add(sensor_name)
        
        return list(sensor_names)

    def analyze_sensor_distributions(self, features_df: pd.DataFrame, 
                                   sensor_names: Optional[List[str]] = None) -> Dict:
        """
        Analyze distribution characteristics of sensor data.
        
        Args:
            features_df: DataFrame with aggregated sensor features
            sensor_names: List of sensor names to analyze
            
        Returns:
            Dictionary with distribution analysis results
        """
        if sensor_names is None:
            sensor_names = self._extract_sensor_names(features_df)
        
        results = {}
        
        for sensor in sensor_names:
            sensor_cols = [col for col in features_df.columns if col.startswith(sensor + '_')]
            
            if not sensor_cols:
                continue
                
            sensor_stats = {}
            
            for col in sensor_cols:
                data = features_df[col].dropna()
                
                if len(data) > 0:
                    # Basic distribution statistics
                    sensor_stats[col] = {
                        'count': len(data),
                        'mean': data.mean(),
                        'std': data.std(),
                        'min': data.min(),
                        'max': data.max(),
                        'median': data.median(),
                        'q25': data.quantile(0.25),
                        'q75': data.quantile(0.75),
                        'skewness': data.skew(),
                        'kurtosis': data.kurtosis(),
                        'normality_test': self._test_normality(data)
                    }
            
            results[sensor] = sensor_stats
        
        return results
    
    def analyze_defect_stratification(self, features_df: pd.DataFrame, 
                                    sensor_names: Optional[List[str]] = None) -> Dict:
        """
        Compare sensor distributions between good and defect classes.
        
        Args:
            features_df: DataFrame with aggregated sensor features
            sensor_names: List of sensor names to analyze
            
        Returns:
            Dictionary with stratification analysis results
        """
        if sensor_names is None:
            sensor_names = self._extract_sensor_names(features_df)
        
        good_data = features_df[features_df['defect_label'] == 0]
        defect_data = features_df[features_df['defect_label'] == 1]
        
        results = {
            'sample_sizes': {
                'good': len(good_data),
                'defect': len(defect_data),
                'total': len(features_df)
            },
            'sensors': {}
        }
        
        for sensor in sensor_names:
            sensor_cols = [col for col in features_df.columns if col.startswith(sensor + '_')]
            sensor_results = {}
            
            for col in sensor_cols:
                good_values = good_data[col].dropna()
                defect_values = defect_data[col].dropna()
                
                if len(good_values) > 0 and len(defect_values) > 0:
                    # Statistical comparison
                    comparison = {
                        'good_stats': {
                            'count': len(good_values),
                            'mean': good_values.mean(),
                            'std': good_values.std(),
                            'median': good_values.median()
                        },
                        'defect_stats': {
                            'count': len(defect_values),
                            'mean': defect_values.mean(),
                            'std': defect_values.std(),
                            'median': defect_values.median()
                        },
                        'statistical_tests': self._compare_distributions(good_values, defect_values)
                    }
                    
                    sensor_results[col] = comparison
            
            results['sensors'][sensor] = sensor_results
        
        return results
    
    def detect_outliers(self, features_df: pd.DataFrame, 
                       method: str = 'iqr',
                       sensor_names: Optional[List[str]] = None) -> Dict:
        """
        Detect outliers in sensor data using various statistical methods.
        
        Args:
            features_df: DataFrame with aggregated sensor features
            method: Outlier detection method ('iqr', 'zscore', 'modified_zscore')
            sensor_names: List of sensor names to analyze
            
        Returns:
            Dictionary with outlier detection results
        """
        if method not in self.outlier_methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.outlier_methods.keys())}")
        
        if sensor_names is None:
            sensor_names = self._extract_sensor_names(features_df)
        
        results = {'method': method, 'sensors': {}}
        
        for sensor in sensor_names:
            sensor_cols = [col for col in features_df.columns if col.startswith(sensor + '_')]
            sensor_outliers = {}
            
            for col in sensor_cols:
                data = features_df[col].dropna()
                
                if len(data) > 0:
                    outlier_mask = self._detect_outliers_by_method(data, method)
                    outlier_indices = features_df.index[features_df[col].notna()][outlier_mask]
                    
                    sensor_outliers[col] = {
                        'outlier_count': outlier_mask.sum(),
                        'outlier_percentage': (outlier_mask.sum() / len(data)) * 100,
                        'outlier_indices': outlier_indices.tolist(),
                        'outlier_values': data[outlier_mask].tolist()
                    }
            
            results['sensors'][sensor] = sensor_outliers
        
        return results
    
    def perform_ks_tests(self, features_df: pd.DataFrame, 
                        reference_distribution: str = 'normal',
                        sensor_names: Optional[List[str]] = None) -> Dict:
        """
        Perform Kolmogorov-Smirnov tests to compare distributions.
        
        Args:
            features_df: DataFrame with aggregated sensor features
            reference_distribution: Reference distribution for testing ('normal', 'uniform')
            sensor_names: List of sensor names to analyze
            
        Returns:
            Dictionary with KS test results
        """
        if sensor_names is None:
            sensor_names = self._extract_sensor_names(features_df)
        
        results = {'reference_distribution': reference_distribution, 'sensors': {}}
        
        for sensor in sensor_names:
            sensor_cols = [col for col in features_df.columns if col.startswith(sensor + '_')]
            sensor_results = {}
            
            for col in sensor_cols:
                data = features_df[col].dropna()
                
                if len(data) > 5:  # Minimum sample size for KS test
                    # Test against reference distribution
                    ks_stat, p_value = self._ks_test_against_reference(data, reference_distribution)
                    
                    sensor_results[col] = {
                        'ks_statistic': ks_stat,
                        'p_value': p_value,
                        'is_significant': p_value < 0.05,
                        'sample_size': len(data)
                    }
            
            results['sensors'][sensor] = sensor_results
        
        return results
    
    def _test_normality(self, data: pd.Series) -> Dict:
        """Test if data follows normal distribution."""
        if len(data) < 3:
            return {'test': 'insufficient_data', 'p_value': None, 'is_normal': None}
        
        try:
            # Shapiro-Wilk test for normality
            stat, p_value = stats.shapiro(data)
            return {
                'test': 'shapiro_wilk',
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            }
        except Exception as e:
            return {'test': 'failed', 'error': str(e), 'is_normal': None}
    
    def _compare_distributions(self, data1: pd.Series, data2: pd.Series) -> Dict:
        """Compare two distributions using statistical tests."""
        tests = {}
        
        try:
            # Mann-Whitney U test (non-parametric)
            stat_mw, p_mw = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            tests['mann_whitney'] = {
                'statistic': stat_mw,
                'p_value': p_mw,
                'is_significant': p_mw < 0.05
            }
        except Exception as e:
            tests['mann_whitney'] = {'error': str(e)}
        
        try:
            # Kolmogorov-Smirnov test
            stat_ks, p_ks = ks_2samp(data1, data2)
            tests['kolmogorov_smirnov'] = {
                'statistic': stat_ks,
                'p_value': p_ks,
                'is_significant': p_ks < 0.05
            }
        except Exception as e:
            tests['kolmogorov_smirnov'] = {'error': str(e)}
        
        try:
            # t-test (assuming normality)
            stat_t, p_t = stats.ttest_ind(data1, data2)
            tests['t_test'] = {
                'statistic': stat_t,
                'p_value': p_t,
                'is_significant': p_t < 0.05
            }
        except Exception as e:
            tests['t_test'] = {'error': str(e)}
        
        return tests
    
    def _detect_outliers_by_method(self, data: pd.Series, method: str) -> np.ndarray:
        """Detect outliers using specified method."""
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            factor = self.outlier_methods['iqr']['factor']
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            return (data < lower_bound) | (data > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            threshold = self.outlier_methods['zscore']['threshold']
            return z_scores > threshold
        
        elif method == 'modified_zscore':
            median = data.median()
            mad = np.median(np.abs(data - median))
            if mad == 0:
                # Handle case where MAD is zero (all values are identical)
                return np.zeros(len(data), dtype=bool)
            modified_z_scores = 0.6745 * (data - median) / mad
            threshold = self.outlier_methods['modified_zscore']['threshold']
            return np.abs(modified_z_scores) > threshold
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    def _ks_test_against_reference(self, data: pd.Series, reference: str) -> Tuple[float, float]:
        """Perform KS test against reference distribution."""
        if len(data) < 8:  # Minimum reasonable sample size for KS test
            return 0.0, 1.0  # Return non-significant result for small samples
            
        if reference == 'normal':
            # Standardize data and test against standard normal
            if data.std() == 0:
                return 0.0, 1.0  # All values identical, cannot standardize
            standardized = (data - data.mean()) / data.std()
            return kstest(standardized, 'norm')
        
        elif reference == 'uniform':
            # Normalize data to [0,1] and test against uniform
            data_range = data.max() - data.min()
            if data_range == 0:
                return 0.0, 1.0  # All values identical
            normalized = (data - data.min()) / data_range
            return kstest(normalized, 'uniform')
        
        else:
            raise ValueError(f"Unknown reference distribution: {reference}")
    
    def generate_summary_report(self, features_df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive statistical summary report.
        
        Args:
            features_df: DataFrame with aggregated sensor features
            
        Returns:
            Dictionary with complete analysis summary
        """
        report = {
            'dataset_overview': {
                'total_casts': len(features_df),
                'defect_rate': (features_df['defect_label'] == 1).mean(),
                'sensors_analyzed': self._extract_sensor_names(features_df)
            }
        }
        
        # Add all analyses
        report['sensor_distributions'] = self.analyze_sensor_distributions(features_df)
        report['defect_stratification'] = self.analyze_defect_stratification(features_df)
        report['outlier_detection'] = self.detect_outliers(features_df)
        report['ks_tests'] = self.perform_ks_tests(features_df)
        
        return report