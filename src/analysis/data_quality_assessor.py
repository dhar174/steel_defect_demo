import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
from datetime import datetime, timedelta
import warnings


class DataQualityAssessor:
    """
    Comprehensive data quality assessment for steel casting sensor data.
    
    Provides functionality for:
    - Missing Value Analysis: Check for gaps in sensor data
    - Data Consistency Checks: Verify sensor readings are within expected ranges
    - Temporal Continuity: Ensure proper time sequencing in generated data
    - Synthetic Data Realism: Compare generated patterns to expected steel casting behavior
    """
    
    def __init__(self, data_path: str = "data", config: Optional[Dict] = None):
        """
        Initialize the data quality assessor.
        
        Args:
            data_path (str): Path to the data directory
            config (Dict, optional): Configuration parameters for quality checks
        """
        self.data_path = Path(data_path)
        self.config = config or self._load_default_config()
        
        # Expected sensor ranges from steel casting domain knowledge
        self.sensor_expected_ranges = {
            'casting_speed': {'min': 0.5, 'max': 2.5, 'typical_min': 0.8, 'typical_max': 1.8},
            'mold_temperature': {'min': 1400, 'max': 1650, 'typical_min': 1480, 'typical_max': 1580},
            'mold_level': {'min': 100, 'max': 200, 'typical_min': 120, 'typical_max': 180},
            'cooling_water_flow': {'min': 100, 'max': 300, 'typical_min': 150, 'typical_max': 250},
            'superheat': {'min': 10, 'max': 50, 'typical_min': 15, 'typical_max': 40}
        }
        
        # Steel casting process physics constraints
        self.physics_constraints = {
            'temperature_gradient_max': 100,  # Max temp change per minute (°C/min)
            'speed_gradient_max': 0.3,        # Max speed change per minute (m/min/min)
            'level_gradient_max': 30,         # Max level change per minute (mm/min)
            'flow_gradient_max': 50,          # Max flow change per minute (L/min/min)
            'superheat_gradient_max': 15      # Max superheat change per minute (°C/min)
        }
        
        self._load_dataset_metadata()
    
    def _load_default_config(self) -> Dict:
        """Load default configuration for quality assessment."""
        return {
            'missing_value_threshold': 0.01,  # Max 1% missing values allowed
            'temporal_gap_threshold_seconds': 5,  # Max gap of 5 seconds
            'outlier_threshold_std': 3.0,     # Standard deviations for outlier detection
            'physics_violation_threshold': 0.05,  # Max 5% physics violations allowed
            'realism_score_threshold': 0.7    # Minimum realism score (0-1)
        }
    
    def _load_dataset_metadata(self) -> None:
        """Load dataset metadata if available."""
        metadata_path = self.data_path / "synthetic" / "dataset_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.dataset_metadata = json.load(f)
        else:
            self.dataset_metadata = None
    
    def assess_missing_values(self, cast_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Analyze missing values in sensor data.
        
        Args:
            cast_data (pd.DataFrame, optional): Single cast data. If None, analyzes all casts.
            
        Returns:
            Dict: Missing value analysis results
        """
        results = {
            'missing_value_analysis': {
                'total_missing_percentage': 0.0,
                'missing_by_sensor': {},
                'missing_patterns': {},
                'temporal_gaps': [],
                'quality_score': 1.0
            }
        }
        
        if cast_data is not None:
            # Analyze single cast
            missing_analysis = self._analyze_single_cast_missing_values(cast_data)
            results['missing_value_analysis'].update(missing_analysis)
        else:
            # Analyze all casts
            missing_analysis = self._analyze_all_casts_missing_values()
            results['missing_value_analysis'].update(missing_analysis)
        
        return results
    
    def _analyze_single_cast_missing_values(self, cast_data: pd.DataFrame) -> Dict:
        """Analyze missing values in a single cast."""
        total_values = cast_data.size
        total_missing = cast_data.isnull().sum().sum()
        missing_percentage = (total_missing / total_values) * 100 if total_values > 0 else 0
        
        # Missing values by sensor
        missing_by_sensor = {}
        for column in cast_data.columns:
            missing_count = cast_data[column].isnull().sum()
            missing_by_sensor[column] = {
                'count': int(missing_count),
                'percentage': (missing_count / len(cast_data)) * 100 if len(cast_data) > 0 else 0
            }
        
        # Identify temporal gaps
        temporal_gaps = self._identify_temporal_gaps(cast_data)
        
        # Calculate quality score
        quality_score = max(0, 1 - (missing_percentage / 100))
        
        return {
            'total_missing_percentage': missing_percentage,
            'missing_by_sensor': missing_by_sensor,
            'temporal_gaps': temporal_gaps,
            'quality_score': quality_score
        }
    
    def _analyze_all_casts_missing_values(self) -> Dict:
        """Analyze missing values across all casts."""
        if not self.dataset_metadata:
            return {'error': 'No dataset metadata available'}
        
        total_missing_percentage = 0.0
        sensor_missing_stats = {}
        all_temporal_gaps = []
        cast_count = 0
        
        # Initialize sensor stats
        for sensor in ['casting_speed', 'mold_temperature', 'mold_level', 'cooling_water_flow', 'superheat']:
            sensor_missing_stats[sensor] = {'total_missing': 0, 'total_values': 0}
        
        # Analyze each cast
        for cast_metadata in self.dataset_metadata['cast_metadata']:
            cast_id = cast_metadata['cast_id']
            cast_file = self.data_path / 'raw' / f'cast_timeseries_{cast_id.split("_")[1]}.parquet'
            
            if cast_file.exists():
                cast_data = pd.read_parquet(cast_file)
                cast_analysis = self._analyze_single_cast_missing_values(cast_data)
                
                total_missing_percentage += cast_analysis['total_missing_percentage']
                all_temporal_gaps.extend(cast_analysis['temporal_gaps'])
                
                # Aggregate sensor stats
                for sensor, stats in cast_analysis['missing_by_sensor'].items():
                    if sensor in sensor_missing_stats:
                        sensor_missing_stats[sensor]['total_missing'] += stats['count']
                        sensor_missing_stats[sensor]['total_values'] += len(cast_data)
                
                cast_count += 1
        
        # Calculate aggregate statistics
        avg_missing_percentage = total_missing_percentage / cast_count if cast_count > 0 else 0
        
        missing_by_sensor = {}
        for sensor, stats in sensor_missing_stats.items():
            percentage = (stats['total_missing'] / stats['total_values']) * 100 if stats['total_values'] > 0 else 0
            missing_by_sensor[sensor] = {
                'count': stats['total_missing'],
                'percentage': percentage
            }
        
        quality_score = max(0, 1 - (avg_missing_percentage / 100))
        
        return {
            'total_missing_percentage': avg_missing_percentage,
            'missing_by_sensor': missing_by_sensor,
            'temporal_gaps': all_temporal_gaps,
            'casts_analyzed': cast_count,
            'quality_score': quality_score
        }
    
    def _identify_temporal_gaps(self, cast_data: pd.DataFrame) -> List[Dict]:
        """Identify gaps in time series data."""
        gaps = []
        
        if len(cast_data) < 2:
            return gaps
        
        # Calculate time differences
        time_diffs = cast_data.index.to_series().diff()
        
        # Expected frequency (1 second)
        expected_freq = pd.Timedelta(seconds=1)
        gap_threshold = pd.Timedelta(seconds=self.config['temporal_gap_threshold_seconds'])
        
        # Find gaps larger than threshold
        large_gaps = time_diffs > gap_threshold
        
        for idx in time_diffs[large_gaps].index:
            gap_duration = time_diffs.loc[idx]
            gaps.append({
                'start_time': str(cast_data.index[cast_data.index.get_loc(idx) - 1]),
                'end_time': str(idx),
                'duration_seconds': gap_duration.total_seconds(),
                'missing_samples': int((gap_duration - expected_freq).total_seconds())
            })
        
        return gaps
    
    def assess_data_consistency(self, cast_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Verify sensor readings are within expected ranges.
        
        Args:
            cast_data (pd.DataFrame, optional): Single cast data. If None, analyzes all casts.
            
        Returns:
            Dict: Data consistency analysis results
        """
        results = {
            'consistency_analysis': {
                'range_violations': {},
                'physics_violations': {},
                'outlier_analysis': {},
                'consistency_score': 1.0
            }
        }
        
        if cast_data is not None:
            # Analyze single cast
            consistency_analysis = self._analyze_single_cast_consistency(cast_data)
            results['consistency_analysis'].update(consistency_analysis)
        else:
            # Analyze all casts
            consistency_analysis = self._analyze_all_casts_consistency()
            results['consistency_analysis'].update(consistency_analysis)
        
        return results
    
    def _analyze_single_cast_consistency(self, cast_data: pd.DataFrame) -> Dict:
        """Analyze data consistency for a single cast."""
        range_violations = {}
        physics_violations = {}
        outlier_analysis = {}
        
        # Check range violations
        for sensor, ranges in self.sensor_expected_ranges.items():
            if sensor in cast_data.columns:
                data = cast_data[sensor]
                
                # Hard limit violations (impossible values)
                hard_violations = ((data < ranges['min']) | (data > ranges['max'])).sum()
                
                # Typical range violations (unusual but possible values)
                typical_violations = ((data < ranges['typical_min']) | (data > ranges['typical_max'])).sum()
                
                range_violations[sensor] = {
                    'hard_violations': int(hard_violations),
                    'hard_violation_percentage': (hard_violations / len(data)) * 100,
                    'typical_violations': int(typical_violations),
                    'typical_violation_percentage': (typical_violations / len(data)) * 100,
                    'value_range': [float(data.min()), float(data.max())],
                    'expected_range': [ranges['min'], ranges['max']]
                }
        
        # Check physics violations (unrealistic rate of change)
        physics_violations = self._check_physics_violations(cast_data)
        
        # Outlier analysis
        outlier_analysis = self._analyze_outliers(cast_data)
        
        # Calculate consistency score
        total_violations = sum(v['hard_violations'] for v in range_violations.values())
        total_values = len(cast_data) * len(cast_data.columns)
        violation_rate = total_violations / total_values if total_values > 0 else 0
        consistency_score = max(0, 1 - violation_rate)
        
        return {
            'range_violations': range_violations,
            'physics_violations': physics_violations,
            'outlier_analysis': outlier_analysis,
            'consistency_score': consistency_score
        }
    
    def _analyze_all_casts_consistency(self) -> Dict:
        """Analyze data consistency across all casts."""
        if not self.dataset_metadata:
            return {'error': 'No dataset metadata available'}
        
        aggregate_range_violations = {}
        aggregate_physics_violations = {}
        aggregate_outlier_analysis = {}
        cast_count = 0
        total_consistency_score = 0.0
        
        # Initialize aggregation structures
        for sensor in self.sensor_expected_ranges.keys():
            aggregate_range_violations[sensor] = {
                'hard_violations': 0,
                'typical_violations': 0,
                'total_values': 0
            }
        
        # Analyze each cast
        for cast_metadata in self.dataset_metadata['cast_metadata']:
            cast_id = cast_metadata['cast_id']
            cast_file = self.data_path / 'raw' / f'cast_timeseries_{cast_id.split("_")[1]}.parquet'
            
            if cast_file.exists():
                cast_data = pd.read_parquet(cast_file)
                cast_analysis = self._analyze_single_cast_consistency(cast_data)
                
                # Aggregate range violations
                for sensor, violations in cast_analysis['range_violations'].items():
                    if sensor in aggregate_range_violations:
                        aggregate_range_violations[sensor]['hard_violations'] += violations['hard_violations']
                        aggregate_range_violations[sensor]['typical_violations'] += violations['typical_violations']
                        aggregate_range_violations[sensor]['total_values'] += len(cast_data)
                
                total_consistency_score += cast_analysis['consistency_score']
                cast_count += 1
        
        # Calculate aggregate statistics
        for sensor in aggregate_range_violations:
            total_values = aggregate_range_violations[sensor]['total_values']
            if total_values > 0:
                aggregate_range_violations[sensor]['hard_violation_percentage'] = (
                    aggregate_range_violations[sensor]['hard_violations'] / total_values) * 100
                aggregate_range_violations[sensor]['typical_violation_percentage'] = (
                    aggregate_range_violations[sensor]['typical_violations'] / total_values) * 100
        
        avg_consistency_score = total_consistency_score / cast_count if cast_count > 0 else 0
        
        return {
            'range_violations': aggregate_range_violations,
            'physics_violations': aggregate_physics_violations,
            'outlier_analysis': aggregate_outlier_analysis,
            'casts_analyzed': cast_count,
            'consistency_score': avg_consistency_score
        }
    
    def _check_physics_violations(self, cast_data: pd.DataFrame) -> Dict:
        """Check for violations of steel casting physics constraints."""
        violations = {}
        
        # Convert sampling rate to per-minute rate (assuming 1Hz sampling)
        samples_per_minute = 60
        
        for sensor, max_gradient in self.physics_constraints.items():
            if sensor.replace('_gradient_max', '') in cast_data.columns:
                column_name = sensor.replace('_gradient_max', '')
                data = cast_data[column_name]
                
                # Calculate rate of change per minute
                rate_of_change = data.diff() * samples_per_minute
                
                # Find violations
                violation_mask = abs(rate_of_change) > max_gradient
                violation_count = violation_mask.sum()
                
                violations[column_name] = {
                    'violations': int(violation_count),
                    'violation_percentage': (violation_count / len(data)) * 100 if len(data) > 0 else 0,
                    'max_rate_observed': float(abs(rate_of_change).max()) if len(rate_of_change) > 0 else 0,
                    'max_rate_allowed': max_gradient
                }
        
        return violations
    
    def _analyze_outliers(self, cast_data: pd.DataFrame) -> Dict:
        """Analyze outliers using statistical methods."""
        outlier_analysis = {}
        
        for column in cast_data.columns:
            data = cast_data[column].dropna()
            
            if len(data) == 0:
                continue
            
            # Z-score outliers
            z_scores = np.abs((data - data.mean()) / data.std())
            z_outliers = (z_scores > self.config['outlier_threshold_std']).sum()
            
            # IQR outliers
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            iqr_outliers = ((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))).sum()
            
            outlier_analysis[column] = {
                'z_score_outliers': int(z_outliers),
                'z_score_outlier_percentage': (z_outliers / len(data)) * 100,
                'iqr_outliers': int(iqr_outliers),
                'iqr_outlier_percentage': (iqr_outliers / len(data)) * 100
            }
        
        return outlier_analysis
    
    def assess_temporal_continuity(self, cast_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Ensure proper time sequencing in generated data.
        
        Args:
            cast_data (pd.DataFrame, optional): Single cast data. If None, analyzes all casts.
            
        Returns:
            Dict: Temporal continuity analysis results
        """
        results = {
            'temporal_continuity': {
                'sampling_rate_analysis': {},
                'time_sequence_analysis': {},
                'continuity_score': 1.0
            }
        }
        
        if cast_data is not None:
            # Analyze single cast
            temporal_analysis = self._analyze_single_cast_temporal_continuity(cast_data)
            results['temporal_continuity'].update(temporal_analysis)
        else:
            # Analyze all casts
            temporal_analysis = self._analyze_all_casts_temporal_continuity()
            results['temporal_continuity'].update(temporal_analysis)
        
        return results
    
    def _analyze_single_cast_temporal_continuity(self, cast_data: pd.DataFrame) -> Dict:
        """Analyze temporal continuity for a single cast."""
        if len(cast_data) < 2:
            return {
                'sampling_rate_analysis': {'error': 'Insufficient data points'},
                'time_sequence_analysis': {'error': 'Insufficient data points'},
                'continuity_score': 0.0
            }
        
        # Sampling rate analysis
        time_diffs = cast_data.index.to_series().diff().dropna()
        
        expected_interval = pd.Timedelta(seconds=1)
        actual_intervals = time_diffs.dt.total_seconds()
        
        sampling_rate_analysis = {
            'expected_interval_seconds': 1.0,
            'mean_interval_seconds': float(actual_intervals.mean()),
            'std_interval_seconds': float(actual_intervals.std()),
            'min_interval_seconds': float(actual_intervals.min()),
            'max_interval_seconds': float(actual_intervals.max()),
            'irregular_intervals': int((abs(actual_intervals - 1.0) > 0.1).sum()),
            'irregular_percentage': float(((abs(actual_intervals - 1.0) > 0.1).sum() / len(actual_intervals)) * 100)
        }
        
        # Time sequence analysis
        is_monotonic = cast_data.index.is_monotonic_increasing
        has_duplicates = cast_data.index.has_duplicates
        
        time_sequence_analysis = {
            'is_monotonic_increasing': is_monotonic,
            'has_duplicate_timestamps': has_duplicates,
            'total_duration_seconds': float((cast_data.index[-1] - cast_data.index[0]).total_seconds()),
            'expected_duration_seconds': float(len(cast_data) - 1),
            'coverage_percentage': float(((len(cast_data) - 1) / (cast_data.index[-1] - cast_data.index[0]).total_seconds()) * 100) if (cast_data.index[-1] - cast_data.index[0]).total_seconds() > 0 else 100
        }
        
        # Calculate continuity score
        continuity_score = 1.0
        if not is_monotonic:
            continuity_score *= 0.5
        if has_duplicates:
            continuity_score *= 0.7
        if sampling_rate_analysis['irregular_percentage'] > 0.01:  # More than 0.01% irregular
            continuity_score *= 0.9
        if time_sequence_analysis['coverage_percentage'] < 99.9:  # Less than 99.9% coverage
            continuity_score *= 0.95
        
        return {
            'sampling_rate_analysis': sampling_rate_analysis,
            'time_sequence_analysis': time_sequence_analysis,
            'continuity_score': continuity_score
        }
    
    def _analyze_all_casts_temporal_continuity(self) -> Dict:
        """Analyze temporal continuity across all casts."""
        if not self.dataset_metadata:
            return {'error': 'No dataset metadata available'}
        
        all_sampling_stats = []
        all_continuity_scores = []
        sequence_issues = {'non_monotonic': 0, 'duplicates': 0}
        cast_count = 0
        
        # Analyze each cast
        for cast_metadata in self.dataset_metadata['cast_metadata']:
            cast_id = cast_metadata['cast_id']
            cast_file = self.data_path / 'raw' / f'cast_timeseries_{cast_id.split("_")[1]}.parquet'
            
            if cast_file.exists():
                cast_data = pd.read_parquet(cast_file)
                cast_analysis = self._analyze_single_cast_temporal_continuity(cast_data)
                
                if 'error' not in cast_analysis['sampling_rate_analysis']:
                    all_sampling_stats.append(cast_analysis['sampling_rate_analysis'])
                    all_continuity_scores.append(cast_analysis['continuity_score'])
                    
                    if not cast_analysis['time_sequence_analysis']['is_monotonic_increasing']:
                        sequence_issues['non_monotonic'] += 1
                    if cast_analysis['time_sequence_analysis']['has_duplicate_timestamps']:
                        sequence_issues['duplicates'] += 1
                
                cast_count += 1
        
        # Aggregate statistics
        if all_sampling_stats:
            aggregate_sampling = {
                'mean_interval_seconds': np.mean([s['mean_interval_seconds'] for s in all_sampling_stats]),
                'std_interval_seconds': np.mean([s['std_interval_seconds'] for s in all_sampling_stats]),
                'irregular_percentage': np.mean([s['irregular_percentage'] for s in all_sampling_stats]),
                'worst_irregular_percentage': max([s['irregular_percentage'] for s in all_sampling_stats])
            }
        else:
            aggregate_sampling = {'error': 'No valid sampling data found'}
        
        aggregate_sequence = {
            'casts_with_non_monotonic_time': sequence_issues['non_monotonic'],
            'casts_with_duplicate_timestamps': sequence_issues['duplicates'],
            'percentage_problematic_casts': ((sequence_issues['non_monotonic'] + sequence_issues['duplicates']) / cast_count) * 100 if cast_count > 0 else 0
        }
        
        avg_continuity_score = np.mean(all_continuity_scores) if all_continuity_scores else 0.0
        
        return {
            'sampling_rate_analysis': aggregate_sampling,
            'time_sequence_analysis': aggregate_sequence,
            'casts_analyzed': cast_count,
            'continuity_score': avg_continuity_score
        }
    
    def assess_synthetic_data_realism(self, cast_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Compare generated patterns to expected steel casting behavior.
        
        Args:
            cast_data (pd.DataFrame, optional): Single cast data. If None, analyzes all casts.
            
        Returns:
            Dict: Synthetic data realism analysis results
        """
        results = {
            'realism_analysis': {
                'distribution_analysis': {},
                'correlation_analysis': {},
                'process_behavior_analysis': {},
                'realism_score': 1.0
            }
        }
        
        if cast_data is not None:
            # Analyze single cast
            realism_analysis = self._analyze_single_cast_realism(cast_data)
            results['realism_analysis'].update(realism_analysis)
        else:
            # Analyze all casts
            realism_analysis = self._analyze_all_casts_realism()
            results['realism_analysis'].update(realism_analysis)
        
        return results
    
    def _analyze_single_cast_realism(self, cast_data: pd.DataFrame) -> Dict:
        """Analyze realism for a single cast."""
        # Distribution analysis
        distribution_analysis = {}
        for column in cast_data.columns:
            data = cast_data[column].dropna()
            if len(data) > 1:
                distribution_analysis[column] = {
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'skewness': float(data.skew()),
                    'kurtosis': float(data.kurtosis()),
                    'coefficient_of_variation': float(data.std() / data.mean()) if data.mean() != 0 else 0
                }
        
        # Correlation analysis
        correlation_matrix = cast_data.corr()
        correlation_analysis = {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': []
        }
        
        # Find strong correlations (expected in steel casting)
        for i, col1 in enumerate(correlation_matrix.columns):
            for j, col2 in enumerate(correlation_matrix.columns):
                if i < j:  # Avoid duplicates
                    corr_value = correlation_matrix.loc[col1, col2]
                    if abs(corr_value) > self.STRONG_CORRELATION_THRESHOLD:  # Threshold for "strong" correlation
                        correlation_analysis['strong_correlations'].append({
                            'sensor_pair': [col1, col2],
                            'correlation': float(corr_value)
                        })
        
        # Process behavior analysis
        process_behavior_analysis = self._analyze_process_behavior(cast_data)
        
        # Calculate realism score
        realism_score = self._calculate_realism_score(distribution_analysis, correlation_analysis, process_behavior_analysis)
        
        return {
            'distribution_analysis': distribution_analysis,
            'correlation_analysis': correlation_analysis,
            'process_behavior_analysis': process_behavior_analysis,
            'realism_score': realism_score
        }
    
    def _analyze_all_casts_realism(self) -> Dict:
        """Analyze realism across all casts."""
        if not self.dataset_metadata:
            return {'error': 'No dataset metadata available'}
        
        all_distributions = {}
        all_correlations = []
        all_process_behaviors = []
        all_realism_scores = []
        cast_count = 0
        
        # Initialize distribution aggregation
        for sensor in ['casting_speed', 'mold_temperature', 'mold_level', 'cooling_water_flow', 'superheat']:
            all_distributions[sensor] = {'means': [], 'stds': [], 'skewnesses': [], 'kurtoses': []}
        
        # Analyze each cast
        for cast_metadata in self.dataset_metadata['cast_metadata']:
            cast_id = cast_metadata['cast_id']
            cast_file = self.data_path / 'raw' / f'cast_timeseries_{cast_id.split("_")[1]}.parquet'
            
            if cast_file.exists():
                cast_data = pd.read_parquet(cast_file)
                cast_analysis = self._analyze_single_cast_realism(cast_data)
                
                # Aggregate distributions
                for sensor, stats in cast_analysis['distribution_analysis'].items():
                    if sensor in all_distributions:
                        all_distributions[sensor]['means'].append(stats['mean'])
                        all_distributions[sensor]['stds'].append(stats['std'])
                        all_distributions[sensor]['skewnesses'].append(stats['skewness'])
                        all_distributions[sensor]['kurtoses'].append(stats['kurtosis'])
                
                all_correlations.append(cast_analysis['correlation_analysis'])
                all_process_behaviors.append(cast_analysis['process_behavior_analysis'])
                all_realism_scores.append(cast_analysis['realism_score'])
                cast_count += 1
        
        # Calculate aggregate statistics
        aggregate_distributions = {}
        for sensor, values in all_distributions.items():
            if values['means']:
                aggregate_distributions[sensor] = {
                    'mean_of_means': float(np.mean(values['means'])),
                    'std_of_means': float(np.std(values['means'])),
                    'mean_of_stds': float(np.mean(values['stds'])),
                    'mean_skewness': float(np.mean(values['skewnesses'])),
                    'mean_kurtosis': float(np.mean(values['kurtoses']))
                }
        
        avg_realism_score = np.mean(all_realism_scores) if all_realism_scores else 0.0
        
        return {
            'distribution_analysis': aggregate_distributions,
            'correlation_analysis': {'average_strong_correlations': np.mean([len(c['strong_correlations']) for c in all_correlations]) if all_correlations else 0},
            'process_behavior_analysis': {'analyzed_casts': len(all_process_behaviors)},
            'casts_analyzed': cast_count,
            'realism_score': avg_realism_score
        }
    
    def _analyze_process_behavior(self, cast_data: pd.DataFrame) -> Dict:
        """Analyze steel casting process behavior patterns."""
        behavior_analysis = {}
        
        # Temperature stability analysis
        if 'mold_temperature' in cast_data.columns:
            temp_data = cast_data['mold_temperature']
            temp_variation = temp_data.std()
            behavior_analysis['temperature_stability'] = {
                'variation_std': float(temp_variation),
                'is_stable': temp_variation < 20,  # Less than 20°C std is considered stable
                'trend': 'increasing' if temp_data.iloc[-1] > temp_data.iloc[0] else 'decreasing'
            }
        
        # Speed consistency analysis
        if 'casting_speed' in cast_data.columns:
            speed_data = cast_data['casting_speed']
            speed_variation = speed_data.std()
            behavior_analysis['speed_consistency'] = {
                'variation_std': float(speed_variation),
                'is_consistent': speed_variation < 0.1,  # Less than 0.1 m/min std is consistent
                'trend': 'increasing' if speed_data.iloc[-1] > speed_data.iloc[0] else 'decreasing'
            }
        
        # Mold level control analysis
        if 'mold_level' in cast_data.columns:
            level_data = cast_data['mold_level']
            level_variation = level_data.std()
            behavior_analysis['mold_level_control'] = {
                'variation_std': float(level_variation),
                'is_well_controlled': level_variation < 10,  # Less than 10mm std is well controlled
                'excursions': int(((level_data < 130) | (level_data > 170)).sum())  # Outside normal range
            }
        
        return behavior_analysis
    
    def _calculate_realism_score(self, distribution_analysis: Dict, correlation_analysis: Dict, process_behavior_analysis: Dict) -> float:
        """Calculate overall realism score."""
        score = 1.0
        
        # Check if distributions are realistic
        for sensor, stats in distribution_analysis.items():
            # Penalize unrealistic coefficient of variation
            cv = stats.get('coefficient_of_variation', 0)
            if cv > self.COEFFICIENT_OF_VARIATION_THRESHOLD:  # High variability might be unrealistic
                score *= 0.95
            
            # Penalize extreme skewness
            if abs(stats.get('skewness', 0)) > self.MAX_ALLOWED_SKEWNESS:
                score *= 0.9
        
        # Check correlations
        strong_corr_count = len(correlation_analysis.get('strong_correlations', []))
        if strong_corr_count < 2:  # Expected some correlations in steel casting
            score *= 0.9
        
        # Check process behavior
        temp_stable = process_behavior_analysis.get('temperature_stability', {}).get('is_stable', True)
        speed_consistent = process_behavior_analysis.get('speed_consistency', {}).get('is_consistent', True)
        level_controlled = process_behavior_analysis.get('mold_level_control', {}).get('is_well_controlled', True)
        
        if not temp_stable:
            score *= 0.95
        if not speed_consistent:
            score *= 0.95
        if not level_controlled:
            score *= 0.9
        
        return max(0.0, min(1.0, score))
    
    def comprehensive_quality_assessment(self, cast_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Perform comprehensive data quality assessment.
        
        Args:
            cast_data (pd.DataFrame, optional): Single cast data. If None, analyzes all casts.
            
        Returns:
            Dict: Comprehensive quality assessment results
        """
        print("Performing comprehensive data quality assessment...")
        
        # Perform all assessments
        missing_values = self.assess_missing_values(cast_data)
        consistency = self.assess_data_consistency(cast_data)
        temporal_continuity = self.assess_temporal_continuity(cast_data)
        realism = self.assess_synthetic_data_realism(cast_data)
        
        # Calculate overall quality score
        scores = [
            missing_values['missing_value_analysis']['quality_score'],
            consistency['consistency_analysis']['consistency_score'],
            temporal_continuity['temporal_continuity']['continuity_score'],
            realism['realism_analysis']['realism_score']
        ]
        overall_score = np.mean(scores)
        
        # Determine quality level
        if overall_score >= 0.9:
            quality_level = "Excellent"
        elif overall_score >= 0.8:
            quality_level = "Good"
        elif overall_score >= 0.7:
            quality_level = "Acceptable"
        elif overall_score >= 0.6:
            quality_level = "Poor"
        else:
            quality_level = "Unacceptable"
        
        # Generate summary
        summary = {
            'overall_quality_score': overall_score,
            'quality_level': quality_level,
            'component_scores': {
                'missing_values': missing_values['missing_value_analysis']['quality_score'],
                'consistency': consistency['consistency_analysis']['consistency_score'],
                'temporal_continuity': temporal_continuity['temporal_continuity']['continuity_score'],
                'realism': realism['realism_analysis']['realism_score']
            },
            'assessment_timestamp': datetime.now().isoformat(),
            'data_scope': 'single_cast' if cast_data is not None else 'all_casts'
        }
        
        # Combine all results
        comprehensive_results = {
            'summary': summary,
            **missing_values,
            **consistency,
            **temporal_continuity,
            **realism
        }
        
        return comprehensive_results
    
    def generate_quality_report(self, assessment_results: Dict, output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive quality assessment report.
        
        Args:
            assessment_results (Dict): Results from comprehensive_quality_assessment
            output_path (str, optional): Path to save the report
            
        Returns:
            str: Path to the generated report file
        """
        if output_path is None:
            output_path = f"data_quality_assessment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Save the report
        with open(output_path, 'w') as f:
            json.dump(assessment_results, f, indent=2, default=str)
        
        print(f"Data quality assessment report saved to: {output_path}")
        return output_path