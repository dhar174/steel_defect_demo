"""Data quality validation utilities for steel casting data"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataQualityValidator:
    """Comprehensive data quality validation for steel casting datasets"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize validator with configuration.
        
        Args:
            config: Data generation configuration containing sensor specs and thresholds
        """
        self.config = config
        self.sensor_config = config.get('data_generation', {}).get('sensors', {})
        self.defect_config = config.get('data_generation', {}).get('defect_simulation', {})
        
    def validate_statistical_properties(self, df: pd.DataFrame, cast_id: str) -> Dict[str, Any]:
        """
        Validate statistical properties of sensor data.
        
        Args:
            df: Time series data for a single cast
            cast_id: Cast identifier
            
        Returns:
            Dict containing validation results
        """
        results = {
            'cast_id': cast_id,
            'validation_type': 'statistical',
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'passed': True,
            'issues': []
        }
        
        for sensor_name, sensor_cfg in self.sensor_config.items():
            if sensor_name == 'mold_level_normal_range':
                continue
                
            if sensor_name not in df.columns:
                results['issues'].append(f"Missing sensor data: {sensor_name}")
                results['passed'] = False
                continue
                
            sensor_data = df[sensor_name]
            
            # Check value ranges
            min_val, max_val = sensor_data.min(), sensor_data.max()
            expected_min, expected_max = sensor_cfg['min_value'], sensor_cfg['max_value']
            
            range_check = {
                'actual_min': float(min_val),
                'actual_max': float(max_val),
                'expected_min': expected_min,
                'expected_max': expected_max,
                'range_valid': expected_min <= min_val <= max_val <= expected_max
            }
            
            if not range_check['range_valid']:
                results['issues'].append(f"{sensor_name}: Values outside expected range")
                results['passed'] = False
            
            # Check noise characteristics
            mean_val = sensor_data.mean()
            std_val = sensor_data.std()
            expected_base = sensor_cfg['base_value']
            expected_std = sensor_cfg['noise_std']
            
            # Allow some deviation from expected values
            mean_deviation = abs(mean_val - expected_base) / expected_base
            std_ratio = std_val / expected_std
            
            noise_check = {
                'actual_mean': float(mean_val),
                'actual_std': float(std_val),
                'expected_base': expected_base,
                'expected_std': expected_std,
                'mean_deviation_pct': float(mean_deviation * 100),
                'std_ratio': float(std_ratio),
                'noise_reasonable': mean_deviation < 0.2 and 0.5 < std_ratio < 2.0
            }
            
            if not noise_check['noise_reasonable']:
                results['issues'].append(f"{sensor_name}: Unusual noise characteristics")
                # Don't fail for noise issues, just warn
            
            results['checks'][sensor_name] = {
                'range_check': range_check,
                'noise_check': noise_check
            }
        
        return results
    
    def validate_defect_logic(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that defect labels match the underlying process conditions.
        
        Args:
            df: Time series data for a single cast
            metadata: Cast metadata including defect information
            
        Returns:
            Dict containing validation results
        """
        results = {
            'cast_id': metadata['cast_id'],
            'validation_type': 'defect_logic',
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'passed': True,
            'issues': []
        }
        
        # Re-detect triggers to compare with stored triggers
        detected_triggers = self._detect_defect_triggers_validation(df)
        stored_triggers = metadata.get('defect_trigger_events', [])
        stored_defect = bool(metadata.get('defect_label', 0))
        
        # Check trigger consistency
        trigger_consistency = set(detected_triggers) == set(stored_triggers)
        if not trigger_consistency:
            results['issues'].append(
                f"Trigger mismatch - Detected: {detected_triggers}, Stored: {stored_triggers}"
            )
            results['passed'] = False
        
        # Check defect logic consistency
        should_have_defect = len(detected_triggers) > 0
        defect_logic_consistent = True
        
        if stored_defect and not detected_triggers:
            results['issues'].append("Cast labeled as defect but no triggers detected")
            defect_logic_consistent = False
            
        results['checks'] = {
            'detected_triggers': detected_triggers,
            'stored_triggers': stored_triggers,
            'trigger_consistency': trigger_consistency,
            'stored_defect_label': stored_defect,
            'triggers_present': len(detected_triggers) > 0,
            'defect_logic_consistent': defect_logic_consistent
        }
        
        if not defect_logic_consistent:
            results['passed'] = False
        
        return results
    
    def _detect_defect_triggers_validation(self, df: pd.DataFrame) -> List[str]:
        """
        Re-implement defect trigger detection for validation.
        This should match the logic in data_generator.py
        """
        triggers = []
        
        # 1. Prolonged mold level deviation
        mold_level_normal_range = self.sensor_config.get('mold_level_normal_range', [130, 170])
        outside_range = (df['mold_level'] < mold_level_normal_range[0]) | (df['mold_level'] > mold_level_normal_range[1])
        
        consecutive_periods = []
        in_deviation = False
        start_idx = 0
        
        for i, is_outside in enumerate(outside_range):
            if is_outside and not in_deviation:
                start_idx = i
                in_deviation = True
            elif not is_outside and in_deviation:
                consecutive_periods.append(i - start_idx)
                in_deviation = False
        
        if in_deviation:
            consecutive_periods.append(len(outside_range) - start_idx)
        
        prolonged_threshold = self.defect_config.get('defect_triggers', {}).get('prolonged_mold_level_deviation', 30)
        if any(period >= prolonged_threshold for period in consecutive_periods):
            triggers.append('prolonged_mold_level_deviation')
        
        # 2. Rapid temperature drop
        temp_diff = df['mold_temperature'].diff(periods=60)
        rapid_drop_threshold = -self.defect_config.get('defect_triggers', {}).get('rapid_temperature_drop', 50)
        if (temp_diff < rapid_drop_threshold).any():
            triggers.append('rapid_temperature_drop')
        
        # 3. High speed with low superheat
        if self.defect_config.get('defect_triggers', {}).get('high_speed_with_low_superheat', True):
            high_speed_low_superheat = (df['casting_speed'] > 1.5) & (df['superheat'] < 20)
            if high_speed_low_superheat.any():
                triggers.append('high_speed_with_low_superheat')
        
        return triggers
    
    def validate_temporal_consistency(self, df: pd.DataFrame, cast_id: str) -> Dict[str, Any]:
        """
        Validate temporal consistency and correlations in the data.
        
        Args:
            df: Time series data for a single cast
            cast_id: Cast identifier
            
        Returns:
            Dict containing validation results
        """
        results = {
            'cast_id': cast_id,
            'validation_type': 'temporal',
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'passed': True,
            'issues': []
        }
        
        # Check for timestamp consistency
        time_diffs = df.index.to_series().diff()[1:]  # Skip first NaN
        expected_freq = pd.Timedelta(seconds=1)  # 1 Hz sampling
        
        freq_consistent = (time_diffs == expected_freq).all()
        if not freq_consistent:
            results['issues'].append("Inconsistent timestamp frequency")
            results['passed'] = False
        
        # Check for reasonable correlations between related sensors
        correlations = {}
        
        # Mold temperature and superheat should be positively correlated
        if 'mold_temperature' in df.columns and 'superheat' in df.columns:
            temp_superheat_corr = df['mold_temperature'].corr(df['superheat'])
            correlations['temp_superheat'] = float(temp_superheat_corr)
            
            if temp_superheat_corr < 0.1:  # Expecting some positive correlation
                results['issues'].append("Unexpectedly low correlation between temperature and superheat")
        
        # Check for excessive drift in sensor values
        for sensor_name in ['casting_speed', 'mold_temperature', 'mold_level', 'cooling_water_flow']:
            if sensor_name in df.columns:
                sensor_data = df[sensor_name]
                drift = abs(sensor_data.iloc[-1] - sensor_data.iloc[0]) / sensor_data.mean()
                
                if drift > 0.5:  # More than 50% drift is suspicious
                    results['issues'].append(f"Excessive drift in {sensor_name}: {drift:.2%}")
        
        results['checks'] = {
            'timestamp_frequency_consistent': freq_consistent,
            'correlations': correlations,
            'data_completeness': float(df.notna().all(axis=1).mean())
        }
        
        return results
    
    def validate_dataset_distribution(self, metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate overall dataset properties and distributions.
        
        Args:
            metadata_list: List of metadata for all casts
            
        Returns:
            Dict containing validation results
        """
        results = {
            'validation_type': 'dataset_distribution',
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'passed': True,
            'issues': []
        }
        
        total_casts = len(metadata_list)
        defect_casts = sum(1 for m in metadata_list if m.get('defect_label', 0))
        actual_defect_rate = defect_casts / total_casts if total_casts > 0 else 0
        target_defect_rate = self.defect_config.get('defect_probability', 0.15)
        
        # Check defect rate is within reasonable bounds
        defect_rate_tolerance = 0.05  # 5% tolerance
        defect_rate_ok = abs(actual_defect_rate - target_defect_rate) <= defect_rate_tolerance
        
        if not defect_rate_ok:
            results['issues'].append(
                f"Defect rate {actual_defect_rate:.2%} too far from target {target_defect_rate:.2%}"
            )
            results['passed'] = False
        
        # Check trigger distribution
        trigger_counts = {}
        for metadata in metadata_list:
            for trigger in metadata.get('defect_trigger_events', []):
                trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
        
        # Check steel grade distribution
        grade_counts = {}
        for metadata in metadata_list:
            grade = metadata.get('steel_grade', 'Unknown')
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
        
        results['checks'] = {
            'total_casts': total_casts,
            'defect_casts': defect_casts,
            'actual_defect_rate': actual_defect_rate,
            'target_defect_rate': target_defect_rate,
            'defect_rate_within_tolerance': defect_rate_ok,
            'trigger_distribution': trigger_counts,
            'grade_distribution': grade_counts
        }
        
        return results
    
    def validate_file_integrity(self, output_dir: Path) -> Dict[str, Any]:
        """
        Validate integrity of generated files.
        
        Args:
            output_dir: Directory containing generated data
            
        Returns:
            Dict containing validation results
        """
        results = {
            'validation_type': 'file_integrity',
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'passed': True,
            'issues': []
        }
        
        # Check directory structure
        expected_dirs = ['raw_timeseries', 'metadata', 'summary']
        missing_dirs = []
        
        for expected_dir in expected_dirs:
            dir_path = output_dir / expected_dir
            if not dir_path.exists():
                missing_dirs.append(expected_dir)
                results['passed'] = False
        
        if missing_dirs:
            results['issues'].append(f"Missing directories: {missing_dirs}")
        
        # Check file counts and sizes
        file_checks = {}
        
        if (output_dir / 'raw_timeseries').exists():
            parquet_files = list((output_dir / 'raw_timeseries').glob('*.parquet'))
            file_checks['raw_timeseries'] = {
                'file_count': len(parquet_files),
                'total_size_mb': sum(f.stat().st_size for f in parquet_files) / 1024 / 1024
            }
        
        if (output_dir / 'metadata').exists():
            metadata_files = list((output_dir / 'metadata').glob('*.json'))
            file_checks['metadata'] = {
                'file_count': len(metadata_files),
                'total_size_mb': sum(f.stat().st_size for f in metadata_files) / 1024 / 1024
            }
        
        results['checks'] = {
            'directory_structure_complete': len(missing_dirs) == 0,
            'missing_directories': missing_dirs,
            'file_checks': file_checks
        }
        
        return results
    
    def generate_quality_report(self, validation_results: List[Dict[str, Any]], 
                              output_path: Path) -> Dict[str, Any]:
        """
        Generate comprehensive quality report from validation results.
        
        Args:
            validation_results: List of all validation results
            output_path: Path to save the quality report
            
        Returns:
            Dict containing the quality report summary
        """
        report = {
            'generation_timestamp': datetime.now().isoformat(),
            'total_validations': len(validation_results),
            'validation_summary': {},
            'overall_passed': True,
            'issues_summary': {},
            'detailed_results': self._convert_for_json(validation_results)
        }
        
        # Summarize by validation type
        by_type = {}
        issue_counts = {}
        
        for result in validation_results:
            val_type = result.get('validation_type', 'unknown')
            
            if val_type not in by_type:
                by_type[val_type] = {'total': 0, 'passed': 0, 'failed': 0}
                issue_counts[val_type] = {}
            
            by_type[val_type]['total'] += 1
            
            if result.get('passed', False):
                by_type[val_type]['passed'] += 1
            else:
                by_type[val_type]['failed'] += 1
                report['overall_passed'] = False
                
                # Count issue types
                for issue in result.get('issues', []):
                    issue_type = issue.split(':')[0] if ':' in issue else issue
                    issue_counts[val_type][issue_type] = issue_counts[val_type].get(issue_type, 0) + 1
        
        report['validation_summary'] = by_type
        report['issues_summary'] = issue_counts
        
        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Quality report saved to {output_path}")
        
        return report
    
    def _convert_for_json(self, obj):
        """Convert numpy types and other non-serializable objects for JSON."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj