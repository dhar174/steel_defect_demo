"""Specialized defect labeling validation for steel casting data"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime
import logging
from collections import defaultdict

from .validation import DataQualityValidator

logger = logging.getLogger(__name__)


class DefectLabelingValidator(DataQualityValidator):
    """
    Specialized validator for defect labeling logic and distribution analysis.
    
    This class extends the base DataQualityValidator to provide focused validation
    of defect labeling consistency, domain knowledge alignment, and edge case detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize defect labeling validator.
        
        Args:
            config: Data generation configuration containing sensor specs and thresholds
        """
        super().__init__(config)
        
        # Domain knowledge thresholds and rules
        self.domain_rules = {
            'mold_level_critical_deviation': 20,  # Seconds outside normal range that's very concerning
            'temperature_drop_severe': 75,  # °C drop that's extremely concerning
            'speed_superheat_critical': (1.7, 18),  # (speed m/min, superheat °C) very risky combination
            'normal_operation_ranges': {
                'casting_speed': (1.0, 1.5),  # Normal operating range
                'mold_temperature': (1500, 1540),  # Normal operating range
                'superheat': (20, 35),  # Normal operating range
                'mold_level': (135, 165)  # Tighter normal range than min/max
            }
        }
    
    def analyze_label_distribution(self, metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Comprehensive analysis of defect label distribution.
        
        Args:
            metadata_list: List of metadata for all casts
            
        Returns:
            Dict containing detailed distribution analysis
        """
        total_casts = len(metadata_list)
        defect_casts = [m for m in metadata_list if m.get('defect_label', 0)]
        good_casts = [m for m in metadata_list if not m.get('defect_label', 0)]
        
        # Basic distribution
        defect_count = len(defect_casts)
        good_count = len(good_casts)
        defect_rate = defect_count / total_casts if total_casts > 0 else 0
        
        # Trigger analysis
        trigger_analysis = self._analyze_trigger_distribution(metadata_list)
        
        # Grade-based analysis
        grade_analysis = self._analyze_grade_distribution(metadata_list)
        
        # Temporal pattern analysis
        temporal_analysis = self._analyze_temporal_patterns(metadata_list)
        
        # Statistical significance tests
        statistical_tests = self._perform_statistical_tests(defect_casts, good_casts)
        
        return {
            'analysis_type': 'label_distribution',
            'timestamp': datetime.now().isoformat(),
            'dataset_summary': {
                'total_casts': total_casts,
                'defect_casts': defect_count,
                'good_casts': good_count,
                'defect_rate': defect_rate,
                'target_defect_rate': self.defect_config.get('defect_probability', 0.15),
                'class_balance_ratio': good_count / defect_count if defect_count > 0 else float('inf')
            },
            'trigger_analysis': trigger_analysis,
            'grade_analysis': grade_analysis,
            'temporal_analysis': temporal_analysis,
            'statistical_tests': statistical_tests,
            'recommendations': self._generate_distribution_recommendations(
                defect_rate, trigger_analysis, grade_analysis
            )
        }
    
    def validate_domain_knowledge_alignment(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that defect labels align with steel casting domain knowledge.
        
        Args:
            df: Time series data for a single cast
            metadata: Cast metadata including defect information
            
        Returns:
            Dict containing domain validation results
        """
        cast_id = metadata['cast_id']
        defect_label = bool(metadata.get('defect_label', 0))
        triggers = metadata.get('defect_trigger_events', [])
        
        results = {
            'cast_id': cast_id,
            'validation_type': 'domain_knowledge',
            'timestamp': datetime.now().isoformat(),
            'defect_label': defect_label,
            'triggers': triggers,
            'domain_checks': {},
            'severity_assessment': {},
            'passed': True,
            'issues': [],
            'warnings': []
        }
        
        # Check each trigger against domain knowledge
        domain_checks = {}
        
        if 'prolonged_mold_level_deviation' in triggers:
            domain_checks['mold_level'] = self._validate_mold_level_domain_logic(df)
        
        if 'rapid_temperature_drop' in triggers:
            domain_checks['temperature'] = self._validate_temperature_domain_logic(df)
        
        if 'high_speed_with_low_superheat' in triggers:
            domain_checks['speed_superheat'] = self._validate_speed_superheat_domain_logic(df)
        
        # Assess overall severity
        severity_assessment = self._assess_defect_severity(df, triggers)
        
        # Check for concerning patterns not captured by triggers
        missed_patterns = self._identify_missed_domain_patterns(df, defect_label, triggers)
        
        # Domain logic consistency checks
        consistency_issues = self._check_domain_consistency(df, defect_label, triggers)
        
        results.update({
            'domain_checks': domain_checks,
            'severity_assessment': severity_assessment,
            'missed_patterns': missed_patterns,
            'consistency_issues': consistency_issues
        })
        
        # Determine overall pass/fail
        if consistency_issues or severity_assessment.get('severity_mismatch', False):
            results['passed'] = False
            results['issues'].extend(consistency_issues)
        
        return results
    
    def identify_edge_cases(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify borderline cases that may be mislabeled or require expert review.
        
        Args:
            df: Time series data for a single cast
            metadata: Cast metadata including defect information
            
        Returns:
            Dict containing edge case analysis
        """
        cast_id = metadata['cast_id']
        defect_label = bool(metadata.get('defect_label', 0))
        triggers = metadata.get('defect_trigger_events', [])
        
        edge_cases = {
            'cast_id': cast_id,
            'analysis_type': 'edge_case_detection',
            'timestamp': datetime.now().isoformat(),
            'defect_label': defect_label,
            'triggers': triggers,
            'edge_case_flags': {},
            'borderline_conditions': {},
            'uncertainty_score': 0.0,
            'requires_expert_review': False,
            'review_reasons': []
        }
        
        # Near-threshold conditions
        borderline_conditions = self._detect_borderline_conditions(df)
        
        # Conflicting signals
        conflicting_signals = self._detect_conflicting_signals(df)
        
        # Inconsistent labeling patterns
        labeling_inconsistencies = self._detect_labeling_inconsistencies(df, defect_label, triggers)
        
        # Calculate uncertainty score
        uncertainty_factors = [
            borderline_conditions.get('near_threshold_count', 0) / 10,
            len(conflicting_signals) / 5,
            len(labeling_inconsistencies) / 3
        ]
        uncertainty_score = min(1.0, sum(uncertainty_factors))
        
        # Determine if expert review is needed
        requires_review = (
            uncertainty_score > 0.7 or
            len(conflicting_signals) > 2 or
            len(labeling_inconsistencies) > 1
        )
        
        if requires_review:
            review_reasons = []
            if uncertainty_score > 0.7:
                review_reasons.append(f"High uncertainty score: {uncertainty_score:.2f}")
            if len(conflicting_signals) > 2:
                review_reasons.append(f"Multiple conflicting signals: {len(conflicting_signals)}")
            if len(labeling_inconsistencies) > 1:
                review_reasons.append(f"Labeling inconsistencies: {len(labeling_inconsistencies)}")
            
            edge_cases['review_reasons'] = review_reasons
        
        edge_cases.update({
            'borderline_conditions': borderline_conditions,
            'conflicting_signals': conflicting_signals,
            'labeling_inconsistencies': labeling_inconsistencies,
            'uncertainty_score': uncertainty_score,
            'requires_expert_review': requires_review
        })
        
        return edge_cases
    
    def generate_expert_review_documentation(self, 
                                           label_distribution: Dict[str, Any],
                                           domain_validations: List[Dict[str, Any]],
                                           edge_cases: List[Dict[str, Any]],
                                           output_path: Path) -> Dict[str, Any]:
        """
        Generate comprehensive documentation for domain expert review.
        
        Args:
            label_distribution: Results from label distribution analysis
            domain_validations: Results from domain knowledge validation
            edge_cases: Results from edge case detection
            output_path: Path to save the expert review document
            
        Returns:
            Dict containing the expert review summary
        """
        expert_review = {
            'document_type': 'expert_review_documentation',
            'generation_timestamp': datetime.now().isoformat(),
            'executive_summary': {},
            'detailed_findings': {},
            'recommendations': {},
            'appendices': {}
        }
        
        # Executive summary
        total_casts = label_distribution['dataset_summary']['total_casts']
        defect_rate = label_distribution['dataset_summary']['defect_rate']
        edge_case_count = len([ec for ec in edge_cases if ec.get('requires_expert_review', False)])
        
        domain_issues = len([dv for dv in domain_validations if not dv.get('passed', True)])
        
        expert_review['executive_summary'] = {
            'dataset_overview': {
                'total_casts_analyzed': total_casts,
                'overall_defect_rate': f"{defect_rate:.1%}",
                'cases_requiring_review': edge_case_count,
                'domain_validation_issues': domain_issues
            },
            'key_findings': self._generate_key_findings(label_distribution, domain_validations, edge_cases),
            'overall_assessment': self._generate_overall_assessment(label_distribution, domain_validations, edge_cases)
        }
        
        # Detailed findings
        expert_review['detailed_findings'] = {
            'label_distribution_analysis': label_distribution,
            'domain_knowledge_validation': self._summarize_domain_validations(domain_validations),
            'edge_case_analysis': self._summarize_edge_cases(edge_cases),
            'trigger_effectiveness': self._analyze_trigger_effectiveness(label_distribution, domain_validations)
        }
        
        # Recommendations
        expert_review['recommendations'] = self._generate_expert_recommendations(
            label_distribution, domain_validations, edge_cases
        )
        
        # Appendices with detailed data
        expert_review['appendices'] = {
            'failed_domain_validations': [dv for dv in domain_validations if not dv.get('passed', True)],
            'high_uncertainty_cases': [ec for ec in edge_cases if ec.get('uncertainty_score', 0) > 0.7],
            'statistical_analysis': label_distribution.get('statistical_tests', {}),
            'methodology_notes': self._generate_methodology_notes()
        }
        
        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(expert_review, f, indent=2)
        
        logger.info(f"Expert review documentation saved to {output_path}")
        
        return expert_review
    
    def _analyze_trigger_distribution(self, metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze distribution of defect triggers across the dataset."""
        trigger_counts = defaultdict(int)
        trigger_combinations = defaultdict(int)
        defect_casts_with_triggers = 0
        defect_casts_without_triggers = 0
        
        for metadata in metadata_list:
            triggers = metadata.get('defect_trigger_events', [])
            defect_label = metadata.get('defect_label', 0)
            
            if defect_label:
                if triggers:
                    defect_casts_with_triggers += 1
                    for trigger in triggers:
                        trigger_counts[trigger] += 1
                    
                    # Track combinations
                    trigger_combo = tuple(sorted(triggers))
                    trigger_combinations[trigger_combo] += 1
                else:
                    defect_casts_without_triggers += 1
        
        return {
            'individual_triggers': dict(trigger_counts),
            'trigger_combinations': {str(k): v for k, v in trigger_combinations.items()},
            'defect_casts_with_triggers': defect_casts_with_triggers,
            'defect_casts_without_triggers': defect_casts_without_triggers,
            'trigger_coverage': defect_casts_with_triggers / (defect_casts_with_triggers + defect_casts_without_triggers) if (defect_casts_with_triggers + defect_casts_without_triggers) > 0 else 0
        }
    
    def _analyze_grade_distribution(self, metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze defect distribution by steel grade."""
        grade_stats = defaultdict(lambda: {'total': 0, 'defects': 0})
        
        for metadata in metadata_list:
            grade = metadata.get('steel_grade', 'Unknown')
            defect_label = metadata.get('defect_label', 0)
            
            grade_stats[grade]['total'] += 1
            if defect_label:
                grade_stats[grade]['defects'] += 1
        
        # Calculate rates
        for grade, stats in grade_stats.items():
            stats['defect_rate'] = stats['defects'] / stats['total'] if stats['total'] > 0 else 0
        
        return dict(grade_stats)
    
    def _analyze_temporal_patterns(self, metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in defect occurrence."""
        # For synthetic data, we'll analyze patterns based on generation order
        defect_positions = [i for i, m in enumerate(metadata_list) if m.get('defect_label', 0)]
        
        if len(defect_positions) < 2:
            return {'sufficient_data': False}
        
        # Calculate intervals between defects
        intervals = [defect_positions[i+1] - defect_positions[i] for i in range(len(defect_positions)-1)]
        
        return {
            'sufficient_data': True,
            'defect_positions': defect_positions,
            'intervals_between_defects': {
                'mean': np.mean(intervals),
                'std': np.std(intervals),
                'min': min(intervals),
                'max': max(intervals)
            }
        }
    
    def _perform_statistical_tests(self, defect_casts: List[Dict], good_casts: List[Dict]) -> Dict[str, Any]:
        """Perform statistical tests on cast characteristics."""
        if not defect_casts or not good_casts:
            return {'sufficient_data': False}
        
        # Extract process summaries
        defect_speeds = [cast.get('process_summary', {}).get('avg_casting_speed', 0) for cast in defect_casts]
        good_speeds = [cast.get('process_summary', {}).get('avg_casting_speed', 0) for cast in good_casts]
        
        defect_temps = [cast.get('process_summary', {}).get('avg_mold_temperature', 0) for cast in defect_casts]
        good_temps = [cast.get('process_summary', {}).get('avg_mold_temperature', 0) for cast in good_casts]
        
        # Simple statistical comparisons (avoiding scipy.stats dependency)
        tests = {
            'sufficient_data': True,
            'casting_speed': {
                'defect_mean': np.mean(defect_speeds),
                'good_mean': np.mean(good_speeds),
                'difference': np.mean(defect_speeds) - np.mean(good_speeds)
            },
            'mold_temperature': {
                'defect_mean': np.mean(defect_temps),
                'good_mean': np.mean(good_temps),
                'difference': np.mean(defect_temps) - np.mean(good_temps)
            }
        }
        
        return tests
    
    def _generate_distribution_recommendations(self, defect_rate: float, 
                                             trigger_analysis: Dict, 
                                             grade_analysis: Dict) -> List[str]:
        """Generate recommendations based on distribution analysis."""
        recommendations = []
        
        target_rate = self.defect_config.get('defect_probability', 0.15)
        
        if abs(defect_rate - target_rate) > 0.05:
            recommendations.append(
                f"Defect rate ({defect_rate:.1%}) significantly differs from target ({target_rate:.1%}). "
                "Consider adjusting base defect probability or trigger thresholds."
            )
        
        trigger_coverage = trigger_analysis.get('trigger_coverage', 0)
        if trigger_coverage < 0.8:
            recommendations.append(
                f"Only {trigger_coverage:.1%} of defect cases have identified triggers. "
                "Consider reviewing trigger logic or adding new trigger conditions."
            )
        
        # Check grade balance
        grade_rates = [stats['defect_rate'] for stats in grade_analysis.values()]
        if len(grade_rates) > 1 and max(grade_rates) - min(grade_rates) > 0.1:
            recommendations.append(
                "Significant variation in defect rates across steel grades. "
                "Consider grade-specific defect probability settings."
            )
        
        return recommendations
    
    def _validate_mold_level_domain_logic(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate mold level deviation trigger against domain knowledge."""
        normal_range = self.sensor_config.get('mold_level_normal_range', [130, 170])
        critical_threshold = self.domain_rules['mold_level_critical_deviation']
        
        outside_range = (df['mold_level'] < normal_range[0]) | (df['mold_level'] > normal_range[1])
        max_consecutive = self._get_max_consecutive_true(outside_range)
        
        return {
            'trigger_type': 'mold_level_deviation',
            'max_consecutive_seconds': max_consecutive,
            'critical_threshold': critical_threshold,
            'exceeds_critical': max_consecutive > critical_threshold,
            'domain_assessment': 'severe' if max_consecutive > critical_threshold else 'moderate',
            'justification': self._get_mold_level_justification(max_consecutive, critical_threshold)
        }
    
    def _validate_temperature_domain_logic(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate temperature drop trigger against domain knowledge."""
        temp_diff = df['mold_temperature'].diff(periods=60)
        max_drop = abs(temp_diff.min()) if len(temp_diff) > 0 else 0
        severe_threshold = self.domain_rules['temperature_drop_severe']
        
        return {
            'trigger_type': 'rapid_temperature_drop',
            'max_drop_60s': max_drop,
            'severe_threshold': severe_threshold,
            'exceeds_severe': max_drop > severe_threshold,
            'domain_assessment': 'severe' if max_drop > severe_threshold else 'moderate',
            'justification': self._get_temperature_justification(max_drop, severe_threshold)
        }
    
    def _validate_speed_superheat_domain_logic(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate speed-superheat combination trigger against domain knowledge."""
        critical_speed, critical_superheat = self.domain_rules['speed_superheat_critical']
        
        risky_conditions = (df['casting_speed'] > critical_speed) & (df['superheat'] < critical_superheat)
        risky_duration = risky_conditions.sum()  # Number of seconds in risky state
        
        return {
            'trigger_type': 'high_speed_low_superheat',
            'critical_speed_threshold': critical_speed,
            'critical_superheat_threshold': critical_superheat,
            'risky_duration_seconds': risky_duration,
            'exceeds_critical': risky_duration > 0,
            'domain_assessment': 'severe' if risky_duration > 30 else 'moderate',
            'justification': self._get_speed_superheat_justification(risky_duration, critical_speed, critical_superheat)
        }
    
    def _assess_defect_severity(self, df: pd.DataFrame, triggers: List[str]) -> Dict[str, Any]:
        """Assess the overall severity of defect conditions."""
        severity_scores = []
        
        for trigger in triggers:
            if trigger == 'prolonged_mold_level_deviation':
                # More severe for longer deviations
                normal_range = self.sensor_config.get('mold_level_normal_range', [130, 170])
                outside_range = (df['mold_level'] < normal_range[0]) | (df['mold_level'] > normal_range[1])
                max_consecutive = self._get_max_consecutive_true(outside_range)
                severity_scores.append(min(1.0, max_consecutive / 60))  # Normalize to 1 at 60 seconds
                
            elif trigger == 'rapid_temperature_drop':
                temp_diff = df['mold_temperature'].diff(periods=60)
                max_drop = abs(temp_diff.min()) if len(temp_diff) > 0 else 0
                severity_scores.append(min(1.0, max_drop / 100))  # Normalize to 1 at 100°C drop
                
            elif trigger == 'high_speed_with_low_superheat':
                risky_conditions = (df['casting_speed'] > 1.5) & (df['superheat'] < 20)
                risky_duration = risky_conditions.sum()
                severity_scores.append(min(1.0, risky_duration / 120))  # Normalize to 1 at 2 minutes
        
        overall_severity = max(severity_scores) if severity_scores else 0
        
        return {
            'individual_severity_scores': severity_scores,
            'overall_severity': overall_severity,
            'severity_level': self._categorize_severity(overall_severity),
            'severity_mismatch': False  # Could add logic to detect mismatches
        }
    
    def _identify_missed_domain_patterns(self, df: pd.DataFrame, defect_label: bool, triggers: List[str]) -> List[str]:
        """Identify concerning patterns not captured by current triggers."""
        missed_patterns = []
        
        # Check for patterns that should trigger defects but don't
        normal_ranges = self.domain_rules['normal_operation_ranges']
        
        # Excessive variability
        for sensor, (min_normal, max_normal) in normal_ranges.items():
            if sensor in df.columns:
                sensor_std = df[sensor].std()
                sensor_range = max_normal - min_normal
                if sensor_std > sensor_range * 0.3:  # High variability
                    missed_patterns.append(f"High variability in {sensor} (std: {sensor_std:.2f})")
        
        # Sustained operation outside normal ranges
        for sensor, (min_normal, max_normal) in normal_ranges.items():
            if sensor in df.columns:
                outside_normal = (df[sensor] < min_normal) | (df[sensor] > max_normal)
                if outside_normal.mean() > 0.3:  # More than 30% of time outside normal
                    missed_patterns.append(f"Sustained operation outside normal range: {sensor}")
        
        return missed_patterns
    
    def _check_domain_consistency(self, df: pd.DataFrame, defect_label: bool, triggers: List[str]) -> List[str]:
        """Check for domain logic consistency issues."""
        issues = []
        
        # Good cast with severe conditions
        if not defect_label:
            # Check if good cast has severe conditions that should cause defects
            severe_conditions = self._check_severe_conditions(df)
            if severe_conditions:
                issues.append(f"Good cast has severe conditions: {severe_conditions}")
        
        # Defect cast with mild triggers
        if defect_label and triggers:
            # Check if triggers are actually mild
            mild_triggers = self._check_trigger_severity(df, triggers)
            if mild_triggers:
                issues.append(f"Defect cast has only mild triggers: {mild_triggers}")
        
        return issues
    
    def _detect_borderline_conditions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect conditions that are near trigger thresholds."""
        borderline = {
            'near_threshold_conditions': [],
            'near_threshold_count': 0
        }
        
        # Mold level near boundaries
        normal_range = self.sensor_config.get('mold_level_normal_range', [130, 170])
        buffer = 5  # 5mm buffer
        
        near_lower = (df['mold_level'] > normal_range[0] - buffer) & (df['mold_level'] < normal_range[0] + buffer)
        near_upper = (df['mold_level'] > normal_range[1] - buffer) & (df['mold_level'] < normal_range[1] + buffer)
        
        if near_lower.any() or near_upper.any():
            borderline['near_threshold_conditions'].append('mold_level_near_boundary')
            borderline['near_threshold_count'] += 1
        
        # Temperature drops near threshold
        temp_diff = df['mold_temperature'].diff(periods=60)
        near_threshold_drop = (temp_diff < -40) & (temp_diff > -50)  # Near 50°C threshold
        
        if near_threshold_drop.any():
            borderline['near_threshold_conditions'].append('temperature_drop_near_threshold')
            borderline['near_threshold_count'] += 1
        
        # Speed-superheat near critical combination
        near_critical = (df['casting_speed'] > 1.4) & (df['casting_speed'] < 1.6) & (df['superheat'] > 18) & (df['superheat'] < 22)
        
        if near_critical.any():
            borderline['near_threshold_conditions'].append('speed_superheat_near_critical')
            borderline['near_threshold_count'] += 1
        
        return borderline
    
    def _detect_conflicting_signals(self, df: pd.DataFrame) -> List[str]:
        """Detect conflicting signals in the data."""
        conflicts = []
        
        # High superheat with rapid cooling (contradictory)
        high_superheat = df['superheat'] > 30
        high_cooling = df['cooling_water_flow'] > 230
        
        if (high_superheat & high_cooling).any():
            conflicts.append("High superheat with excessive cooling")
        
        # Low casting speed with high temperature (inefficient)
        low_speed = df['casting_speed'] < 1.0
        high_temp = df['mold_temperature'] > 1550
        
        if (low_speed & high_temp).any():
            conflicts.append("Low casting speed with high temperature")
        
        # Stable mold level with high flow variations
        stable_level = df['mold_level'].std() < 3
        high_flow_variation = df['cooling_water_flow'].std() > 20
        
        if stable_level and high_flow_variation:
            conflicts.append("Stable mold level with high cooling flow variation")
        
        return conflicts
    
    def _detect_labeling_inconsistencies(self, df: pd.DataFrame, defect_label: bool, triggers: List[str]) -> List[str]:
        """Detect potential labeling inconsistencies."""
        inconsistencies = []
        
        # No triggers but labeled as defect
        if defect_label and not triggers:
            # Check if conditions are actually normal
            normal_ranges = self.domain_rules['normal_operation_ranges']
            all_normal = True
            
            for sensor, (min_normal, max_normal) in normal_ranges.items():
                if sensor in df.columns:
                    sensor_data = df[sensor]
                    outside_normal_pct = ((sensor_data < min_normal) | (sensor_data > max_normal)).mean()
                    if outside_normal_pct > 0.1:  # More than 10% outside normal
                        all_normal = False
                        break
            
            if all_normal:
                inconsistencies.append("Labeled as defect but all parameters in normal ranges")
        
        # Triggers present but labeled as good
        if not defect_label and triggers:
            inconsistencies.append(f"Triggers present ({triggers}) but labeled as good")
        
        return inconsistencies
    
    # Helper methods
    def _get_max_consecutive_true(self, boolean_series: pd.Series) -> int:
        """Get maximum consecutive True values in a boolean series."""
        if not boolean_series.any():
            return 0
        
        groups = boolean_series.groupby((boolean_series != boolean_series.shift()).cumsum())
        return max(group.sum() for name, group in groups if group.iloc[0])
    
    def _get_mold_level_justification(self, max_consecutive: int, critical_threshold: int) -> str:
        """Generate justification for mold level assessment."""
        if max_consecutive > critical_threshold:
            return f"Extended mold level deviation ({max_consecutive}s) exceeds critical threshold ({critical_threshold}s), indicating severe shell formation issues"
        else:
            return f"Mold level deviation ({max_consecutive}s) within concerning but manageable range"
    
    def _get_temperature_justification(self, max_drop: float, severe_threshold: float) -> str:
        """Generate justification for temperature assessment."""
        if max_drop > severe_threshold:
            return f"Rapid temperature drop ({max_drop:.1f}°C) exceeds severe threshold ({severe_threshold}°C), indicating potential shell cracking"
        else:
            return f"Temperature drop ({max_drop:.1f}°C) concerning but within manageable range"
    
    def _get_speed_superheat_justification(self, risky_duration: int, critical_speed: float, critical_superheat: float) -> str:
        """Generate justification for speed-superheat assessment."""
        if risky_duration > 30:
            return f"Extended operation ({risky_duration}s) with high speed (>{critical_speed}) and low superheat (<{critical_superheat}°C) increases breakout risk"
        else:
            return f"Brief operation with risky speed-superheat combination, manageable if corrected quickly"
    
    def _categorize_severity(self, severity_score: float) -> str:
        """Categorize severity based on numerical score."""
        if severity_score >= 0.8:
            return "Critical"
        elif severity_score >= 0.6:
            return "High"
        elif severity_score >= 0.4:
            return "Moderate"
        elif severity_score >= 0.2:
            return "Low"
        else:
            return "Minimal"
    
    def _check_severe_conditions(self, df: pd.DataFrame) -> List[str]:
        """Check for severe conditions that should trigger defects."""
        severe_conditions = []
        
        # Very long mold level deviations
        normal_range = self.sensor_config.get('mold_level_normal_range', [130, 170])
        outside_range = (df['mold_level'] < normal_range[0]) | (df['mold_level'] > normal_range[1])
        if self._get_max_consecutive_true(outside_range) > 60:
            severe_conditions.append("Very long mold level deviation (>60s)")
        
        # Extreme temperature drops
        temp_diff = df['mold_temperature'].diff(periods=60)
        if temp_diff.min() < -80:
            severe_conditions.append("Extreme temperature drop (>80°C)")
        
        return severe_conditions
    
    def _check_trigger_severity(self, df: pd.DataFrame, triggers: List[str]) -> List[str]:
        """Check if triggers are actually mild."""
        mild_triggers = []
        
        for trigger in triggers:
            if trigger == 'prolonged_mold_level_deviation':
                normal_range = self.sensor_config.get('mold_level_normal_range', [130, 170])
                outside_range = (df['mold_level'] < normal_range[0]) | (df['mold_level'] > normal_range[1])
                max_consecutive = self._get_max_consecutive_true(outside_range)
                if max_consecutive < 40:  # Less than 40 seconds
                    mild_triggers.append(f"{trigger} (only {max_consecutive}s)")
        
        return mild_triggers
    
    def _generate_key_findings(self, label_distribution: Dict, domain_validations: List[Dict], edge_cases: List[Dict]) -> List[str]:
        """Generate key findings for expert review."""
        findings = []
        
        # Distribution findings
        defect_rate = label_distribution['dataset_summary']['defect_rate']
        target_rate = label_distribution['dataset_summary']['target_defect_rate']
        
        if abs(defect_rate - target_rate) > 0.05:
            findings.append(f"Defect rate ({defect_rate:.1%}) differs significantly from target ({target_rate:.1%})")
        
        # Domain validation findings
        failed_validations = len([dv for dv in domain_validations if not dv.get('passed', True)])
        if failed_validations > 0:
            findings.append(f"{failed_validations} casts failed domain knowledge validation")
        
        # Edge case findings
        high_uncertainty_cases = len([ec for ec in edge_cases if ec.get('uncertainty_score', 0) > 0.7])
        if high_uncertainty_cases > 0:
            findings.append(f"{high_uncertainty_cases} casts identified as high uncertainty/edge cases")
        
        return findings
    
    def _generate_overall_assessment(self, label_distribution: Dict, domain_validations: List[Dict], edge_cases: List[Dict]) -> str:
        """Generate overall assessment for expert review."""
        total_issues = 0
        
        # Count various issues
        defect_rate = label_distribution['dataset_summary']['defect_rate']
        target_rate = label_distribution['dataset_summary']['target_defect_rate']
        
        if abs(defect_rate - target_rate) > 0.05:
            total_issues += 1
        
        failed_validations = len([dv for dv in domain_validations if not dv.get('passed', True)])
        total_issues += failed_validations
        
        high_uncertainty_cases = len([ec for ec in edge_cases if ec.get('uncertainty_score', 0) > 0.7])
        total_issues += high_uncertainty_cases
        
        if total_issues == 0:
            return "Excellent: Defect labeling appears highly consistent with domain knowledge"
        elif total_issues <= 5:
            return "Good: Minor issues identified, defect labeling is generally sound"
        elif total_issues <= 15:
            return "Fair: Several issues identified, recommend review and refinement"
        else:
            return "Poor: Significant issues identified, major review required"
    
    def _summarize_domain_validations(self, domain_validations: List[Dict]) -> Dict[str, Any]:
        """Summarize domain validation results."""
        total = len(domain_validations)
        passed = len([dv for dv in domain_validations if dv.get('passed', True)])
        failed = total - passed
        
        # Categorize failure reasons
        failure_categories = defaultdict(int)
        for dv in domain_validations:
            if not dv.get('passed', True):
                for issue in dv.get('issues', []):
                    category = issue.split(':')[0] if ':' in issue else 'Other'
                    failure_categories[category] += 1
        
        return {
            'total_validations': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / total if total > 0 else 0,
            'failure_categories': dict(failure_categories)
        }
    
    def _summarize_edge_cases(self, edge_cases: List[Dict]) -> Dict[str, Any]:
        """Summarize edge case analysis results."""
        total = len(edge_cases)
        requiring_review = len([ec for ec in edge_cases if ec.get('requires_expert_review', False)])
        
        uncertainty_scores = [ec.get('uncertainty_score', 0) for ec in edge_cases]
        
        return {
            'total_cases': total,
            'requiring_expert_review': requiring_review,
            'review_rate': requiring_review / total if total > 0 else 0,
            'uncertainty_statistics': {
                'mean': np.mean(uncertainty_scores),
                'max': max(uncertainty_scores) if uncertainty_scores else 0,
                'std': np.std(uncertainty_scores)
            }
        }
    
    def _analyze_trigger_effectiveness(self, label_distribution: Dict, domain_validations: List[Dict]) -> Dict[str, Any]:
        """Analyze the effectiveness of defect triggers."""
        trigger_analysis = label_distribution.get('trigger_analysis', {})
        trigger_counts = trigger_analysis.get('individual_triggers', {})
        
        # Calculate trigger reliability based on domain validations
        trigger_reliability = {}
        for trigger_name in trigger_counts.keys():
            # Count how many times this trigger led to valid defects
            valid_occurrences = 0
            total_occurrences = 0
            
            for dv in domain_validations:
                if trigger_name in dv.get('triggers', []):
                    total_occurrences += 1
                    if dv.get('passed', True):
                        valid_occurrences += 1
            
            if total_occurrences > 0:
                trigger_reliability[trigger_name] = valid_occurrences / total_occurrences
        
        return {
            'trigger_frequency': trigger_counts,
            'trigger_reliability': trigger_reliability,
            'coverage': trigger_analysis.get('trigger_coverage', 0)
        }
    
    def _generate_expert_recommendations(self, label_distribution: Dict, domain_validations: List[Dict], edge_cases: List[Dict]) -> List[str]:
        """Generate expert recommendations based on analysis."""
        recommendations = []
        
        # Add distribution-based recommendations
        recommendations.extend(label_distribution.get('recommendations', []))
        
        # Add domain validation recommendations
        failed_validations = [dv for dv in domain_validations if not dv.get('passed', True)]
        if len(failed_validations) > len(domain_validations) * 0.1:  # More than 10% failed
            recommendations.append(
                "High rate of domain validation failures suggests need to review trigger thresholds or logic"
            )
        
        # Add edge case recommendations
        high_uncertainty_rate = len([ec for ec in edge_cases if ec.get('uncertainty_score', 0) > 0.7]) / len(edge_cases)
        if high_uncertainty_rate > 0.1:  # More than 10% high uncertainty
            recommendations.append(
                f"High uncertainty rate ({high_uncertainty_rate:.1%}) suggests need for additional validation criteria"
            )
        
        return recommendations
    
    def _generate_methodology_notes(self) -> Dict[str, str]:
        """Generate notes about the validation methodology."""
        return {
            'domain_rules_basis': "Domain rules based on steel casting industry standards and thermal/mechanical principles",
            'trigger_validation': "Triggers validated against configurable thresholds with domain knowledge checks",
            'edge_case_detection': "Edge cases identified using threshold proximity, conflicting signals, and labeling consistency",
            'uncertainty_scoring': "Uncertainty scores calculated from borderline conditions, conflicts, and inconsistencies",
            'statistical_methods': "Basic statistical comparisons used to avoid external dependencies",
            'limitations': "Analysis based on synthetic data; real-world validation with domain experts recommended"
        }