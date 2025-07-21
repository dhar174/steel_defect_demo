#!/usr/bin/env python3
"""
Demonstration script for Data Quality Assessment functionality.

This script showcases the comprehensive data quality assessment capabilities
for steel casting defect detection including:
- Missing Value Analysis: Check for gaps in sensor data
- Data Consistency Checks: Verify sensor readings are within expected ranges
- Temporal Continuity: Ensure proper time sequencing in generated data
- Synthetic Data Realism: Compare generated patterns to expected steel casting behavior
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from analysis.data_quality_assessor import DataQualityAssessor
from data.data_generator import SteelCastingDataGenerator


def generate_demo_data():
    """Generate sample data for demonstration if not available."""
    data_path = Path('data')
    
    if not (data_path / 'synthetic' / 'dataset_metadata.json').exists():
        print("Generating sample dataset for demonstration...")
        
        # Generate a small dataset for demo
        generator = SteelCastingDataGenerator('configs/data_generation.yaml')
        generator.data_config['num_casts'] = 50  # Reasonable size for demo
        generator.generate_dataset()
        print("Sample dataset generated successfully!\n")
    else:
        print("Using existing dataset for demonstration.\n")


def demonstrate_single_cast_analysis():
    """Demonstrate data quality assessment for a single cast."""
    print("=" * 70)
    print("SINGLE CAST DATA QUALITY ANALYSIS")
    print("=" * 70)
    
    # Load a sample cast
    cast_file = Path('data/raw/cast_timeseries_0001.parquet')
    if not cast_file.exists():
        print("Error: No cast data found. Please run generate_demo_data() first.")
        return
    
    cast_data = pd.read_parquet(cast_file)
    print(f"Analyzing cast data: {cast_data.shape[0]} samples, {cast_data.shape[1]} sensors")
    print(f"Time range: {cast_data.index[0]} to {cast_data.index[-1]}")
    print(f"Duration: {(cast_data.index[-1] - cast_data.index[0]).total_seconds()} seconds\n")
    
    # Initialize assessor
    assessor = DataQualityAssessor(data_path='data')
    
    # Perform comprehensive assessment
    results = assessor.comprehensive_quality_assessment(cast_data)
    
    # Display results
    print_assessment_summary(results)
    print_detailed_analysis(results)
    
    return results


def demonstrate_dataset_analysis():
    """Demonstrate data quality assessment for the entire dataset."""
    print("\n" + "=" * 70)
    print("COMPLETE DATASET DATA QUALITY ANALYSIS")
    print("=" * 70)
    
    # Initialize assessor
    assessor = DataQualityAssessor(data_path='data')
    
    if not assessor.dataset_metadata:
        print("Error: No dataset metadata found. Please run generate_demo_data() first.")
        return
    
    total_casts = assessor.dataset_metadata['dataset_info']['total_casts']
    print(f"Analyzing complete dataset: {total_casts} casts")
    
    # Perform comprehensive assessment across all casts
    results = assessor.comprehensive_quality_assessment()
    
    # Display results
    print_assessment_summary(results)
    print_detailed_analysis(results)
    
    return results


def demonstrate_degraded_data_analysis():
    """Demonstrate data quality assessment with intentionally degraded data."""
    print("\n" + "=" * 70)
    print("DEGRADED DATA QUALITY ANALYSIS DEMONSTRATION")
    print("=" * 70)
    
    # Load a sample cast
    cast_file = Path('data/raw/cast_timeseries_0001.parquet')
    if not cast_file.exists():
        print("Error: No cast data found. Please run generate_demo_data() first.")
        return
    
    original_data = pd.read_parquet(cast_file)
    
    print("Creating intentionally degraded data for demonstration...")
    
    # Create degraded version with various quality issues
    degraded_data = original_data.copy()
    
    # 1. Add missing values (simulate sensor failures)
    print("- Adding missing values (simulating sensor failures)")
    degraded_data.iloc[1000:1050, degraded_data.columns.get_loc('casting_speed')] = np.nan
    degraded_data.iloc[2000:2010, degraded_data.columns.get_loc('mold_temperature')] = np.nan
    
    # 2. Add range violations (simulate sensor malfunctions)
    print("- Adding range violations (simulating sensor malfunctions)")
    degraded_data.iloc[3000, degraded_data.columns.get_loc('casting_speed')] = 10.0  # Impossible value
    degraded_data.iloc[3500, degraded_data.columns.get_loc('mold_temperature')] = 1000.0  # Too low
    degraded_data.iloc[4000, degraded_data.columns.get_loc('mold_level')] = 500.0  # Too high
    
    # 3. Add outliers (simulate measurement noise)
    print("- Adding outliers (simulating measurement noise)")
    np.random.seed(42)
    outlier_indices = np.random.choice(len(degraded_data), 20, replace=False)
    for idx in outlier_indices:
        sensor = np.random.choice(degraded_data.columns)
        # Add significant noise
        degraded_data.iloc[idx, degraded_data.columns.get_loc(sensor)] *= np.random.choice([0.5, 1.5, 2.0])
    
    # 4. Create temporal gaps (simulate data transmission issues)
    print("- Creating temporal gaps (simulating data transmission issues)")
    gap_indices = degraded_data.index[5000:5030]  # 30-second gap
    degraded_data = degraded_data.drop(gap_indices)
    
    print("\nAnalyzing degraded data...\n")
    
    # Initialize assessor
    assessor = DataQualityAssessor(data_path='data')
    
    # Analyze original data
    print("ORIGINAL DATA QUALITY:")
    print("-" * 40)
    original_results = assessor.comprehensive_quality_assessment(original_data)
    print_assessment_summary(original_results, detailed=False)
    
    # Analyze degraded data
    print("\nDEGRADED DATA QUALITY:")
    print("-" * 40)
    degraded_results = assessor.comprehensive_quality_assessment(degraded_data)
    print_assessment_summary(degraded_results, detailed=False)
    
    # Compare results
    print("\nQUALITY COMPARISON:")
    print("-" * 40)
    compare_quality_results(original_results, degraded_results)
    
    return original_results, degraded_results


def demonstrate_individual_assessments():
    """Demonstrate individual assessment components."""
    print("\n" + "=" * 70)
    print("INDIVIDUAL ASSESSMENT COMPONENTS DEMONSTRATION")
    print("=" * 70)
    
    # Load sample data
    cast_file = Path('data/raw/cast_timeseries_0001.parquet')
    if not cast_file.exists():
        print("Error: No cast data found. Please run generate_demo_data() first.")
        return
    
    cast_data = pd.read_parquet(cast_file)
    assessor = DataQualityAssessor(data_path='data')
    
    # 1. Missing Value Analysis
    print("1. MISSING VALUE ANALYSIS")
    print("-" * 40)
    missing_results = assessor.assess_missing_values(cast_data)
    analysis = missing_results['missing_value_analysis']
    
    print(f"Total missing percentage: {analysis['total_missing_percentage']:.2f}%")
    print(f"Quality score: {analysis['quality_score']:.3f}")
    print(f"Temporal gaps detected: {len(analysis['temporal_gaps'])}")
    
    print("\nMissing values by sensor:")
    for sensor, stats in analysis['missing_by_sensor'].items():
        print(f"  {sensor}: {stats['count']} values ({stats['percentage']:.2f}%)")
    
    # 2. Data Consistency Checks
    print("\n2. DATA CONSISTENCY CHECKS")
    print("-" * 40)
    consistency_results = assessor.assess_data_consistency(cast_data)
    analysis = consistency_results['consistency_analysis']
    
    print(f"Consistency score: {analysis['consistency_score']:.3f}")
    
    print("\nRange violations by sensor:")
    for sensor, violations in analysis['range_violations'].items():
        print(f"  {sensor}:")
        print(f"    Hard violations: {violations['hard_violations']} ({violations['hard_violation_percentage']:.2f}%)")
        print(f"    Value range: [{violations['value_range'][0]:.2f}, {violations['value_range'][1]:.2f}]")
        print(f"    Expected range: [{violations['expected_range'][0]}, {violations['expected_range'][1]}]")
    
    # 3. Temporal Continuity
    print("\n3. TEMPORAL CONTINUITY ANALYSIS")
    print("-" * 40)
    temporal_results = assessor.assess_temporal_continuity(cast_data)
    analysis = temporal_results['temporal_continuity']
    
    print(f"Continuity score: {analysis['continuity_score']:.3f}")
    
    sampling_analysis = analysis['sampling_rate_analysis']
    print(f"Mean interval: {sampling_analysis['mean_interval_seconds']:.3f} seconds")
    print(f"Std interval: {sampling_analysis['std_interval_seconds']:.3f} seconds")
    print(f"Irregular intervals: {sampling_analysis['irregular_percentage']:.2f}%")
    
    sequence_analysis = analysis['time_sequence_analysis']
    print(f"Monotonic time sequence: {sequence_analysis['is_monotonic_increasing']}")
    print(f"Duplicate timestamps: {sequence_analysis['has_duplicate_timestamps']}")
    print(f"Coverage: {sequence_analysis['coverage_percentage']:.2f}%")
    
    # 4. Synthetic Data Realism
    print("\n4. SYNTHETIC DATA REALISM ANALYSIS")
    print("-" * 40)
    realism_results = assessor.assess_synthetic_data_realism(cast_data)
    analysis = realism_results['realism_analysis']
    
    print(f"Realism score: {analysis['realism_score']:.3f}")
    
    print("\nDistribution statistics:")
    for sensor, stats in analysis['distribution_analysis'].items():
        print(f"  {sensor}:")
        print(f"    Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
        print(f"    Skewness: {stats['skewness']:.3f}, Kurtosis: {stats['kurtosis']:.3f}")
    
    correlations = analysis['correlation_analysis']['strong_correlations']
    print(f"\nStrong correlations found: {len(correlations)}")
    for corr in correlations[:3]:  # Show first 3
        print(f"  {corr['sensor_pair'][0]} ↔ {corr['sensor_pair'][1]}: {corr['correlation']:.3f}")
    
    # Process behavior
    behavior = analysis['process_behavior_analysis']
    if 'temperature_stability' in behavior:
        temp_stable = behavior['temperature_stability']['is_stable']
        print(f"\nTemperature stability: {'Good' if temp_stable else 'Poor'}")
    
    if 'speed_consistency' in behavior:
        speed_consistent = behavior['speed_consistency']['is_consistent']
        print(f"Speed consistency: {'Good' if speed_consistent else 'Poor'}")
    
    if 'mold_level_control' in behavior:
        level_controlled = behavior['mold_level_control']['is_well_controlled']
        excursions = behavior['mold_level_control']['excursions']
        print(f"Mold level control: {'Good' if level_controlled else 'Poor'} ({excursions} excursions)")


def print_assessment_summary(results, detailed=True):
    """Print a summary of assessment results."""
    summary = results['summary']
    
    print(f"OVERALL QUALITY ASSESSMENT")
    print(f"{'=' * 40}")
    print(f"Overall Quality Score: {summary['overall_quality_score']:.3f}")
    print(f"Quality Level: {summary['quality_level']}")
    print(f"Assessment Timestamp: {summary['assessment_timestamp']}")
    print(f"Data Scope: {summary['data_scope']}")
    
    print(f"\nComponent Scores:")
    component_scores = summary['component_scores']
    print(f"  Missing Values:      {component_scores['missing_values']:.3f}")
    print(f"  Data Consistency:    {component_scores['consistency']:.3f}")
    print(f"  Temporal Continuity: {component_scores['temporal_continuity']:.3f}")
    print(f"  Data Realism:        {component_scores['realism']:.3f}")
    
    if not detailed:
        return
    
    # Quality interpretation
    overall_score = summary['overall_quality_score']
    print(f"\nQuality Interpretation:")
    if overall_score >= 0.9:
        print("  ✓ Excellent - Data meets all quality standards")
    elif overall_score >= 0.8:
        print("  ✓ Good - Data is suitable for analysis with minor issues")
    elif overall_score >= 0.7:
        print("  ⚠ Acceptable - Data has some quality issues that should be addressed")
    elif overall_score >= 0.6:
        print("  ⚠ Poor - Data has significant quality issues")
    else:
        print("  ✗ Unacceptable - Data quality is too poor for reliable analysis")


def print_detailed_analysis(results):
    """Print detailed analysis results."""
    print(f"\nDETAILED ANALYSIS RESULTS")
    print(f"{'=' * 40}")
    
    # Missing values analysis
    if 'missing_value_analysis' in results:
        analysis = results['missing_value_analysis']
        print(f"\nMissing Values Analysis:")
        print(f"  Total missing: {analysis['total_missing_percentage']:.2f}%")
        print(f"  Temporal gaps: {len(analysis.get('temporal_gaps', []))}")
        
        if 'casts_analyzed' in analysis:
            print(f"  Casts analyzed: {analysis['casts_analyzed']}")
    
    # Consistency analysis
    if 'consistency_analysis' in results:
        analysis = results['consistency_analysis']
        print(f"\nConsistency Analysis:")
        
        if 'range_violations' in analysis:
            total_hard_violations = sum(v.get('hard_violations', 0) for v in analysis['range_violations'].values())
            print(f"  Hard range violations: {total_hard_violations}")
        
        if 'physics_violations' in analysis:
            total_physics_violations = sum(v.get('violations', 0) for v in analysis['physics_violations'].values())
            print(f"  Physics violations: {total_physics_violations}")
        
        if 'casts_analyzed' in analysis:
            print(f"  Casts analyzed: {analysis['casts_analyzed']}")
    
    # Temporal continuity analysis
    if 'temporal_continuity' in results:
        analysis = results['temporal_continuity']
        print(f"\nTemporal Continuity Analysis:")
        
        if 'sampling_rate_analysis' in analysis:
            sampling = analysis['sampling_rate_analysis']
            if 'irregular_percentage' in sampling:
                print(f"  Irregular intervals: {sampling['irregular_percentage']:.2f}%")
        
        if 'time_sequence_analysis' in analysis:
            sequence = analysis['time_sequence_analysis']
            if 'percentage_problematic_casts' in sequence:
                print(f"  Problematic casts: {sequence['percentage_problematic_casts']:.2f}%")
        
        if 'casts_analyzed' in analysis:
            print(f"  Casts analyzed: {analysis['casts_analyzed']}")
    
    # Realism analysis
    if 'realism_analysis' in results:
        analysis = results['realism_analysis']
        print(f"\nRealism Analysis:")
        
        if 'correlation_analysis' in analysis:
            if 'average_strong_correlations' in analysis['correlation_analysis']:
                avg_corr = analysis['correlation_analysis']['average_strong_correlations']
                print(f"  Average strong correlations: {avg_corr:.1f}")
            elif 'strong_correlations' in analysis['correlation_analysis']:
                strong_corr = len(analysis['correlation_analysis']['strong_correlations'])
                print(f"  Strong correlations found: {strong_corr}")
        
        if 'casts_analyzed' in analysis:
            print(f"  Casts analyzed: {analysis['casts_analyzed']}")


def compare_quality_results(original_results, degraded_results):
    """Compare quality results between original and degraded data."""
    original_score = original_results['summary']['overall_quality_score']
    degraded_score = degraded_results['summary']['overall_quality_score']
    
    print(f"Overall Quality Score Change: {original_score:.3f} → {degraded_score:.3f}")
    print(f"Quality Degradation: {((original_score - degraded_score) / original_score * 100):.1f}%")
    
    print(f"\nComponent Score Changes:")
    original_components = original_results['summary']['component_scores']
    degraded_components = degraded_results['summary']['component_scores']
    
    for component in original_components:
        original_val = original_components[component]
        degraded_val = degraded_components[component]
        change = degraded_val - original_val
        print(f"  {component.replace('_', ' ').title()}: {original_val:.3f} → {degraded_val:.3f} ({change:+.3f})")


def generate_and_save_reports(results_list):
    """Generate and save comprehensive reports."""
    print("\n" + "=" * 70)
    print("GENERATING QUALITY ASSESSMENT REPORTS")
    print("=" * 70)
    
    assessor = DataQualityAssessor(data_path='data')
    
    for i, (description, results) in enumerate(results_list):
        if results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = f"data_quality_report_{description.lower().replace(' ', '_')}_{timestamp}.json"
            
            saved_path = assessor.generate_quality_report(results, report_path)
            print(f"✓ {description} report saved: {saved_path}")


def main():
    """Main demonstration function."""
    print("=" * 70)
    print("STEEL DEFECT DETECTION - DATA QUALITY ASSESSMENT DEMO")
    print("=" * 70)
    print()
    print("This demonstration showcases comprehensive data quality assessment")
    print("capabilities for steel casting sensor data including:")
    print("- Missing Value Analysis")
    print("- Data Consistency Checks") 
    print("- Temporal Continuity Validation")
    print("- Synthetic Data Realism Assessment")
    print()
    
    # Generate sample data if needed
    generate_demo_data()
    
    # Store results for report generation
    all_results = []
    
    # 1. Single cast analysis
    print("Starting demonstrations...")
    single_cast_results = demonstrate_single_cast_analysis()
    if single_cast_results:
        all_results.append(("Single Cast Analysis", single_cast_results))
    
    # 2. Complete dataset analysis
    dataset_results = demonstrate_dataset_analysis()
    if dataset_results:
        all_results.append(("Complete Dataset Analysis", dataset_results))
    
    # 3. Degraded data analysis
    original_results, degraded_results = demonstrate_degraded_data_analysis()
    if original_results and degraded_results:
        all_results.append(("Original Data Analysis", original_results))
        all_results.append(("Degraded Data Analysis", degraded_results))
    
    # 4. Individual assessment components
    demonstrate_individual_assessments()
    
    # 5. Generate comprehensive reports
    generate_and_save_reports(all_results)
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    print("\nKey Capabilities Demonstrated:")
    print("✓ Missing value detection and temporal gap analysis")
    print("✓ Range violation and physics constraint validation")
    print("✓ Temporal continuity and sampling rate analysis")
    print("✓ Data realism assessment with distribution and correlation analysis")
    print("✓ Comprehensive quality scoring and reporting")
    print("✓ Comparative analysis of original vs degraded data")
    print("✓ Automated report generation in JSON format")
    
    print(f"\nFiles generated:")
    for description, results in all_results:
        if results:
            print(f"  - {description.lower().replace(' ', '_')}_report_*.json")
    
    print("\nThe data quality assessment system is ready for production use!")
    print("It can be integrated into data pipelines for continuous monitoring")
    print("and quality assurance of steel casting sensor data.")


if __name__ == "__main__":
    main()