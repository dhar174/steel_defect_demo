#!/usr/bin/env python3
"""
Demonstration script for Statistical Distribution Analysis functionality.

This script showcases the comprehensive statistical analysis capabilities
for steel casting defect detection including:
- Sensor value distributions
- Defect class stratification 
- Outlier detection
- Kolmogorov-Smirnov tests
"""

import sys
import os
from pathlib import Path
import json

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from analysis.statistical_analyzer import StatisticalAnalyzer
from visualization.plotting_utils import PlottingUtils
from data.data_generator import SteelCastingDataGenerator


def generate_demo_data():
    """Generate sample data for demonstration if not available."""
    data_path = Path('data')
    
    if not (data_path / 'synthetic' / 'dataset_metadata.json').exists():
        print("Generating sample dataset for demonstration...")
        
        # Generate a small dataset for demo
        generator = SteelCastingDataGenerator('configs/data_generation.yaml')
        generator.data_config['num_casts'] = 100  # Reasonable size for demo
        generator.generate_dataset()
        print("Sample dataset generated successfully!\n")
    else:
        print("Using existing dataset for demonstration.\n")


def demonstrate_statistical_analysis():
    """Demonstrate the statistical analysis capabilities."""
    print("=" * 70)
    print("STEEL DEFECT DETECTION - STATISTICAL DISTRIBUTION ANALYSIS DEMO")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = StatisticalAnalyzer(data_path='data')
    
    # Load data
    print("Loading cast data...")
    features_df, metadata_df = analyzer.load_cast_data()
    
    print(f"Dataset Overview:")
    print(f"  - Total casts: {len(features_df)}")
    print(f"  - Defect rate: {(features_df['defect_label'] == 1).mean():.1%}")
    print(f"  - Sensors analyzed: {len(analyzer._extract_sensor_names(features_df))}")
    print()
    
    # 1. Sensor Distribution Analysis
    print("1. SENSOR DISTRIBUTION ANALYSIS")
    print("-" * 40)
    
    dist_results = analyzer.analyze_sensor_distributions(features_df)
    
    for sensor_name, sensor_stats in dist_results.items():
        print(f"\n{sensor_name.upper()} Sensor:")
        
        # Show stats for mean feature
        mean_feature = f"{sensor_name}_mean"
        if mean_feature in sensor_stats:
            stats = sensor_stats[mean_feature]
            print(f"  Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"  Skewness: {stats['skewness']:.3f}")
            print(f"  Normal distribution: {stats['normality_test']['is_normal']}")
    
    print()
    
    # 2. Defect Class Stratification
    print("2. DEFECT CLASS STRATIFICATION ANALYSIS")
    print("-" * 45)
    
    stratification_results = analyzer.analyze_defect_stratification(features_df)
    
    print(f"Sample sizes:")
    print(f"  - Good casts: {stratification_results['sample_sizes']['good']}")
    print(f"  - Defect casts: {stratification_results['sample_sizes']['defect']}")
    
    print(f"\nStatistical Comparisons:")
    for sensor_name, sensor_results in stratification_results['sensors'].items():
        mean_feature = f"{sensor_name}_mean"
        if mean_feature in sensor_results:
            comparison = sensor_results[mean_feature]
            good_mean = comparison['good_stats']['mean']
            defect_mean = comparison['defect_stats']['mean']
            
            # Check for significant difference
            ks_test = comparison['statistical_tests'].get('kolmogorov_smirnov', {})
            is_significant = ks_test.get('is_significant', False)
            p_value = ks_test.get('p_value', 1.0)
            
            print(f"  {sensor_name}:")
            print(f"    Good: {good_mean:.3f}, Defect: {defect_mean:.3f}")
            print(f"    Significant difference: {is_significant} (p={p_value:.4f})")
    
    print()
    
    # 3. Outlier Detection
    print("3. OUTLIER DETECTION")
    print("-" * 25)
    
    for method in ['iqr', 'zscore', 'modified_zscore']:
        outlier_results = analyzer.detect_outliers(features_df, method=method)
        
        total_outliers = 0
        for sensor_results in outlier_results['sensors'].values():
            for feature_results in sensor_results.values():
                total_outliers += feature_results['outlier_count']
        
        print(f"{method.upper()} method: {total_outliers} total outliers detected")
    
    # Show detailed outliers for IQR method
    iqr_results = analyzer.detect_outliers(features_df, method='iqr')
    print(f"\nDetailed IQR Outlier Analysis:")
    for sensor_name, sensor_outliers in iqr_results['sensors'].items():
        mean_feature = f"{sensor_name}_mean"
        if mean_feature in sensor_outliers:
            outlier_info = sensor_outliers[mean_feature]
            count = outlier_info['outlier_count']
            percentage = outlier_info['outlier_percentage']
            print(f"  {sensor_name}: {count} outliers ({percentage:.1f}%)")
    
    print()
    
    # 4. Kolmogorov-Smirnov Tests
    print("4. KOLMOGOROV-SMIRNOV DISTRIBUTION TESTS")
    print("-" * 45)
    
    ks_results = analyzer.perform_ks_tests(features_df, reference_distribution='normal')
    
    print(f"Testing against normal distribution:")
    significant_features = []
    
    for sensor_name, sensor_results in ks_results['sensors'].items():
        for feature_name, test_results in sensor_results.items():
            p_value = test_results['p_value']
            is_significant = test_results['is_significant']
            
            if is_significant:
                significant_features.append((feature_name, p_value))
    
    print(f"Features significantly different from normal distribution:")
    if significant_features:
        for feature, p_val in sorted(significant_features, key=lambda x: x[1]):
            print(f"  {feature}: p = {p_val:.4f}")
    else:
        print("  None detected (all features appear normally distributed)")
    
    print()
    
    # 5. Generate Complete Report
    print("5. COMPREHENSIVE ANALYSIS REPORT")
    print("-" * 40)
    
    report = analyzer.generate_summary_report(features_df)
    
    print("Report generated with the following sections:")
    for section in report.keys():
        print(f"  - {section}")
    
    # Save report to file
    report_path = Path('statistical_analysis_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nComplete report saved to: {report_path}")
    
    return features_df, analyzer


def demonstrate_visualizations(features_df, analyzer):
    """Demonstrate the visualization capabilities."""
    print("\n" + "=" * 70)
    print("VISUALIZATION DEMONSTRATIONS")
    print("=" * 70)
    
    plotter = PlottingUtils()
    
    # 1. Defect Distribution
    print("1. Creating defect distribution plot...")
    labels = features_df['defect_label'].values
    fig1 = plotter.plot_defect_distribution(labels)
    print(f"   Created bar chart showing {len(labels)} casts")
    
    # 2. Sensor Histograms
    print("2. Creating sensor histograms...")
    sensor_names = analyzer._extract_sensor_names(features_df)
    
    for sensor in sensor_names[:2]:  # Demo first 2 sensors
        fig = plotter.plot_sensor_histograms(features_df, sensor)
        print(f"   Created histogram for {sensor}")
    
    # 3. Box Plots
    print("3. Creating sensor box plots...")
    fig3 = plotter.plot_sensor_boxplots(features_df, sensor_names)
    print(f"   Created box plots for {len(sensor_names)} sensors")
    
    # 4. Correlation Heatmap
    print("4. Creating correlation heatmap...")
    numeric_features = features_df.select_dtypes(include=['float64', 'int64'])
    fig4 = plotter.plot_correlation_heatmap(numeric_features)
    print(f"   Created heatmap for {len(numeric_features.columns)} features")
    
    # 5. Outlier Detection Plot
    print("5. Creating outlier detection plots...")
    outlier_results = analyzer.detect_outliers(features_df, method='iqr')
    
    for sensor in sensor_names[:1]:  # Demo first sensor
        fig5 = plotter.plot_outlier_detection(features_df, outlier_results, sensor)
        print(f"   Created outlier plot for {sensor}")
    
    # 6. KS Test Results
    print("6. Creating KS test results plot...")
    ks_results = analyzer.perform_ks_tests(features_df)
    fig6 = plotter.plot_ks_test_results(ks_results)
    print(f"   Created KS test results visualization")
    
    print("\nAll visualizations created successfully!")
    print("Note: In a real application, these would be displayed or saved as images.")


def main():
    """Main demonstration function."""
    try:
        # Generate data if needed
        generate_demo_data()
        
        # Demonstrate statistical analysis
        features_df, analyzer = demonstrate_statistical_analysis()
        
        # Demonstrate visualizations
        demonstrate_visualizations(features_df, analyzer)
        
        print("\n" + "=" * 70)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nKey Capabilities Demonstrated:")
        print("✓ Sensor value distribution analysis")
        print("✓ Defect class stratification")  
        print("✓ Multiple outlier detection methods")
        print("✓ Kolmogorov-Smirnov statistical tests")
        print("✓ Comprehensive visualization suite")
        print("✓ Automated report generation")
        
        print(f"\nFiles generated:")
        print(f"  - statistical_analysis_report.json")
        print(f"  - data/synthetic/dataset_metadata.json")
        print(f"  - data/raw/ (cast time series files)")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())