"""
Simple example showing how to use the Statistical Distribution Analysis module.

This example demonstrates the basic API for analyzing steel casting sensor data.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from analysis.statistical_analyzer import StatisticalAnalyzer
from visualization.plotting_utils import PlottingUtils


def basic_analysis_example():
    """Basic example of statistical analysis usage."""
    
    # 1. Initialize the analyzer
    analyzer = StatisticalAnalyzer(data_path='data')
    
    # 2. Load cast data
    features_df, metadata_df = analyzer.load_cast_data()
    print(f"Loaded {len(features_df)} casts with {len(features_df.columns)} features")
    
    # 3. Analyze sensor distributions
    distributions = analyzer.analyze_sensor_distributions(features_df)
    print(f"Analyzed distributions for {len(distributions)} sensors")
    
    # 4. Compare good vs defect classes
    stratification = analyzer.analyze_defect_stratification(features_df)
    print(f"Compared {stratification['sample_sizes']['good']} good vs "
          f"{stratification['sample_sizes']['defect']} defect casts")
    
    # 5. Detect outliers
    outliers = analyzer.detect_outliers(features_df, method='iqr')
    total_outliers = sum(
        result['outlier_count']
        for sensor_results in outliers['sensors'].values()
        for result in sensor_results.values()
    )
    print(f"Detected {total_outliers} outliers using IQR method")
    
    # 6. Perform statistical tests
    ks_tests = analyzer.perform_ks_tests(features_df)
    significant_features = []
    for sensor_results in ks_tests['sensors'].values():
        for feature, results in sensor_results.items():
            if results['is_significant']:
                significant_features.append(feature)
    
    print(f"Found {len(significant_features)} features significantly different from normal")
    
    # 7. Generate comprehensive report
    report = analyzer.generate_summary_report(features_df)
    print(f"Generated complete report with {len(report)} sections")
    
    return features_df, analyzer


def basic_visualization_example(features_df, analyzer):
    """Basic example of visualization usage."""
    
    plotter = PlottingUtils()
    
    # 1. Plot defect distribution
    labels = features_df['defect_label'].values
    defect_fig = plotter.plot_defect_distribution(labels, title="Steel Cast Quality Distribution")
    
    # 2. Plot sensor histograms for first sensor
    sensor_names = analyzer._extract_sensor_names(features_df)
    first_sensor = sensor_names[0]
    histogram_fig = plotter.plot_sensor_histograms(
        features_df, 
        first_sensor,
        title=f"{first_sensor.title()} Distribution by Class"
    )
    
    # 3. Plot box plots for all sensors
    boxplot_fig = plotter.plot_sensor_boxplots(
        features_df, 
        sensor_names,
        title="Sensor Distributions by Class"
    )
    
    # 4. Plot correlation heatmap
    numeric_features = features_df.select_dtypes(include=['float64', 'int64'])
    correlation_fig = plotter.plot_correlation_heatmap(
        numeric_features,
        title="Sensor Feature Correlations"
    )
    
    print("Created 4 visualization plots")
    print("- Defect distribution bar chart")
    print("- Sensor histogram by class")
    print("- Multi-sensor box plots")
    print("- Feature correlation heatmap")


if __name__ == "__main__":
    print("Steel Defect Detection - Statistical Analysis Example")
    print("=" * 55)
    
    # Run basic analysis
    print("\nRunning basic statistical analysis...")
    features_df, analyzer = basic_analysis_example()
    
    # Run basic visualization  
    print("\nCreating basic visualizations...")
    basic_visualization_example(features_df, analyzer)
    
    print("\n✓ Example completed successfully!")
    print("\nFor a more comprehensive demonstration, run:")
    print("  python demo_statistical_analysis.py")