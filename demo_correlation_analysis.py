#!/usr/bin/env python3
"""
Quick demo script to showcase correlation analysis functionality.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data.data_generator import SteelCastingDataGenerator
from src.features.correlation_analyzer import SensorCorrelationAnalyzer
from src.visualization.plotting_utils import PlottingUtils


def main():
    print("🔍 Steel Casting Sensor Correlation Analysis Demo")
    print("=" * 50)
    
    # Initialize components
    generator = SteelCastingDataGenerator('configs/data_generation.yaml')
    analyzer = SensorCorrelationAnalyzer()
    plotter = PlottingUtils()
    
    # Generate sample data with both good and defective casts
    print("\n📊 Generating sample casting data...")
    cast_data_list = []
    
    for i in range(8):
        cast_id = f"demo_cast_{i+1:03d}"
        time_series, metadata = generator.generate_cast_sequence(cast_id)
        cast_data_list.append((time_series, metadata))
        
        status = "🔴 DEFECTIVE" if metadata['defect_label'] else "🟢 GOOD"
        triggers = ", ".join(metadata['defect_trigger_events']) if metadata['defect_trigger_events'] else "None"
        print(f"  {cast_id}: {status} - Triggers: {triggers}")
    
    # Basic statistics
    good_count = sum(1 for _, metadata in cast_data_list if metadata['defect_label'] == 0)
    defect_count = len(cast_data_list) - good_count
    print(f"\n📈 Dataset Summary: {good_count} good casts, {defect_count} defective casts")
    
    # 1. Cross-sensor correlations
    print("\n🔗 1. Cross-Sensor Correlation Analysis")
    sample_data = cast_data_list[0][0]
    corr_matrix = analyzer.compute_cross_sensor_correlations(sample_data)
    
    print("Correlation Matrix:")
    print(corr_matrix.round(3))
    
    # 2. Defect-specific correlations
    print("\n🎯 2. Defect-Specific Correlation Analysis")
    defect_analysis = analyzer.compute_defect_specific_correlations(cast_data_list)
    
    if 'difference' in defect_analysis:
        diff_matrix = defect_analysis['difference']
        print("Correlation Differences (Defective - Good):")
        print(diff_matrix.round(3))
        
        # Find most significant difference
        abs_diff = diff_matrix.abs()
        max_diff_idx = abs_diff.values.argmax()
        max_row, max_col = divmod(max_diff_idx, abs_diff.shape[1])
        max_sensors = (abs_diff.index[max_row], abs_diff.columns[max_col])
        max_value = diff_matrix.loc[max_sensors[0], max_sensors[1]]
        
        print(f"\nLargest difference: {max_sensors[0]} ↔ {max_sensors[1]}: {max_value:.3f}")
    
    # 3. Time-lagged correlations
    print("\n⏰ 3. Time-Lagged Correlation Analysis")
    lagged_corr = analyzer.compute_time_lagged_correlations(
        sample_data, 
        max_lag=30, 
        target_sensor='mold_temperature'
    )
    
    print(f"Analyzing lags for mold_temperature with {len(lagged_corr)} other sensors:")
    for pair, lag_data in lagged_corr.items():
        max_corr_idx = lag_data['correlation'].abs().idxmax()
        max_lag = lag_data.loc[max_corr_idx, 'lag']
        max_corr = lag_data.loc[max_corr_idx, 'correlation']
        sensor_name = pair.replace('_mold_temperature', '').replace('mold_temperature_', '')
        print(f"  {sensor_name}: Peak correlation {max_corr:.3f} at {max_lag}s lag")
    
    # 4. Predictive features
    print("\n🧠 4. Predictive Feature Analysis")
    importance_df = analyzer.identify_predictive_sensor_combinations(cast_data_list, top_k=8)
    
    print("Top predictive features:")
    for _, row in importance_df.head(5).iterrows():
        feature_type = "📊 Statistical" if any(x in row['feature'] for x in ['_mean', '_std', '_min', '_max']) else "🔗 Correlation"
        print(f"  {feature_type}: {row['feature']} (importance: {row['importance']:.4f})")
    
    # 5. Generate visualizations
    print("\n🎨 5. Generating Visualizations")
    
    # Create output directory
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # Correlation heatmap
    fig_heatmap = plotter.plot_correlation_heatmap(
        sample_data, 
        title="Cross-Sensor Correlations"
    )
    plotter.save_plot(fig_heatmap, str(output_dir / "correlation_heatmap.html"), format="html")
    print("  ✅ Saved correlation heatmap")
    
    # Defect comparison (if available)
    if 'good_casts' in defect_analysis and 'defect_casts' in defect_analysis:
        fig_comparison = plotter.plot_defect_correlation_comparison(
            defect_analysis['good_casts'],
            defect_analysis['defect_casts'],
            defect_analysis.get('difference')
        )
        plotter.save_plot(fig_comparison, str(output_dir / "defect_comparison.html"), format="html")
        print("  ✅ Saved defect correlation comparison")
    
    # Time-lagged correlations
    fig_lagged = plotter.plot_time_lagged_correlations(lagged_corr)
    plotter.save_plot(fig_lagged, str(output_dir / "time_lagged_correlations.html"), format="html")
    print("  ✅ Saved time-lagged correlation analysis")
    
    # Feature importance
    fig_importance = plotter.plot_feature_importance_ranking(importance_df)
    plotter.save_plot(fig_importance, str(output_dir / "feature_importance.html"), format="html")
    print("  ✅ Saved feature importance ranking")
    
    # Export comprehensive analysis
    analyzer.export_correlation_analysis(cast_data_list, str(output_dir / "full_analysis.json"))
    print("  ✅ Saved comprehensive analysis data")
    
    print(f"\n🎉 Demo Complete! Check the '{output_dir}' directory for results.")
    print("\nKey insights from correlation analysis:")
    print("• Cross-sensor relationships reveal process interdependencies")
    print("• Defect-specific patterns help identify failure precursors")
    print("• Time-lagged correlations show causal relationships")
    print("• Feature importance guides sensor monitoring priorities")


if __name__ == "__main__":
    main()