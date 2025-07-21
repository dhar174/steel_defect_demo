#!/usr/bin/env python3
"""
Demonstration script for sensor time series visualization capabilities.

This script shows the core visualization features for steel defect prediction:
1. Side-by-side time series plots comparing good vs defective casts
2. Multi-sensor dashboard showing all channels simultaneously  
3. Pattern recognition visualization
4. Temporal analysis of sensor evolution

Since the data generator has an issue, this uses synthetic data for demonstration.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visualization.plotting_utils import PlottingUtils

def generate_demo_data(num_casts: int = 10, cast_duration_minutes: int = 30, defect_rate: float = 0.3):
    """
    Generate synthetic steel casting data for demonstration purposes.
    
    Args:
        num_casts: Number of casts to generate
        cast_duration_minutes: Duration of each cast in minutes
        defect_rate: Proportion of casts that should be defective
        
    Returns:
        List of (cast_data, metadata) tuples
    """
    np.random.seed(42)  # For reproducible results
    
    cast_data_list = []
    
    # Sensor configurations (realistic steel casting ranges)
    sensor_configs = {
        'casting_speed': {'base': 1.2, 'range': (0.8, 1.8), 'noise': 0.05},
        'mold_temperature': {'base': 1520, 'range': (1480, 1580), 'noise': 10},
        'mold_level': {'base': 150, 'range': (120, 180), 'noise': 5},
        'cooling_water_flow': {'base': 200, 'range': (150, 250), 'noise': 15},
        'superheat': {'base': 25, 'range': (15, 40), 'noise': 3}
    }
    
    for cast_idx in range(num_casts):
        # Determine if this cast is defective
        is_defect = np.random.random() < defect_rate
        cast_id = f"cast_{cast_idx:04d}"
        
        # Create time index
        num_points = cast_duration_minutes * 60  # 1 Hz sampling
        start_time = datetime.now() + timedelta(hours=cast_idx)
        time_index = pd.date_range(start=start_time, periods=num_points, freq='1S')
        
        # Generate sensor data
        cast_data = pd.DataFrame(index=time_index)
        
        for sensor_name, config in sensor_configs.items():
            base_value = config['base']
            noise_std = config['noise']
            min_val, max_val = config['range']
            
            # Generate base signal with trend and noise
            trend = np.linspace(0, np.random.uniform(-0.1, 0.1) * base_value, num_points)
            noise = np.random.normal(0, noise_std, num_points)
            
            # Add defect patterns for defective casts
            if is_defect:
                # Add various defect signatures
                if sensor_name == 'mold_level' and np.random.random() < 0.7:
                    # Prolonged mold level deviation
                    deviation_start = num_points // 3
                    deviation_end = deviation_start + num_points // 4
                    deviation_magnitude = np.random.uniform(20, 40)
                    trend[deviation_start:deviation_end] += deviation_magnitude
                
                elif sensor_name == 'mold_temperature' and np.random.random() < 0.6:
                    # Rapid temperature drop
                    drop_start = num_points // 2
                    drop_end = drop_start + num_points // 10
                    drop_magnitude = np.random.uniform(30, 60)
                    trend[drop_start:drop_end] -= drop_magnitude
                
                elif sensor_name == 'casting_speed' and np.random.random() < 0.5:
                    # High speed with instability
                    trend += np.random.uniform(0.2, 0.4) * base_value
                    noise_std *= 2  # Increased instability
                    noise = np.random.normal(0, noise_std, num_points)
            
            # Combine signals and apply constraints
            signal = base_value + trend + noise
            signal = np.clip(signal, min_val, max_val)
            
            cast_data[sensor_name] = signal
        
        # Add cast metadata
        cast_data['cast_id'] = cast_id
        metadata = {
            'cast_id': cast_id,
            'defect_label': 1 if is_defect else 0,
            'steel_grade': np.random.choice(['A1', 'B2', 'C3']),
            'start_time': start_time.isoformat(),
            'duration_minutes': cast_duration_minutes
        }
        
        cast_data_list.append((cast_data, metadata))
    
    return cast_data_list

def demonstrate_visualizations():
    """Demonstrate the various visualization capabilities."""
    print("🔧 Steel Defect Demo: Sensor Time Series Visualization")
    print("=" * 60)
    
    # Initialize plotting utilities
    plotter = PlottingUtils()
    
    # Generate demonstration data
    print("📊 Generating synthetic demonstration data...")
    cast_data_list = generate_demo_data(num_casts=8, cast_duration_minutes=20, defect_rate=0.4)
    
    # Separate good and defect casts
    good_casts = []
    defect_casts = []
    all_metadata = []
    
    for cast_data, metadata in cast_data_list:
        all_metadata.append(metadata)
        if metadata['defect_label'] == 0:
            good_casts.append(cast_data)
        else:
            defect_casts.append(cast_data)
    
    print(f"✓ Generated {len(good_casts)} good casts and {len(defect_casts)} defect casts")
    
    # Create output directory
    os.makedirs('demo_outputs', exist_ok=True)
    
    # 1. Defect Distribution
    print("\n1️⃣ Creating defect distribution visualization...")
    labels = np.array([meta['defect_label'] for meta in all_metadata])
    defect_dist_fig = plotter.plot_defect_distribution(labels, "Cast Defect Distribution")
    defect_dist_fig.write_html('demo_outputs/defect_distribution.html')
    print("   ✓ Saved: demo_outputs/defect_distribution.html")
    
    # 2. Side-by-side comparison
    print("\n2️⃣ Creating side-by-side normal vs defect comparison...")
    if good_casts and defect_casts:
        comparison_fig = plotter.create_side_by_side_comparison(
            good_casts[:2], defect_casts[:2]  # Use first 2 of each type
        )
        comparison_fig.write_html('demo_outputs/side_by_side_comparison.html')
        print("   ✓ Saved: demo_outputs/side_by_side_comparison.html")
    
    # 3. Multi-sensor dashboard for individual casts
    print("\n3️⃣ Creating multi-sensor dashboards...")
    
    # Dashboard for a good cast
    if good_casts:
        good_dashboard = plotter.create_multi_sensor_dashboard(
            good_casts[0], all_metadata[0]
        )
        good_dashboard.write_html('demo_outputs/good_cast_dashboard.html')
        print("   ✓ Saved: demo_outputs/good_cast_dashboard.html")
    
    # Dashboard for a defect cast
    defect_cast_idx = next(i for i, meta in enumerate(all_metadata) if meta['defect_label'] == 1)
    if defect_casts:
        defect_dashboard = plotter.create_multi_sensor_dashboard(
            defect_casts[0], all_metadata[defect_cast_idx]
        )
        defect_dashboard.write_html('demo_outputs/defect_cast_dashboard.html')
        print("   ✓ Saved: demo_outputs/defect_cast_dashboard.html")
    
    # 4. Individual sensor comparisons
    print("\n4️⃣ Creating individual sensor comparison plots...")
    if good_casts and defect_casts:
        sensors = ['casting_speed', 'mold_temperature', 'mold_level', 'cooling_water_flow', 'superheat']
        
        for sensor in sensors:
            if sensor in good_casts[0].columns and sensor in defect_casts[0].columns:
                sensor_fig = plotter.plot_cast_comparison(
                    good_casts[0], defect_casts[0], sensor
                )
                filename = f'demo_outputs/{sensor}_comparison.html'
                sensor_fig.write_html(filename)
                print(f"   ✓ Saved: {filename}")
    
    # 5. Correlation heatmap
    print("\n5️⃣ Creating correlation analysis...")
    if cast_data_list:
        # Combine all cast data for correlation analysis
        all_sensor_data = []
        for cast_data, metadata in cast_data_list:
            sensor_data = cast_data.drop('cast_id', axis=1).copy()
            sensor_data['defect_label'] = metadata['defect_label']
            all_sensor_data.append(sensor_data)
        
        combined_data = pd.concat(all_sensor_data, ignore_index=True)
        correlation_fig = plotter.plot_correlation_heatmap(
            combined_data, "Sensor Correlation Matrix"
        )
        correlation_fig.write_html('demo_outputs/correlation_heatmap.html')
        print("   ✓ Saved: demo_outputs/correlation_heatmap.html")
    
    # 6. Prediction timeline simulation
    print("\n6️⃣ Creating prediction timeline simulation...")
    # Simulate prediction probabilities
    timestamps = [meta['start_time'][:19] for meta in all_metadata]  # Remove timezone info
    predictions = np.random.beta(2, 5, len(all_metadata))  # Skewed towards low probabilities
    # Make defect casts have higher prediction probabilities
    for i, meta in enumerate(all_metadata):
        if meta['defect_label'] == 1:
            predictions[i] = np.random.beta(5, 2)  # Higher probability for defects
    
    true_labels = np.array([meta['defect_label'] for meta in all_metadata])
    timeline_fig = plotter.plot_prediction_timeline(
        timestamps, predictions, true_labels, "Simulated Defect Prediction Timeline"
    )
    timeline_fig.write_html('demo_outputs/prediction_timeline.html')
    print("   ✓ Saved: demo_outputs/prediction_timeline.html")
    
    # 7. Summary statistics
    print("\n📈 Summary Statistics:")
    print(f"   • Total casts analyzed: {len(all_metadata)}")
    print(f"   • Good casts: {len(good_casts)} ({len(good_casts)/len(all_metadata)*100:.1f}%)")
    print(f"   • Defect casts: {len(defect_casts)} ({len(defect_casts)/len(all_metadata)*100:.1f}%)")
    print(f"   • Sensors monitored: {len(sensors)}")
    print(f"   • Data points per cast: ~{len(cast_data_list[0][0])} (20 minutes @ 1Hz)")
    
    print(f"\n✅ Demonstration complete! All visualizations saved to 'demo_outputs/' directory")
    print("\n🌐 Open the HTML files in a web browser to view interactive plots:")
    for filename in os.listdir('demo_outputs'):
        if filename.endswith('.html'):
            print(f"   • {filename}")

if __name__ == "__main__":
    demonstrate_visualizations()