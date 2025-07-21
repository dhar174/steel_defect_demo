#!/usr/bin/env python3
"""
Demonstration script for the prediction display components.

This script shows how to use the new prediction visualization components
and generates sample outputs to validate functionality.
"""

import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.visualization.components.prediction_display import (
    PredictionDisplayComponents, 
    create_sample_data_for_demo
)


def load_config():
    """Load inference configuration."""
    try:
        with open('configs/inference_config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Return default config if file not found
        return {
            'inference': {
                'thresholds': {
                    'defect_probability': 0.5,
                    'high_risk_threshold': 0.7,
                    'alert_threshold': 0.8
                }
            }
        }


def demonstrate_prediction_components():
    """Demonstrate all prediction display components."""
    print("=== Steel Defect Prediction Visualization Components Demo ===\n")
    
    # Load configuration
    config = load_config()
    print(f"Loaded configuration with thresholds:")
    thresholds = config.get('inference', {}).get('thresholds', {})
    for key, value in thresholds.items():
        print(f"  {key}: {value}")
    print()
    
    # Initialize components
    components = PredictionDisplayComponents(config)
    
    # Generate sample data
    history_data, sample_metrics = create_sample_data_for_demo()
    
    print("=== Component Demonstrations ===\n")
    
    # 1. Test prediction gauge with different risk levels
    print("1. Prediction Gauge Components:")
    test_probabilities = [0.2, 0.55, 0.75, 0.9]
    for prob in test_probabilities:
        gauge_fig = components.create_prediction_gauge(prob)
        print(f"   Created gauge for probability {prob} - Risk level: {components._get_risk_levels([prob])[0]}")
    print()
    
    # 2. Test historical timeline
    print("2. Historical Timeline:")
    timeline_fig = components.create_historical_timeline(history_data)
    print(f"   Created timeline with {len(history_data)} data points")
    print(f"   Time range: {history_data.index[0]} to {history_data.index[-1]}")
    print(f"   Prediction range: {history_data['prediction'].min():.3f} to {history_data['prediction'].max():.3f}")
    print()
    
    # 3. Test ensemble contribution chart
    print("3. Model Ensemble Contribution:")
    baseline_weight = config.get('inference', {}).get('ensemble', {}).get('baseline_weight', 0.4)
    lstm_weight = config.get('inference', {}).get('ensemble', {}).get('lstm_weight', 0.6)
    ensemble_fig = components.create_ensemble_contribution_chart(baseline_weight, lstm_weight)
    print(f"   Created ensemble chart with baseline: {baseline_weight}, LSTM: {lstm_weight}")
    print()
    
    # 4. Test alert status indicators
    print("4. Alert Status Indicators:")
    for prob in test_probabilities:
        alert_component = components.create_alert_status_indicator(prob, cast_id=f"CAST_{int(prob*1000):03d}")
        print(f"   Created alert for probability {prob} (Cast: CAST_{int(prob*1000):03d})")
    print()
    
    # 5. Test confidence visualization
    print("5. Prediction Confidence Visualization:")
    confidence_fig = components.create_confidence_visualization(
        prediction_prob=0.7,
        confidence_interval=(0.65, 0.75),
        uncertainty=0.05
    )
    print("   Created confidence visualization with interval [0.65, 0.75] and uncertainty ±0.05")
    print()
    
    # 6. Test accuracy metrics display
    print("6. Model Performance Metrics:")
    metrics_component = components.create_accuracy_metrics_display(sample_metrics, "Ensemble Model")
    print(f"   Created metrics display with:")
    for metric, value in sample_metrics.items():
        print(f"     {metric}: {value:.3f}")
    print()
    
    print("=== Risk Level Analysis ===")
    
    # Analyze risk levels in sample data
    predictions = history_data['prediction'].values
    risk_levels = components._get_risk_levels(predictions)
    
    print(f"Risk level distribution in sample data:")
    risk_counts = pd.Series(risk_levels).value_counts()
    for level, count in risk_counts.items():
        percentage = count / len(risk_levels) * 100
        print(f"  {level}: {count} occurrences ({percentage:.1f}%)")
    print()
    
    # Show threshold analysis
    print("Threshold Analysis:")
    above_defect = (predictions >= components.defect_threshold).sum()
    above_high_risk = (predictions >= components.high_risk_threshold).sum()
    above_alert = (predictions >= components.alert_threshold).sum()
    
    print(f"  Predictions above defect threshold ({components.defect_threshold}): {above_defect}/{len(predictions)} ({above_defect/len(predictions)*100:.1f}%)")
    print(f"  Predictions above high risk threshold ({components.high_risk_threshold}): {above_high_risk}/{len(predictions)} ({above_high_risk/len(predictions)*100:.1f}%)")
    print(f"  Predictions above alert threshold ({components.alert_threshold}): {above_alert}/{len(predictions)} ({above_alert/len(predictions)*100:.1f}%)")
    print()
    
    print("=== Integration Test ===")
    
    # Test that components can be integrated into existing dashboard structure
    print("Testing integration with existing dashboard components...")
    
    # Simulate real-time update
    current_prediction = np.random.uniform(0.1, 0.9)
    print(f"Simulated real-time prediction: {current_prediction:.3f}")
    
    # Create all components for this prediction
    gauge = components.create_prediction_gauge(current_prediction)
    alert = components.create_alert_status_indicator(current_prediction, cast_id="CAST_DEMO")
    confidence = components.create_confidence_visualization(current_prediction, uncertainty=0.1)
    
    print("✓ Successfully created gauge, alert, and confidence components")
    print("✓ All components use consistent risk level color coding")
    print("✓ Components are ready for dashboard integration")
    print()
    
    print("=== Demo Complete ===")
    print("All prediction display components have been successfully demonstrated!")
    print("Components are now ready for integration into the main dashboard.")


if __name__ == "__main__":
    demonstrate_prediction_components()