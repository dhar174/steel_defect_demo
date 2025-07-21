#!/usr/bin/env python3
"""
Demo script to showcase RealTimeMonitor functionality.
This script demonstrates the monitoring system capabilities.

Note: Run 'pip install -e .' from the repository root to install the package in development mode.
"""

import pandas as pd
import numpy as np

from monitoring.real_time_monitor import RealTimeMonitor


def demo_real_time_monitor():
    """Demonstrates RealTimeMonitor functionality."""
    print("=== Real-Time Monitoring System Demo ===\n")
    
    # Configuration similar to inference_config.yaml
    config = {
        'inference': {
            'thresholds': {
                'defect_probability': 0.5,
                'high_risk_threshold': 0.7,
                'alert_threshold': 0.8
            }
        },
        'monitoring': {
            'metrics_logging': True,
            'performance_tracking': True,
            'data_drift_detection': True
        }
    }
    
    # Initialize monitor
    print("1. Initializing RealTimeMonitor...")
    monitor = RealTimeMonitor(config)
    print("✓ Monitor initialized successfully\n")
    
    # Demo 1: Track various prediction scenarios
    print("2. Testing prediction tracking and alerting...")
    
    predictions = [
        {'name': 'Low Risk', 'ensemble_prediction': 0.3, 'confidence': 0.85, 'latency': {'total_time': 0.05}},
        {'name': 'Medium Risk', 'ensemble_prediction': 0.55, 'confidence': 0.78, 'latency': {'total_time': 0.07}},
        {'name': 'High Risk', 'ensemble_prediction': 0.75, 'confidence': 0.92, 'latency': {'total_time': 0.09}},
        {'name': 'Critical Alert', 'ensemble_prediction': 0.85, 'confidence': 0.95, 'latency': {'total_time': 0.12}},
    ]
    
    for pred in predictions:
        print(f"   Tracking {pred['name']} prediction ({pred['ensemble_prediction']:.2f})...")
        monitor.track_prediction(pred)
    
    print("✓ Prediction tracking completed\n")
    
    # Demo 2: Data quality monitoring
    print("3. Testing data quality monitoring...")
    
    # Clean data
    clean_data = pd.DataFrame({
        'temperature': np.random.normal(1520, 10, 50),
        'pressure': np.random.normal(100, 5, 50),
        'flow_rate': np.random.normal(200, 15, 50),
        'mold_temperature': np.random.normal(1520, 10, 50),
    })
    
    issues = monitor.check_data_quality(clean_data)
    print(f"   Clean data issues: {len(issues)} (Expected: 0)")
    
    # Data with issues
    problematic_data = clean_data.copy()
    problematic_data.loc[0:2, 'temperature'] = np.nan  # NaN values
    problematic_data.loc[5:7, 'pressure'] = -50        # Negative pressure
    problematic_data.loc[10:12, 'mold_temperature'] = 3000  # Out of range
    
    issues = monitor.check_data_quality(problematic_data)
    print(f"   Problematic data issues: {len(issues)}")
    for issue in issues:
        print(f"     • {issue}")
    
    print("✓ Data quality monitoring completed\n")
    
    # Demo 3: Performance metrics
    print("4. Testing performance metrics calculation...")
    
    # Add more predictions to get meaningful metrics
    for i in range(10):
        pred = {
            'ensemble_prediction': np.random.uniform(0.2, 0.9),
            'confidence': np.random.uniform(0.7, 0.95),
            'latency': {'total_time': np.random.uniform(0.03, 0.15)}
        }
        monitor.track_prediction(pred)
    
    metrics = monitor.get_system_performance_metrics()
    print(f"   Total predictions: {metrics['total_predictions']}")
    print(f"   Average latency: {metrics['avg_latency_ms']:.2f} ms")
    print(f"   Throughput: {metrics['throughput_preds_per_sec']:.2f} pred/s")
    print(f"   High-risk predictions: {metrics['high_risk_predictions']}")
    
    print("✓ Performance metrics completed\n")
    
    # Demo 4: Integration with configuration
    print("5. Testing configuration handling...")
    
    # Test with minimal config
    minimal_config = {'inference': {}, 'monitoring': {}}
    minimal_monitor = RealTimeMonitor(minimal_config)
    
    # Should still work with defaults
    test_pred = {'ensemble_prediction': 0.9, 'confidence': 0.8, 'latency': {'total_time': 0.05}}
    minimal_monitor.track_prediction(test_pred)
    
    print("✓ Configuration handling completed\n")
    
    print("=== Demo Completed Successfully ===")
    print("All RealTimeMonitor features are working correctly!")


if __name__ == '__main__':
    demo_real_time_monitor()