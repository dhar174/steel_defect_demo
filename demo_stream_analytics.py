#!/usr/bin/env python3
"""
Demonstration script for the StreamAnalyticsEngine.
Shows how the engine performs SPC checks, trend detection, and anomaly detection.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from analytics.stream_analytics import StreamAnalyticsEngine


def main():
    """Demonstrate StreamAnalyticsEngine functionality."""
    print("StreamAnalyticsEngine Demonstration")
    print("=" * 50)
    
    # Configuration
    config = {
        'spc_sigma_threshold': 3.0,
        'trend_min_points': 10,
        'trend_significance_level': 0.05
    }
    
    # Initialize the engine
    engine = StreamAnalyticsEngine(config)
    print(f"Initialized StreamAnalyticsEngine with config: {config}")
    print()
    
    # Generate sample data with known characteristics
    np.random.seed(42)
    
    # 1. Normal data
    print("1. Testing with normal sensor data...")
    normal_data = pd.DataFrame({
        'temperature': np.random.normal(100, 5, 50),
        'pressure': np.random.normal(200, 10, 50),
        'vibration': np.random.normal(50, 3, 50),
        'timestamp': pd.date_range('2023-01-01', periods=50, freq='1h')
    })
    
    result = engine.update_with_new_data(normal_data)
    print(f"   Total points: {result['summary']['total_points']}")
    print(f"   SPC violations: {result['summary']['spc_violations_count']}")
    print(f"   Trends detected: {result['summary']['trends_count']}")
    print(f"   Anomalies detected: {result['summary']['anomalies_count']}")
    print()
    
    # 2. Data with outliers
    print("2. Testing with data containing outliers...")
    outlier_data = normal_data.copy()
    outlier_data.loc[10, 'temperature'] = 200  # Clear outlier
    outlier_data.loc[20, 'pressure'] = 400     # Clear outlier
    
    result = engine.update_with_new_data(outlier_data)
    print(f"   Total points: {result['summary']['total_points']}")
    print(f"   SPC violations: {result['summary']['spc_violations_count']}")
    print(f"   Trends detected: {result['summary']['trends_count']}")
    print(f"   Anomalies detected: {result['summary']['anomalies_count']}")
    
    if result['spc_violations']:
        print("   SPC violations found:")
        for sensor, violation in result['spc_violations'].items():
            print(f"     {sensor}: {violation}")
    print()
    
    # 3. Data with trends
    print("3. Testing with data containing trends...")
    trend_data = normal_data.copy()
    # Add increasing trend to temperature
    trend_values = np.linspace(0, 20, 50)
    trend_data['temperature'] = trend_data['temperature'] + trend_values
    
    result = engine.update_with_new_data(trend_data)
    print(f"   Total points: {result['summary']['total_points']}")
    print(f"   SPC violations: {result['summary']['spc_violations_count']}")
    print(f"   Trends detected: {result['summary']['trends_count']}")
    print(f"   Anomalies detected: {result['summary']['anomalies_count']}")
    
    if result['trends']:
        print("   Trends found:")
        for sensor, trend in result['trends'].items():
            print(f"     {sensor}: {trend}")
    print()
    
    # 4. Complex scenario with multiple issues
    print("4. Testing complex scenario with multiple issues...")
    complex_data = trend_data.copy()
    complex_data.loc[15, 'pressure'] = 500      # Outlier
    complex_data.loc[25, 'vibration'] = 100     # Outlier
    
    result = engine.update_with_new_data(complex_data)
    print(f"   Total points: {result['summary']['total_points']}")
    print(f"   SPC violations: {result['summary']['spc_violations_count']}")
    print(f"   Trends detected: {result['summary']['trends_count']}")
    print(f"   Anomalies detected: {result['summary']['anomalies_count']}")
    
    print("\n   Detailed Results:")
    if result['spc_violations']:
        print("   SPC Violations:")
        for sensor, violation in result['spc_violations'].items():
            print(f"     {sensor}: {violation}")
    
    if result['trends']:
        print("   Trends:")
        for sensor, trend in result['trends'].items():
            print(f"     {sensor}: {trend}")
    
    if result['anomalies']:
        print(f"   Anomaly indices: {result['anomalies']}")
    
    print("\nDemonstration completed successfully!")


if __name__ == '__main__':
    main()