#!/usr/bin/env python3
"""
Demonstration script for the Steel Casting Feature Engineering Pipeline.

This script shows how to use the CastingFeatureEngineer to extract comprehensive
features from time series sensor data for steel casting defect prediction.
"""

import numpy as np
import pandas as pd
import time
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from features.feature_engineer import CastingFeatureEngineer
from features.feature_validation import FeatureValidator

def create_sample_data(n_casts=5, n_points=300):
    """Create sample casting data for demonstration."""
    print(f"Creating {n_casts} sample casting sequences with {n_points} data points each...")
    
    data_dict = {}
    np.random.seed(42)
    
    for i in range(n_casts):
        cast_id = f"cast_{i+1:03d}"
        
        # Create realistic sensor patterns
        time_idx = np.arange(n_points)
        
        # Temperature: starts high, gradually decreases with some oscillation
        base_temp = 1520 + np.random.normal(0, 5)
        temp_trend = -0.5 * time_idx / n_points * 50  # Cooling trend
        temp_noise = np.random.normal(0, 8, n_points)
        temp_oscillation = 3 * np.sin(time_idx * 2 * np.pi / 50)
        temperature = base_temp + temp_trend + temp_noise + temp_oscillation
        
        # Pressure: correlated with temperature but with different dynamics
        pressure = 150 + 0.05 * (temperature - 1520) + np.random.normal(0, 3, n_points)
        
        # Flow rate: more stable with occasional spikes
        flow_rate = np.random.normal(200, 10, n_points)
        # Add some spikes
        spike_indices = np.random.choice(n_points, size=5, replace=False)
        flow_rate[spike_indices] += np.random.normal(50, 10, 5)
        
        # Vibration: lower values with some correlation to flow
        vibration = 1.0 + 0.002 * (flow_rate - 200) + np.random.normal(0, 0.1, n_points)
        vibration = np.maximum(vibration, 0.5)  # Ensure positive
        
        # Power consumption: related to temperature and flow
        power_consumption = (20 + 0.003 * (temperature - 1520) + 
                           0.02 * (flow_rate - 200) + 
                           np.random.normal(0, 2, n_points))
        
        # Create DataFrame
        data_dict[cast_id] = pd.DataFrame({
            'temperature': temperature,
            'pressure': pressure,
            'flow_rate': flow_rate,
            'vibration': vibration,
            'power_consumption': power_consumption
        })
        
        # Add some missing data to make it realistic
        if i == 2:  # Add missing data to one cast
            data_dict[cast_id].loc[50:60, 'temperature'] = np.nan
            data_dict[cast_id].loc[100:105, 'vibration'] = np.nan
    
    return data_dict

def demonstrate_feature_engineering():
    """Demonstrate the feature engineering pipeline."""
    print("=" * 60)
    print("Steel Casting Feature Engineering Pipeline Demo")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create sample data
    data_dict = create_sample_data(n_casts=10, n_points=300)
    
    # Initialize feature engineer
    print("\n1. Initializing Feature Engineer...")
    engineer = CastingFeatureEngineer(
        scaling_method='standard',
        spike_threshold=2.5,
        extreme_percentiles=(5, 95)
    )
    
    # Initialize validator
    validator = FeatureValidator()
    
    print(f"   - Configured for sensors: {engineer.sensor_columns}")
    print(f"   - Scaling method: {engineer.scaling_method}")
    print(f"   - Spike threshold: {engineer.spike_threshold}")
    
    # Test single cast processing
    print("\n2. Processing Single Cast...")
    cast_id = 'cast_001'
    single_cast_data = data_dict[cast_id]
    
    start_time = time.time()
    features = engineer.engineer_features(single_cast_data, cast_id=cast_id)
    single_time = time.time() - start_time
    
    print(f"   - Processed {len(single_cast_data)} data points in {single_time:.3f} seconds")
    print(f"   - Extracted {len(features.columns)} features")
    print(f"   - Feature categories:")
    
    # Count features by category
    stat_features = [col for col in features.columns if any(stat in col for stat in ['_mean', '_std', '_min', '_max', '_median', '_p'])]
    stab_features = [col for col in features.columns if any(metric in col for metric in ['_spike_count', '_excursion_freq', '_cv', '_range_ratio'])]
    dur_features = [col for col in features.columns if any(metric in col for metric in ['_time_extremes', '_threshold_crossings', '_consec_extremes'])]
    inter_features = [col for col in features.columns if any(metric in col for metric in ['_corr', '_ratio', '_diff', '_product', 'overall_', 'sensor_variance'])]
    temp_features = [col for col in features.columns if any(metric in col for metric in ['_trend', '_gradient'])]
    
    print(f"     * Statistical features: {len(stat_features)}")
    print(f"     * Stability features: {len(stab_features)}")
    print(f"     * Duration features: {len(dur_features)}")
    print(f"     * Interaction features: {len(inter_features)}")
    print(f"     * Temporal features: {len(temp_features)}")
    
    # Show sample features
    print(f"\n   - Sample feature values:")
    sample_features = ['temperature_mean', 'pressure_std', 'flow_rate_spike_count', 
                      'temp_pressure_corr', 'vibration_trend']
    for feat in sample_features:
        if feat in features.columns:
            value = features[feat].iloc[0]
            print(f"     * {feat}: {value:.4f}")
    
    # Validate single cast
    print("\n3. Validating Data Quality...")
    input_validation = validator.validate_input_data(single_cast_data, cast_id=cast_id)
    feature_validation = validator.validate_features(features)
    
    print(f"   - Input data valid: {input_validation['is_valid']}")
    print(f"   - Features valid: {feature_validation['is_valid']}")
    print(f"   - Feature count sufficient: {feature_validation['sufficient_features']}")
    
    # Test batch processing
    print("\n4. Processing Multiple Casts (Batch)...")
    start_time = time.time()
    all_features = engineer.engineer_features_batch(data_dict, n_jobs=2)
    batch_time = time.time() - start_time
    
    print(f"   - Processed {len(data_dict)} casts in {batch_time:.3f} seconds")
    print(f"   - Average time per cast: {batch_time/len(data_dict):.3f} seconds")
    print(f"   - Output shape: {all_features.shape}")
    
    # Feature scaling demonstration
    print("\n5. Feature Scaling...")
    print("   - Fitting scaler on training data...")
    engineer.fit_scaler(all_features)
    
    print("   - Scaling features...")
    scaled_features = engineer.scale_features(all_features)
    
    # Show scaling effect
    numeric_cols = scaled_features.select_dtypes(include=[np.number]).columns
    if 'cast_id' in numeric_cols:
        numeric_cols = numeric_cols.drop('cast_id')
    
    original_means = all_features[numeric_cols].mean()
    scaled_means = scaled_features[numeric_cols].mean()
    
    print(f"   - Original feature means (sample): {original_means.head(3).values}")
    print(f"   - Scaled feature means (sample): {scaled_means.head(3).values}")
    print(f"   - Scaling successful: {np.allclose(scaled_means, 0, atol=1e-10)}")
    
    # Correlation analysis
    print("\n6. Feature Correlation Analysis...")
    correlation_df = validator.check_feature_correlations(all_features)
    
    if not correlation_df.empty:
        print(f"   - Found {len(correlation_df)} highly correlated feature pairs")
        print(f"   - Highest correlation: {correlation_df['correlation'].abs().max():.3f}")
        print("   - Top correlated pairs:")
        top_corr = correlation_df.nlargest(3, 'correlation', keep='all')
        for _, row in top_corr.iterrows():
            print(f"     * {row['feature1']} ↔ {row['feature2']}: {row['correlation']:.3f}")
    else:
        print("   - No highly correlated features found")
    
    # Performance assessment
    print("\n7. Performance Assessment...")
    data_points_per_cast = len(single_cast_data)
    total_data_points = len(data_dict) * data_points_per_cast
    processing_rate = total_data_points / batch_time
    
    print(f"   - Total data points processed: {total_data_points:,}")
    print(f"   - Processing rate: {processing_rate:,.0f} points/second")
    
    # Estimate for 1,200 casts requirement
    estimated_time_1200 = 1200 * (batch_time / len(data_dict))
    print(f"   - Estimated time for 1,200 casts: {estimated_time_1200:.1f} seconds ({estimated_time_1200/60:.1f} minutes)")
    
    requirement_met = estimated_time_1200 < 120  # Less than 2 minutes
    print(f"   - Performance requirement met: {requirement_met}")
    
    # Generate comprehensive report
    print("\n8. Generating Validation Report...")
    report = validator.generate_validation_report(
        input_validation, feature_validation, correlation_df, cast_id
    )
    
    print(f"   - Overall validation status: {report['overall_valid']}")
    if report['recommendations']:
        print("   - Recommendations:")
        for rec in report['recommendations'][:3]:
            print(f"     * {rec}")
    
    print("\n" + "=" * 60)
    print("Feature Engineering Pipeline Demo Complete!")
    print("=" * 60)
    
    return all_features, scaled_features, report

if __name__ == "__main__":
    demonstrate_feature_engineering()