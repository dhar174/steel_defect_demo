#!/usr/bin/env python3
"""
Demonstration script for PredictionPipeline functionality.

This script shows how to use the PredictionPipeline to orchestrate
multiple concurrent data streams for real-time steel defect prediction.
"""

import os
import sys
import asyncio
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from inference.prediction_pipeline import PredictionPipeline

def create_sample_cast_data(cast_id: int, num_points: int = 100) -> pd.DataFrame:
    """Create sample cast data for demonstration."""
    np.random.seed(42 + cast_id)  # Different seed per cast
    
    timestamps = pd.date_range('2023-01-01', periods=num_points, freq='1s')
    
    # Create realistic sensor data with some variation per cast
    base_temp = 1520 + cast_id * 10  # Slight temperature variation per cast
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': np.random.normal(base_temp, 10, num_points),
        'pressure': np.random.normal(100, 5, num_points),
        'flow_rate': np.random.normal(200, 15, num_points),
        'vibration': np.random.normal(0.5, 0.1, num_points),
        'power_consumption': np.random.normal(500, 50, num_points)
    })
    
    return data

def create_demo_config():
    """Create demonstration configuration."""
    return {
        'inference': {
            'real_time_simulation': {
                'playback_speed_multiplier': 10,  # 10x real time for demo
                'update_interval_seconds': 2,     # Predict every 2 seconds
                'buffer_size_seconds': 60         # 1 minute buffer
            },
            'ensemble': {
                'baseline_weight': 0.4,
                'lstm_weight': 0.6
            },
            'thresholds': {
                'defect_probability': 0.5,
                'high_risk_threshold': 0.7,
                'alert_threshold': 0.8
            }
        }
    }

async def run_demo():
    """Run the PredictionPipeline demonstration."""
    print("PredictionPipeline Demonstration")
    print("=" * 50)
    
    # Create temporary cast files
    temp_files = []
    num_casts = 3
    
    try:
        print(f"Creating {num_casts} sample cast data files...")
        
        for i in range(num_casts):
            # Create sample data
            cast_data = create_sample_cast_data(i, num_points=30)  # Smaller for demo
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'_cast_{i}.csv', delete=False) as f:
                cast_data.to_csv(f.name, index=False)
                temp_files.append(f.name)
                print(f"  Created cast {i}: {os.path.basename(f.name)} ({len(cast_data)} points)")
        
        # Create configuration
        config = create_demo_config()
        
        # Initialize PredictionPipeline
        print(f"\nInitializing PredictionPipeline with {len(temp_files)} streams...")
        pipeline = PredictionPipeline(config=config, cast_files=temp_files)
        
        print(f"Pipeline created:")
        print(f"  - Cast files: {len(pipeline.cast_files)}")
        print(f"  - Running: {pipeline.running}")
        print(f"  - Update interval: {config['inference']['real_time_simulation']['update_interval_seconds']}s")
        
        # Demonstrate pipeline functionality without actually running
        # (since we don't have trained models available)
        print(f"\nPipeline is ready to:")
        print(f"  - Process {len(temp_files)} concurrent data streams")
        print(f"  - Generate predictions every {config['inference']['real_time_simulation']['update_interval_seconds']} seconds")
        print(f"  - Handle errors gracefully without stopping other streams")
        print(f"  - Use asyncio for efficient concurrency")
        
        # Test stop functionality
        print(f"\nTesting graceful shutdown...")
        pipeline.stop_pipeline()
        print(f"Pipeline stopped successfully")
        
        # Show what would happen during actual execution
        print(f"\nDuring actual execution, the pipeline would:")
        print(f"  1. Start {len(temp_files)} RealTimeStreamSimulator instances")
        print(f"  2. Create {len(temp_files)} DefectPredictionEngine instances")
        print(f"  3. Run {len(temp_files)} concurrent async tasks")
        print(f"  4. Each task would:")
        print(f"     - Get data from simulator.get_data_buffer()")
        print(f"     - Run engine.predict_ensemble() in thread pool")
        print(f"     - Log prediction results and metrics")
        print(f"     - Sleep for update interval")
        print(f"  5. Handle errors without stopping other streams")
        print(f"  6. Allow graceful shutdown via stop_pipeline()")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        
    finally:
        # Clean up temporary files
        print(f"\nCleaning up {len(temp_files)} temporary files...")
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    print(f"  Removed: {os.path.basename(temp_file)}")
            except Exception as e:
                print(f"  Error removing {temp_file}: {e}")
    
    print(f"\nDemo completed successfully!")

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(run_demo())