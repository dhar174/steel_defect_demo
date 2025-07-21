#!/usr/bin/env python3
"""Run real-time inference demonstration"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from inference.inference_engine import DefectPredictionEngine
from inference.stream_simulator import RealTimeStreamSimulator
from visualization.dashboard import DefectMonitoringDashboard
from utils.config_loader import ConfigLoader
from utils.logger import get_logger
import argparse
import pandas as pd
import threading
import time

from configs.schemas.inference_config_schema import FullInferenceConfig

def main():
    """Main function to run real-time inference demo."""
    parser = argparse.ArgumentParser(description='Run real-time inference demonstration')
    parser.add_argument('--test-data',
                       default='data/synthetic/test_cast.parquet',
                       help='Path to test cast data for streaming simulation')
    parser.add_argument('--model-dir',
                       default='models',
                       help='Directory containing trained models')
    parser.add_argument('--dashboard',
                       action='store_true',
                       help='Launch dashboard along with inference')
    parser.add_argument('--duration',
                       type=int,
                       default=300,
                       help='Duration to run simulation (seconds)')
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = get_logger(__name__)
    
    if args.verbose:
        logger.info(f"Test data: {args.test_data}")
        logger.info(f"Model directory: {args.model_dir}")
        logger.info(f"Simulation duration: {args.duration} seconds")
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_config("inference_config", FullInferenceConfig)
        
        # Initialize inference engine
        logger.info("Initializing inference engine...")
        inference_engine = DefectPredictionEngine(config)
        inference_engine.load_models()
        
        # Load test cast data for simulation
        logger.info("Loading test cast data...")
        # TODO: Implement test data loading
        test_data = None  # Placeholder
        
        # Initialize stream simulator
        stream_simulator = RealTimeStreamSimulator(test_data, config)
        
        # Start dashboard if requested
        dashboard_thread = None
        if args.dashboard:
            logger.info("Starting dashboard...")
            dashboard = DefectMonitoringDashboard(config)
            dashboard_thread = threading.Thread(
                target=dashboard.run,
                kwargs={'debug': False}
            )
            dashboard_thread.daemon = True
            dashboard_thread.start()
            time.sleep(2)  # Give dashboard time to start
        
        # Start streaming simulation
        logger.info("Starting real-time inference simulation...")
        stream_simulator.start_stream()
        
        # Process stream with inference engine
        logger.info("Processing streaming data...")
        stream_simulator.process_stream(inference_engine)
        
        # Run simulation for specified duration
        start_time = time.time()
        while time.time() - start_time < args.duration:
            # Get stream status
            status = stream_simulator.get_stream_status()
            
            if args.verbose and int(time.time() - start_time) % 30 == 0:
                logger.info(f"Simulation running... Status: {status}")
            
            time.sleep(1)
        
        # Stop streaming
        logger.info("Stopping simulation...")
        stream_simulator.stop_stream()
        
        # Generate summary
        final_status = stream_simulator.get_stream_status()
        logger.info(f"Simulation completed. Final status: {final_status}")
        
        if args.dashboard:
            logger.info("Dashboard is still running. Press Ctrl+C to exit.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutting down...")
        
        logger.info("Real-time inference demo completed successfully.")
        
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()