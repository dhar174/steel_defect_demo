import argparse
import asyncio
import yaml
import logging
from pathlib import Path
import sys
import glob

# Add src to python path
sys.path.append(str((Path(__file__).parent.parent / 'src').resolve()))

from src.inference.prediction_pipeline import PredictionPipeline

def setup_logging():
    """Configures basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

async def main():
    """Main function to run the inference demo."""
    parser = argparse.ArgumentParser(description="Run the real-time steel defect inference demo.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference_config.yaml",
        help="Path to the inference configuration file."
    )
    parser.add_argument(
        "--cast-file",
        type=str,
        help="Path to a single cast data file to run the simulation."
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run in benchmark mode to measure performance."
    )
    parser.add_argument(
        "--streams",
        type=int,
        default=1,
        help="Number of concurrent streams (for benchmarking)."
    )
    # Add other arguments as needed...
    
    args = parser.parse_args()
    setup_logging()

    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        sys.exit(1)

    # Determine which cast files to process
    if args.cast_file:
        cast_files = [args.cast_file]
    else:
        # Logic to get a default list of test cast files
        test_data_patterns = [
            'data/examples/*.csv',
            'data/test/*.csv',
            'data/test/*.parquet',
            'data/synthetic/*.csv',
            'data/synthetic/*.parquet'
        ]
        cast_files = []
        for pattern in test_data_patterns:
            found_files = glob.glob(pattern)
            if found_files:
                cast_files.extend(found_files)
                break
        
        # If no files found, use the sample file as fallback
        if not cast_files:
            sample_file = "data/examples/steel_defect_sample.csv"
            if Path(sample_file).exists():
                cast_files = [sample_file]
            else:
                logging.error("No test cast files found. Please specify --cast-file or ensure test data exists.")
                sys.exit(1)
        
        # For benchmark mode, multiply the files to create concurrent streams
        if args.benchmark and args.streams > 1:
            # Generate unique identifiers for each stream
            import uuid
            original_files = cast_files.copy()
            cast_files = []
            for i in range(args.streams):
                for file in original_files:
                    unique_id = uuid.uuid4().hex
                    cast_files.append(f"{file}_stream_{unique_id}")

    logging.info(f"Processing {len(cast_files)} cast files across {args.streams if args.benchmark else 1} streams")

    # Initialize and run the pipeline
    pipeline = PredictionPipeline(config, cast_files)
    
    try:
        logging.info("Starting inference pipeline. Press Ctrl+C to exit.")
        if args.benchmark:
            logging.info("Running in benchmark mode...")
            # For benchmark mode, we'll run for a fixed duration
            import time
            start_time = time.time()
            
            # Start the pipeline
            task = asyncio.create_task(pipeline.run_pipeline())
            
            # Run for a fixed duration in benchmark mode (e.g., 60 seconds)
            benchmark_duration = 60
            await asyncio.sleep(benchmark_duration)
            
            # Stop the pipeline
            pipeline.stop_pipeline()
            
            # Wait for the task to complete cleanup
            try:
                await asyncio.wait_for(task, timeout=PIPELINE_CLEANUP_TIMEOUT)
            except asyncio.TimeoutError:
                logging.warning("Pipeline cleanup took longer than expected")
            
            # Get performance metrics from the monitor
            metrics = pipeline.monitor.get_system_performance_metrics()
            end_time = time.time()
            
            # Print benchmark summary
            logging.info("=" * 50)
            logging.info("BENCHMARK SUMMARY")
            logging.info("=" * 50)
            logging.info(f"Duration: {end_time - start_time:.2f} seconds")
            logging.info(f"Total Predictions: {metrics['total_predictions']}")
            logging.info(f"Average Latency: {metrics['avg_latency_ms']:.2f} ms")
            logging.info(f"Throughput: {metrics['throughput_preds_per_sec']:.2f} predictions/sec")
            logging.info(f"High Risk Predictions: {metrics['high_risk_predictions']}")
            logging.info(f"Streams: {args.streams}")
            logging.info("=" * 50)
        else:
            # Normal mode - run until interrupted
            await pipeline.run_pipeline()
            
    except KeyboardInterrupt:
        logging.info("Shutdown signal received.")
    finally:
        logging.info("Stopping pipeline...")
        pipeline.stop_pipeline()
        logging.info("Pipeline stopped gracefully.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Application interrupted by user")
        sys.exit(0)