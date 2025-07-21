import asyncio
import logging
from typing import List, Dict
import pandas as pd
from src.inference.inference_engine import DefectPredictionEngine
from src.inference.stream_simulator import RealTimeStreamSimulator

class PredictionPipeline:
    """Orchestrates multiple real-time prediction streams."""
    
    def __init__(self, config: Dict, cast_files: List[str]):
        """
        Initializes the prediction pipeline orchestrator.
        
        Args:
            config (Dict): The main configuration dictionary.
            cast_files (List[str]): A list of file paths to the historical cast data.
        """
        self.config = config
        self.cast_files = cast_files
        self.streams = []
        self.tasks = []
        self.running = False
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
    async def _run_single_stream(self, stream_id: int, simulator: RealTimeStreamSimulator, engine: DefectPredictionEngine):
        """
        The core asynchronous loop for processing a single data stream.
        
        Args:
            stream_id (int): Unique identifier for this stream
            simulator (RealTimeStreamSimulator): Stream simulator instance
            engine (DefectPredictionEngine): Prediction engine instance
        """
        self.logger.info(f"Starting stream {stream_id}")
        
        # Get update interval from config
        update_interval = self.config.get('inference', {}).get('real_time_simulation', {}).get('update_interval_seconds', 30)
        
        while self.running:
            try:
                # Get the latest data window from the simulator
                data_buffer = simulator.get_data_buffer()
                
                if data_buffer is not None and not data_buffer.empty:
                    # Run prediction in thread pool to avoid blocking the event loop
                    prediction_result = await asyncio.to_thread(engine.predict_ensemble, data_buffer)
                    
                    # Log prediction results
                    ensemble_score = prediction_result.get('ensemble_prediction', 0.0)
                    confidence = prediction_result.get('confidence', 0.0)
                    latency = prediction_result.get('latency', {}).get('total_time', 0.0)
                    
                    self.logger.info(f"Stream {stream_id} - Prediction: {ensemble_score:.4f}, "
                                   f"Confidence: {confidence:.4f}, Latency: {latency:.4f}s")
                    
                    # Log individual model predictions for debugging
                    baseline_pred = prediction_result.get('baseline_prediction', 0.0)
                    lstm_pred = prediction_result.get('lstm_prediction', 0.0)
                    self.logger.debug(f"Stream {stream_id} - Baseline: {baseline_pred:.4f}, "
                                    f"LSTM: {lstm_pred:.4f}")
                else:
                    self.logger.debug(f"Stream {stream_id} - No data available in buffer")
                
                # Wait for the next prediction cycle
                await asyncio.sleep(update_interval)
                
            except pd.errors.EmptyDataError as e:
                # Handle empty data buffer error
                self.logger.error(f"Stream {stream_id} - Empty data buffer: {e}", exc_info=True)
            except asyncio.TimeoutError as e:
                # Handle timeout errors
                self.logger.error(f"Stream {stream_id} - Timeout error: {e}", exc_info=True)
            except RuntimeError as e:
                # Handle runtime errors specific to the prediction engine or simulator
                self.logger.error(f"Stream {stream_id} - Runtime error: {e}", exc_info=True)
            except Exception as e:
                # Log unexpected errors but don't crash the entire pipeline
                self.logger.error(f"Stream {stream_id} - Unexpected error: {e}", exc_info=True)
                
                # Use a fallback prediction strategy
                try:
                    fallback_result = {
                        'ensemble_prediction': 0.0,
                        'confidence': 0.0,
                        'latency': {'total_time': 0.0},
                        'error': str(e)
                    }
                    self.logger.info(f"Stream {stream_id} - Fallback prediction: {fallback_result['ensemble_prediction']:.4f}")
                except Exception as fallback_error:
                    self.logger.error(f"Fallback prediction failed for stream {stream_id}: {fallback_error}")
                
                # Continue with the next cycle after a brief delay
                await asyncio.sleep(update_interval)
        
        self.logger.info(f"Stream {stream_id} stopped")
        
    async def run_pipeline(self):
        """
        Initializes and runs the prediction pipeline for all configured streams.
        """
        if self.running:
            self.logger.warning("Pipeline is already running")
            return
        
        self.logger.info(f"Starting prediction pipeline with {len(self.cast_files)} streams")
        
        # Set running flag
        self.running = True
        
        try:
            # Clear any previous streams and tasks
            self.streams = []
            self.tasks = []
            
            # Create stream instances for each cast file
            for i, cast_file in enumerate(self.cast_files):
                try:
                    # Load cast data
                    cast_data = pd.read_csv(cast_file)
                    
                    # Create DefectPredictionEngine instance
                    engine = DefectPredictionEngine(config_path=None)
                    engine.config = self.config  # Pass config directly
                    
                    # Create RealTimeStreamSimulator instance
                    simulator = RealTimeStreamSimulator(
                        cast_data=cast_data,
                        config=self.config,
                        inference_engine=engine
                    )
                    
                    # Start the simulator's stream
                    simulator.start_stream()
                    
                    # Store the simulator for cleanup later
                    self.streams.append(simulator)
                    
                    # Create asyncio task for this stream
                    task = asyncio.create_task(
                        self._run_single_stream(i, simulator, engine)
                    )
                    self.tasks.append(task)
                    
                    self.logger.info(f"Created stream {i} for cast file: {cast_file}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to create stream {i} for file {cast_file}: {e}")
                    continue
            
            if not self.tasks:
                self.logger.error("No streams were successfully created")
                self.running = False
                return
            
            self.logger.info(f"Running {len(self.tasks)} concurrent prediction streams")
            
            # Run all tasks concurrently
            results = await asyncio.gather(*self.tasks, return_exceptions=True)
            
            # Process and log exceptions from tasks
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Exception in task {i}: {result}", exc_info=True)
                    
        except Exception as e:
            self.logger.error(f"Error in pipeline execution: {e}", exc_info=True)
        finally:
            # Ensure cleanup happens even if there's an exception
            self.stop_pipeline()
    
    def stop_pipeline(self):
        """
        Gracefully stops all running streams and asynchronous tasks.
        """
        if not self.running:
            self.logger.debug("Pipeline is not running")
            return
        
        self.logger.info("Stopping prediction pipeline")
        
        # Set running flag to False to signal all loops to exit
        self.running = False
        
        # Cancel all asyncio tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Stop all simulator streams to ensure threads are cleanly shut down
        for simulator in self.streams:
            try:
                simulator.stop_stream()
            except Exception as e:
                self.logger.error(f"Error stopping simulator: {e}")
        
        self.logger.info("Pipeline stopped successfully")
        
        # Clear streams and tasks
        self.streams = []
        self.tasks = []