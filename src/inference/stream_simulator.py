import time
import threading
import queue
from queue import Queue
import queue
import pandas as pd
from typing import Dict
from src.inference.inference_engine import DefectPredictionEngine

class RealTimeStreamSimulator:
    """Simulates real-time streaming data from historical casting data."""
    
    def __init__(self, cast_data: pd.DataFrame, config: Dict, inference_engine: DefectPredictionEngine):
        """
        Initializes the stream simulator.
        
        Args:
            cast_data (pd.DataFrame): The historical data for a single cast.
            config (Dict): The inference configuration.
            inference_engine (DefectPredictionEngine): The engine to send data to.
        """
        self.cast_data = cast_data
        self.config = config['inference']['real_time_simulation']
        self.inference_engine = inference_engine
        self.data_queue = Queue()
        self.running = False
        self.producer_thread = None
        self._data_finished = False
        self._data_interval = self._calculate_data_interval()
        
    def _calculate_data_interval(self) -> float:
        """
        Calculate the time interval between data points.
        
        Returns:
            float: Time interval in seconds between data points.
                   Uses actual timestamp differences if available,
                   otherwise falls back to configured value.
        """
        # Check if data has timestamp column and enough rows to calculate interval
        if 'timestamp' in self.cast_data.columns and len(self.cast_data) >= 2:
            try:
                # Convert timestamp column to datetime if it's not already
                timestamps = pd.to_datetime(self.cast_data['timestamp'])
                
                # Calculate differences between consecutive timestamps
                time_diffs = timestamps.diff().dropna()
                
                if len(time_diffs) > 0:
                    # Use median of time differences to handle potential irregularities
                    median_diff = time_diffs.median()
                    interval_seconds = median_diff.total_seconds()
                    
                    # Sanity check: interval should be positive and reasonable (between 0.001s and 3600s)
                    if 0.001 <= interval_seconds <= 3600:
                        return interval_seconds
            except (ValueError, TypeError):
                # If timestamp parsing fails, fall back to configured value
                pass
        
        # Fallback to configured value
        return self.config.get('inference', {}).get('real_time_simulation', {}).get('data_interval_seconds', 1.0)
        
    def _produce_data(self) -> None:
        """
        Internal method to read historical data and push it to the queue
        at a simulated real-time speed.
        """
        playback_speed = self.config.get('playback_speed_multiplier', 1.0)
        
        for index, row in self.cast_data.iterrows():
            if not self.running:
                break
                
            # Put data row into queue
            self.data_queue.put(row)
            
            # Sleep to simulate real-time data ingestion rate
            # Use calculated or configured data interval
            time.sleep(self._data_interval / playback_speed)
        
        # Mark that we've finished processing all data
        self._data_finished = True
    
    def start_stream(self) -> None:
        """
        Starts the data streaming simulation in a separate thread.
        """
        if not self.running:
            self.running = True
            self._data_finished = False
            self.producer_thread = threading.Thread(target=self._produce_data)
            self.producer_thread.start()
    
    def stop_stream(self) -> None:
        """
        Stops the streaming simulation and ensures graceful shutdown.
        """
        if self.running:
            self.running = False
            if self.producer_thread and self.producer_thread.is_alive():
                self.producer_thread.join()
    
    def get_data_buffer(self) -> pd.DataFrame:
        """
        Consumes data from the queue to form a sliding window buffer for prediction.
        """
        buffer_size_seconds = self.config.get('buffer_size_seconds', 300)
        buffer_rows = []
        
        # Get all available data from queue
        while not self.data_queue.empty():
            try:
                row = self.data_queue.get_nowait()
                buffer_rows.append(row)
            except queue.Empty:
                break
        
        if not buffer_rows:
            return pd.DataFrame()
        
        # Convert to DataFrame
        buffer_df = pd.DataFrame(buffer_rows)
        
        # Maintain sliding window of specified size
        # Use calculated data interval for buffer size calculation
        max_rows = int(buffer_size_seconds / self._data_interval)
        if len(buffer_df) > max_rows:
            buffer_df = buffer_df.tail(max_rows)
        
        # Reset index to maintain continuity
        buffer_df = buffer_df.reset_index(drop=True)
        
        return buffer_df