import time
import threading
from queue import Queue
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
            # Assuming data points are 1 second apart in real time
            time.sleep(1.0 / playback_speed)
        
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
            except:
                break
        
        if not buffer_rows:
            return pd.DataFrame()
        
        # Convert to DataFrame
        buffer_df = pd.DataFrame(buffer_rows)
        
        # Maintain sliding window of specified size
        # Assuming each row represents 1 second of data
        if len(buffer_df) > buffer_size_seconds:
            buffer_df = buffer_df.tail(buffer_size_seconds)
        
        # Reset index to maintain continuity
        buffer_df = buffer_df.reset_index(drop=True)
        
        return buffer_df