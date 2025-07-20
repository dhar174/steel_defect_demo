import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class DataLoader:
    """Load and prepare steel casting data for training and inference"""
    
    def __init__(self, data_dir: str):
        """
        Initialize data loader with data directory path.
        
        Args:
            data_dir (str): Path to the data directory
        """
        self.data_dir = Path(data_dir)
    
    def load_raw_data(self, file_pattern: str = "*.parquet") -> pd.DataFrame:
        """
        Load raw time series data from files.
        
        Args:
            file_pattern (str): File pattern to match for loading data
            
        Returns:
            pd.DataFrame: Combined raw data from all matching files
        """
        # TODO: Implement raw data loading
        pass
    
    def load_processed_data(self, split: str = "train") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load processed features and labels.
        
        Args:
            split (str): Data split to load ('train', 'test', 'val')
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and labels
        """
        # TODO: Implement processed data loading
        pass
    
    def load_cast_metadata(self) -> pd.DataFrame:
        """
        Load cast metadata including steel grade, composition, etc.
        
        Returns:
            pd.DataFrame: Cast metadata
        """
        # TODO: Implement metadata loading
        pass
    
    def get_train_test_split(self, test_size: float = 0.2, 
                           random_state: int = 42) -> Tuple[List[str], List[str]]:
        """
        Get train/test split of cast IDs.
        
        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            Tuple[List[str], List[str]]: Train and test cast IDs
        """
        # TODO: Implement train/test split
        pass