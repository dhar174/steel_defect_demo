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
    
    def load_raw_data(self, file_pattern: str = "*.parquet", cast_id_col: str = "cast_id") -> pd.DataFrame:
        """
        Load raw time series data from files.
        
        Args:
            file_pattern (str): File pattern to match for loading data
            cast_id_col (str): Name of the column to store the cast ID
            
        Returns:
            pd.DataFrame: Combined raw data from all matching files
        """
        raw_data_dir = self.data_dir / "raw"
        all_files = list(raw_data_dir.glob(file_pattern))

        if not all_files:
            raise FileNotFoundError(f"No files matching pattern '{file_pattern}' found in {raw_data_dir}")

        df_list = []
        for file_path in all_files:
            df = pd.read_parquet(file_path)
            df[cast_id_col] = file_path.stem
            df_list.append(df)

        return pd.concat(df_list, ignore_index=True)
    
    def load_processed_data(self, split: str = "train") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load processed features and labels.
        
        Args:
            split (str): Data split to load ('train', 'test', 'val')
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and labels
        """
        processed_data_dir = self.data_dir / "processed"
        features_path = processed_data_dir / f"{split}_features.parquet"
        labels_path = processed_data_dir / f"{split}_labels.parquet"

        if not features_path.exists() or not labels_path.exists():
            raise FileNotFoundError(f"Processed data for split '{split}' not found in {processed_data_dir}")

        X = pd.read_parquet(features_path)
        y = pd.read_parquet(labels_path).squeeze()

        return X, y
    
    def load_cleaned_data(self, file_path: str) -> pd.DataFrame:
        """
        Load cleaned data from a CSV file.
        
        Args:
            file_path (str): Path to the cleaned data file
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            # If the specific path doesn't exist, try to load sample data
            sample_path = self.data_dir / "examples" / "steel_defect_sample.csv"
            if sample_path.exists():
                return pd.read_csv(sample_path)
            else:
                raise FileNotFoundError(f"Neither {file_path} nor {sample_path} exists")
    
    def load_cast_metadata(self) -> pd.DataFrame:
        """
        Load cast metadata including steel grade, composition, etc.
        
        Returns:
            pd.DataFrame: Cast metadata
        """
        metadata_path = self.data_dir / "synthetic" / "dataset_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        return pd.json_normalize(metadata['cast_metadata'])
    
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
        metadata = self.load_cast_metadata()
        cast_ids = metadata['cast_id'].unique()

        # Use scikit-learn for stratified split if labels are available
        try:
            from sklearn.model_selection import train_test_split

            labels = metadata.set_index('cast_id')['defect_label']
            train_ids, test_ids = train_test_split(
                cast_ids,
                test_size=test_size,
                random_state=random_state,
                stratify=labels
            )
            return list(train_ids), list(test_ids)

        except ImportError:
            # Fallback to random split if scikit-learn is not available
            np.random.seed(random_state)
            np.random.shuffle(cast_ids)
            split_idx = int(len(cast_ids) * (1 - test_size))
            return list(cast_ids[:split_idx]), list(cast_ids[split_idx:])