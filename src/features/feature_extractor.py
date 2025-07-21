import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Tuple, Dict, Optional
from scipy import interpolate
import warnings

class SequenceFeatureExtractor:
    """Prepare sequences for LSTM model"""
    
    def __init__(self, sequence_length: int = 300, stride: int = 1, augmentation_config: dict = None):
        """
        Initialize sequence feature extractor.
        
        Args:
            sequence_length (int): Fixed length for sequences (default: 300 for 5 minutes at 1Hz)
            stride (int): Stride for sliding window extraction (default: 1)
            augmentation_config (dict): Configuration for data augmentation:
                - noise_std (float): Standard deviation for Gaussian noise (default: 0.01)
                - time_warp_max (float): Maximum time warping factor (default: 0.1)
                - augmentation_ratio (float): Fraction of data to augment (default: 0.2)
                - min_max_scaling (bool): Whether to use MinMax scaling instead of Z-score (default: False)
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.augmentation_config = augmentation_config or {}
        
        # Default augmentation parameters
        self.noise_std = self.augmentation_config.get('noise_std', 0.01)
        self.time_warp_max = self.augmentation_config.get('time_warp_max', 0.1)
        self.augmentation_ratio = self.augmentation_config.get('augmentation_ratio', 0.2)
        self.use_minmax = self.augmentation_config.get('min_max_scaling', False)
        
        # Initialize scalers
        if self.use_minmax:
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        
        self.fitted = False
    
    def extract_sequences(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract overlapping sequences from continuous sensor data using sliding window.
        
        Args:
            data (pd.DataFrame): Time series data with sensor readings
            
        Returns:
            np.ndarray: Array of sequences with shape (n_sequences, sequence_length, n_features)
        """
        if data.empty:
            return np.array([])
        
        # Convert to numpy array for efficient processing
        values = data.values
        n_samples, n_features = values.shape
        
        if n_samples < self.sequence_length:
            warnings.warn(f"Data length ({n_samples}) is less than sequence_length ({self.sequence_length}). "
                         "Consider using padding or reducing sequence_length.")
            return np.array([])
        
        # Calculate number of sequences that can be extracted
        n_sequences = (n_samples - self.sequence_length) // self.stride + 1
        
        sequences = []
        for i in range(n_sequences):
            start_idx = i * self.stride
            end_idx = start_idx + self.sequence_length
            sequence = values[start_idx:end_idx]
            sequences.append(sequence)
        
        return np.array(sequences)
    def normalize_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization to sequences per feature channel.
        
        Args:
            sequences (np.ndarray): Array of sequences to normalize
                                  Shape: (n_sequences, sequence_length, n_features)
            
        Returns:
            np.ndarray: Normalized sequences with same shape
        """
        if sequences.size == 0:
            return sequences
        
        # Handle both 2D and 3D arrays
        if sequences.ndim == 2:
            # Single sequence: (sequence_length, n_features)
            sequences = sequences.reshape(1, *sequences.shape)
            single_sequence = True
        else:
            single_sequence = False
        
        n_sequences, sequence_length, n_features = sequences.shape
        
        # Reshape to (n_sequences * sequence_length, n_features) for normalization
        reshaped = sequences.reshape(-1, n_features)
        
        if not self.fitted:
            # Fit scaler on the data
            self.scaler.fit(reshaped)
            self.fitted = True
        
        # Transform the data
        normalized_reshaped = self.scaler.transform(reshaped)
        
        # Reshape back to original shape
        normalized_sequences = normalized_reshaped.reshape(n_sequences, sequence_length, n_features)
        
        if single_sequence:
            return normalized_sequences[0]
        
        return normalized_sequences
    
    def pad_sequences(self, sequences: List[np.ndarray]) -> np.ndarray:
        """
        Pad/truncate sequences to fixed length.
        
        Args:
            sequences (List[np.ndarray]): List of variable-length sequences
            
        Returns:
            np.ndarray: Fixed-length sequences with shape (n_sequences, sequence_length, n_features)
        """
        if not sequences:
            return np.array([])
        
        # Determine number of features from the first sequence
        n_features = sequences[0].shape[1] if sequences[0].ndim > 1 else 1
        
        padded_sequences = []
        
        for sequence in sequences:
            seq_length = len(sequence)
            
            if seq_length >= self.sequence_length:
                # Truncate to fixed length
                padded_seq = sequence[:self.sequence_length]
            else:
                # Pad with zeros
                if sequence.ndim == 1:
                    padding_shape = (self.sequence_length - seq_length,)
                else:
                    padding_shape = (self.sequence_length - seq_length, n_features)
                
                padding = np.zeros(padding_shape)
                padded_seq = np.vstack([sequence, padding]) if sequence.ndim > 1 else np.concatenate([sequence, padding])
            
            padded_sequences.append(padded_seq)
        
        return np.array(padded_sequences)
    
    def augment_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation techniques including noise injection and time warping.
        
        Args:
            sequences (np.ndarray): Original sequences to augment
                                  Shape: (n_sequences, sequence_length, n_features)
            
        Returns:
            np.ndarray: Augmented sequences (original + augmented)
        """
        if sequences.size == 0:
            return sequences
        
        n_sequences, sequence_length, n_features = sequences.shape
        n_augment = int(n_sequences * self.augmentation_ratio)
        
        if n_augment == 0:
            return sequences
        
        # Randomly select sequences to augment
        np.random.seed(42)  # For reproducibility
        indices_to_augment = np.random.choice(n_sequences, n_augment, replace=True)
        
        augmented_sequences = []
        
        for idx in indices_to_augment:
            original_seq = sequences[idx].copy()
            
            # Apply noise injection
            noise = np.random.normal(0, self.noise_std, original_seq.shape)
            noisy_seq = original_seq + noise
            
            # Apply time warping
            warped_seq = self._time_warp_sequence(original_seq)
            
            augmented_sequences.extend([noisy_seq, warped_seq])
        
        # Combine original and augmented sequences
        augmented_array = np.array(augmented_sequences)
        
        if augmented_array.size > 0:
            return np.concatenate([sequences, augmented_array], axis=0)
        else:
            return sequences
    
    def _time_warp_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Apply time warping to a sequence by stretching/compressing the time axis.
        
        Args:
            sequence (np.ndarray): Input sequence of shape (sequence_length, n_features)
            
        Returns:
            np.ndarray: Time-warped sequence with same shape
        """
        sequence_length, n_features = sequence.shape
        
        # Generate random warping factor
        warp_factor = 1 + np.random.uniform(-self.time_warp_max, self.time_warp_max)
        
        # Create original and warped time indices
        original_indices = np.arange(sequence_length)
        warped_length = int(sequence_length * warp_factor)
        
        if warped_length <= 1:
            return sequence
        
        warped_indices = np.linspace(0, sequence_length - 1, warped_length)
        
        # Interpolate each feature channel
        warped_sequence = np.zeros((sequence_length, n_features))
        
        for feature_idx in range(n_features):
            # Use linear interpolation
            f = interpolate.interp1d(original_indices, sequence[:, feature_idx], 
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
            warped_values = f(warped_indices)
            
            # Resample back to original length
            resample_indices = np.linspace(0, len(warped_values) - 1, sequence_length)
            f_resample = interpolate.interp1d(np.arange(len(warped_values)), warped_values,
                                            kind='linear', bounds_error=False, fill_value='extrapolate')
            warped_sequence[:, feature_idx] = f_resample(resample_indices)
        
        return warped_sequence
    
    def split_sequences(self, sequences: np.ndarray, labels: np.ndarray, 
                       split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> Tuple:
        """
        Temporal-aware train/validation/test split preserving data order.
        
        Args:
            sequences (np.ndarray): Sequences to split
            labels (np.ndarray): Corresponding labels
            split_ratios (Tuple[float, float, float]): Train, validation, test split ratios
            
        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if sequences.size == 0 or labels.size == 0:
            empty_array = np.array([])
            return empty_array, empty_array, empty_array, empty_array, empty_array, empty_array
        
        # Validate split ratios
        train_ratio, val_ratio, test_ratio = split_ratios
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Split ratios must sum to 1.0")
        
        n_sequences = len(sequences)
        
        # Calculate split indices (temporal order preserved)
        train_end = int(n_sequences * train_ratio)
        val_end = int(n_sequences * (train_ratio + val_ratio))
        
        # Split sequences
        X_train = sequences[:train_end]
        X_val = sequences[train_end:val_end]
        X_test = sequences[val_end:]
        
        # Split labels
        y_train = labels[:train_end]
        y_val = labels[train_end:val_end]
        y_test = labels[val_end:]
        
    def prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences and labels for training.
        
        Args:
            data (pd.DataFrame): Time series data with labels
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Sequences and corresponding labels
        """
        if data.empty:
            return np.array([]), np.array([])
        
        # Assume last column is labels, others are features
        if 'label' in data.columns:
            feature_data = data.drop('label', axis=1)
            labels = data['label'].values
        elif 'defect' in data.columns:
            feature_data = data.drop('defect', axis=1)
            labels = data['defect'].values
        else:
            # If no explicit label column, use last column
            feature_data = data.iloc[:, :-1]
            labels = data.iloc[:, -1].values
        
        # Extract sequences
        sequences = self.extract_sequences(feature_data)
        
        if sequences.size == 0:
            return np.array([]), np.array([])
        
        # Create labels for each sequence (use label at the end of each sequence)
        sequence_labels = []
        for i in range(len(sequences)):
            start_idx = i * self.stride
            end_idx = start_idx + self.sequence_length - 1
            if end_idx < len(labels):
                sequence_labels.append(labels[end_idx])
            else:
                sequence_labels.append(labels[-1])
        
        return sequences, np.array(sequence_labels)
    
    def extract_sliding_windows(self, time_series: pd.DataFrame, 
                              window_size: int = None, 
                              stride: int = 1) -> List[np.ndarray]:
        """
        Extract sliding windows from time series.
        
        Args:
            time_series (pd.DataFrame): Input time series
            window_size (int): Size of sliding window (default: self.sequence_length)
            stride (int): Stride for sliding window
            
        Returns:
            List[np.ndarray]: List of sliding windows
        """
        if window_size is None:
            window_size = self.sequence_length
        
        if time_series.empty:
            return []
        
        values = time_series.values
        n_samples = len(values)
        
        if n_samples < window_size:
            return []
        
        windows = []
        for i in range(0, n_samples - window_size + 1, stride):
            window = values[i:i + window_size]
            windows.append(window)
        
        return windows
    
    def fit_normalizer(self, sequences: np.ndarray) -> None:
        """
        Fit normalizer on training sequences.
        
        Args:
            sequences (np.ndarray): Training sequences
        """
        if sequences.size == 0:
            return
        
        # Handle both 2D and 3D arrays
        if sequences.ndim == 3:
            # Reshape to (n_samples, n_features) for fitting
            n_sequences, sequence_length, n_features = sequences.shape
            reshaped = sequences.reshape(-1, n_features)
        else:
            reshaped = sequences
        
        self.scaler.fit(reshaped)
        self.fitted = True
    
    def create_sequences_from_cast(self, cast_data: pd.DataFrame) -> np.ndarray:
        """
        Create sequence representation from single cast data.
        
        Args:
            cast_data (pd.DataFrame): Time series data for a single cast
            
        Returns:
            np.ndarray: Sequence representation
        """
        if cast_data.empty:
            return np.array([])
        
        # Extract sequences using sliding window
        sequences = self.extract_sequences(cast_data)
        
        if sequences.size == 0:
            # If data is too short, pad it to sequence length
            padded_data = self.pad_sequences([cast_data.values])
            return padded_data[0] if len(padded_data) > 0 else np.array([])
        
        return sequences