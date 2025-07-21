"""
PyTorch Dataset and DataLoader implementation for steel casting time series sequences.

This module provides comprehensive data pipeline components for LSTM model training,
including memory-efficient loading, data augmentation, and class imbalance handling.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import h5py
from pathlib import Path
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CastingSequenceDataset(Dataset):
    """PyTorch Dataset for steel casting time series sequences"""
    
    def __init__(self, 
                 sequences: np.ndarray, 
                 labels: np.ndarray, 
                 transform: Optional[Callable] = None, 
                 augment: bool = False,
                 sequence_length: int = 300):
        """
        Initialize dataset with casting sequences and defect labels
        
        Parameters:
        - sequences: (N, sequence_length, num_features) array of sensor data
        - labels: (N,) array of binary defect labels
        - transform: Optional preprocessing transforms
        - augment: Enable data augmentation during training
        - sequence_length: Target sequence length for padding/truncation
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
        self.transform = transform
        self.augment = augment
        self.sequence_length = sequence_length
        
        # Initialize augmentation if enabled
        if self.augment:
            self.augmentation = SequenceAugmentation()
        
        logger.info(f"Initialized CastingSequenceDataset with {len(self)} sequences")
        logger.info(f"Sequence shape: {self.sequences.shape}")
        logger.info(f"Defect rate: {self.labels.mean():.3f}")
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset"""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence and its label
        
        Parameters:
        - idx: Index of the sequence to retrieve
        
        Returns:
        - Tuple of (sequence, label) tensors
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Apply data augmentation if enabled and this is a non-defect sample
        if self.augment and label.item() == 0:  # Only augment non-defect sequences
            sequence = self.augmentation.apply_random_augmentation(sequence)
        
        # Apply custom transforms if provided
        if self.transform:
            sequence = self.transform(sequence)
        
        return sequence, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset"""
        pos_count = self.labels.sum().item()
        neg_count = len(self.labels) - pos_count
        
        if pos_count == 0 or neg_count == 0:
            return torch.tensor([1.0, 1.0])
        
        total = len(self.labels)
        pos_weight = total / (2 * pos_count)
        neg_weight = total / (2 * neg_count)
        
        return torch.tensor([neg_weight, pos_weight])


class MemoryEfficientSequenceDataset(Dataset):
    """Memory-efficient dataset for large steel casting sequences"""
    
    def __init__(self, 
                 data_path: Union[str, Path], 
                 sequence_indices: List[int], 
                 labels: np.ndarray,
                 sequence_length: int = 300,
                 transform: Optional[Callable] = None,
                 augment: bool = False):
        """
        Initialize memory-efficient dataset that loads data on-demand
        
        Parameters:
        - data_path: Path to HDF5 file containing sequences
        - sequence_indices: Indices of sequences to use
        - labels: Binary defect labels
        - sequence_length: Target sequence length
        - transform: Optional preprocessing transforms
        - augment: Enable data augmentation
        """
        self.data_path = Path(data_path)
        self.sequence_indices = sequence_indices
        self.labels = torch.FloatTensor(labels)
        self.sequence_length = sequence_length
        self.transform = transform
        self.augment = augment
        
        if self.augment:
            self.augmentation = SequenceAugmentation()
        
        # Verify data file exists
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        logger.info(f"Initialized MemoryEfficientSequenceDataset with {len(self)} sequences")
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset"""
        return len(self.sequence_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load sequence data on-demand from disk
        
        Parameters:
        - idx: Index of the sequence to retrieve
        
        Returns:
        - Tuple of (sequence, label) tensors
        """
        sequence_idx = self.sequence_indices[idx]
        label = self.labels[idx]
        
        # Load sequence from HDF5 file
        with h5py.File(self.data_path, 'r') as f:
            sequence = torch.FloatTensor(f['sequences'][sequence_idx])
        
        # Pad or truncate to target length
        sequence = self._pad_or_truncate(sequence)
        
        # Apply data augmentation if enabled and this is a non-defect sample
        if self.augment and label.item() == 0:
            sequence = self.augmentation.apply_random_augmentation(sequence)
        
        # Apply custom transforms if provided
        if self.transform:
            sequence = self.transform(sequence)
        
        return sequence, label
    
    def _pad_or_truncate(self, sequence: torch.Tensor) -> torch.Tensor:
        """Pad or truncate sequence to target length"""
        current_length = sequence.shape[0]
        
        if current_length >= self.sequence_length:
            # Truncate to target length
            return sequence[:self.sequence_length]
        else:
            # Pad with zeros
            padding = torch.zeros(self.sequence_length - current_length, sequence.shape[1])
            return torch.cat([sequence, padding], dim=0)


class SequenceAugmentation:
    """Data augmentation techniques for time series sequences"""
    
    def __init__(self, 
                 noise_std: float = 0.01,
                 time_warp_sigma: float = 0.2,
                 magnitude_warp_sigma: float = 0.2,
                 time_warp_probability: float = 0.3,
                 magnitude_warp_probability: float = 0.3,
                 noise_probability: float = 0.5):
        """
        Initialize sequence augmentation with configurable parameters
        
        Parameters:
        - noise_std: Standard deviation for Gaussian noise
        - time_warp_sigma: Sigma parameter for time warping
        - magnitude_warp_sigma: Sigma parameter for magnitude warping
        - time_warp_probability: Probability of applying time warping
        - magnitude_warp_probability: Probability of applying magnitude warping
        - noise_probability: Probability of applying noise
        """
        self.noise_std = noise_std
        self.time_warp_sigma = time_warp_sigma
        self.magnitude_warp_sigma = magnitude_warp_sigma
        self.time_warp_probability = time_warp_probability
        self.magnitude_warp_probability = magnitude_warp_probability
        self.noise_probability = noise_probability
    
    def add_noise(self, sequence: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to sensor readings"""
        noise = torch.randn_like(sequence) * self.noise_std
        return sequence + noise
    
    def time_warp(self, sequence: torch.Tensor) -> torch.Tensor:
        """Apply time warping augmentation"""
        seq_len, n_features = sequence.shape
        
        # Generate smooth warping function
        warp_steps = torch.linspace(0, seq_len - 1, seq_len)
        warp_noise = torch.randn(seq_len) * self.time_warp_sigma
        
        # Smooth the noise with a simple moving average
        kernel_size = max(1, seq_len // 20)
        if kernel_size > 1:
            # Pad the noise to handle convolution boundaries
            padded_noise = torch.nn.functional.pad(warp_noise.unsqueeze(0).unsqueeze(0), 
                                                 (kernel_size // 2, kernel_size // 2), 
                                                 mode='replicate')
            smoothed_noise = torch.nn.functional.conv1d(
                padded_noise,
                torch.ones(1, 1, kernel_size) / kernel_size,
                padding=0
            ).squeeze()
            
            # Ensure same length as original
            if len(smoothed_noise) != seq_len:
                smoothed_noise = smoothed_noise[:seq_len]
            warp_noise = smoothed_noise
        
        warped_indices = warp_steps + warp_noise
        warped_indices = torch.clamp(warped_indices, 0, seq_len - 1)
        
        # Simple interpolation approach
        warped_sequence = torch.zeros_like(sequence)
        for i in range(seq_len):
            src_idx = int(warped_indices[i].item())
            src_idx = min(max(src_idx, 0), seq_len - 1)
            warped_sequence[i] = sequence[src_idx]
        
        return warped_sequence
    
    def magnitude_warp(self, sequence: torch.Tensor) -> torch.Tensor:
        """Apply magnitude warping augmentation"""
        seq_len, n_features = sequence.shape
        
        # Generate smooth magnitude scaling factors
        magnitude_factors = 1.0 + torch.randn(seq_len, n_features) * self.magnitude_warp_sigma
        
        # Smooth the factors
        kernel_size = max(1, seq_len // 20)
        if kernel_size > 1:
            for i in range(n_features):
                padded_factors = torch.nn.functional.pad(
                    magnitude_factors[:, i].unsqueeze(0).unsqueeze(0),
                    (kernel_size // 2, kernel_size // 2),
                    mode='replicate'
                )
                smoothed_factors = torch.nn.functional.conv1d(
                    padded_factors,
                    torch.ones(1, 1, kernel_size) / kernel_size,
                    padding=0
                ).squeeze()
                
                # Ensure same length
                if len(smoothed_factors) != seq_len:
                    smoothed_factors = smoothed_factors[:seq_len]
                magnitude_factors[:, i] = smoothed_factors
        
        return sequence * magnitude_factors
    
    def apply_random_augmentation(self, sequence: torch.Tensor) -> torch.Tensor:
        """Apply random combination of augmentations"""
        augmented_sequence = sequence.clone()
        
        # Apply noise with probability
        if torch.rand(1).item() < self.noise_probability:
            augmented_sequence = self.add_noise(augmented_sequence)
        
        # Apply time warping with probability
        if torch.rand(1).item() < self.time_warp_probability:
            augmented_sequence = self.time_warp(augmented_sequence)
        
        # Apply magnitude warping with probability
        if torch.rand(1).item() < self.magnitude_warp_probability:
            augmented_sequence = self.magnitude_warp(augmented_sequence)
        
        return augmented_sequence


class ImbalancedSequenceSampler(WeightedRandomSampler):
    """Custom sampler to handle class imbalance in defect detection"""
    
    def __init__(self, 
                 labels: torch.Tensor, 
                 defect_weight_multiplier: float = 3.0,
                 replacement: bool = True):
        """
        Create weighted sampler based on defect frequency
        
        Parameters:
        - labels: Binary defect labels
        - defect_weight_multiplier: Weight multiplier for defect samples
        - replacement: Whether to sample with replacement
        """
        self.defect_weight_multiplier = defect_weight_multiplier
        
        # Calculate sample weights
        weights = torch.ones_like(labels, dtype=torch.float)
        defect_mask = labels == 1
        weights[defect_mask] = defect_weight_multiplier
        
        # Log sampling statistics
        total_samples = len(labels)
        defect_samples = defect_mask.sum().item()
        normal_samples = total_samples - defect_samples
        
        logger.info(f"Sampler statistics:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Normal samples: {normal_samples} (weight: 1.0)")
        logger.info(f"  Defect samples: {defect_samples} (weight: {defect_weight_multiplier})")
        
        super().__init__(weights, total_samples, replacement=replacement)


def sequence_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for variable-length sequences
    
    Handles:
    - Padding sequences to max length in batch
    - Creating attention masks for padded sequences
    - Proper tensor stacking and device placement
    
    Parameters:
    - batch: List of (sequence, label) tuples
    
    Returns:
    - Tuple of (padded_sequences, labels, attention_masks)
    """
    sequences, labels = zip(*batch)
    
    # Find maximum sequence length in batch
    max_length = max(seq.shape[0] for seq in sequences)
    batch_size = len(sequences)
    n_features = sequences[0].shape[1]
    
    # Create padded tensor
    padded_sequences = torch.zeros(batch_size, max_length, n_features)
    attention_masks = torch.zeros(batch_size, max_length, dtype=torch.bool)
    
    # Fill padded tensor and create attention masks
    for i, seq in enumerate(sequences):
        seq_len = seq.shape[0]
        padded_sequences[i, :seq_len] = seq
        attention_masks[i, :seq_len] = True
    
    # Stack labels
    labels = torch.stack(labels)
    
    return padded_sequences, labels, attention_masks


def create_data_loaders(dataset_train: Dataset, 
                       dataset_val: Dataset, 
                       config: Dict[str, Any],
                       use_weighted_sampling: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Create optimized DataLoaders for training and validation
    
    Parameters:
    - dataset_train: Training dataset
    - dataset_val: Validation dataset
    - config: Configuration dictionary
    - use_weighted_sampling: Whether to use weighted sampling for training
    
    Returns:
    - Tuple of (train_loader, val_loader)
    """
    # Extract configuration parameters
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4)
    pin_memory = config.get('pin_memory', True)
    defect_weight_multiplier = config.get('defect_weight_multiplier', 3.0)
    
    # Create weighted sampler for training if requested
    train_sampler = None
    shuffle_train = True
    
    if use_weighted_sampling and hasattr(dataset_train, 'labels'):
        train_sampler = ImbalancedSequenceSampler(
            dataset_train.labels, 
            defect_weight_multiplier=defect_weight_multiplier
        )
        shuffle_train = False  # Cannot shuffle when using sampler
    
    # Create training DataLoader
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=sequence_collate_fn,
        drop_last=True  # Drop incomplete batches for consistent training
    )
    
    # Create validation DataLoader (no sampling, no augmentation)
    val_loader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=sequence_collate_fn,
        drop_last=False
    )
    
    logger.info(f"Created DataLoaders:")
    logger.info(f"  Training batches: {len(train_loader)}")
    logger.info(f"  Validation batches: {len(val_loader)}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Weighted sampling: {use_weighted_sampling}")
    
    return train_loader, val_loader


def validate_dataset_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and set default values for dataset configuration
    
    Parameters:
    - config: Configuration dictionary
    
    Returns:
    - Validated configuration dictionary
    """
    validated_config = config.copy()
    
    # Set defaults
    defaults = {
        'sequence_length': 300,
        'batch_size': 32,
        'num_workers': 4,
        'pin_memory': True,
        'defect_weight_multiplier': 3.0,
        'augmentation': {
            'enabled': True,
            'noise_std': 0.01,
            'time_warp_probability': 0.3,
            'magnitude_warp_probability': 0.3,
            'noise_probability': 0.5
        }
    }
    
    # Apply defaults
    for key, default_value in defaults.items():
        if key not in validated_config:
            validated_config[key] = default_value
        elif isinstance(default_value, dict) and isinstance(validated_config[key], dict):
            # Merge nested dictionaries
            for nested_key, nested_default in default_value.items():
                if nested_key not in validated_config[key]:
                    validated_config[key][nested_key] = nested_default
    
    return validated_config


# Utility functions for data preparation
def prepare_sequences_from_dataframe(df: pd.DataFrame, 
                                   sequence_length: int = 300,
                                   feature_columns: Optional[List[str]] = None,
                                   label_column: str = 'defect',
                                   cast_id_column: str = 'cast_id') -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences from a pandas DataFrame
    
    Parameters:
    - df: DataFrame with time series data
    - sequence_length: Target sequence length
    - feature_columns: List of feature column names (auto-detect if None)
    - label_column: Name of label column
    - cast_id_column: Name of cast ID column
    
    Returns:
    - Tuple of (sequences, labels) arrays
    """
    if feature_columns is None:
        # Auto-detect feature columns (exclude label and cast_id)
        feature_columns = [col for col in df.columns 
                          if col not in [label_column, cast_id_column]]
    
    sequences = []
    labels = []
    
    # Group by cast ID and create sequences
    for cast_id, group in df.groupby(cast_id_column):
        # Sort by timestamp if available
        if 'timestamp' in group.columns:
            group = group.sort_values('timestamp')
        
        # Extract features and label
        features = group[feature_columns].values
        label = group[label_column].iloc[0]  # Assume constant label per cast
        
        # Create sequence (pad or truncate as needed)
        if len(features) >= sequence_length:
            sequence = features[:sequence_length]
        else:
            # Pad with zeros
            padding = np.zeros((sequence_length - len(features), len(feature_columns)))
            sequence = np.vstack([features, padding])
        
        sequences.append(sequence)
        labels.append(label)
    
    return np.array(sequences), np.array(labels)