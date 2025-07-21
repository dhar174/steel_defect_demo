"""
Test suite for sequence dataset implementation
"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader
import tempfile
from pathlib import Path
import h5py

from src.data.sequence_dataset import (
    CastingSequenceDataset,
    MemoryEfficientSequenceDataset,
    SequenceAugmentation,
    ImbalancedSequenceSampler,
    sequence_collate_fn,
    create_data_loaders,
    validate_dataset_config,
    prepare_sequences_from_dataframe
)


class TestCastingSequenceDataset:
    """Test cases for CastingSequenceDataset"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample sequences and labels for testing"""
        # Create synthetic sequences: (N=100, seq_len=50, features=5)
        np.random.seed(42)
        sequences = np.random.randn(100, 50, 5).astype(np.float32)
        labels = np.random.choice([0, 1], size=100, p=[0.8, 0.2]).astype(np.float32)
        return sequences, labels
    
    def test_dataset_initialization(self, sample_data):
        """Test dataset creation with various input formats"""
        sequences, labels = sample_data
        
        # Test basic initialization
        dataset = CastingSequenceDataset(sequences, labels)
        assert len(dataset) == 100
        assert dataset.sequences.shape == (100, 50, 5)
        assert dataset.labels.shape == (100,)
        
        # Test with augmentation enabled
        dataset_aug = CastingSequenceDataset(sequences, labels, augment=True)
        assert hasattr(dataset_aug, 'augmentation')
    
    def test_dataset_getitem(self, sample_data):
        """Test data retrieval from dataset"""
        sequences, labels = sample_data
        dataset = CastingSequenceDataset(sequences, labels)
        
        # Test single item retrieval
        seq, label = dataset[0]
        assert isinstance(seq, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert seq.shape == (50, 5)
        assert label.shape == ()
    
    def test_class_weights(self, sample_data):
        """Test class weight calculation"""
        sequences, labels = sample_data
        dataset = CastingSequenceDataset(sequences, labels)
        
        weights = dataset.get_class_weights()
        assert isinstance(weights, torch.Tensor)
        assert len(weights) == 2
        assert weights[1] > weights[0]  # Defect class should have higher weight


class TestSequenceAugmentation:
    """Test cases for SequenceAugmentation"""
    
    @pytest.fixture
    def sample_sequence(self):
        """Create a sample sequence for augmentation testing"""
        np.random.seed(42)
        return torch.randn(100, 5)
    
    def test_augmentation_initialization(self):
        """Test augmentation object creation"""
        aug = SequenceAugmentation()
        assert hasattr(aug, 'noise_std')
        assert hasattr(aug, 'time_warp_sigma')
        assert hasattr(aug, 'magnitude_warp_sigma')
    
    def test_add_noise(self, sample_sequence):
        """Test noise addition augmentation"""
        aug = SequenceAugmentation(noise_std=0.1)
        augmented = aug.add_noise(sample_sequence)
        
        assert augmented.shape == sample_sequence.shape
        assert not torch.equal(augmented, sample_sequence)
    
    def test_magnitude_warp(self, sample_sequence):
        """Test magnitude warping augmentation"""
        aug = SequenceAugmentation(magnitude_warp_sigma=0.1)
        augmented = aug.magnitude_warp(sample_sequence)
        
        assert augmented.shape == sample_sequence.shape
        assert not torch.equal(augmented, sample_sequence)
    
    def test_random_augmentation(self, sample_sequence):
        """Test random augmentation application"""
        aug = SequenceAugmentation()
        augmented = aug.apply_random_augmentation(sample_sequence)
        
        assert augmented.shape == sample_sequence.shape


class TestImbalancedSequenceSampler:
    """Test cases for ImbalancedSequenceSampler"""
    
    def test_sampler_initialization(self):
        """Test sampler creation with imbalanced labels"""
        labels = torch.tensor([0, 0, 0, 0, 1, 1])  # 4 normal, 2 defect
        sampler = ImbalancedSequenceSampler(labels, defect_weight_multiplier=3.0)
        
        assert len(sampler) == len(labels)
    
    def test_sampler_weights(self):
        """Test that defect samples get higher weights"""
        labels = torch.tensor([0, 0, 0, 1])
        sampler = ImbalancedSequenceSampler(labels, defect_weight_multiplier=2.0)
        
        # Check that weights are applied correctly
        weights = sampler.weights
        assert weights[3] == 2.0  # Defect sample gets multiplier
        assert weights[0] == 1.0   # Normal sample gets weight 1.0


class TestSequenceCollateFunction:
    """Test cases for sequence_collate_fn"""
    
    def test_collate_same_length(self):
        """Test collation of same-length sequences"""
        sequences = [torch.randn(50, 5) for _ in range(4)]
        labels = [torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0), torch.tensor(1.0)]
        batch = list(zip(sequences, labels))
        
        padded_seqs, batch_labels, attention_masks = sequence_collate_fn(batch)
        
        assert padded_seqs.shape == (4, 50, 5)
        assert batch_labels.shape == (4,)
        assert attention_masks.shape == (4, 50)
        assert attention_masks.all()  # All positions should be valid
    
    def test_collate_variable_length(self):
        """Test collation of variable-length sequences"""
        sequences = [
            torch.randn(30, 5),  # Short sequence
            torch.randn(50, 5),  # Medium sequence
            torch.randn(40, 5),  # Medium sequence
        ]
        labels = [torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0)]
        batch = list(zip(sequences, labels))
        
        padded_seqs, batch_labels, attention_masks = sequence_collate_fn(batch)
        
        assert padded_seqs.shape == (3, 50, 5)  # Padded to max length
        assert batch_labels.shape == (3,)
        assert attention_masks.shape == (3, 50)
        
        # Check attention masks are correct
        assert attention_masks[0][:30].all() and not attention_masks[0][30:].any()
        assert attention_masks[1].all()  # Full sequence
        assert attention_masks[2][:40].all() and not attention_masks[2][40:].any()


class TestDataLoaderCreation:
    """Test cases for create_data_loaders function"""
    
    @pytest.fixture
    def sample_datasets(self):
        """Create sample train/val datasets"""
        np.random.seed(42)
        train_sequences = np.random.randn(80, 50, 5).astype(np.float32)
        train_labels = np.random.choice([0, 1], size=80, p=[0.8, 0.2]).astype(np.float32)
        val_sequences = np.random.randn(20, 50, 5).astype(np.float32)
        val_labels = np.random.choice([0, 1], size=20, p=[0.8, 0.2]).astype(np.float32)
        
        train_dataset = CastingSequenceDataset(train_sequences, train_labels)
        val_dataset = CastingSequenceDataset(val_sequences, val_labels)
        
        return train_dataset, val_dataset
    
    def test_dataloader_creation(self, sample_datasets):
        """Test DataLoader creation with default config"""
        train_dataset, val_dataset = sample_datasets
        config = {'batch_size': 16, 'num_workers': 0}  # Use 0 workers for testing
        
        train_loader, val_loader = create_data_loaders(
            train_dataset, val_dataset, config
        )
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert train_loader.batch_size == 16
        assert val_loader.batch_size == 16
    
    def test_dataloader_batching(self, sample_datasets):
        """Test that DataLoader produces correct batch shapes"""
        train_dataset, val_dataset = sample_datasets
        config = {'batch_size': 8, 'num_workers': 0}
        
        train_loader, _ = create_data_loaders(
            train_dataset, val_dataset, config
        )
        
        # Test first batch
        batch = next(iter(train_loader))
        sequences, labels, attention_masks = batch
        
        assert sequences.shape[0] == 8  # Batch size
        assert sequences.shape[2] == 5   # Feature dimension
        assert labels.shape == (8,)
        assert attention_masks.shape[0] == 8


class TestConfigValidation:
    """Test cases for validate_dataset_config"""
    
    def test_config_defaults(self):
        """Test that default values are applied correctly"""
        config = {}
        validated = validate_dataset_config(config)
        
        assert validated['sequence_length'] == 300
        assert validated['batch_size'] == 32
        assert validated['defect_weight_multiplier'] == 3.0
        assert 'augmentation' in validated
        assert validated['augmentation']['enabled'] is True
    
    def test_config_override(self):
        """Test that provided values override defaults"""
        config = {
            'batch_size': 64,
            'augmentation': {'enabled': False, 'noise_std': 0.05}
        }
        validated = validate_dataset_config(config)
        
        assert validated['batch_size'] == 64
        assert validated['augmentation']['enabled'] is False
        assert validated['augmentation']['noise_std'] == 0.05
        # Default values should still be present for other keys
        assert validated['sequence_length'] == 300


class TestMemoryEfficientDataset:
    """Test cases for MemoryEfficientSequenceDataset"""
    
    @pytest.fixture
    def temp_h5_file(self):
        """Create temporary HDF5 file with sample data"""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        # Create sample data and save to HDF5
        np.random.seed(42)
        sequences = np.random.randn(50, 100, 5).astype(np.float32)
        
        with h5py.File(tmp_path, 'w') as f:
            f.create_dataset('sequences', data=sequences)
        
        yield tmp_path
        
        # Cleanup
        tmp_path.unlink()
    
    def test_memory_efficient_initialization(self, temp_h5_file):
        """Test memory-efficient dataset initialization"""
        labels = np.random.choice([0, 1], size=30, p=[0.8, 0.2]).astype(np.float32)
        indices = list(range(30))
        
        dataset = MemoryEfficientSequenceDataset(
            temp_h5_file, indices, labels, sequence_length=80
        )
        
        assert len(dataset) == 30
        assert dataset.sequence_length == 80
    
    def test_memory_efficient_getitem(self, temp_h5_file):
        """Test data loading from memory-efficient dataset"""
        labels = np.random.choice([0, 1], size=10, p=[0.8, 0.2]).astype(np.float32)
        indices = list(range(10))
        
        dataset = MemoryEfficientSequenceDataset(
            temp_h5_file, indices, labels, sequence_length=80
        )
        
        seq, label = dataset[0]
        assert isinstance(seq, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert seq.shape == (80, 5)  # Truncated from 100 to 80


def test_integration():
    """Integration test with complete pipeline"""
    # Create sample data
    np.random.seed(42)
    sequences = np.random.randn(50, 60, 4).astype(np.float32)
    labels = np.random.choice([0, 1], size=50, p=[0.7, 0.3]).astype(np.float32)
    
    # Split into train/val
    train_sequences, train_labels = sequences[:40], labels[:40]
    val_sequences, val_labels = sequences[40:], labels[40:]
    
    # Create datasets
    train_dataset = CastingSequenceDataset(train_sequences, train_labels, augment=True)
    val_dataset = CastingSequenceDataset(val_sequences, val_labels, augment=False)
    
    # Create data loaders
    config = validate_dataset_config({
        'batch_size': 8,
        'num_workers': 0,
        'defect_weight_multiplier': 2.0
    })
    
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, config)
    
    # Test batch loading
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    
    assert len(train_batch) == 3  # sequences, labels, attention_masks
    assert len(val_batch) == 3
    
    train_seqs, train_labels_batch, train_masks = train_batch
    assert train_seqs.shape[0] == 8  # Batch size
    assert train_seqs.shape[2] == 4  # Feature dimension
    
    print("Integration test passed successfully!")


if __name__ == "__main__":
    # Run integration test
    test_integration()