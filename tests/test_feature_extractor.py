import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from features.feature_extractor import SequenceFeatureExtractor


class TestSequenceFeatureExtractor:
    """Test suite for SequenceFeatureExtractor"""
    
    def setup_method(self):
        """Setup test fixtures."""
        np.random.seed(42)
        
        # Create sample time series data
        self.sample_data = pd.DataFrame({
            'temperature': np.random.normal(1520, 10, 100),
            'pressure': np.random.normal(150, 5, 100),
            'flow_rate': np.random.normal(200, 15, 100),
            'vibration': np.random.normal(1.2, 0.05, 100),
            'power_consumption': np.random.normal(25, 3, 100),
            'label': np.random.choice([0, 1], 100)
        })
        
        # Create shorter data for padding tests
        self.short_data = pd.DataFrame({
            'temperature': np.random.normal(1520, 10, 20),
            'pressure': np.random.normal(150, 5, 20),
            'flow_rate': np.random.normal(200, 15, 20),
            'vibration': np.random.normal(1.2, 0.05, 20),
            'power_consumption': np.random.normal(25, 3, 20)
        })
        
        self.extractor = SequenceFeatureExtractor(
            sequence_length=30,
            stride=5,
            augmentation_config={
                'noise_std': 0.01,
                'time_warp_max': 0.1,
                'augmentation_ratio': 0.2
            }
        )
    
    def test_extract_sequences(self):
        """Test sliding window sequence extraction."""
        feature_data = self.sample_data.drop('label', axis=1)
        sequences = self.extractor.extract_sequences(feature_data)
        
        # Expected number of sequences: (100 - 30) // 5 + 1 = 15
        expected_sequences = (100 - 30) // 5 + 1
        assert sequences.shape[0] == expected_sequences
        assert sequences.shape[1] == 30  # sequence_length
        assert sequences.shape[2] == 5   # number of features
        
        # Test with empty data
        empty_sequences = self.extractor.extract_sequences(pd.DataFrame())
        assert empty_sequences.size == 0
        
        # Test with insufficient data
        short_sequences = self.extractor.extract_sequences(self.short_data)
        assert short_sequences.size == 0  # Too short for sequence_length=30
    
    def test_normalize_sequences(self):
        """Test Z-score normalization per feature channel."""
        feature_data = self.sample_data.drop('label', axis=1)
        sequences = self.extractor.extract_sequences(feature_data)
        
        normalized = self.extractor.normalize_sequences(sequences)
        
        # Shape should be preserved
        assert normalized.shape == sequences.shape
        
        # After normalization, mean should be close to 0 and std close to 1
        # (across all sequences and time steps for each feature)
        reshaped = normalized.reshape(-1, normalized.shape[2])
        means = np.mean(reshaped, axis=0)
        stds = np.std(reshaped, axis=0)
        
        assert np.allclose(means, 0, atol=1e-10)
        assert np.allclose(stds, 1, atol=1e-10)
        
        # Test with empty sequences
        empty_normalized = self.extractor.normalize_sequences(np.array([]))
        assert empty_normalized.size == 0
    
    def test_pad_sequences(self):
        """Test sequence padding and truncation."""
        # Create variable length sequences
        seq1 = np.random.random((20, 5))  # Shorter than sequence_length
        seq2 = np.random.random((30, 5))  # Exactly sequence_length
        seq3 = np.random.random((50, 5))  # Longer than sequence_length
        
        sequences = [seq1, seq2, seq3]
        padded = self.extractor.pad_sequences(sequences)
        
        # All should be padded/truncated to sequence_length
        assert padded.shape[0] == 3
        assert padded.shape[1] == 30  # sequence_length
        assert padded.shape[2] == 5   # features
        
        # Check that seq1 was padded with zeros
        assert np.allclose(padded[0, 20:, :], 0)  # Last 10 timesteps should be zero
        
        # Check that seq2 is unchanged
        assert np.allclose(padded[1], seq2)
        
        # Check that seq3 was truncated
        assert np.allclose(padded[2], seq3[:30])
        
        # Test with empty list
        empty_padded = self.extractor.pad_sequences([])
        assert empty_padded.size == 0
    
    def test_augment_sequences(self):
        """Test data augmentation with noise injection and time warping."""
        feature_data = self.sample_data.drop('label', axis=1)
        sequences = self.extractor.extract_sequences(feature_data)
        
        augmented = self.extractor.augment_sequences(sequences)
        
        # Should have more sequences after augmentation
        assert len(augmented) > len(sequences)
        
        # Original sequences should be included
        assert np.allclose(augmented[:len(sequences)], sequences)
        
        # Test with empty sequences
        empty_augmented = self.extractor.augment_sequences(np.array([]))
        assert empty_augmented.size == 0
        
        # Test with zero augmentation ratio
        extractor_no_aug = SequenceFeatureExtractor(
            sequence_length=30,
            augmentation_config={'augmentation_ratio': 0.0}
        )
        no_augmented = extractor_no_aug.augment_sequences(sequences)
        assert np.allclose(no_augmented, sequences)
    
    def test_split_sequences(self):
        """Test temporal-aware train/validation/test split."""
        feature_data = self.sample_data.drop('label', axis=1)
        sequences = self.extractor.extract_sequences(feature_data)
        labels = np.random.choice([0, 1], len(sequences))
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.extractor.split_sequences(
            sequences, labels, split_ratios=(0.6, 0.2, 0.2)
        )
        
        # Check that splits sum to original length
        assert len(X_train) + len(X_val) + len(X_test) == len(sequences)
        assert len(y_train) + len(y_val) + len(y_test) == len(labels)
        
        # Check that temporal order is preserved
        expected_train_end = int(len(sequences) * 0.6)
        expected_val_end = int(len(sequences) * 0.8)
        
        assert np.allclose(X_train, sequences[:expected_train_end])
        assert np.allclose(X_val, sequences[expected_train_end:expected_val_end])
        assert np.allclose(X_test, sequences[expected_val_end:])
        
        # Test with invalid split ratios
        with pytest.raises(ValueError):
            self.extractor.split_sequences(sequences, labels, (0.5, 0.3, 0.3))
        
        # Test with empty sequences
        empty_splits = self.extractor.split_sequences(np.array([]), np.array([]))
        assert all(split.size == 0 for split in empty_splits)
    
    def test_prepare_sequences(self):
        """Test end-to-end sequence preparation."""
        sequences, seq_labels = self.extractor.prepare_sequences(self.sample_data)
        
        # Should extract sequences and corresponding labels
        assert sequences.ndim == 3
        assert len(seq_labels) == len(sequences)
        
        # Check that labels are taken from the end of each sequence window
        expected_sequences = (100 - 30) // 5 + 1
        assert len(sequences) == expected_sequences
        
        # Test with data without explicit label column
        data_no_label = self.sample_data.drop('label', axis=1)
        data_no_label['target'] = self.sample_data['label']
        
        sequences2, labels2 = self.extractor.prepare_sequences(data_no_label)
        assert len(sequences2) == len(labels2)
        
        # Test with empty data
        empty_seq, empty_labels = self.extractor.prepare_sequences(pd.DataFrame())
        assert empty_seq.size == 0
        assert empty_labels.size == 0
    
    def test_extract_sliding_windows(self):
        """Test sliding window extraction method."""
        windows = self.extractor.extract_sliding_windows(
            self.sample_data.drop('label', axis=1),
            window_size=20,
            stride=10
        )
        
        # Expected number of windows: (100 - 20) // 10 + 1 = 9
        expected_windows = (100 - 20) // 10 + 1
        assert len(windows) == expected_windows
        assert windows[0].shape == (20, 5)  # window_size x features
        
        # Test with default parameters
        default_windows = self.extractor.extract_sliding_windows(
            self.sample_data.drop('label', axis=1)
        )
        expected_default = (100 - 30) // 1 + 1  # default stride=1, window_size=sequence_length
        assert len(default_windows) == expected_default
        
        # Test with empty data
        empty_windows = self.extractor.extract_sliding_windows(pd.DataFrame())
        assert len(empty_windows) == 0
    
    def test_fit_normalizer(self):
        """Test normalizer fitting."""
        feature_data = self.sample_data.drop('label', axis=1)
        sequences = self.extractor.extract_sequences(feature_data)
        
        # Test fitting
        assert not self.extractor.fitted
        self.extractor.fit_normalizer(sequences)
        assert self.extractor.fitted
        
        # Test with empty sequences
        extractor2 = SequenceFeatureExtractor()
        extractor2.fit_normalizer(np.array([]))
        assert not extractor2.fitted
    
    def test_create_sequences_from_cast(self):
        """Test sequence creation from single cast data."""
        cast_data = self.sample_data.drop('label', axis=1)
        sequences = self.extractor.create_sequences_from_cast(cast_data)
        
        # Should return multiple sequences from the cast
        assert sequences.ndim == 3
        assert sequences.shape[1] == 30  # sequence_length
        assert sequences.shape[2] == 5   # features
        
        # Test with short cast data (requires padding)
        short_sequences = self.extractor.create_sequences_from_cast(self.short_data)
        assert short_sequences.shape == (20, 5)  # Should be padded
        
        # Test with empty data
        empty_cast = self.extractor.create_sequences_from_cast(pd.DataFrame())
        assert empty_cast.size == 0
    
    def test_augmentation_config(self):
        """Test different augmentation configurations."""
        # Test with MinMax scaling
        extractor_minmax = SequenceFeatureExtractor(
            sequence_length=30,
            augmentation_config={'min_max_scaling': True}
        )
        assert extractor_minmax.use_minmax
        
        # Test with different noise levels
        extractor_noise = SequenceFeatureExtractor(
            sequence_length=30,
            augmentation_config={'noise_std': 0.1}
        )
        assert extractor_noise.noise_std == 0.1
        
        # Test with different time warp settings
        extractor_warp = SequenceFeatureExtractor(
            sequence_length=30,
            augmentation_config={'time_warp_max': 0.2}
        )
        assert extractor_warp.time_warp_max == 0.2
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with single-row DataFrame
        single_row = self.sample_data.iloc[:1]
        sequences = self.extractor.extract_sequences(single_row.drop('label', axis=1))
        assert sequences.size == 0  # Too short
        
        # Test with data containing NaN
        nan_data = self.sample_data.copy()
        nan_data.iloc[10:20, 0] = np.nan
        sequences = self.extractor.extract_sequences(nan_data.drop('label', axis=1))
        # Should still work but contain NaN values
        assert sequences.shape[2] == 5
    
    def test_integration(self):
        """Test complete pipeline integration."""
        # Full pipeline test
        sequences, labels = self.extractor.prepare_sequences(self.sample_data)
        
        # Normalize
        normalized = self.extractor.normalize_sequences(sequences)
        
        # Augment
        augmented = self.extractor.augment_sequences(normalized)
        
        # Split
        X_train, X_val, X_test, y_train, y_val, y_test = self.extractor.split_sequences(
            augmented, np.random.choice([0, 1], len(augmented))
        )
        
        # Verify final shapes are reasonable
        assert X_train.ndim == 3
        assert X_val.ndim == 3
        assert X_test.ndim == 3
        assert len(y_train) == len(X_train)
        assert len(y_val) == len(X_val)
        assert len(y_test) == len(X_test)