import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.lstm_model import (
    SteelDefectLSTM, CastingSequenceDataset, LSTMModelVariants,
    LSTMPerformanceTracker, load_lstm_config, create_default_lstm_config,
    TORCH_AVAILABLE
)
from models.model_config import ModelConfig


class TestSteelDefectLSTM:
    """Comprehensive test suite for SteelDefectLSTM model"""
    
    def setup_method(self):
        """Setup test fixtures."""
        np.random.seed(42)
        
        # Sample data
        self.batch_size = 16
        self.seq_len = 100
        self.input_size = 5
        
        self.sample_sequences = np.random.normal(0, 1, (self.batch_size, self.seq_len, self.input_size))
        self.sample_labels = np.random.binomial(1, 0.15, self.batch_size)
        self.sample_sequence_lengths = np.random.randint(50, self.seq_len + 1, self.batch_size)
        
        # Basic configuration
        self.basic_config = {
            'architecture': {
                'input_size': self.input_size,
                'hidden_size': 32,
                'num_layers': 1,
                'bidirectional': False,
                'dropout': 0.1
            },
            'classifier': {
                'hidden_dims': [16],
                'activation': 'relu',
                'dropout': 0.2
            },
            'normalization': {
                'batch_norm': False,
                'layer_norm': False,
                'input_norm': False
            },
            'regularization': {
                'weight_decay': 1e-4,
                'gradient_clip': 1.0
            }
        }
        
        # Bidirectional configuration
        self.bidirectional_config = {
            'architecture': {
                'input_size': self.input_size,
                'hidden_size': 64,
                'num_layers': 2,
                'bidirectional': True,
                'dropout': 0.1
            },
            'classifier': {
                'hidden_dims': [16],
                'activation': 'relu',
                'dropout': 0.2
            },
            'normalization': {
                'batch_norm': False,
                'layer_norm': False,
                'input_norm': False
            },
            'regularization': {
                'weight_decay': 1e-4,
                'gradient_clip': 1.0
            }
        }
    
    def test_model_initialization(self):
        """Test model initialization with different configurations."""
        # Test basic initialization
        model = SteelDefectLSTM(self.basic_config)
        assert model.input_size == self.input_size
        assert model.hidden_size == 32  # Should match the config we set
        assert model.num_layers == 1
        assert not model.bidirectional
        
        # Test bidirectional initialization
        model_bidir = SteelDefectLSTM(self.bidirectional_config)
        assert model_bidir.bidirectional
        assert model_bidir.hidden_size == 64
        assert model_bidir.num_layers == 2
    
    def test_forward_pass(self):
        """Test forward pass with different input configurations."""
        model = SteelDefectLSTM(self.basic_config)
        
        # Test basic forward pass
        if TORCH_AVAILABLE:
            import torch
            x = torch.FloatTensor(self.sample_sequences)
        else:
            # Use numpy array for mock implementation
            x = self.sample_sequences
        
        output = model.forward(x)
        
        # Check output shape - should be [batch_size, 1] for binary classification
        if hasattr(output, 'shape'):
            expected_shape = (self.batch_size, 1)
            assert output.shape == expected_shape or output.shape == expected_shape[0:1]
    
    def test_bidirectional_processing(self):
        """Test bidirectional LSTM processing."""
        model = SteelDefectLSTM(self.bidirectional_config)
        
        if TORCH_AVAILABLE:
            import torch
            x = torch.FloatTensor(self.sample_sequences)
        else:
            x = self.sample_sequences
        
        output = model.forward(x)
        
        # Bidirectional model should handle input correctly
        assert output is not None
        
        # Test that bidirectional flag affects internal structure
        assert model.bidirectional == True
    
    def test_variable_sequence_lengths(self):
        """Test handling of variable sequence lengths."""
        model = SteelDefectLSTM(self.bidirectional_config)
        
        if TORCH_AVAILABLE:
            import torch
            x = torch.FloatTensor(self.sample_sequences)
            sequence_lengths = self.sample_sequence_lengths.tolist()
        else:
            x = self.sample_sequences
            sequence_lengths = self.sample_sequence_lengths.tolist()
        
        # Test forward pass with sequence lengths
        output = model.forward(x, sequence_lengths=sequence_lengths)
        assert output is not None
    
    def test_hidden_state_initialization(self):
        """Test hidden state initialization."""
        model = SteelDefectLSTM(self.basic_config)
        
        hidden, cell = model.init_hidden(self.batch_size)
        
        # Check hidden state shapes
        expected_hidden_shape = (model.num_layers, self.batch_size, model.hidden_size)
        if hasattr(hidden, 'shape'):
            assert hidden.shape == expected_hidden_shape
            assert cell.shape == expected_hidden_shape
        
        # Test with bidirectional model
        model_bidir = SteelDefectLSTM(self.bidirectional_config)
        hidden_bidir, cell_bidir = model_bidir.init_hidden(self.batch_size)
        
        expected_bidir_shape = (model_bidir.num_layers * 2, self.batch_size, model_bidir.hidden_size)
        if hasattr(hidden_bidir, 'shape'):
            assert hidden_bidir.shape == expected_bidir_shape
            assert cell_bidir.shape == expected_bidir_shape
    
    def test_attention_weights(self):
        """Test attention weight computation."""
        model = SteelDefectLSTM(self.bidirectional_config)
        
        if TORCH_AVAILABLE:
            import torch
            x = torch.FloatTensor(self.sample_sequences)
        else:
            x = self.sample_sequences
        
        # Forward pass to compute attention weights
        output = model.forward(x)
        
        # Get attention weights
        attention_weights = model.get_attention_weights()
        
        if attention_weights is not None:
            assert isinstance(attention_weights, np.ndarray)
            # Should have attention weights for each sample and time step
            assert attention_weights.shape[0] == self.batch_size or attention_weights.shape == (1, 10)
    
    def test_layer_freezing(self):
        """Test layer freezing functionality."""
        model = SteelDefectLSTM(self.bidirectional_config)
        
        # Test freezing LSTM layers
        model.freeze_layers(['lstm'])
        
        # In mock implementation, this should complete without error
        assert True  # Test passes if no exception is raised
    
    def test_layer_outputs(self):
        """Test layer output extraction."""
        model = SteelDefectLSTM(self.bidirectional_config)
        
        if TORCH_AVAILABLE:
            import torch
            x = torch.FloatTensor(self.sample_sequences)
        else:
            x = self.sample_sequences
        
        # Test extracting LSTM outputs
        lstm_outputs = model.get_layer_outputs(x, layer_name='lstm')
        assert lstm_outputs is not None
    
    def test_normalization_layers(self):
        """Test different normalization options."""
        # Test with batch normalization
        config_with_batch_norm = self.basic_config.copy()
        config_with_batch_norm['normalization']['batch_norm'] = True
        config_with_batch_norm['normalization']['input_norm'] = True
        
        model_batch_norm = SteelDefectLSTM(config_with_batch_norm)
        assert model_batch_norm.use_batch_norm
        assert model_batch_norm.use_input_norm
        
        # Test with layer normalization
        config_with_layer_norm = self.basic_config.copy()
        config_with_layer_norm['normalization']['layer_norm'] = True
        
        model_layer_norm = SteelDefectLSTM(config_with_layer_norm)
        assert model_layer_norm.use_layer_norm
    
    def test_dropout_functionality(self):
        """Test dropout configuration."""
        config_with_dropout = self.basic_config.copy()
        config_with_dropout['architecture']['dropout'] = 0.5
        config_with_dropout['classifier']['dropout'] = 0.3
        
        model = SteelDefectLSTM(config_with_dropout)
        assert model.dropout == 0.5
        assert model.classifier_dropout == 0.3
    
    def test_model_variants(self):
        """Test different model variant configurations."""
        # Test standard LSTM
        standard_config = LSTMModelVariants.standard_lstm()
        model_standard = SteelDefectLSTM(standard_config)
        assert not model_standard.bidirectional
        
        # Test bidirectional LSTM
        bidir_config = LSTMModelVariants.bidirectional_lstm()
        model_bidir = SteelDefectLSTM(bidir_config)
        assert model_bidir.bidirectional
        
        # Test deep LSTM
        deep_config = LSTMModelVariants.deep_lstm()
        model_deep = SteelDefectLSTM(deep_config)
        assert model_deep.num_layers == 4
        assert model_deep.hidden_size == 128
        
        # Test lightweight LSTM
        light_config = LSTMModelVariants.lightweight_lstm()
        model_light = SteelDefectLSTM(light_config)
        assert model_light.num_layers == 1
        assert model_light.hidden_size == 32


class TestCastingSequenceDataset:
    """Test suite for CastingSequenceDataset"""
    
    def setup_method(self):
        """Setup test fixtures."""
        np.random.seed(42)
        self.batch_size = 32
        self.seq_len = 100
        self.input_size = 5
        
        self.sequences = np.random.normal(0, 1, (self.batch_size, self.seq_len, self.input_size))
        self.labels = np.random.binomial(1, 0.15, self.batch_size)
        self.sequence_lengths = np.random.randint(50, self.seq_len + 1, self.batch_size)
    
    def test_dataset_creation(self):
        """Test dataset creation and basic functionality."""
        dataset = CastingSequenceDataset(self.sequences, self.labels)
        
        assert len(dataset) == self.batch_size
        
        # Test item retrieval
        seq, label = dataset[0]
        if hasattr(seq, 'shape'):
            assert seq.shape == (self.seq_len, self.input_size)
        assert isinstance(label, (int, float, np.number)) or hasattr(label, 'item')
    
    def test_variable_length_dataset(self):
        """Test dataset with variable sequence lengths."""
        dataset = CastingSequenceDataset(
            self.sequences, self.labels, sequence_lengths=self.sequence_lengths.tolist()
        )
        
        # Test item retrieval with sequence length
        seq, label, seq_len = dataset[0]
        assert seq_len == self.sequence_lengths[0]
    
    def test_sequence_statistics(self):
        """Test sequence length statistics."""
        dataset = CastingSequenceDataset(
            self.sequences, self.labels, sequence_lengths=self.sequence_lengths.tolist()
        )
        
        stats = dataset.get_sequence_stats()
        
        assert 'min_length' in stats
        assert 'max_length' in stats
        assert 'mean_length' in stats
        assert 'std_length' in stats
        
        assert stats['min_length'] == np.min(self.sequence_lengths)
        assert stats['max_length'] == np.max(self.sequence_lengths)


class TestModelConfiguration:
    """Test suite for LSTM model configuration"""
    
    def test_default_config_creation(self):
        """Test creation of default LSTM configuration."""
        config = create_default_lstm_config()
        
        # Check required sections
        assert 'architecture' in config
        assert 'classifier' in config
        assert 'normalization' in config
        assert 'regularization' in config
        
        # Check architecture parameters
        arch = config['architecture']
        assert arch['input_size'] == 5
        assert arch['hidden_size'] == 64
        assert arch['bidirectional'] == True
        
        # Check classifier parameters
        classifier = config['classifier']
        assert classifier['hidden_dims'] == [32, 16]
        assert classifier['activation'] == 'relu'
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Create a test config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            test_config = {
                'lstm_model': create_default_lstm_config()
            }
            yaml.dump(test_config, f)
            config_path = f.name
        
        try:
            # Test loading and validation
            config_manager = ModelConfig(config_path)
            assert config_manager.config is not None
            
            # Test LSTM-specific parameter extraction
            lstm_config = config_manager.get_lstm_config()
            assert 'architecture' in lstm_config
            
            arch_params = config_manager.get_lstm_architecture_params()
            assert 'input_size' in arch_params
            
        finally:
            os.unlink(config_path)
    
    def test_invalid_config_validation(self):
        """Test validation of invalid configurations."""
        # Test with invalid hidden size
        invalid_config = create_default_lstm_config()
        invalid_config['architecture']['hidden_size'] = -1
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump({'lstm_model': invalid_config}, f)
            config_path = f.name
        
        try:
            config_manager = ModelConfig()
            # This should fail validation
            is_valid = config_manager._validate_lstm_config(invalid_config)
            assert not is_valid
            
        finally:
            os.unlink(config_path)


class TestLSTMPerformanceTracker:
    """Test suite for LSTM performance tracking"""
    
    def test_performance_tracking(self):
        """Test performance metrics tracking."""
        tracker = LSTMPerformanceTracker()
        
        # Log some training metrics
        tracker.log_training_metrics(1, {'loss': 0.5, 'auc_roc': 0.75})
        tracker.log_validation_metrics(1, {'loss': 0.6, 'auc_roc': 0.80})
        
        # Log inference metrics
        tracker.log_inference_time(32, 0.1)  # 100ms for batch of 32
        tracker.log_memory_usage(2048)  # 2GB memory usage
        
        # Get performance summary
        summary = tracker.get_performance_summary()
        
        assert 'latest_training_metrics' in summary
        assert 'latest_validation_metrics' in summary
        assert 'avg_inference_time_per_sample' in summary
        assert 'peak_memory_usage_mb' in summary
        
        # Check target achievement
        assert 'meets_auc_target' in summary
        assert 'meets_inference_target' in summary
        assert 'meets_memory_target' in summary


class TestIntegration:
    """Integration tests for LSTM components"""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from config to model training."""
        # Create default configuration
        config = create_default_lstm_config()
        
        # Create model
        model = SteelDefectLSTM(config)
        
        # Create dataset
        sequences = np.random.normal(0, 1, (16, 50, 5))
        labels = np.random.binomial(1, 0.15, 16)
        dataset = CastingSequenceDataset(sequences, labels)
        
        # Test forward pass
        if TORCH_AVAILABLE:
            import torch
            x = torch.FloatTensor(sequences)
        else:
            x = sequences
        
        output = model.forward(x)
        assert output is not None
        
        # Test performance tracking
        tracker = LSTMPerformanceTracker()
        tracker.log_training_metrics(1, {'loss': 0.5, 'auc_roc': 0.85})
        
        summary = tracker.get_performance_summary()
        assert summary is not None
    
    def test_model_serialization_compatibility(self):
        """Test model compatibility with serialization requirements."""
        config = create_default_lstm_config()
        model = SteelDefectLSTM(config)
        
        # Test state dict access (required for saving/loading)
        if TORCH_AVAILABLE:
            state_dict = model.state_dict()
            assert isinstance(state_dict, dict)
        
        # Test parameter access
        params = list(model.parameters())
        assert isinstance(params, list)