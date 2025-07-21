import pytest
import torch
import torch.nn as nn
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
                'input_size': 5,
                'hidden_size': 32,
                'num_layers': 2,
                'bidirectional': True,
                'dropout': 0.2
            },
            'classifier': {
                'hidden_dims': [32, 16],
                'activation': 'relu',
                'dropout': 0.3
            },
            'normalization': {
                'batch_norm': True,
                'layer_norm': False,
                'input_norm': True
            },
            'regularization': {
                'weight_decay': 1e-4,
                'gradient_clip': 1.0
            }
        }
        
        # Standard configuration (same as basic for compatibility)
        self.standard_config = {
            'architecture': {
                'input_size': self.input_size,
                'hidden_size': 32,
                'num_layers': 2,
                'bidirectional': False,
                'dropout': 0.2
            },
            'classifier': {
                'hidden_dims': [32, 16],
                'activation': 'relu',
                'dropout': 0.3
            },
            'normalization': {
                'batch_norm': True,
                'layer_norm': False,
                'input_norm': True
            },
            'regularization': {
                'weight_decay': 1e-4,
                'gradient_clip': 1.0
            }
        }
         
        # Create sample sequences and labels
        if TORCH_AVAILABLE:
            self.sample_sequences = torch.randn(self.batch_size, self.seq_len, self.input_size)
            self.sample_labels = torch.randint(0, 2, (self.batch_size,)).float()
            self.sample_lengths = torch.randint(30, self.seq_len + 1, (self.batch_size,))
        else:
            # Use numpy arrays for mock implementation
            self.sample_sequences = np.random.randn(self.batch_size, self.seq_len, self.input_size)
            self.sample_labels = np.random.randint(0, 2, self.batch_size).astype(np.float32)
            self.sample_lengths = np.random.randint(30, self.seq_len + 1, self.batch_size)
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
        assert model_bidir.hidden_size == 32  # This is the LSTM hidden size, not output size
        assert model_bidir.num_layers == 2
            

    
        #reinitialize the model and test another way
        """Test model initialization with different configurations."""
        # Test standard configuration
        model = SteelDefectLSTM(self.standard_config)
        assert model.input_size == 5
        assert model.hidden_size == 32
        assert model.num_layers == 2
        assert not model.bidirectional
        assert model.dropout == 0.2
        
        # Test bidirectional configuration
        model_bi = SteelDefectLSTM(self.bidirectional_config)
        assert model_bi.bidirectional
        
        # Check parameter count is higher for bidirectional
        params_std = sum(p.numel() for p in model.parameters())
        params_bi = sum(p.numel() for p in model_bi.parameters())
        assert params_bi > params_std
    

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
            
        #reinitialize the model and test another way
        model = SteelDefectLSTM(self.standard_config)
        model.eval()
        
        # Test without lengths
        outputs = model(self.sample_sequences)
        assert outputs.shape == (self.batch_size, 1)
        
        # Test with lengths
        outputs_with_lengths = model(self.sample_sequences, self.sample_lengths)
        assert outputs_with_lengths.shape == (self.batch_size, 1)
        
        # Outputs should be different when using lengths
        assert not torch.allclose(outputs, outputs_with_lengths, atol=1e-6)
    
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
        #reinitialize the model and test another way
        """Test bidirectional LSTM processing."""
        model_std = SteelDefectLSTM(self.standard_config)
        model_bi = SteelDefectLSTM(self.bidirectional_config)
        
        model_std.eval()
        model_bi.eval()
        
        outputs_std = model_std(self.sample_sequences)
        outputs_bi = model_bi(self.sample_sequences)
        
        # Outputs should be different between unidirectional and bidirectional
        assert not torch.allclose(outputs_std, outputs_bi, atol=1e-6)
        
        # Both should produce valid outputs
        assert outputs_std.shape == (self.batch_size, 1)
        assert outputs_bi.shape == (self.batch_size, 1)
    
    def test_variable_sequence_lengths(self):
        """Test handling of variable sequence lengths."""
        model = SteelDefectLSTM(self.bidirectional_config)
        model.eval()
        
        if TORCH_AVAILABLE:
            import torch
            # Create sequences with different lengths
            max_len = 60
            variable_sequences = torch.randn(self.batch_size, max_len, self.input_size)
            lengths = torch.randint(20, max_len + 1, (self.batch_size,))
            
            # Pad sequences according to lengths
            for i, length in enumerate(lengths):
                if length < max_len:
                    variable_sequences[i, length:] = 0
            
            outputs = model(variable_sequences, lengths)
            assert outputs.shape == (self.batch_size, 1)
            assert not torch.isnan(outputs).any()
            assert not torch.isinf(outputs).any()
            
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
        #reinitialize the model and test another way

        model = SteelDefectLSTM(self.bidirectional_config)
        
        batch_size = 10
        hidden, cell = model.init_hidden(batch_size)
        
        expected_layers = model.num_layers * (2 if model.bidirectional else 1)
        
        assert hidden.shape == (expected_layers, batch_size, model.hidden_size)
        assert cell.shape == (expected_layers, batch_size, model.hidden_size)
        assert torch.allclose(hidden, torch.zeros_like(hidden))
        assert torch.allclose(cell, torch.zeros_like(cell))
    

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
            # The shape depends on the actual sequence length used
            assert attention_weights.shape[0] >= 1  # Should have at least one sample
            assert len(attention_weights.shape) == 2  # Should be 2D (batch_size, sequence_length)
    
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
        #reinitialize the model and test another way

        # Test with batch norm
        config_bn = self.standard_config.copy()
        config_bn['normalization'] = {'batch_norm': True, 'layer_norm': False, 'input_norm': True}
        model_bn = SteelDefectLSTM(config_bn)
        
        # Test with layer norm
        config_ln = self.standard_config.copy()
        config_ln['normalization'] = {'batch_norm': False, 'layer_norm': True, 'input_norm': True}
        model_ln = SteelDefectLSTM(config_ln)
        
        # Test with no norm
        config_no_norm = self.standard_config.copy()
        config_no_norm['normalization'] = {'batch_norm': False, 'layer_norm': False, 'input_norm': False}
        model_no_norm = SteelDefectLSTM(config_no_norm)
        
        # All should work
        out_bn = model_bn(self.sample_sequences)
        out_ln = model_ln(self.sample_sequences)
        out_no_norm = model_no_norm(self.sample_sequences)
        
        assert out_bn.shape == (self.batch_size, 1)
        assert out_ln.shape == (self.batch_size, 1)
        assert out_no_norm.shape == (self.batch_size, 1)
        
        # Outputs should be different due to normalization
        assert not torch.allclose(out_bn, out_no_norm, atol=1e-6)    
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
    
    def setup_method(self):
        """Setup test fixtures."""
        np.random.seed(42)
        
        # Test data
        self.batch_size = 8
        self.seq_len = 50
        self.input_size = 5
        
        # Standard configuration for tests
        self.standard_config = {
            'architecture': {
                'input_size': self.input_size,
                'hidden_size': 32,
                'num_layers': 2,
                'bidirectional': False,
                'dropout': 0.2
            },
            'classifier': {
                'hidden_dims': [32, 16],
                'activation': 'relu',
                'dropout': 0.3
            },
            'normalization': {
                'batch_norm': True,
                'layer_norm': False,
                'input_norm': True
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
                'hidden_size': 32,
                'num_layers': 2,
                'bidirectional': True,
                'dropout': 0.2
            },
            'classifier': {
                'hidden_dims': [32, 16],
                'activation': 'relu',
                'dropout': 0.3
            },
            'normalization': {
                'batch_norm': True,
                'layer_norm': False,
                'input_norm': True
            },
            'regularization': {
                'weight_decay': 1e-4,
                'gradient_clip': 1.0
            }
        }
        
        # Create sample sequences and labels
        if TORCH_AVAILABLE:
            self.sample_sequences = torch.randn(self.batch_size, self.seq_len, self.input_size)
            self.sample_labels = torch.randint(0, 2, (self.batch_size,)).float()
            self.sample_lengths = torch.randint(30, self.seq_len + 1, (self.batch_size,))
        else:
            # Use numpy arrays for mock implementation
            self.sample_sequences = np.random.randn(self.batch_size, self.seq_len, self.input_size)
            self.sample_labels = np.random.randint(0, 2, self.batch_size).astype(np.float32)
            self.sample_lengths = np.random.randint(30, self.seq_len + 1, self.batch_size)
    
    
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

    
    def test_gradient_computation(self):
        """Test gradient computation and backpropagation."""
        model = SteelDefectLSTM(self.standard_config)
        criterion = nn.BCEWithLogitsLoss()
        
        # Forward pass
        outputs = model(self.sample_sequences)
        loss = criterion(outputs.squeeze(), self.sample_labels)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are computed
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    

    
    def test_dropout_functionality(self):
        """Test dropout during training and evaluation."""
        model = SteelDefectLSTM(self.standard_config)
        
        # Training mode - dropout active
        model.train()
        outputs_train_1 = model(self.sample_sequences)
        outputs_train_2 = model(self.sample_sequences)
        
        # Outputs should be different due to dropout (with high probability)
        # Note: Small chance they could be same, so we test multiple times
        different_outputs = False
        for _ in range(5):
            out1 = model(self.sample_sequences)
            out2 = model(self.sample_sequences)
            if not torch.allclose(out1, out2, atol=1e-6):
                different_outputs = True
                break
        assert different_outputs, "Dropout should cause different outputs in training mode"
        
        # Evaluation mode - dropout disabled
        model.eval()
        outputs_eval_1 = model(self.sample_sequences)
        outputs_eval_2 = model(self.sample_sequences)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(outputs_eval_1, outputs_eval_2, atol=1e-6)
    
    def test_gpu_compatibility(self):
        """Test GPU compatibility if CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device('cuda')
        model = SteelDefectLSTM(self.standard_config)
        model = model.to(device)
        
        sequences_gpu = self.sample_sequences.to(device)
        outputs = model(sequences_gpu)
        
        assert outputs.device == device
        assert outputs.shape == (self.batch_size, 1)
    
    def test_model_serialization(self):
        """Test model save and load functionality."""
        model = SteelDefectLSTM(self.standard_config)
        
        # Generate some outputs before saving
        model.eval()
        original_outputs = model(self.sample_sequences)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, 'test_lstm_model.pth')
            
            # Save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': self.standard_config
            }, model_path)
            
            # Load model
            checkpoint = torch.load(model_path, map_location='cpu')
            new_model = SteelDefectLSTM(checkpoint['config'])
            new_model.load_state_dict(checkpoint['model_state_dict'])
            new_model.eval()
            
            # Test that outputs are identical
            loaded_outputs = new_model(self.sample_sequences)
            assert torch.allclose(original_outputs, loaded_outputs, atol=1e-6)
    
    def test_attention_weights(self):
        """Test attention weights computation."""
        model = SteelDefectLSTM(self.bidirectional_config)
        model.eval()
        
        # Test without lengths
        attention = model.get_attention_weights(self.sample_sequences)
        assert attention.shape == (self.batch_size, self.seq_len)
        
        # Test that attention weights sum to 1
        attention_sums = attention.sum(dim=1)
        assert torch.allclose(attention_sums, torch.ones_like(attention_sums), atol=1e-6)
        
        # Test with lengths
        attention_with_lengths = model.get_attention_weights(self.sample_sequences, self.sample_lengths)
        assert attention_with_lengths.shape == (self.batch_size, self.seq_len)
        
        # Check that attention is zero for padded positions
        for i, length in enumerate(self.sample_lengths):
            if length < self.seq_len:
                assert torch.allclose(
                    attention_with_lengths[i, length:], 
                    torch.zeros(self.seq_len - length),
                    atol=1e-6
                )
    
    def test_freeze_layers(self):
        """Test layer freezing functionality."""
        model = SteelDefectLSTM(self.standard_config)
        
        # Initially all parameters should require gradients
        lstm_params = list(model.lstm.parameters())
        classifier_params = list(model.classifier.parameters())
        
        assert all(p.requires_grad for p in lstm_params)
        assert all(p.requires_grad for p in classifier_params)
        
        # Freeze LSTM layers
        model.freeze_layers(freeze_lstm=True, freeze_classifier=False)
        
        assert all(not p.requires_grad for p in lstm_params)
        assert all(p.requires_grad for p in classifier_params)
        
        # Freeze classifier layers
        model.freeze_layers(freeze_lstm=False, freeze_classifier=True)
        
        # LSTM should still be frozen, classifier should now be frozen too
        assert all(not p.requires_grad for p in lstm_params)
        assert all(not p.requires_grad for p in classifier_params)
    
    def test_get_layer_outputs(self):
        """Test layer output extraction."""
        model = SteelDefectLSTM(self.bidirectional_config)
        model.eval()
        
        layer_outputs = model.get_layer_outputs(self.sample_sequences)
        
        expected_keys = ['input_normalized', 'lstm_output', 'lstm_hidden', 'lstm_cell', 
                        'final_features', 'normalized_features']
        
        for key in expected_keys:
            assert key in layer_outputs, f"Missing key: {key}"
        
        # Check shapes
        assert layer_outputs['input_normalized'].shape == self.sample_sequences.shape
        assert layer_outputs['lstm_output'].shape[0] == self.batch_size
        assert layer_outputs['final_features'].shape == (self.batch_size, model.hidden_size * 2)  # bidirectional
    
    def test_model_info(self):
        """Test model information extraction."""
        model = SteelDefectLSTM(self.standard_config)
        
        info = model.get_model_info()
        
        required_keys = ['total_parameters', 'trainable_parameters', 'model_size_mb', 
                        'architecture', 'regularization', 'normalization']
        
        for key in required_keys:
            assert key in info, f"Missing info key: {key}"
        
        assert info['total_parameters'] > 0
        assert info['trainable_parameters'] == info['total_parameters']  # No frozen layers
        assert info['model_size_mb'] > 0
        assert info['architecture']['input_size'] == 5
        assert info['architecture']['hidden_size'] == 32
    
    def test_different_activations(self):
        """Test different activation functions in classifier."""
        activations = ['relu', 'tanh', 'sigmoid']
        
        for activation in activations:
            config = self.standard_config.copy()
            config['classifier']['activation'] = activation
            
            model = SteelDefectLSTM(config)
            outputs = model(self.sample_sequences)
            
            assert outputs.shape == (self.batch_size, 1)
            assert not torch.isnan(outputs).any()
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with minimum configuration
        minimal_config = {
            'architecture': {'input_size': 5}
        }
        model = SteelDefectLSTM(minimal_config)
        outputs = model(self.sample_sequences)
        assert outputs.shape == (self.batch_size, 1)
        
        # Test with single layer
        single_layer_config = self.standard_config.copy()
        single_layer_config['architecture']['num_layers'] = 1
        single_layer_config['architecture']['dropout'] = 0.0  # No dropout for single layer
        
        model_single = SteelDefectLSTM(single_layer_config)
        outputs_single = model_single(self.sample_sequences)
        assert outputs_single.shape == (self.batch_size, 1)
    
    def test_casting_sequence_dataset(self):
        """Test the CastingSequenceDataset class."""
        sequences = np.random.randn(100, 50, 5)
        labels = np.random.binomial(1, 0.15, 100)
        
        dataset = CastingSequenceDataset(sequences, labels)
        
        assert len(dataset) == 100
        
        # Test getitem
        seq, label = dataset[0]
        assert isinstance(seq, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert seq.shape == (50, 5)
        assert label.shape == ()
        
        # Test DataLoader compatibility
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        for batch_seq, batch_labels in dataloader:
            assert batch_seq.shape[0] <= 16
            assert batch_labels.shape[0] <= 16
            break  # Just test first batch
