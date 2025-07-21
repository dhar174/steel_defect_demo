import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
from pathlib import Path
import yaml

import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.lstm_model import SteelDefectLSTM, CastingSequenceDataset


class TestSteelDefectLSTM:
    """Comprehensive test suite for enhanced LSTM model"""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Standard configuration for testing
        self.standard_config = {
            'architecture': {
                'input_size': 5,
                'hidden_size': 32,
                'num_layers': 2,
                'bidirectional': False,
                'dropout': 0.2
            },
            'classifier': {
                'hidden_dims': [16, 8],
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
        
        # Test data
        self.batch_size = 8
        self.seq_len = 50
        self.input_size = 5
        
        # Create sample sequences and labels
        self.sample_sequences = torch.randn(self.batch_size, self.seq_len, self.input_size)
        self.sample_labels = torch.randint(0, 2, (self.batch_size,)).float()
        self.sample_lengths = torch.randint(30, self.seq_len + 1, (self.batch_size,))
    
    def test_model_initialization(self):
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
        """Test forward pass functionality."""
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
    
    def test_hidden_state_initialization(self):
        """Test hidden state initialization."""
        model = SteelDefectLSTM(self.bidirectional_config)
        
        batch_size = 10
        hidden, cell = model.init_hidden(batch_size)
        
        expected_layers = model.num_layers * (2 if model.bidirectional else 1)
        
        assert hidden.shape == (expected_layers, batch_size, model.hidden_size)
        assert cell.shape == (expected_layers, batch_size, model.hidden_size)
        assert torch.allclose(hidden, torch.zeros_like(hidden))
        assert torch.allclose(cell, torch.zeros_like(cell))
    
    def test_normalization_layers(self):
        """Test different normalization options."""
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