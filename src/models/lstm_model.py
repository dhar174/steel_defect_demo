try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    # Create mock classes for development when PyTorch is not available
    TORCH_AVAILABLE = False
    class MockModule:
        def __init__(self):
            self.training = True
        def train(self): self.training = True
        def eval(self): self.training = False
        def parameters(self): return []
        def state_dict(self): return {}
        def load_state_dict(self, state): pass
        def to(self, device): return self
        def named_parameters(self): return [('mock_param', type('MockParam', (), {'requires_grad': True})())]
    
    class MockTensor:
        def __init__(self, *args, **kwargs):
            self.shape = args[0] if args else (1,)
        def __getitem__(self, key): 
            # Handle slicing like [:, -1, :] to extract last timestep
            if isinstance(key, tuple) and len(key) == 3:
                if key[1] == -1:  # Getting last timestep
                    # Return tensor with batch and feature dimensions only
                    return MockTensor((self.shape[0], self.shape[2]))
            return MockTensor(self.shape)
        def size(self, dim=None): return self.shape[dim] if dim is not None else self.shape
        def view(self, *args): return MockTensor(*args)
        def unsqueeze(self, dim): return MockTensor(*self.shape)
        def squeeze(self, dim=None): return MockTensor(*self.shape)
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): 
            import numpy as np
            return np.ones(self.shape)
    
    class MockLSTM:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, x): return MockTensor(x.shape), (MockTensor((1,)), MockTensor((1,)))
    
    class MockLinear:
        def __init__(self, in_features, out_features, *args, **kwargs): 
            self.out_features = out_features
        def __call__(self, x): 
            # Return correct output shape for linear layer
            if hasattr(x, 'shape'):
                return MockTensor(x.shape[:-1] + (self.out_features,))
            return MockTensor((1, self.out_features))
    
    class MockDropout:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, x): return x
    
    class MockBatchNorm1d:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, x): return x
    
    class MockLayerNorm:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, x): return x
    
    class MockReLU:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, x): return x
    
    class MockSigmoid:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, x): return x
    
    class MockTanh:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, x): return x
    
    class MockSequential:
        def __init__(self, *layers):
            self.layers = layers
        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
        def parameters(self): return []
    
    # Mock torch namespace
    class torch:
        nn = type('nn', (), {
            'Module': MockModule,
            'LSTM': MockLSTM,
            'Linear': MockLinear,
            'Dropout': MockDropout,
            'BatchNorm1d': MockBatchNorm1d,
            'LayerNorm': MockLayerNorm,
            'ReLU': MockReLU,
            'Sigmoid': MockSigmoid,
            'Tanh': MockTanh,
            'Sequential': MockSequential
        })()
        
        @staticmethod
        def zeros(*args, **kwargs):
            return MockTensor(*args)
        
        @staticmethod
        def FloatTensor(*args):
            return MockTensor(*args)
        
        @staticmethod
        def LongTensor(*args):
            return MockTensor(*args)
        
        @staticmethod
        def softmax(x, dim):
            return MockTensor(x.shape)
        
        @staticmethod
        def sum(x, dim):
            return MockTensor(x.shape)

import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path


class SteelDefectLSTM(torch.nn.Module if TORCH_AVAILABLE else MockModule):
    """Enhanced LSTM model for sequence-based defect prediction with bidirectional support"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize enhanced LSTM model with comprehensive configuration.
        
        Args:
            config (Dict[str, Any]): Model configuration dictionary containing:
                - architecture: LSTM architecture parameters
                - classifier: Classifier head parameters  
                - normalization: Normalization options
                - regularization: Regularization parameters
        """
        super(SteelDefectLSTM, self).__init__()
        
        # Extract configuration
        arch_config = config.get('architecture', {})
        classifier_config = config.get('classifier', {})
        norm_config = config.get('normalization', {})
        reg_config = config.get('regularization', {})
        
        # Architecture parameters
        self.input_size = arch_config.get('input_size', 5)
        self.hidden_size = arch_config.get('hidden_size', 64)
        self.num_layers = arch_config.get('num_layers', 2)
        self.bidirectional = arch_config.get('bidirectional', True)
        self.dropout = arch_config.get('dropout', 0.2)
        
        # Classifier parameters
        self.classifier_hidden_dims = classifier_config.get('hidden_dims', [32, 16])
        self.classifier_activation = classifier_config.get('activation', 'relu')
        self.classifier_dropout = classifier_config.get('dropout', 0.3)
        
        # Normalization parameters
        self.use_batch_norm = norm_config.get('batch_norm', True)
        self.use_layer_norm = norm_config.get('layer_norm', False)
        self.use_input_norm = norm_config.get('input_norm', True)
        
        # Regularization parameters
        self.weight_decay = reg_config.get('weight_decay', 1e-4)
        self.gradient_clip = reg_config.get('gradient_clip', 1.0)
        
        # Calculate LSTM output size (doubled if bidirectional)
        lstm_output_size = self.hidden_size * (2 if self.bidirectional else 1)
        
        # Input normalization layer
        if self.use_input_norm:
            if TORCH_AVAILABLE:
                self.input_norm = torch.nn.BatchNorm1d(self.input_size)
            else:
                self.input_norm = MockBatchNorm1d(self.input_size)
        else:
            self.input_norm = None
        
        # LSTM layers
        if TORCH_AVAILABLE:
            self.lstm = torch.nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0,
                bidirectional=self.bidirectional,
                batch_first=True
            )
        else:
            self.lstm = MockLSTM()
        
        # Post-LSTM normalization
        if self.use_layer_norm:
            if TORCH_AVAILABLE:
                self.lstm_norm = torch.nn.LayerNorm(lstm_output_size)
            else:
                self.lstm_norm = MockLayerNorm(lstm_output_size)
        elif self.use_batch_norm:
            if TORCH_AVAILABLE:
                self.lstm_norm = torch.nn.BatchNorm1d(lstm_output_size)
            else:
                self.lstm_norm = MockBatchNorm1d(lstm_output_size)
        else:
            self.lstm_norm = None
        
        # Build classifier head
        self.classifier = self._build_classifier(lstm_output_size)
        
        # Store attention weights for interpretability
        self.attention_weights = None
        
        # Initialize weights
        if TORCH_AVAILABLE:
            self._init_weights()
    
    def _build_classifier(self, input_dim: int) -> torch.nn.Sequential if TORCH_AVAILABLE else MockSequential:
        """
        Build configurable classifier head.
        
        Args:
            input_dim (int): Input dimension from LSTM output
            
        Returns:
            Sequential classifier network
        """
        layers = []
        current_dim = input_dim
        
        # Add hidden layers
        for hidden_dim in self.classifier_hidden_dims:
            if TORCH_AVAILABLE:
                layers.append(torch.nn.Linear(current_dim, hidden_dim))
                layers.append(self._get_activation(self.classifier_activation))
                layers.append(torch.nn.Dropout(self.classifier_dropout))
            else:
                layers.append(MockLinear(current_dim, hidden_dim))
                layers.append(self._get_activation(self.classifier_activation))
                layers.append(MockDropout(self.classifier_dropout))
            current_dim = hidden_dim
        
        # Output layer (binary classification)
        if TORCH_AVAILABLE:
            layers.append(torch.nn.Linear(current_dim, 1))
        else:
            layers.append(MockLinear(current_dim, 1))
        
        if TORCH_AVAILABLE:
            return torch.nn.Sequential(*layers)
        else:
            return MockSequential(*layers)
    
    def _get_activation(self, activation: str):
        """Get activation function by name."""
        if TORCH_AVAILABLE:
            activations = {
                'relu': torch.nn.ReLU(),
                'sigmoid': torch.nn.Sigmoid(),
                'tanh': torch.nn.Tanh()
            }
            return activations.get(activation.lower(), torch.nn.ReLU())
        else:
            activations = {
                'relu': MockReLU(),
                'sigmoid': MockSigmoid(),
                'tanh': MockTanh()
            }
            return activations.get(activation.lower(), MockReLU())
    
    def _init_weights(self):
        """Initialize model weights."""
        if not TORCH_AVAILABLE:
            return
            
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM weight initialization
                    torch.nn.init.xavier_uniform_(param)
                else:
                    # Linear layer weight initialization
                    torch.nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
    
    def forward(self, x, sequence_lengths: Optional[List[int]] = None):
        """
        Enhanced forward pass with bidirectional processing and variable sequence lengths.
        
        Args:
            x: Input sequences [batch_size, seq_len, input_size]
            sequence_lengths: Optional list of actual sequence lengths for each sample
            
        Returns:
            Predicted logits for binary classification
        """
        batch_size, seq_len, _ = x.shape if hasattr(x, 'shape') else (1, 1, self.input_size)
        
        # Input normalization
        if self.input_norm is not None:
            # Reshape for batch norm: [batch_size * seq_len, input_size]
            if TORCH_AVAILABLE:
                x_reshaped = x.view(-1, self.input_size)
                x_normalized = self.input_norm(x_reshaped)
                x = x_normalized.view(batch_size, seq_len, self.input_size)
            else:
                x = self.input_norm(x)
        
        # Handle variable sequence lengths with padding
        if sequence_lengths is not None and TORCH_AVAILABLE:
            # Pack padded sequence for efficient processing
            x = torch.nn.utils.rnn.pack_padded_sequence(
                x, sequence_lengths, batch_first=True, enforce_sorted=False
            )
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Unpack if we used packed sequences
        if sequence_lengths is not None and TORCH_AVAILABLE:
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        
        # Extract final output considering variable lengths
        if sequence_lengths is not None:
            # Get the last actual output for each sequence
            if TORCH_AVAILABLE:
                indices = torch.LongTensor(sequence_lengths).to(lstm_out.device) - 1
                indices = indices.view(-1, 1, 1).expand(-1, -1, lstm_out.size(-1))
                final_output = lstm_out.gather(1, indices).squeeze(1)
            else:
                # For mock implementation, just use last output with correct shape
                final_output = MockTensor((batch_size, lstm_out.shape[-1] if hasattr(lstm_out, 'shape') else self.hidden_size * (2 if self.bidirectional else 1)))
        else:
            # Use last time step output
            if hasattr(lstm_out, '__getitem__'):
                final_output = lstm_out[:, -1, :]
            else:
                # Mock implementation - create tensor with correct shape
                output_size = self.hidden_size * (2 if self.bidirectional else 1)
                final_output = MockTensor((batch_size, output_size))
        
        # Post-LSTM normalization
        if self.lstm_norm is not None:
            final_output = self.lstm_norm(final_output)
        
        # Store attention weights for interpretability
        if TORCH_AVAILABLE and hasattr(lstm_out, 'shape'):
            # Simple attention mechanism for interpretability
            attention_scores = torch.softmax(
                torch.sum(lstm_out, dim=-1), dim=1
            )
            self.attention_weights = attention_scores.detach()
        
        # Classifier forward pass
        prediction = self.classifier(final_output)
        
        return prediction
    
    def init_hidden(self, batch_size: int, device: Optional[str] = None) -> Tuple:
        """
        Initialize hidden and cell states.
        
        Args:
            batch_size (int): Batch size
            device (str, optional): Device to place tensors on
            
        Returns:
            Tuple of (hidden_state, cell_state)
        """
        num_directions = 2 if self.bidirectional else 1
        hidden_dim = (self.num_layers * num_directions, batch_size, self.hidden_size)
        
        if not TORCH_AVAILABLE:
            return (MockTensor(hidden_dim), MockTensor(hidden_dim))
        
        if device:
            hidden = torch.zeros(hidden_dim).to(device)
            cell = torch.zeros(hidden_dim).to(device)
        else:
            hidden = torch.zeros(hidden_dim)
            cell = torch.zeros(hidden_dim)
        
        return (hidden, cell)
    
    def get_attention_weights(self) -> Optional[np.ndarray]:
        """
        Get attention weights for model interpretability.
        
        Returns:
            Attention weights array or None if not available
        """
        if self.attention_weights is not None:
            if TORCH_AVAILABLE:
                return self.attention_weights.cpu().numpy()
            else:
                attention_shape = (1, self.hidden_size) if hasattr(self, 'hidden_size') else (1, 10)
                return np.ones(attention_shape)  # Mock attention weights
        return None
    
    def freeze_layers(self, layer_names: List[str]) -> None:
        """
        Freeze specified layers for transfer learning.
        
        Args:
            layer_names: List of layer names to freeze
        """
        if not TORCH_AVAILABLE:
            return
            
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
    
    def get_layer_outputs(self, x, layer_name: str = 'lstm'):
        """
        Extract outputs from specified layer for feature analysis.
        
        Args:
            x: Input tensor
            layer_name: Name of layer to extract outputs from
            
        Returns:
            Layer outputs
        """
        if layer_name == 'lstm':
            if self.input_norm is not None:
                if TORCH_AVAILABLE:
                    batch_size, seq_len, _ = x.shape
                    x_reshaped = x.view(-1, self.input_size)
                    x_normalized = self.input_norm(x_reshaped)
                    x = x_normalized.view(batch_size, seq_len, self.input_size)
                else:
                    x = self.input_norm(x)
            
            lstm_out, _ = self.lstm(x)
            return lstm_out
        
        # For other layers, would need more complex implementation
        return None


def load_lstm_config(config_path: str) -> Dict[str, Any]:
    """
    Load LSTM configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        LSTM configuration dictionary
    """
    config_manager = ModelConfig(config_path)
    return config_manager.get_lstm_config()


def create_default_lstm_config() -> Dict[str, Any]:
    """
    Create default LSTM configuration.
    
    Returns:
        Default LSTM configuration dictionary
    """
    return {
        'architecture': {
            'input_size': 5,
            'hidden_size': 64,
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
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 100,
            'early_stopping_patience': 15,
            'weight_decay': 0.0001
        },
        'data_processing': {
            'sequence_length': 300,
            'normalization': 'z_score',
            'padding': 'zero'
        }
    }


class CastingSequenceDataset:
    """PyTorch Dataset for casting sequences with enhanced functionality"""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray, 
                 sequence_lengths: Optional[List[int]] = None,
                 transform: Optional[callable] = None):
        """
        Initialize dataset with variable sequence length support.
        
        Args:
            sequences (np.ndarray): Input sequences [batch_size, max_seq_len, features]
            labels (np.ndarray): Target labels
            sequence_lengths (List[int], optional): Actual lengths of each sequence
            transform (callable, optional): Optional transform to apply to sequences
        """
        if TORCH_AVAILABLE:
            self.sequences = torch.FloatTensor(sequences)
            self.labels = torch.FloatTensor(labels)
        else:
            self.sequences = sequences
            self.labels = labels
        
        self.sequence_lengths = sequence_lengths
        self.transform = transform
    
    def __len__(self):
        """Return dataset length."""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get item by index with optional sequence length.
        
        Args:
            idx: Index
            
        Returns:
            Sequence, label, and optionally sequence length
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
        
        if self.sequence_lengths is not None:
            return sequence, label, self.sequence_lengths[idx]
        else:
            return sequence, label
    
    def get_sequence_stats(self) -> Dict[str, float]:
        """
        Get statistics about sequence lengths.
        
        Returns:
            Dictionary with sequence length statistics
        """
        if self.sequence_lengths is not None:
            lengths = np.array(self.sequence_lengths)
            return {
                'min_length': int(np.min(lengths)),
                'max_length': int(np.max(lengths)),
                'mean_length': float(np.mean(lengths)),
                'std_length': float(np.std(lengths))
            }
        else:
            seq_len = self.sequences.shape[1] if hasattr(self.sequences, 'shape') else 300
            return {
                'min_length': seq_len,
                'max_length': seq_len,
                'mean_length': seq_len,
                'std_length': 0.0
            }


class LSTMModelVariants:
    """Factory class for creating different LSTM model configurations"""
    
    @staticmethod
    def standard_lstm() -> Dict[str, Any]:
        """Standard unidirectional LSTM configuration"""
        config = create_default_lstm_config()
        config['architecture']['bidirectional'] = False
        config['architecture']['hidden_size'] = 64
        config['architecture']['num_layers'] = 2
        return config
    
    @staticmethod
    def bidirectional_lstm() -> Dict[str, Any]:
        """Bidirectional LSTM configuration"""
        config = create_default_lstm_config()
        config['architecture']['bidirectional'] = True
        config['architecture']['hidden_size'] = 64
        config['architecture']['num_layers'] = 2
        return config
    
    @staticmethod
    def deep_lstm() -> Dict[str, Any]:
        """Deep bidirectional LSTM configuration"""
        config = create_default_lstm_config()
        config['architecture']['bidirectional'] = True
        config['architecture']['hidden_size'] = 128
        config['architecture']['num_layers'] = 4
        config['architecture']['dropout'] = 0.3
        config['classifier']['hidden_dims'] = [64, 32, 16]
        return config
    
    @staticmethod
    def lightweight_lstm() -> Dict[str, Any]:
        """Lightweight LSTM for faster inference"""
        config = create_default_lstm_config()
        config['architecture']['bidirectional'] = False
        config['architecture']['hidden_size'] = 32
        config['architecture']['num_layers'] = 1
        config['architecture']['dropout'] = 0.1
        config['classifier']['hidden_dims'] = [16]
        return config


# Model performance tracking
class LSTMPerformanceTracker:
    """Track LSTM model performance metrics"""
    
    def __init__(self):
        self.training_history = []
        self.validation_history = []
        self.inference_times = []
        self.memory_usage = []
    
    def log_training_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log training metrics for an epoch"""
        metrics['epoch'] = epoch
        self.training_history.append(metrics.copy())
    
    def log_validation_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log validation metrics for an epoch"""
        metrics['epoch'] = epoch
        self.validation_history.append(metrics.copy())
    
    def log_inference_time(self, batch_size: int, inference_time: float):
        """Log inference time for performance tracking"""
        self.inference_times.append({
            'batch_size': batch_size,
            'inference_time': inference_time,
            'time_per_sample': inference_time / batch_size
        })
    
    def log_memory_usage(self, memory_mb: float):
        """Log memory usage during training/inference"""
        self.memory_usage.append(memory_mb)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of model performance"""
        summary = {}
        
        if self.training_history:
            latest_train = self.training_history[-1]
            summary['latest_training_metrics'] = latest_train
        
        if self.validation_history:
            latest_val = self.validation_history[-1]
            summary['latest_validation_metrics'] = latest_val
            
            # Check if performance targets are met
            val_auc = latest_val.get('auc_roc', 0)
            summary['meets_auc_target'] = val_auc > 0.88
        
        if self.inference_times:
            avg_inference_time = np.mean([t['time_per_sample'] for t in self.inference_times])
            summary['avg_inference_time_per_sample'] = avg_inference_time
            summary['meets_inference_target'] = avg_inference_time < 0.2  # 200ms target
        
        if self.memory_usage:
            max_memory = max(self.memory_usage)
            summary['peak_memory_usage_mb'] = max_memory
            summary['meets_memory_target'] = max_memory < 4096  # 4GB target
        
        return summary
