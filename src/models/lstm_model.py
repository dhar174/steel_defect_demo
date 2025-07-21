import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union, List
import math

class SteelDefectLSTM(nn.Module):
    """Enhanced LSTM model for sequence-based defect prediction"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize enhanced LSTM model.
        
        Args:
            config (Dict[str, Any]): Model configuration containing:
                - architecture: LSTM architecture parameters
                - classifier: Classifier head configuration
                - normalization: Normalization options
                - regularization: Regularization parameters
        """
        super(SteelDefectLSTM, self).__init__()
        
        # Extract configuration parameters
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
        
        # Normalization options
        self.use_batch_norm = norm_config.get('batch_norm', True)
        self.use_layer_norm = norm_config.get('layer_norm', False)
        self.use_input_norm = norm_config.get('input_norm', True)
        
        # Regularization parameters
        self.weight_decay = reg_config.get('weight_decay', 1e-4)
        self.gradient_clip = reg_config.get('gradient_clip', 1.0)
        
        # Store config for reference
        self.config = config
        
        # Calculate LSTM output size (bidirectional doubles the output)
        lstm_output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        
        # Input normalization layer
        if self.use_input_norm:
            self.input_norm = nn.BatchNorm1d(self.input_size)
        
        # Main LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        
        # Normalization after LSTM
        if self.use_batch_norm:
            self.lstm_batch_norm = nn.BatchNorm1d(lstm_output_size)
        if self.use_layer_norm:
            self.lstm_layer_norm = nn.LayerNorm(lstm_output_size)
        
        # Build classifier head
        self.classifier = self._build_classifier(lstm_output_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_classifier(self, input_dim: int) -> nn.Module:
        """
        Build configurable classifier head.
        
        Args:
            input_dim (int): Input dimension from LSTM output
            
        Returns:
            nn.Module: Classifier sequential module
        """
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers
        for hidden_dim in self.classifier_hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Add activation
            if self.classifier_activation.lower() == 'relu':
                layers.append(nn.ReLU())
            elif self.classifier_activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            elif self.classifier_activation.lower() == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())  # Default
            
            # Add dropout
            if self.classifier_dropout > 0:
                layers.append(nn.Dropout(self.classifier_dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                if 'bias_ih' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input sequences [batch_size, seq_len, input_size]
            lengths (Optional[torch.Tensor]): Actual sequence lengths for each sample
            
        Returns:
            torch.Tensor: Predicted logits [batch_size, 1]
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Input normalization
        if self.use_input_norm:
            # Reshape for batch norm: [batch_size * seq_len, input_size]
            x_reshaped = x.view(-1, self.input_size)
            x_reshaped = self.input_norm(x_reshaped)
            x = x_reshaped.view(batch_size, seq_len, self.input_size)
        
        # Handle variable length sequences
        if lengths is not None:
            # Pack sequences for efficient processing
            x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out_packed, (hidden, cell) = self.lstm(x_packed)
            lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)
        else:
            lstm_out, (hidden, cell) = self.lstm(x)
        
        # Get final output (last time step or last valid time step for each sequence)
        if lengths is not None:
            # Use the last valid output for each sequence
            batch_indices = torch.arange(batch_size, device=x.device)
            last_indices = (lengths - 1).long()
            final_output = lstm_out[batch_indices, last_indices]
        else:
            # Use the last time step
            final_output = lstm_out[:, -1, :]
        
        # Apply normalization after LSTM
        if self.use_batch_norm:
            final_output = self.lstm_batch_norm(final_output)
        if self.use_layer_norm:
            final_output = self.lstm_layer_norm(final_output)
        
        # Classification
        prediction = self.classifier(final_output)
        
        return prediction
    
    def init_hidden(self, batch_size: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden states.
        
        Args:
            batch_size (int): Batch size
            device (Optional[torch.device]): Device to create tensors on
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Initialized hidden and cell states
        """
        if device is None:
            device = next(self.parameters()).device
        
        num_directions = 2 if self.bidirectional else 1
        
        hidden = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=device)
        cell = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=device)
        
        return hidden, cell
    
    def get_attention_weights(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get attention weights for model interpretability.
        
        Args:
            x (torch.Tensor): Input sequences
            lengths (Optional[torch.Tensor]): Actual sequence lengths
            
        Returns:
            torch.Tensor: Attention weights [batch_size, seq_len]
        """
        with torch.no_grad():
            # Get LSTM outputs
            if lengths is not None:
                x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
                lstm_out_packed, _ = self.lstm(x_packed)
                lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)
            else:
                lstm_out, _ = self.lstm(x)
            
            # Simple attention mechanism based on output magnitude
            attention_scores = torch.norm(lstm_out, dim=2)  # [batch_size, seq_len]
            
            # Apply softmax to get attention weights
            if lengths is not None:
                # Create mask for padding
                mask = torch.arange(lstm_out.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
                attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
            
            attention_weights = torch.softmax(attention_scores, dim=1)
            
            return attention_weights
    
    def freeze_layers(self, freeze_lstm: bool = True, freeze_classifier: bool = False):
        """
        Freeze layers for transfer learning.
        
        Args:
            freeze_lstm (bool): Whether to freeze LSTM layers
            freeze_classifier (bool): Whether to freeze classifier layers
        """
        if freeze_lstm:
            for param in self.lstm.parameters():
                param.requires_grad = False
            if hasattr(self, 'lstm_batch_norm'):
                for param in self.lstm_batch_norm.parameters():
                    param.requires_grad = False
            if hasattr(self, 'lstm_layer_norm'):
                for param in self.lstm_layer_norm.parameters():
                    param.requires_grad = False
        
        if freeze_classifier:
            for param in self.classifier.parameters():
                param.requires_grad = False
    
    def get_layer_outputs(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Get outputs from different layers for feature extraction.
        
        Args:
            x (torch.Tensor): Input sequences
            lengths (Optional[torch.Tensor]): Actual sequence lengths
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing outputs from different layers
        """
        outputs = {}
        
        # Input after normalization
        if self.use_input_norm:
            batch_size, seq_len = x.size(0), x.size(1)
            x_reshaped = x.view(-1, self.input_size)
            x_norm = self.input_norm(x_reshaped).view(batch_size, seq_len, self.input_size)
            outputs['input_normalized'] = x_norm
        else:
            outputs['input_normalized'] = x
        
        # LSTM outputs
        if lengths is not None:
            x_packed = pack_padded_sequence(outputs['input_normalized'], lengths.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out_packed, (hidden, cell) = self.lstm(x_packed)
            lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)
        else:
            lstm_out, (hidden, cell) = self.lstm(outputs['input_normalized'])
        
        outputs['lstm_output'] = lstm_out
        outputs['lstm_hidden'] = hidden
        outputs['lstm_cell'] = cell
        
        # Final features before classification
        if lengths is not None:
            batch_indices = torch.arange(x.size(0), device=x.device)
            last_indices = (lengths - 1).long()
            final_features = lstm_out[batch_indices, last_indices]
        else:
            final_features = lstm_out[:, -1, :]
        
        outputs['final_features'] = final_features
        
        # Apply normalization
        if self.use_batch_norm:
            final_features = self.lstm_batch_norm(final_features)
        if self.use_layer_norm:
            final_features = self.lstm_layer_norm(final_features)
        
        outputs['normalized_features'] = final_features
        
        return outputs
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and statistics.
        
        Returns:
            Dict[str, Any]: Model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'architecture': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'bidirectional': self.bidirectional,
                'classifier_dims': self.classifier_hidden_dims
            },
            'regularization': {
                'dropout': self.dropout,
                'classifier_dropout': self.classifier_dropout,
                'weight_decay': self.weight_decay,
                'gradient_clip': self.gradient_clip
            },
            'normalization': {
                'batch_norm': self.use_batch_norm,
                'layer_norm': self.use_layer_norm,
                'input_norm': self.use_input_norm
            }
        }

class CastingSequenceDataset(Dataset):
    """PyTorch Dataset for casting sequences"""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            sequences (np.ndarray): Input sequences
            labels (np.ndarray): Target labels
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        """Return dataset length."""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get item by index.
        
        Args:
            idx: Index
            
        Returns:
            Sequence and label pair
        """
        return self.sequences[idx], self.labels[idx]
