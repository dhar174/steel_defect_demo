"""
Integration test to verify sequence dataset works with LSTM model
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
import os

# Ensure the src package is available when running tests directly
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.fspath(REPO_ROOT))

from src.data.sequence_dataset import CastingSequenceDataset, create_data_loaders, validate_dataset_config


def test_lstm_integration():
    """Test that our dataset integrates properly with a simple LSTM model"""
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    sequence_length = 50
    n_features = 5
    
    sequences = np.random.randn(n_samples, sequence_length, n_features).astype(np.float32)
    labels = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]).astype(np.float32)
    
    # Split data
    n_train = 80
    train_sequences, train_labels = sequences[:n_train], labels[:n_train]
    val_sequences, val_labels = sequences[n_train:], labels[n_train:]
    
    # Create datasets
    train_dataset = CastingSequenceDataset(train_sequences, train_labels, augment=True)
    val_dataset = CastingSequenceDataset(val_sequences, val_labels, augment=False)
    
    # Create data loaders
    config = validate_dataset_config({
        'batch_size': 16,
        'num_workers': 0
    })
    
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, config)
    
    # Simple LSTM model for testing
    class SimpleLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(SimpleLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.classifier = nn.Linear(hidden_size, output_size)
            
        def forward(self, x, attention_mask=None):
            # x shape: (batch, seq_len, features)
            lstm_out, (hidden, cell) = self.lstm(x)
            
            # Use last timestep output
            if attention_mask is not None:
                # Find actual sequence lengths
                seq_lengths = attention_mask.sum(dim=1)
                batch_size = x.size(0)
                
                # Gather last valid timestep for each sequence
                last_outputs = []
                for i in range(batch_size):
                    last_idx = seq_lengths[i] - 1
                    last_outputs.append(lstm_out[i, last_idx])
                output = torch.stack(last_outputs)
            else:
                output = lstm_out[:, -1, :]  # Use last timestep
            
            return torch.sigmoid(self.classifier(output))
    
    # Initialize model
    model = SimpleLSTM(
        input_size=n_features,
        hidden_size=32,
        num_layers=2,
        output_size=1
    )
    
    # Test forward pass with training data
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            sequences, labels, attention_masks = batch
            
            # Forward pass
            outputs = model(sequences, attention_masks)
            
            # Check shapes
            assert outputs.shape == (sequences.shape[0], 1), f"Expected {(sequences.shape[0], 1)}, got {outputs.shape}"
            assert sequences.shape == (16, 50, 5), f"Expected (16, 50, 5), got {sequences.shape}"
            assert labels.shape == (16,), f"Expected (16,), got {labels.shape}"
            assert attention_masks.shape == (16, 50), f"Expected (16, 50), got {attention_masks.shape}"
            
            print(f"Batch {batch_idx}: Shapes verified successfully")
            print(f"  Sequences: {sequences.shape}")
            print(f"  Labels: {labels.shape}")
            print(f"  Attention masks: {attention_masks.shape}")
            print(f"  Model outputs: {outputs.shape}")
            print(f"  Output range: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
            
            # Only test first batch
            break
    
    # Test with validation data
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            sequences, labels, attention_masks = batch
            outputs = model(sequences, attention_masks)
            
            print(f"Validation batch: Shapes verified successfully")
            print(f"  Sequences: {sequences.shape}")
            print(f"  Model outputs: {outputs.shape}")
            break
    
    print("\n✅ LSTM integration test passed!")
    return True


def test_training_step():
    """Test a single training step with loss calculation"""
    
    # Create sample data
    np.random.seed(42)
    sequences = np.random.randn(32, 100, 5).astype(np.float32)
    labels = np.random.choice([0, 1], size=32, p=[0.8, 0.2]).astype(np.float32)
    
    # Create dataset and loader
    dataset = CastingSequenceDataset(sequences, labels)
    config = validate_dataset_config({'batch_size': 8, 'num_workers': 0})
    loader = create_data_loaders(dataset, dataset, config)[0]
    
    # Simple model
    model = nn.LSTM(input_size=5, hidden_size=32, num_layers=1, batch_first=True)
    classifier = nn.Linear(32, 1)
    criterion = nn.BCEWithLogitsLoss()
    
    # Test training step
    model.train()
    batch = next(iter(loader))
    sequences_batch, labels_batch, attention_masks = batch
    
    # Forward pass
    lstm_out, _ = model(sequences_batch)
    outputs = classifier(lstm_out[:, -1, :])  # Use last timestep
    outputs = outputs.squeeze()
    
    # Loss calculation
    loss = criterion(outputs, labels_batch)
    
    print(f"Training step test:")
    print(f"  Batch size: {sequences_batch.shape[0]}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Outputs shape: {outputs.shape}")
    print(f"  Labels shape: {labels_batch.shape}")
    
    print("\n✅ Training step test passed!")
    return True


if __name__ == "__main__":
    print("=== LSTM Integration Tests ===\n")
    
    test_lstm_integration()
    print()
    test_training_step()
    
    print("\n🎉 All integration tests passed!")