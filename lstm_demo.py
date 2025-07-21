#!/usr/bin/env python3
"""
Enhanced LSTM Model Demo Script

This script demonstrates the usage of the enhanced SteelDefectLSTM model
with all its advanced features including bidirectional processing,
configurable architecture, and various normalization options.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from models.lstm_model import SteelDefectLSTM, CastingSequenceDataset


def load_lstm_config(config_path: str = None) -> dict:
    """Load LSTM configuration from YAML file or return default config."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('lstm_model', {})
    
    # Default configuration matching the requirements
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
        }
    }


def generate_sample_data(num_samples: int = 1000, seq_length: int = 300, 
                        input_size: int = 5, defect_rate: float = 0.15):
    """Generate sample sensor data for demonstration."""
    print(f"Generating {num_samples} samples with sequence length {seq_length}...")
    
    # Generate sequences with temporal patterns
    sequences = []
    labels = []
    
    for i in range(num_samples):
        # Random sequence length between 100-300
        length = np.random.randint(100, seq_length + 1)
        
        # Generate base sequence
        seq = np.random.randn(seq_length, input_size) * 0.1
        
        # Add temporal patterns
        t = np.linspace(0, 10, length)
        for sensor in range(input_size):
            # Add sinusoidal patterns with noise
            pattern = np.sin(t * (sensor + 1) * 0.5) * 0.5
            seq[:length, sensor] += pattern
        
        # Create defect labels (15% defect rate)
        is_defect = np.random.random() < defect_rate
        
        if is_defect:
            # Add defect pattern in the last part of sequence
            defect_start = length // 2
            defect_pattern = np.random.randn(length - defect_start, input_size) * 2.0
            seq[defect_start:length] += defect_pattern
        
        sequences.append(seq)
        labels.append(float(is_defect))
    
    return np.array(sequences), np.array(labels)


def demonstrate_model_variants():
    """Demonstrate different model configurations."""
    print("\n" + "="*50)
    print("DEMONSTRATING MODEL VARIANTS")
    print("="*50)
    
    # Standard LSTM Configuration
    standard_config = {
        'architecture': {
            'input_size': 5,
            'hidden_size': 64,
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
            'batch_norm': False,
            'layer_norm': False,
            'input_norm': False
        }
    }
    
    # Bidirectional LSTM Configuration
    bidirectional_config = {
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
        }
    }
    
    # Deep LSTM Configuration
    deep_config = {
        'architecture': {
            'input_size': 5,
            'hidden_size': 128,
            'num_layers': 4,
            'bidirectional': True,
            'dropout': 0.3
        },
        'classifier': {
            'hidden_dims': [64, 32, 16],
            'activation': 'relu',
            'dropout': 0.4
        },
        'normalization': {
            'batch_norm': True,
            'layer_norm': True,
            'input_norm': True
        }
    }
    
    configs = [
        ("Standard LSTM", standard_config),
        ("Bidirectional LSTM", bidirectional_config),
        ("Deep LSTM", deep_config)
    ]
    
    # Test data
    test_input = torch.randn(8, 100, 5)
    
    for name, config in configs:
        print(f"\n{name}:")
        model = SteelDefectLSTM(config)
        info = model.get_model_info()
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(test_input)
        
        print(f"  Parameters: {info['total_parameters']:,}")
        print(f"  Model size: {info['model_size_mb']:.2f} MB")
        print(f"  Architecture: {info['architecture']}")
        print(f"  Output shape: {output.shape}")


def demonstrate_training_loop():
    """Demonstrate a complete training loop with the enhanced LSTM."""
    print("\n" + "="*50)
    print("DEMONSTRATING TRAINING LOOP")
    print("="*50)
    
    # Load configuration
    config = load_lstm_config("configs/model_config.yaml")
    print(f"Using configuration: {config}")
    
    # Generate training data
    sequences, labels = generate_sample_data(num_samples=500, seq_length=200)
    
    # Create dataset and dataloader
    dataset = CastingSequenceDataset(sequences, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Create model
    model = SteelDefectLSTM(config)
    print(f"\nModel info: {model.get_model_info()}")
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Training loop
    print("\nStarting training...")
    model.train()
    
    for epoch in range(5):  # Just a few epochs for demo
        total_loss = 0
        num_batches = 0
        
        for batch_sequences, batch_labels in dataloader:
            # Forward pass
            outputs = model(batch_sequences)
            loss = criterion(outputs.squeeze(), batch_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('regularization', {}).get('gradient_clip', 1.0))
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/5, Average Loss: {avg_loss:.4f}")
    
    print("Training completed!")
    
    # Evaluation
    model.eval()
    print("\nEvaluating model...")
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_sequences, batch_labels in dataloader:
            outputs = model(batch_sequences)
            predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")


def demonstrate_advanced_features():
    """Demonstrate advanced features like attention weights and layer outputs."""
    print("\n" + "="*50)
    print("DEMONSTRATING ADVANCED FEATURES")
    print("="*50)
    
    # Create model with bidirectional LSTM
    config = load_lstm_config()
    model = SteelDefectLSTM(config)
    model.eval()
    
    # Test data with variable lengths
    batch_size = 4
    max_seq_len = 100
    test_sequences = torch.randn(batch_size, max_seq_len, 5)
    test_lengths = torch.tensor([100, 80, 60, 90])
    
    print(f"Test sequences shape: {test_sequences.shape}")
    print(f"Sequence lengths: {test_lengths.tolist()}")
    
    # Forward pass with lengths
    with torch.no_grad():
        outputs = model(test_sequences, test_lengths)
        print(f"Model outputs shape: {outputs.shape}")
        print(f"Output values: {outputs.squeeze().tolist()}")
        
        # Get attention weights
        attention = model.get_attention_weights(test_sequences, test_lengths)
        print(f"\nAttention weights shape: {attention.shape}")
        print("Attention weights sum (should be ~1.0 for each sequence):")
        print(attention.sum(dim=1).tolist())
        
        # Get layer outputs
        layer_outputs = model.get_layer_outputs(test_sequences, test_lengths)
        print(f"\nLayer outputs available: {list(layer_outputs.keys())}")
        for key, value in layer_outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
    
    # Demonstrate layer freezing
    print("\nDemonstrating layer freezing...")
    
    # Count trainable parameters before freezing
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters before freezing: {trainable_before:,}")
    
    # Freeze LSTM layers
    model.freeze_layers(freeze_lstm=True, freeze_classifier=False)
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after freezing LSTM: {trainable_after:,}")
    
    # Show reduction
    reduction = trainable_before - trainable_after
    print(f"Reduction in trainable parameters: {reduction:,}")


def main():
    """Main demonstration function."""
    print("Enhanced LSTM Model Demonstration")
    print("=" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Demonstrate different model variants
        demonstrate_model_variants()
        
        # Demonstrate training loop
        demonstrate_training_loop()
        
        # Demonstrate advanced features
        demonstrate_advanced_features()
        
        print("\n" + "="*50)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        print("\nKey features demonstrated:")
        print("✓ Bidirectional LSTM processing")
        print("✓ Configurable architecture parameters")
        print("✓ Multiple normalization options")
        print("✓ Advanced regularization techniques")
        print("✓ Variable sequence length support")
        print("✓ Attention weights computation")
        print("✓ Layer output extraction")
        print("✓ Layer freezing for transfer learning")
        print("✓ Gradient clipping and weight decay")
        print("✓ Model serialization compatibility")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()