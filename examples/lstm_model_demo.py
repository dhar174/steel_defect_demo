#!/usr/bin/env python3
"""
Example usage of the enhanced SteelDefectLSTM model.

This script demonstrates how to:
1. Load LSTM configuration
2. Create different model variants
3. Handle variable sequence lengths
4. Train and evaluate models
5. Use performance tracking
"""

import sys
from pathlib import Path
import numpy as np

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.lstm_model import (
    SteelDefectLSTM, CastingSequenceDataset, LSTMModelVariants,
    LSTMPerformanceTracker, load_lstm_config, create_default_lstm_config
)
from models.model_config import ModelConfig


def demonstrate_model_variants():
    """Demonstrate different LSTM model variants."""
    print("=== LSTM Model Variants Demo ===")
    
    # Standard LSTM
    standard_config = LSTMModelVariants.standard_lstm()
    model_standard = SteelDefectLSTM(standard_config)
    print(f"Standard LSTM: {model_standard.num_layers} layers, "
          f"hidden_size={model_standard.hidden_size}, "
          f"bidirectional={model_standard.bidirectional}")
    
    # Bidirectional LSTM
    bidir_config = LSTMModelVariants.bidirectional_lstm()
    model_bidir = SteelDefectLSTM(bidir_config)
    print(f"Bidirectional LSTM: {model_bidir.num_layers} layers, "
          f"hidden_size={model_bidir.hidden_size}, "
          f"bidirectional={model_bidir.bidirectional}")
    
    # Deep LSTM
    deep_config = LSTMModelVariants.deep_lstm()
    model_deep = SteelDefectLSTM(deep_config)
    print(f"Deep LSTM: {model_deep.num_layers} layers, "
          f"hidden_size={model_deep.hidden_size}, "
          f"bidirectional={model_deep.bidirectional}")
    
    # Lightweight LSTM
    light_config = LSTMModelVariants.lightweight_lstm()
    model_light = SteelDefectLSTM(light_config)
    print(f"Lightweight LSTM: {model_light.num_layers} layers, "
          f"hidden_size={model_light.hidden_size}, "
          f"bidirectional={model_light.bidirectional}")
    
    print()


def demonstrate_variable_sequences():
    """Demonstrate handling of variable sequence lengths."""
    print("=== Variable Sequence Lengths Demo ===")
    
    # Create synthetic data with variable lengths
    np.random.seed(42)
    batch_size = 8
    max_seq_len = 200
    input_size = 5
    
    # Generate sequences with different lengths
    sequence_lengths = np.random.randint(50, max_seq_len + 1, batch_size)
    
    # Create padded sequences
    sequences = np.zeros((batch_size, max_seq_len, input_size))
    labels = np.random.binomial(1, 0.15, batch_size)
    
    for i, seq_len in enumerate(sequence_lengths):
        sequences[i, :seq_len, :] = np.random.normal(0, 1, (seq_len, input_size))
    
    print(f"Batch size: {batch_size}")
    print(f"Sequence lengths: {sequence_lengths}")
    print(f"Min length: {np.min(sequence_lengths)}, Max length: {np.max(sequence_lengths)}")
    
    # Create dataset
    dataset = CastingSequenceDataset(
        sequences, labels, sequence_lengths=sequence_lengths.tolist()
    )
    
    # Get sequence statistics
    stats = dataset.get_sequence_stats()
    print(f"Dataset statistics: {stats}")
    
    # Test model with variable sequences
    config = LSTMModelVariants.bidirectional_lstm()
    model = SteelDefectLSTM(config)
    
    # Forward pass with variable lengths
    outputs = model.forward(sequences, sequence_lengths=sequence_lengths.tolist())
    print(f"Output shape: {outputs.shape}")
    
    print()


def demonstrate_normalization_options():
    """Demonstrate different normalization options."""
    print("=== Normalization Options Demo ===")
    
    # Create configs with different normalization settings
    configs = {
        'No normalization': {
            'architecture': {'input_size': 5, 'hidden_size': 32, 'num_layers': 1, 'bidirectional': False, 'dropout': 0.1},
            'classifier': {'hidden_dims': [16], 'activation': 'relu', 'dropout': 0.2},
            'normalization': {'batch_norm': False, 'layer_norm': False, 'input_norm': False},
            'regularization': {'weight_decay': 1e-4, 'gradient_clip': 1.0}
        },
        'Batch normalization': {
            'architecture': {'input_size': 5, 'hidden_size': 32, 'num_layers': 1, 'bidirectional': False, 'dropout': 0.1},
            'classifier': {'hidden_dims': [16], 'activation': 'relu', 'dropout': 0.2},
            'normalization': {'batch_norm': True, 'layer_norm': False, 'input_norm': True},
            'regularization': {'weight_decay': 1e-4, 'gradient_clip': 1.0}
        },
        'Layer normalization': {
            'architecture': {'input_size': 5, 'hidden_size': 32, 'num_layers': 1, 'bidirectional': False, 'dropout': 0.1},
            'classifier': {'hidden_dims': [16], 'activation': 'relu', 'dropout': 0.2},
            'normalization': {'batch_norm': False, 'layer_norm': True, 'input_norm': False},
            'regularization': {'weight_decay': 1e-4, 'gradient_clip': 1.0}
        }
    }
    
    for name, config in configs.items():
        model = SteelDefectLSTM(config)
        print(f"{name}: batch_norm={model.use_batch_norm}, "
              f"layer_norm={model.use_layer_norm}, "
              f"input_norm={model.use_input_norm}")
    
    print()


def demonstrate_performance_tracking():
    """Demonstrate performance tracking capabilities."""
    print("=== Performance Tracking Demo ===")
    
    tracker = LSTMPerformanceTracker()
    
    # Simulate training progress
    epochs = [1, 2, 3, 4, 5]
    train_losses = [0.8, 0.6, 0.5, 0.4, 0.35]
    train_aucs = [0.70, 0.75, 0.80, 0.85, 0.88]
    val_losses = [0.9, 0.7, 0.6, 0.5, 0.45]
    val_aucs = [0.68, 0.73, 0.78, 0.83, 0.86]
    
    for epoch, train_loss, train_auc, val_loss, val_auc in zip(
        epochs, train_losses, train_aucs, val_losses, val_aucs
    ):
        tracker.log_training_metrics(epoch, {
            'loss': train_loss,
            'auc_roc': train_auc
        })
        tracker.log_validation_metrics(epoch, {
            'loss': val_loss,
            'auc_roc': val_auc
        })
    
    # Log inference metrics
    tracker.log_inference_time(32, 0.15)  # 150ms for batch of 32
    tracker.log_inference_time(64, 0.25)  # 250ms for batch of 64
    tracker.log_memory_usage(3500)  # 3.5GB memory usage
    
    # Get performance summary
    summary = tracker.get_performance_summary()
    
    print("Training Progress:")
    print(f"Latest training AUC: {summary['latest_training_metrics']['auc_roc']:.3f}")
    print(f"Latest validation AUC: {summary['latest_validation_metrics']['auc_roc']:.3f}")
    print(f"Meets AUC target (>0.88): {summary['meets_auc_target']}")
    
    print("\nPerformance Metrics:")
    print(f"Avg inference time per sample: {summary['avg_inference_time_per_sample']:.3f}s")
    print(f"Meets inference target (<0.2s): {summary['meets_inference_target']}")
    print(f"Peak memory usage: {summary['peak_memory_usage_mb']:.0f}MB")
    print(f"Meets memory target (<4GB): {summary['meets_memory_target']}")
    
    print()


def demonstrate_configuration_loading():
    """Demonstrate configuration loading and validation."""
    print("=== Configuration Loading Demo ===")
    
    # Load default configuration
    default_config = create_default_lstm_config()
    print("Default configuration loaded successfully")
    print(f"Hidden size: {default_config['architecture']['hidden_size']}")
    print(f"Bidirectional: {default_config['architecture']['bidirectional']}")
    
    # Try loading from config file
    config_path = Path(__file__).parent.parent / 'configs' / 'model_config.yaml'
    if config_path.exists():
        try:
            config_manager = ModelConfig(str(config_path))
            lstm_config = config_manager.get_lstm_config()
            print(f"Loaded config from {config_path}")
            print(f"Architecture: {lstm_config.get('architecture', {})}")
        except Exception as e:
            print(f"Could not load config file: {e}")
    else:
        print(f"Config file not found: {config_path}")
    
    print()


def demonstrate_attention_and_interpretability():
    """Demonstrate attention weights and model interpretability."""
    print("=== Attention and Interpretability Demo ===")
    
    # Create model with bidirectional LSTM
    config = LSTMModelVariants.bidirectional_lstm()
    model = SteelDefectLSTM(config)
    
    # Generate sample data
    np.random.seed(42)
    sequences = np.random.normal(0, 1, (4, 50, 5))
    
    # Forward pass
    outputs = model.forward(sequences)
    
    # Get attention weights
    attention_weights = model.get_attention_weights()
    if attention_weights is not None:
        print(f"Attention weights shape: {attention_weights.shape}")
        print(f"Sample attention weights (first sequence): {attention_weights[0][:10]}")
    else:
        print("Attention weights not available (normal for mock implementation)")
    
    # Extract layer outputs
    lstm_outputs = model.get_layer_outputs(sequences, layer_name='lstm')
    if lstm_outputs is not None:
        print(f"LSTM outputs extracted successfully")
        if hasattr(lstm_outputs, 'shape'):
            print(f"LSTM output shape: {lstm_outputs.shape}")
    
    # Demonstrate layer freezing
    print("Freezing LSTM layers for transfer learning...")
    model.freeze_layers(['lstm'])
    print("LSTM layers frozen successfully")
    
    print()


def main():
    """Main demonstration function."""
    print("Enhanced SteelDefectLSTM Model Demonstration")
    print("=" * 50)
    
    demonstrate_model_variants()
    demonstrate_variable_sequences()
    demonstrate_normalization_options()
    demonstrate_performance_tracking()
    demonstrate_configuration_loading()
    demonstrate_attention_and_interpretability()
    
    print("Demo completed successfully!")
    print("\nKey Features Demonstrated:")
    print("✓ Bidirectional LSTM with configurable parameters")
    print("✓ Variable sequence length handling")
    print("✓ Multiple normalization options")
    print("✓ Performance tracking and targets")
    print("✓ Configuration management")
    print("✓ Model interpretability features")
    print("✓ Transfer learning capabilities")


if __name__ == "__main__":
    main()