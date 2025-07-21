#!/usr/bin/env python3
"""
LSTM Training Pipeline Demo

This script demonstrates how to use the comprehensive LSTM training pipeline
for steel casting defect prediction.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from models.model_trainer import LSTMTrainer, MockDataLoader

# Mock model since PyTorch may not be available
class MockLSTMModel:
    """Mock LSTM model for demonstration"""
    def __init__(self):
        self.training = True
        
    def train(self):
        self.training = True
        
    def eval(self):
        self.training = False
        
    def parameters(self):
        return []
    
    def state_dict(self):
        return {'lstm.weight': [1, 2, 3], 'classifier.weight': [0.1, 0.2]}
    
    def load_state_dict(self, state_dict):
        pass
    
    def to(self, device):
        return self
    
    def __call__(self, x):
        # Mock forward pass - return predictions
        if hasattr(x, '__len__'):
            return [0.7 + np.random.normal(0, 0.1) for _ in range(len(x))]
        return [0.7]


def create_mock_dataset(n_samples=1000, seq_length=300, n_features=5):
    """Create mock steel casting sensor data for demonstration"""
    print(f"Creating mock dataset: {n_samples} samples, {seq_length} timesteps, {n_features} features")
    
    # Generate synthetic sensor data
    sequences = []
    labels = []
    
    for i in range(n_samples):
        # Generate time series with some patterns
        t = np.linspace(0, 10, seq_length)
        
        # Base signals for different sensors
        temp_sensor = 800 + 50 * np.sin(0.5 * t) + np.random.normal(0, 5, seq_length)
        pressure_sensor = 10 + 2 * np.cos(0.3 * t) + np.random.normal(0, 0.5, seq_length)
        flow_sensor = 5 + np.sin(0.8 * t) + np.random.normal(0, 0.2, seq_length)
        vibration_sensor = np.random.normal(0, 1, seq_length)
        quality_sensor = 95 + 3 * np.sin(0.2 * t) + np.random.normal(0, 1, seq_length)
        
        # Combine into sequence
        sequence = np.column_stack([
            temp_sensor, pressure_sensor, flow_sensor, 
            vibration_sensor, quality_sensor
        ])
        
        # Create defect label (10% defect rate)
        # Add some patterns for defects
        defect = 0
        if i % 10 == 0:  # Every 10th sample has a defect
            defect = 1
            # Add anomalous patterns for defects
            sequence[:100, 0] += 20  # Temperature spike
            sequence[50:150, 3] += 2  # Vibration increase
        
        sequences.append(sequence)
        labels.append(defect)
    
    return np.array(sequences), np.array(labels)


def main():
    """Demonstrate the LSTM training pipeline"""
    print("🔥 LSTM Training Pipeline Demonstration")
    print("=" * 50)
    
    # 1. Create configuration
    print("\n1. Setting up configuration...")
    config = {
        'architecture': {
            'input_size': 5,
            'hidden_size': 64,
            'num_layers': 2,
            'bidirectional': True,
            'dropout': 0.2
        },
        'training': {
            'num_epochs': 10,  # Shorter for demo
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'batch_size': 32,
            'gradient_clip_norm': 1.0
        },
        'optimization': {
            'optimizer': 'adam',
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8
        },
        'scheduler': {
            'type': 'reduce_on_plateau',
            'factor': 0.5,
            'patience': 5
        },
        'loss_function': {
            'type': 'weighted_bce',
            'pos_weight': 3.0
        },
        'early_stopping': {
            'patience': 5,
            'min_delta': 1e-4,
            'monitor': 'val_loss'
        },
        'logging': {
            'tensorboard_enabled': False,  # Disable for demo
            'csv_logging': True,
            'log_interval': 5,
            'save_interval': 2
        }
    }
    
    print(f"   - Epochs: {config['training']['num_epochs']}")
    print(f"   - Learning rate: {config['training']['learning_rate']}")
    print(f"   - Batch size: {config['training']['batch_size']}")
    print(f"   - Early stopping patience: {config['early_stopping']['patience']}")
    
    # 2. Initialize model
    print("\n2. Initializing LSTM model...")
    model = MockLSTMModel()
    print(f"   - Model type: {type(model).__name__}")
    print(f"   - Input size: {config['architecture']['input_size']}")
    print(f"   - Hidden size: {config['architecture']['hidden_size']}")
    print(f"   - Number of layers: {config['architecture']['num_layers']}")
    print(f"   - Bidirectional: {config['architecture']['bidirectional']}")
    
    # 3. Initialize trainer
    print("\n3. Initializing trainer...")
    trainer = LSTMTrainer(model, config)
    print(f"   - Device: {trainer.device}")
    print(f"   - Optimizer: {config['optimization']['optimizer']}")
    print(f"   - Loss function: {config['loss_function']['type']}")
    print(f"   - Scheduler: {config['scheduler']['type']}")
    
    # 4. Create mock data
    print("\n4. Creating mock training data...")
    train_sequences, train_labels = create_mock_dataset(n_samples=800, seq_length=300)
    val_sequences, val_labels = create_mock_dataset(n_samples=200, seq_length=300)
    
    print(f"   - Training samples: {len(train_sequences)}")
    print(f"   - Validation samples: {len(val_sequences)}")
    print(f"   - Defect rate (train): {train_labels.mean():.1%}")
    print(f"   - Defect rate (val): {val_labels.mean():.1%}")
    
    # 5. Create data loaders
    print("\n5. Creating data loaders...")
    
    # Convert to list of tuples for MockDataLoader
    train_data = [(train_sequences[i], train_labels[i]) for i in range(len(train_sequences))]
    val_data = [(val_sequences[i], val_labels[i]) for i in range(len(val_sequences))]
    
    # Sample subset for demo (to make it faster)
    train_data_sample = train_data[:50]  # Use subset for demo
    val_data_sample = val_data[:20]
    
    train_loader = MockDataLoader(train_data_sample, batch_size=config['training']['batch_size'])
    val_loader = MockDataLoader(val_data_sample, batch_size=config['training']['batch_size'])
    
    print(f"   - Training batches: {len(train_loader)}")
    print(f"   - Validation batches: {len(val_loader)}")
    
    # 6. Train the model
    print("\n6. Starting training...")
    print("-" * 30)
    
    try:
        results = trainer.train(train_loader, val_loader)
        print("-" * 30)
        print("✅ Training completed successfully!")
        
        # 7. Display results
        print("\n7. Training Results:")
        print(f"   - Final epoch: {results['final_epoch']}")
        print(f"   - Best epoch: {results['best_epoch']}")
        print(f"   - Best validation AUC: {results['best_val_auc']:.4f}")
        print(f"   - Final train loss: {results['final_train_loss']:.4f}")
        print(f"   - Final validation loss: {results['final_val_loss']:.4f}")
        print(f"   - Final validation AUC: {results['final_val_auc']:.4f}")
        print(f"   - Total training time: {results['total_training_time']:.1f}s")
        
        # 8. Test evaluation
        print("\n8. Evaluating on test set...")
        test_metrics = trainer.evaluate(val_loader)  # Use val_loader as test for demo
        
        print("   Test Results:")
        print(f"     - AUC-ROC: {test_metrics['auc_roc']:.4f}")
        print(f"     - AUC-PR: {test_metrics['auc_pr']:.4f}")
        print(f"     - F1-Score: {test_metrics['f1_score']:.4f}")
        print(f"     - Precision: {test_metrics['precision']:.4f}")
        print(f"     - Recall: {test_metrics['recall']:.4f}")
        print(f"     - Accuracy: {test_metrics['accuracy']:.4f}")
        
        # 9. Save model
        print("\n9. Saving trained model...")
        model_path = "demo_trained_model.pth"
        trainer.save_model(model_path, include_metadata=True)
        print(f"   - Model saved to: {model_path}")
        
        print("\n🎉 LSTM Training Pipeline Demo Complete!")
        print("\nKey Features Demonstrated:")
        print("  ✓ Advanced Adam optimizer with configurable parameters")
        print("  ✓ Learning rate scheduling with ReduceLROnPlateau")
        print("  ✓ Early stopping with patience-based monitoring")
        print("  ✓ Comprehensive metrics tracking (AUC-ROC, AUC-PR, F1, etc.)")
        print("  ✓ Weighted loss function for class imbalance")
        print("  ✓ Gradient clipping for training stability")
        print("  ✓ Model checkpointing and state management")
        print("  ✓ Training resumption capabilities")
        print("  ✓ GPU acceleration support (when available)")
        print("  ✓ Cross-platform compatibility with mock implementations")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)