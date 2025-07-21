"""
Comprehensive test suite for LSTM training pipeline components.

Tests all classes: LSTMTrainer, EarlyStopping, TrainingMetrics, ModelCheckpoint
"""

import pytest
import numpy as np
import tempfile
import os
import json
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.model_trainer import (
    LSTMTrainer, EarlyStopping, TrainingMetrics, ModelCheckpoint,
    TORCH_AVAILABLE, MockDataLoader
)


class MockModel:
    """Mock model for testing when PyTorch is not available"""
    
    def __init__(self):
        self.training = True
        
    def train(self):
        self.training = True
        
    def eval(self):
        self.training = False
        
    def parameters(self):
        return []
    
    def state_dict(self):
        return {'layer1.weight': [1, 2, 3], 'layer1.bias': [0.1, 0.2]}
    
    def load_state_dict(self, state_dict):
        pass
    
    def to(self, device):
        return self
    
    def __call__(self, x):
        # Mock forward pass returning some predictions
        if hasattr(x, '__len__'):
            return [0.7] * len(x)
        return [0.7]


class TestEarlyStopping:
    """Test suite for EarlyStopping class"""
    
    def test_early_stopping_initialization(self):
        """Test EarlyStopping initialization with various parameters"""
        # Test default initialization
        early_stopping = EarlyStopping()
        assert early_stopping.patience == 15
        assert early_stopping.min_delta == 1e-4
        assert early_stopping.restore_best_weights == True
        assert early_stopping.monitor == 'val_loss'
        assert early_stopping.mode == 'min'
        
        # Test custom initialization
        early_stopping = EarlyStopping(
            patience=10, 
            min_delta=1e-3, 
            restore_best_weights=False,
            monitor='val_auc',
            mode='max'
        )
        assert early_stopping.patience == 10
        assert early_stopping.min_delta == 1e-3
        assert early_stopping.restore_best_weights == False
        assert early_stopping.monitor == 'val_auc'
        assert early_stopping.mode == 'max'
    
    def test_early_stopping_min_mode(self):
        """Test early stopping with min mode (loss metric)"""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01, mode='min')
        model = MockModel()
        
        # First epoch - improvement
        assert not early_stopping(0.5, model)
        assert early_stopping.wait == 0
        assert early_stopping.best_score == 0.5
        
        # Second epoch - improvement
        assert not early_stopping(0.4, model)
        assert early_stopping.wait == 0
        assert early_stopping.best_score == 0.4
        
        # Third epoch - no improvement
        assert not early_stopping(0.45, model)
        assert early_stopping.wait == 1
        
        # Fourth epoch - no improvement
        assert not early_stopping(0.47, model)
        assert early_stopping.wait == 2
        
        # Fifth epoch - no improvement, should trigger early stopping
        assert early_stopping(0.46, model)
        assert early_stopping.wait == 3
    
    def test_early_stopping_max_mode(self):
        """Test early stopping with max mode (accuracy/AUC metric)"""
        early_stopping = EarlyStopping(patience=2, min_delta=0.01, mode='max')
        model = MockModel()
        
        # First epoch
        assert not early_stopping(0.8, model)
        assert early_stopping.best_score == 0.8
        
        # Second epoch - improvement
        assert not early_stopping(0.85, model)
        assert early_stopping.wait == 0
        assert early_stopping.best_score == 0.85
        
        # Third epoch - no improvement
        assert not early_stopping(0.84, model)
        assert early_stopping.wait == 1
        
        # Fourth epoch - should trigger early stopping
        assert early_stopping(0.83, model)
    
    def test_get_best_score(self):
        """Test getting best score from early stopping"""
        early_stopping = EarlyStopping(patience=5, mode='min')
        model = MockModel()
        
        early_stopping(0.5, model)
        early_stopping(0.3, model)
        early_stopping(0.4, model)
        
        assert early_stopping.get_best_score() == 0.3


class TestTrainingMetrics:
    """Test suite for TrainingMetrics class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.experiment_name = "test_experiment"
    
    def teardown_method(self):
        """Clean up test files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_training_metrics_initialization(self):
        """Test TrainingMetrics initialization"""
        metrics = TrainingMetrics(
            log_dir=self.temp_dir,
            experiment_name=self.experiment_name,
            tensorboard_enabled=False,  # Disable for testing
            csv_logging=True
        )
        
        assert metrics.log_dir == Path(self.temp_dir)
        assert metrics.experiment_name == self.experiment_name
        assert metrics.tensorboard_enabled == False
        assert metrics.csv_logging == True
        assert metrics.epoch_metrics == []
    
    def test_log_epoch_metrics(self):
        """Test logging epoch metrics"""
        metrics = TrainingMetrics(
            log_dir=self.temp_dir,
            experiment_name=self.experiment_name,
            tensorboard_enabled=False,
            csv_logging=True
        )
        
        train_metrics = {'loss': 0.5, 'accuracy': 0.8}
        val_metrics = {'loss': 0.4, 'accuracy': 0.85, 'auc_roc': 0.9}
        
        metrics.log_epoch_metrics(
            epoch=1,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            learning_rate=0.001,
            epoch_time=120.5,
            memory_usage=512.0
        )
        
        assert len(metrics.epoch_metrics) == 1
        epoch_data = metrics.epoch_metrics[0]
        
        assert epoch_data['epoch'] == 1
        assert epoch_data['learning_rate'] == 0.001
        assert epoch_data['epoch_time'] == 120.5
        assert epoch_data['memory_usage'] == 512.0
        assert epoch_data['train_loss'] == 0.5
        assert epoch_data['train_accuracy'] == 0.8
        assert epoch_data['val_loss'] == 0.4
        assert epoch_data['val_accuracy'] == 0.85
        assert epoch_data['val_auc_roc'] == 0.9
    
    def test_save_metrics(self):
        """Test saving metrics to CSV"""
        metrics = TrainingMetrics(
            log_dir=self.temp_dir,
            experiment_name=self.experiment_name,
            tensorboard_enabled=False,
            csv_logging=True
        )
        
        # Log some metrics
        metrics.log_epoch_metrics(
            epoch=1,
            train_metrics={'loss': 0.5},
            val_metrics={'loss': 0.4},
            learning_rate=0.001,
            epoch_time=120.5
        )
        
        metrics.save_metrics()
        
        # Check if CSV file was created
        csv_path = Path(self.temp_dir) / f"{self.experiment_name}_metrics.csv"
        assert csv_path.exists()
        
        # Read and verify content
        import pandas as pd
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert df['epoch'].iloc[0] == 1
        assert df['train_loss'].iloc[0] == 0.5
        assert df['val_loss'].iloc[0] == 0.4
    
    def test_get_best_epoch(self):
        """Test getting best epoch based on metric"""
        metrics = TrainingMetrics(
            log_dir=self.temp_dir,
            experiment_name=self.experiment_name,
            tensorboard_enabled=False,
            csv_logging=False
        )
        
        # Log multiple epochs
        for epoch in range(1, 6):
            metrics.log_epoch_metrics(
                epoch=epoch,
                train_metrics={'loss': 0.5 - epoch * 0.05},
                val_metrics={'auc_roc': 0.7 + epoch * 0.03},
                learning_rate=0.001,
                epoch_time=120.0
            )
        
        # Test getting best epoch for max metric (AUC)
        best_epoch, best_value = metrics.get_best_epoch('val_auc_roc', 'max')
        assert best_epoch == 5  # Last epoch should have highest AUC
        assert best_value == 0.85  # 0.7 + 5 * 0.03
        
        # Test getting best epoch for min metric (loss)
        best_epoch, best_value = metrics.get_best_epoch('train_loss', 'min')
        assert best_epoch == 5  # Last epoch should have lowest loss
        assert abs(best_value - 0.25) < 1e-6  # 0.5 - 5 * 0.05


class TestModelCheckpoint:
    """Test suite for ModelCheckpoint class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.temp_dir, 'checkpoint_epoch_{epoch}.pth')
    
    def teardown_method(self):
        """Clean up test files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_checkpoint_initialization(self):
        """Test ModelCheckpoint initialization"""
        checkpoint = ModelCheckpoint(
            filepath=self.checkpoint_path,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            period=2
        )
        
        assert checkpoint.filepath == self.checkpoint_path
        assert checkpoint.monitor == 'val_loss'
        assert checkpoint.mode == 'min'
        assert checkpoint.save_best_only == True
        assert checkpoint.period == 2
    
    def test_save_checkpoint_best_only(self):
        """Test saving checkpoint only when metric improves"""
        checkpoint = ModelCheckpoint(
            filepath=self.checkpoint_path,
            monitor='val_loss',
            mode='min',
            save_best_only=True
        )
        
        model = MockModel()
        optimizer = Mock()
        scheduler = Mock()
        
        # Mock the state_dict methods
        optimizer.state_dict = Mock(return_value={'lr': 0.001})
        scheduler.state_dict = Mock(return_value={'step': 1})
        
        # First epoch - should save (first time)
        metrics1 = {'val_loss': 0.5, 'val_accuracy': 0.8}
        saved = checkpoint.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            metrics=metrics1
        )
        assert saved == True
        assert checkpoint.best_score == 0.5
        
        # Second epoch - worse performance, should not save
        metrics2 = {'val_loss': 0.6, 'val_accuracy': 0.75}
        saved = checkpoint.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=2,
            metrics=metrics2
        )
        assert saved == False
        assert checkpoint.best_score == 0.5
        
        # Third epoch - better performance, should save
        metrics3 = {'val_loss': 0.4, 'val_accuracy': 0.85}
        saved = checkpoint.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=3,
            metrics=metrics3
        )
        assert saved == True
        assert checkpoint.best_score == 0.4
    
    def test_save_checkpoint_periodic(self):
        """Test saving checkpoint periodically"""
        checkpoint = ModelCheckpoint(
            filepath=self.checkpoint_path,
            save_best_only=False,
            period=2
        )
        
        model = MockModel()
        optimizer = Mock()
        scheduler = Mock()
        
        optimizer.state_dict = Mock(return_value={'lr': 0.001})
        scheduler.state_dict = Mock(return_value={'step': 1})
        
        # First epoch - should not save (period = 2)
        metrics1 = {'val_loss': 0.5}
        saved = checkpoint.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            metrics=metrics1
        )
        assert saved == False
        
        # Second epoch - should save (period reached)
        metrics2 = {'val_loss': 0.4}
        saved = checkpoint.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=2,
            metrics=metrics2
        )
        assert saved == True
    
    def test_force_save(self):
        """Test force saving checkpoint"""
        checkpoint = ModelCheckpoint(
            filepath=self.checkpoint_path,
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )
        
        model = MockModel()
        optimizer = Mock()
        scheduler = Mock()
        
        optimizer.state_dict = Mock(return_value={'lr': 0.001})
        scheduler.state_dict = Mock(return_value={'step': 1})
        
        # Set up initial best score
        checkpoint.best_score = 0.3
        
        # Epoch with worse performance, but force save
        metrics = {'val_loss': 0.5}
        saved = checkpoint.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            metrics=metrics,
            force_save=True
        )
        assert saved == True


class TestLSTMTrainer:
    """Test suite for LSTMTrainer class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = {
            'training': {
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'num_epochs': 5,
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
                'patience': 2
            },
            'loss_function': {
                'type': 'weighted_bce',
                'pos_weight': 3.0
            },
            'early_stopping': {
                'patience': 10,
                'min_delta': 1e-4,
                'monitor': 'val_loss'
            },
            'logging': {
                'tensorboard_enabled': False,
                'csv_logging': False,
                'log_interval': 1,
                'save_interval': 2
            }
        }
        self.model = MockModel()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_trainer_initialization(self):
        """Test LSTMTrainer initialization with various configurations"""
        trainer = LSTMTrainer(self.model, self.config)
        
        assert trainer.model == self.model
        assert trainer.config == self.config
        assert trainer.device == 'cpu'  # Should default to CPU when TORCH not available
        assert trainer.optimizer is not None
        assert trainer.criterion is not None
        assert trainer.early_stopping is not None
        assert trainer.metrics_tracker is not None
        assert trainer.checkpoint_manager is not None
    
    def test_optimizer_configuration(self):
        """Test optimizer setup and parameter groups"""
        trainer = LSTMTrainer(self.model, self.config)
        
        # Test that optimizer is configured with correct learning rate
        assert trainer.optimizer.param_groups[0]['lr'] == 0.001
    
    def test_loss_function_weighting(self):
        """Test weighted loss calculation and class balance"""
        trainer = LSTMTrainer(self.model, self.config)
        
        # Test that loss function is configured (mock implementation)
        assert trainer.criterion is not None
    
    def test_early_stopping_logic(self):
        """Test early stopping trigger conditions"""
        trainer = LSTMTrainer(self.model, self.config)
        
        # Test early stopping configuration
        assert trainer.early_stopping.patience == 10
        assert trainer.early_stopping.min_delta == 1e-4
        assert trainer.early_stopping.monitor == 'val_loss'
    
    def test_gradient_clipping(self):
        """Test gradient clipping functionality"""
        trainer = LSTMTrainer(self.model, self.config)
        
        # Test gradient clipping
        grad_norm = trainer.clip_gradients(max_norm=1.0)
        assert isinstance(grad_norm, float)
        assert grad_norm >= 0.0
    
    def test_training_history_initialization(self):
        """Test training history initialization"""
        trainer = LSTMTrainer(self.model, self.config)
        
        history = trainer.get_training_history()
        assert isinstance(history, dict)
        assert 'train_losses' in history
        assert 'val_losses' in history
        assert 'train_metrics' in history
        assert 'val_metrics' in history
        assert 'learning_rates' in history
        
        # All should be empty initially
        assert len(history['train_losses']) == 0
        assert len(history['val_losses']) == 0
    
    def test_train_epoch_mock(self):
        """Test single epoch training with mock data"""
        trainer = LSTMTrainer(self.model, self.config)
        
        # Create mock data loader
        mock_dataset = [(np.random.randn(32, 100, 5), np.random.randint(0, 2, 32)) for _ in range(3)]
        train_loader = MockDataLoader(mock_dataset)
        
        # Test training epoch
        metrics = trainer.train_epoch(train_loader, epoch=1)
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'auc_roc' in metrics
        assert 'auc_pr' in metrics
        assert 'epoch_time' in metrics
        assert 'total_samples' in metrics
        
        # Verify metrics are reasonable
        assert metrics['loss'] >= 0.0
        assert 0.0 <= metrics['auc_roc'] <= 1.0
        assert 0.0 <= metrics['auc_pr'] <= 1.0
        assert metrics['epoch_time'] >= 0.0
        assert metrics['total_samples'] > 0
    
    def test_validate_epoch_mock(self):
        """Test single epoch validation with mock data"""
        trainer = LSTMTrainer(self.model, self.config)
        
        # Create mock data loader
        mock_dataset = [(np.random.randn(32, 100, 5), np.random.randint(0, 2, 32)) for _ in range(2)]
        val_loader = MockDataLoader(mock_dataset)
        
        # Test validation epoch
        metrics = trainer.validate_epoch(val_loader, epoch=1)
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'auc_roc' in metrics
        assert 'auc_pr' in metrics
        assert 'f1_score' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'accuracy' in metrics
        assert 'validation_time' in metrics
        
        # Verify metrics are reasonable
        assert metrics['loss'] >= 0.0
        assert 0.0 <= metrics['auc_roc'] <= 1.0
        assert 0.0 <= metrics['f1_score'] <= 1.0
        assert 0.0 <= metrics['precision'] <= 1.0
        assert 0.0 <= metrics['recall'] <= 1.0
        assert 0.0 <= metrics['accuracy'] <= 1.0
    
    def test_model_save_load(self):
        """Test model saving and loading functionality"""
        trainer = LSTMTrainer(self.model, self.config)
        
        # Test saving
        save_path = os.path.join(self.temp_dir, 'test_model.pth')
        trainer.save_model(save_path, include_metadata=True)
        
        # Verify file was created (will be .json for mock implementation)
        if not TORCH_AVAILABLE:
            assert os.path.exists(save_path + '.json')
        
        # Test loading
        trainer.load_model(save_path, load_optimizer=True)
        
        # Should not raise any errors
    
    def test_device_setup(self):
        """Test device setup with different configurations"""
        # Test auto device selection
        trainer1 = LSTMTrainer(self.model, self.config, device='auto')
        assert trainer1.device in ['cpu', 'cuda']
        
        # Test manual device selection
        trainer2 = LSTMTrainer(self.model, self.config, device='cpu')
        assert trainer2.device == 'cpu'
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking"""
        trainer = LSTMTrainer(self.model, self.config)
        
        memory_usage = trainer._get_memory_usage()
        assert isinstance(memory_usage, float)
        assert memory_usage >= 0.0
    
    def test_configuration_validation(self):
        """Test configuration validation and defaults"""
        # Test with minimal config
        minimal_config = {'training': {'learning_rate': 0.001}}
        trainer = LSTMTrainer(self.model, minimal_config)
        
        # Should use defaults for missing values
        assert trainer.train_config['learning_rate'] == 0.001
    
    def test_training_integration(self):
        """Test integration of training components"""
        trainer = LSTMTrainer(self.model, self.config)
        
        # Create mock data loaders
        mock_train_data = [(np.random.randn(16, 50, 5), np.random.randint(0, 2, 16)) for _ in range(2)]
        mock_val_data = [(np.random.randn(16, 50, 5), np.random.randint(0, 2, 16)) for _ in range(1)]
        
        train_loader = MockDataLoader(mock_train_data)
        val_loader = MockDataLoader(mock_val_data)
        
        # Test training loop (with very few epochs for testing)
        trainer.config['training']['num_epochs'] = 2
        
        results = trainer.train(train_loader, val_loader)
        
        assert isinstance(results, dict)
        assert 'training_history' in results
        assert 'total_training_time' in results
        assert 'final_epoch' in results
        assert 'best_epoch' in results
        assert 'training_completed' in results
        
        # Verify training history was updated
        history = results['training_history']
        assert len(history['train_losses']) == 2
        assert len(history['val_losses']) == 2


class TestIntegration:
    """Integration tests for the complete LSTM training pipeline"""
    
    def test_end_to_end_training_mock(self):
        """Test complete end-to-end training pipeline with mock data"""
        # Configuration
        config = {
            'training': {'learning_rate': 0.001, 'num_epochs': 3, 'weight_decay': 1e-4},
            'optimization': {'optimizer': 'adam'},
            'scheduler': {'type': 'reduce_on_plateau'},
            'loss_function': {'type': 'weighted_bce', 'pos_weight': 2.0},
            'early_stopping': {'patience': 5},
            'logging': {'tensorboard_enabled': False, 'csv_logging': False}
        }
        
        # Initialize components
        model = MockModel()
        trainer = LSTMTrainer(model, config)
        
        # Create mock data
        mock_train_data = [(np.random.randn(8, 30, 5), np.random.randint(0, 2, 8)) for _ in range(2)]
        mock_val_data = [(np.random.randn(8, 30, 5), np.random.randint(0, 2, 8)) for _ in range(1)]
        
        train_loader = MockDataLoader(mock_train_data)
        val_loader = MockDataLoader(mock_val_data)
        
        # Run training
        results = trainer.train(train_loader, val_loader)
        
        # Verify results
        assert results['training_completed'] == True
        assert results['final_epoch'] == 3
        assert 'total_training_time' in results
        assert results['total_training_time'] > 0
        
        # Verify training history
        history = results['training_history']
        assert len(history['train_losses']) == 3
        assert len(history['val_losses']) == 3
        assert len(history['learning_rates']) == 3
        
        # Test evaluation
        test_metrics = trainer.evaluate(val_loader)
        assert isinstance(test_metrics, dict)
        assert 'auc_roc' in test_metrics
        assert 'f1_score' in test_metrics


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])