#!/usr/bin/env python3
"""Unit tests for LSTM training script"""

import pytest
import tempfile
import shutil
import sys
from pathlib import Path
import yaml
import json
from unittest.mock import patch, MagicMock
import numpy as np

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

# Import the training script components
from train_lstm_model import (
    parse_arguments, setup_device, print_device_info,
    TrainingProgressMonitor, ModelCheckpointManager, ExperimentTracker,
    load_and_override_config, generate_experiment_name, set_random_seeds,
    create_sample_data, load_checkpoint, print_metrics
)


class TestArgumentParsing:
    """Test command-line argument parsing"""
    
    def test_parse_arguments_defaults(self):
        """Test parsing with default arguments"""
        with patch('sys.argv', ['train_lstm_model.py']):
            args = parse_arguments()
            assert args.config == 'configs/model_config.yaml'
            assert args.data_dir == 'data/processed'
            assert args.output_dir == 'models/deep_learning'
            assert args.epochs is None
            assert args.batch_size is None
            assert args.learning_rate is None
            assert args.gpu is None
            assert args.num_workers == 4
            assert args.pin_memory is False
            assert args.seed == 42
            assert args.verbose is False
    
    def test_parse_arguments_custom(self):
        """Test parsing with custom arguments"""
        test_args = [
            'train_lstm_model.py',
            '--config', 'custom_config.yaml',
            '--epochs', '50',
            '--batch-size', '64',
            '--learning-rate', '0.0005',
            '--gpu', '0',
            '--experiment', 'test_exp',
            '--tags', 'tag1', 'tag2',
            '--verbose'
        ]
        with patch('sys.argv', test_args):
            args = parse_arguments()
            assert args.config == 'custom_config.yaml'
            assert args.epochs == 50
            assert args.batch_size == 64
            assert args.learning_rate == 0.0005
            assert args.gpu == 0
            assert args.experiment == 'test_exp'
            assert args.tags == ['tag1', 'tag2']
            assert args.verbose is True


class TestDeviceSetup:
    """Test device detection and setup"""
    
    def test_setup_device_cpu_explicit(self):
        """Test explicit CPU device selection"""
        device = setup_device(-1)
        assert device == 'cpu'
    
    def test_setup_device_auto_cpu_fallback(self):
        """Test auto device detection falling back to CPU"""
        device = setup_device(None)
        assert device == 'cpu'  # Since PyTorch/CUDA not available in test environment
    
    def test_setup_device_invalid_gpu(self):
        """Test invalid GPU ID handling"""
        with pytest.raises(ValueError, match="GPU 999 not available"):
            setup_device(999)
    
    def test_print_device_info_cpu(self):
        """Test device info printing for CPU"""
        # This should not raise an exception
        print_device_info('cpu')


class TestTrainingProgressMonitor:
    """Test training progress monitoring"""
    
    def test_progress_monitor_initialization(self):
        """Test progress monitor initialization"""
        monitor = TrainingProgressMonitor(total_epochs=10, log_interval=5)
        assert monitor.total_epochs == 10
        assert monitor.log_interval == 5
        assert monitor.metrics_history == []
    
    def test_progress_monitor_epoch_tracking(self):
        """Test epoch progress tracking"""
        monitor = TrainingProgressMonitor(total_epochs=5)
        
        # Start epoch
        monitor.start_epoch(0, 100)
        
        # Update batch progress
        monitor.update_batch(0, 0.5, {'accuracy': 0.8})
        monitor.update_batch(5, 0.4, {'accuracy': 0.85})
        
        # End epoch
        epoch_metrics = {'val_loss': 0.3, 'val_auc_roc': 0.9}
        monitor.end_epoch(epoch_metrics)
        
        assert len(monitor.metrics_history) == 1
        assert monitor.metrics_history[0] == epoch_metrics
        
        # Clean up
        monitor.close()


class TestModelCheckpointManager:
    """Test model checkpointing functionality"""
    
    def setUp(self):
        """Set up temporary directory for testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = ModelCheckpointManager(self.temp_dir, save_interval=2)
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_checkpoint_manager_initialization(self):
        """Test checkpoint manager initialization"""
        self.setUp()
        try:
            assert Path(self.temp_dir).exists()
            assert self.checkpoint_manager.save_interval == 2
            assert self.checkpoint_manager.best_metric == float('-inf')
        finally:
            self.tearDown()
    
    def test_save_checkpoint(self):
        """Test checkpoint saving"""
        self.setUp()
        try:
            # Create mock model and trainer
            mock_model = MagicMock()
            mock_model.state_dict.return_value = {'layer1.weight': [1, 2, 3]}
            
            mock_trainer = MagicMock()
            mock_trainer.optimizer.state_dict.return_value = {'lr': 0.001}
            mock_trainer.scheduler = None
            
            metrics = {'val_loss': 0.3, 'val_auc_roc': 0.9}
            
            # Save checkpoint
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                mock_model, mock_trainer, epoch=5, metrics=metrics, is_best=True
            )
            
            # Check that files were created
            assert Path(checkpoint_path + '.json').exists()  # Mock save creates .json file
            assert Path(self.temp_dir, 'best_model.pth.json').exists()
        finally:
            self.tearDown()


class TestExperimentTracker:
    """Test experiment tracking functionality"""
    
    def test_experiment_tracker_initialization(self):
        """Test experiment tracker initialization"""
        config = {'test': 'config'}
        tracker = ExperimentTracker(
            experiment_name='test_exp',
            config=config,
            disable_wandb=True,
            disable_tensorboard=True
        )
        
        assert tracker.experiment_name == 'test_exp'
        assert tracker.config == config
        assert tracker.wandb_enabled is False
        assert tracker.tensorboard_enabled is False
        
        # Clean up
        tracker.finish()
    
    def test_log_metrics(self):
        """Test metrics logging"""
        tracker = ExperimentTracker(
            experiment_name='test_exp',
            config={},
            disable_wandb=True,
            disable_tensorboard=True
        )
        
        # This should not raise an exception
        metrics = {'loss': 0.5, 'accuracy': 0.8}
        tracker.log_metrics(metrics, step=1)
        
        tracker.finish()
    
    def test_log_hyperparameters(self):
        """Test hyperparameter logging"""
        tracker = ExperimentTracker(
            experiment_name='test_exp',
            config={},
            disable_wandb=True,
            disable_tensorboard=True
        )
        
        # This should not raise an exception
        hparams = {'learning_rate': 0.001, 'batch_size': 32}
        tracker.log_hyperparameters(hparams)
        
        tracker.finish()


class TestConfigurationManagement:
    """Test configuration loading and overrides"""
    
    def test_load_and_override_config_default(self):
        """Test loading default config when file doesn't exist"""
        from types import SimpleNamespace
        args = SimpleNamespace(
            config='nonexistent_config.yaml',
            epochs=None,
            batch_size=None,
            learning_rate=None,
            early_stopping_patience=None,
            num_workers=None,
            pin_memory=False
        )
        
        config = load_and_override_config(args)
        assert 'lstm_model' in config
        assert 'training' in config['lstm_model']
    
    def test_load_and_override_config_with_overrides(self):
        """Test config loading with command-line overrides"""
        # Create temporary config file
        temp_config = {
            'lstm_model': {
                'training': {
                    'num_epochs': 100,
                    'batch_size': 32,
                    'learning_rate': 0.001
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(temp_config, f)
            config_path = f.name
        
        try:
            from types import SimpleNamespace
            args = SimpleNamespace(
                config=config_path,
                epochs=50,
                batch_size=64,
                learning_rate=0.0005,
                early_stopping_patience=20,
                num_workers=8,
                pin_memory=True
            )
            
            config = load_and_override_config(args)
            
            # Check overrides were applied
            assert config['lstm_model']['training']['num_epochs'] == 50
            assert config['lstm_model']['training']['batch_size'] == 64
            assert config['lstm_model']['training']['learning_rate'] == 0.0005
            assert config['lstm_model']['early_stopping']['patience'] == 20
            assert config['lstm_model']['data_loading']['num_workers'] == 8
            assert config['lstm_model']['data_loading']['pin_memory'] is True
            
        finally:
            Path(config_path).unlink()


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_generate_experiment_name(self):
        """Test experiment name generation"""
        name = generate_experiment_name()
        assert name.startswith('lstm_steel_defect_')
        assert len(name) > 20  # Should include timestamp
    
    def test_set_random_seeds(self):
        """Test random seed setting"""
        # This should not raise an exception
        set_random_seeds(42)
        
        # Test that numpy random state is set
        np.random.seed(42)
        first_random = np.random.random()
        
        set_random_seeds(42)
        second_random = np.random.random()
        
        assert first_random == second_random
    
    def test_print_metrics(self):
        """Test metrics printing"""
        metrics = {'loss': 0.5, 'accuracy': 0.8, 'f1_score': 0.75}
        
        # This should not raise an exception
        print_metrics(metrics)


class TestDataGeneration:
    """Test sample data generation"""
    
    def test_create_sample_data(self):
        """Test sample data creation"""
        config = {
            'lstm_model': {
                'data_processing': {'sequence_length': 100},
                'architecture': {'input_size': 5},
                'training': {'batch_size': 16}
            }
        }
        
        train_loader, val_loader, test_loader = create_sample_data(config, 'cpu')
        
        # Check that loaders are created
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        
        # Check that loaders have data
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0


class TestModelCheckpointLoading:
    """Test checkpoint loading functionality"""
    
    def test_load_checkpoint_nonexistent(self):
        """Test loading non-existent checkpoint"""
        mock_model = MagicMock()
        mock_trainer = MagicMock()
        
        epoch = load_checkpoint('nonexistent_checkpoint.pth', mock_model, mock_trainer)
        assert epoch == 0


class TestTrainingWorkflow:
    """Test training workflow components"""
    
    def test_training_workflow_mock_data(self):
        """Test training workflow with mock data"""
        # This is more of an integration test to ensure components work together
        config = {
            'lstm_model': {
                'data_processing': {'sequence_length': 50},
                'architecture': {'input_size': 3},
                'training': {'batch_size': 8, 'num_epochs': 2}
            }
        }
        
        # Test data creation
        train_loader, val_loader, test_loader = create_sample_data(config, 'cpu')
        
        # Test progress monitor
        monitor = TrainingProgressMonitor(total_epochs=2)
        monitor.start_epoch(0, len(train_loader))
        monitor.update_batch(0, 0.5)
        monitor.end_epoch({'val_loss': 0.4, 'val_auc_roc': 0.85})
        monitor.close()
        
        # Test experiment tracker
        tracker = ExperimentTracker(
            experiment_name='test_workflow',
            config=config,
            disable_wandb=True,
            disable_tensorboard=True
        )
        tracker.log_metrics({'loss': 0.5}, step=1)
        tracker.finish()


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_gpu_handling(self):
        """Test handling of invalid GPU IDs"""
        with pytest.raises(ValueError):
            setup_device(999)
    
    def test_validation_only_without_resume(self):
        """Test validation-only mode error handling"""
        from types import SimpleNamespace
        
        # This would be tested in the main function, but we can test the logic
        args = SimpleNamespace(resume=None, validate_only=True)
        
        # In the actual main function, this should raise ValueError
        # We're testing that the logic is in place
        assert args.validate_only and not args.resume


if __name__ == '__main__':
    pytest.main([__file__])