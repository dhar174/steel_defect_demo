#!/usr/bin/env python3
"""
LSTM Model Training Script for Steel Casting Defect Prediction

Usage:
    python scripts/train_lstm_model.py --config configs/model_config.yaml
    python scripts/train_lstm_model.py --config configs/model_config.yaml --gpu 0
    python scripts/train_lstm_model.py --experiment steel_lstm_v1 --epochs 150
"""

import argparse
import yaml
from pathlib import Path
import sys
import json
import random
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Conditional imports with graceful fallback
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Running in mock mode for development.")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Mock tqdm for development
    class tqdm:
        def __init__(self, iterable=None, desc="", total=None, position=0, initial=0, leave=True):
            self.iterable = iterable or []
            self.desc = desc
            self.total = total or len(self.iterable) if hasattr(self.iterable, '__len__') else 100
            self.position = position
            self.n = initial
            
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.update(1)
            
        def update(self, n=1):
            self.n += n
            
        def set_description(self, desc):
            self.desc = desc
            
        def close(self):
            pass

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Optional imports for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Import our models and utilities
from models.lstm_model import SteelDefectLSTM, CastingSequenceDataset, create_default_lstm_config
from models.model_trainer import LSTMTrainer


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments with comprehensive options
    
    Returns:
    - args: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='Train LSTM model for steel casting defect prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration arguments
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory containing processed data')
    parser.add_argument('--output-dir', type=str, default='models/deep_learning',
                       help='Directory to save trained models')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--early-stopping-patience', type=int, default=None,
                       help='Early stopping patience (overrides config)')
    
    # Device and performance arguments
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU device ID (-1 for CPU, None for auto-detect)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--pin-memory', action='store_true',
                       help='Use pin memory for data loading')
    
    # Experiment tracking arguments
    parser.add_argument('--experiment', type=str, default=None,
                       help='Experiment name for tracking')
    parser.add_argument('--tags', nargs='+', default=None,
                       help='Tags for experiment tracking')
    parser.add_argument('--wandb-project', type=str, default='steel-defect-lstm',
                       help='Weights & Biases project name')
    parser.add_argument('--disable-wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    
    # Model and data arguments
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run validation (requires --resume)')
    
    # Logging and monitoring arguments
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Logging interval (batches)')
    parser.add_argument('--save-interval', type=int, default=5,
                       help='Model save interval (epochs)')
    parser.add_argument('--disable-tensorboard', action='store_true',
                       help='Disable TensorBoard logging')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()


def setup_device(gpu_id: Optional[int] = None) -> str:
    """
    Setup training device with automatic GPU detection
    
    Parameters:
    - gpu_id: Specific GPU ID, -1 for CPU, None for auto-detect
    
    Returns:
    - device: PyTorch device string for training
    """
    if gpu_id == -1:
        device = 'cpu'
        print("Using CPU for training")
    elif gpu_id is not None:
        if TORCH_AVAILABLE and torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            device = f'cuda:{gpu_id}'
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            raise ValueError(f"GPU {gpu_id} not available")
    else:
        # Auto-detect best device
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Auto-detected GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            device = 'cpu'
            print("CUDA not available, using CPU")
    
    return device


def print_device_info(device: str) -> None:
    """Print detailed device information"""
    if device.startswith('cuda') and TORCH_AVAILABLE:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print(f"GPU Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
    else:
        if PSUTIL_AVAILABLE:
            print(f"CPU Cores: {psutil.cpu_count()}")
            print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f}GB")
        else:
            print("CPU training mode")


class TrainingProgressMonitor:
    """Real-time training progress monitoring and visualization"""
    
    def __init__(self, total_epochs: int, log_interval: int = 10):
        self.total_epochs = total_epochs
        self.log_interval = log_interval
        self.epoch_progress = None
        self.batch_progress = None
        self.metrics_history = []
        
    def start_epoch(self, epoch: int, total_batches: int) -> None:
        """Initialize epoch progress tracking"""
        if TQDM_AVAILABLE:
            self.epoch_progress = tqdm(
                total=self.total_epochs,
                desc="Training Progress",
                position=0,
                initial=epoch
            )
            self.batch_progress = tqdm(
                total=total_batches,
                desc=f"Epoch {epoch+1}/{self.total_epochs}",
                position=1,
                leave=False
            )
        else:
            print(f"Starting Epoch {epoch+1}/{self.total_epochs}")
    
    def update_batch(self, batch_idx: int, loss: float, 
                    metrics: Dict[str, float] = None) -> None:
        """Update batch progress with metrics"""
        if self.batch_progress:
            self.batch_progress.update(1)
            
            if batch_idx % self.log_interval == 0:
                desc = f"Loss: {loss:.4f}"
                if metrics:
                    for key, value in metrics.items():
                        desc += f" | {key}: {value:.4f}"
                self.batch_progress.set_description(desc)
    
    def end_epoch(self, epoch_metrics: Dict[str, float]) -> None:
        """Complete epoch and update progress"""
        if self.batch_progress:
            self.batch_progress.close()
        if self.epoch_progress:
            self.epoch_progress.update(1)
            
            # Update epoch description with metrics
            desc = "Training Progress"
            if epoch_metrics:
                val_loss = epoch_metrics.get('val_loss', 0)
                val_auc = epoch_metrics.get('val_auc_roc', 0)
                desc += f" | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
            self.epoch_progress.set_description(desc)
        
        self.metrics_history.append(epoch_metrics)
    
    def close(self) -> None:
        """Clean up progress bars"""
        if self.epoch_progress:
            self.epoch_progress.close()
        if self.batch_progress:
            self.batch_progress.close()


class ModelCheckpointManager:
    """Advanced model checkpointing with best model tracking"""
    
    def __init__(self, output_dir: str, save_interval: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval
        self.best_metric = float('-inf')
        self.best_model_path = None
        
    def save_checkpoint(self, model, trainer: 'LSTMTrainer',
                       epoch: int, metrics: Dict[str, float],
                       is_best: bool = False) -> str:
        """
        Save model checkpoint with comprehensive state
        
        Parameters:
        - model: LSTM model to save
        - trainer: Trainer instance with optimizer and scheduler
        - epoch: Current epoch number
        - metrics: Training and validation metrics
        - is_best: Whether this is the best model so far
        
        Returns:
        - checkpoint_path: Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'metrics': metrics,
            'random_states': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
            }
        }
        
        if TORCH_AVAILABLE:
            checkpoint.update({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict() if trainer.scheduler else None,
                'random_states': {
                    **checkpoint['random_states'],
                    'torch': torch.get_rng_state(),
                }
            })
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch:03d}.pth'
        
        if TORCH_AVAILABLE:
            torch.save(checkpoint, checkpoint_path)
        else:
            # Mock save for development
            with open(str(checkpoint_path) + '.json', 'w') as f:
                json.dump({k: str(v) for k, v in checkpoint.items()}, f, indent=2)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            if TORCH_AVAILABLE:
                torch.save(checkpoint, best_path)
            else:
                with open(str(best_path) + '.json', 'w') as f:
                    json.dump({k: str(v) for k, v in checkpoint.items()}, f, indent=2)
            self.best_model_path = best_path
            print(f"New best model saved: {best_path}")
        
        # Clean up old checkpoints (keep last 3)
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def _cleanup_old_checkpoints(self, keep_last: int = 3) -> None:
        """Remove old checkpoint files to save disk space"""
        pattern = 'checkpoint_epoch_*.pth*'
        checkpoints = sorted(self.output_dir.glob(pattern))
        if len(checkpoints) > keep_last:
            for old_checkpoint in checkpoints[:-keep_last]:
                old_checkpoint.unlink()


class ExperimentTracker:
    """Comprehensive experiment tracking with multiple backends"""
    
    def __init__(self, experiment_name: str, config: Dict,
                 wandb_project: str = None, disable_wandb: bool = False,
                 disable_tensorboard: bool = False):
        self.experiment_name = experiment_name
        self.config = config
        self.wandb_enabled = not disable_wandb and self._check_wandb()
        self.tensorboard_enabled = not disable_tensorboard and TENSORBOARD_AVAILABLE
        
        # Initialize tracking backends
        if self.wandb_enabled:
            self._init_wandb(wandb_project)
        
        if self.tensorboard_enabled:
            self._init_tensorboard()
    
    def _check_wandb(self) -> bool:
        """Check if wandb is available and user is logged in"""
        if not WANDB_AVAILABLE:
            print("Weights & Biases not installed, skipping wandb logging")
            return False
        try:
            wandb.api.api_key
            return True
        except Exception:
            print("Weights & Biases not configured, skipping wandb logging")
            return False
    
    def _init_wandb(self, project: str) -> None:
        """Initialize Weights & Biases tracking"""
        if WANDB_AVAILABLE:
            wandb.init(
                project=project,
                name=self.experiment_name,
                config=self.config,
                save_code=True
            )
            print(f"Weights & Biases tracking initialized: {wandb.run.url}")
    
    def _init_tensorboard(self) -> None:
        """Initialize TensorBoard logging"""
        log_dir = Path('logs') / 'tensorboard' / self.experiment_name
        self.tb_writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging initialized: {log_dir}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics to all enabled backends"""
        if self.wandb_enabled and WANDB_AVAILABLE:
            wandb.log(metrics, step=step)
        
        if self.tensorboard_enabled:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)
    
    def log_hyperparameters(self, hparams: Dict[str, Any]) -> None:
        """Log hyperparameters"""
        if self.wandb_enabled and WANDB_AVAILABLE:
            wandb.config.update(hparams)
        
        if self.tensorboard_enabled:
            # Convert complex objects to strings for tensorboard
            simple_hparams = {}
            for k, v in hparams.items():
                if isinstance(v, (int, float, str, bool)):
                    simple_hparams[k] = v
                else:
                    simple_hparams[k] = str(v)
            self.tb_writer.add_hparams(simple_hparams, {})
    
    def log_model_graph(self, model, input_sample) -> None:
        """Log model architecture"""
        if self.tensorboard_enabled and TORCH_AVAILABLE:
            try:
                self.tb_writer.add_graph(model, input_sample)
            except Exception as e:
                print(f"Warning: Could not log model graph: {e}")
    
    def finish(self) -> None:
        """Clean up tracking resources"""
        if self.wandb_enabled and WANDB_AVAILABLE:
            wandb.finish()
        
        if self.tensorboard_enabled:
            self.tb_writer.close()


def load_and_override_config(args: argparse.Namespace) -> Dict:
    """Load configuration and apply command-line overrides"""
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file {args.config} not found, using default configuration")
        config = {'lstm_model': create_default_lstm_config()}
    
    # Apply command-line overrides
    lstm_config = config.get('lstm_model', create_default_lstm_config())
    
    if args.epochs:
        lstm_config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        lstm_config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        lstm_config['training']['learning_rate'] = args.learning_rate
    if args.early_stopping_patience:
        lstm_config.setdefault('early_stopping', {})['patience'] = args.early_stopping_patience
    
    # Update data loading configuration
    if args.num_workers:
        lstm_config.setdefault('data_loading', {})['num_workers'] = args.num_workers
    if args.pin_memory:
        lstm_config.setdefault('data_loading', {})['pin_memory'] = True
    
    config['lstm_model'] = lstm_config
    return config


def generate_experiment_name() -> str:
    """Generate unique experiment name with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"lstm_steel_defect_{timestamp}"


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Ensure deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_sample_data(config: Dict, device: str) -> Tuple:
    """Create sample data loaders for demonstration purposes"""
    lstm_config = config['lstm_model']
    
    # Sample data parameters
    num_samples = 1000
    sequence_length = lstm_config['data_processing']['sequence_length']
    input_size = lstm_config['architecture']['input_size']
    batch_size = lstm_config['training']['batch_size']
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(num_samples, sequence_length, input_size).astype(np.float32)
    y = np.random.randint(0, 2, num_samples).astype(np.float32)
    
    # Split data
    train_size = int(0.6 * num_samples)
    val_size = int(0.2 * num_samples)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    # Create datasets and loaders
    if TORCH_AVAILABLE:
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        # Mock data loaders for development
        class MockDataLoader:
            def __init__(self, X, y, batch_size):
                self.X, self.y = X, y
                self.batch_size = batch_size
                
            def __iter__(self):
                for i in range(0, len(self.X), self.batch_size):
                    yield (self.X[i:i+self.batch_size], self.y[i:i+self.batch_size])
                    
            def __len__(self):
                return (len(self.X) + self.batch_size - 1) // self.batch_size
        
        train_loader = MockDataLoader(X_train, y_train, batch_size)
        val_loader = MockDataLoader(X_val, y_val, batch_size)
        test_loader = MockDataLoader(X_test, y_test, batch_size)
    
    print(f"Created sample datasets: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    return train_loader, val_loader, test_loader


def load_checkpoint(checkpoint_path: str, model, trainer) -> int:
    """Load checkpoint and return epoch number"""
    if TORCH_AVAILABLE and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and checkpoint.get('scheduler_state_dict'):
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint.get('epoch', 0)
    else:
        print(f"Checkpoint {checkpoint_path} not found or PyTorch not available")
        return 0


def print_metrics(metrics: Dict[str, float]) -> None:
    """Print metrics in a formatted way"""
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


def main():
    """Main training function with comprehensive workflow"""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set random seeds for reproducibility
    set_random_seeds(args.seed)
    
    # Load and override configuration
    config = load_and_override_config(args)
    
    # Setup device and print system info
    device = setup_device(args.gpu)
    print_device_info(device)
    
    # Initialize experiment tracking
    experiment_name = args.experiment or generate_experiment_name()
    tracker = ExperimentTracker(
        experiment_name=experiment_name,
        config=config,
        wandb_project=args.wandb_project,
        disable_wandb=args.disable_wandb,
        disable_tensorboard=args.disable_tensorboard
    )
    
    # Log hyperparameters
    tracker.log_hyperparameters(config['lstm_model'])
    
    # Load data and create data loaders (using sample data for demonstration)
    print("Loading data...")
    train_loader, val_loader, test_loader = create_sample_data(config, device)
    
    # Initialize model and move to device
    lstm_config = config['lstm_model']
    model = SteelDefectLSTM(lstm_config)
    if TORCH_AVAILABLE:
        model = model.to(device)
    
    # Log model architecture
    if TORCH_AVAILABLE:
        sample_batch = next(iter(train_loader))
        sample_input = sample_batch[0][:1]  # Take first sample
        if hasattr(sample_input, 'to'):
            sample_input = sample_input.to(device)
        tracker.log_model_graph(model, sample_input)
    
    # Create training configuration for LSTMTrainer
    # Extract training configuration from LSTM config and add missing sections
    training_config = {
        'training': lstm_config.get('training', {}),
        'loss_function': lstm_config.get('loss_function', {}),
        'optimization': {
            'optimizer': 'adam',
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8
        },
        'scheduler': {
            'type': 'reduce_on_plateau',
            'factor': 0.5,
            'patience': 5,
            'min_lr': 1e-6
        },
        'early_stopping': {
            'enabled': True,
            'patience': lstm_config.get('training', {}).get('early_stopping_patience', 15),
            'min_delta': 1e-4,
            'monitor': 'val_loss',
            'restore_best_weights': True
        },
        'logging': {
            'tensorboard_enabled': True,
            'csv_logging': True,
            'log_interval': 10,
            'save_interval': 5
        }
    }
    
    # Initialize trainer
    trainer = LSTMTrainer(model, training_config, device)
    
    # Setup checkpointing
    checkpoint_manager = ModelCheckpointManager(
        args.output_dir, 
        args.save_interval
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, trainer)
        print(f"Resumed training from epoch {start_epoch}")
    
    # Handle validation-only mode
    if args.validate_only:
        if not args.resume:
            raise ValueError("--validate-only requires --resume")
        validation_metrics = trainer.validate_epoch(val_loader, start_epoch)
        print("Validation Results:")
        print_metrics(validation_metrics)
        return
    
    # Initialize progress monitoring
    num_epochs = lstm_config['training']['num_epochs']
    progress_monitor = TrainingProgressMonitor(num_epochs, args.log_interval)
    
    try:
        # Training loop
        best_metric = float('-inf')
        for epoch in range(start_epoch, num_epochs):
            progress_monitor.start_epoch(epoch, len(train_loader))
            
            # Training epoch
            train_metrics = trainer.train_epoch(train_loader, epoch)
            
            # Validation epoch
            val_metrics = trainer.validate_epoch(val_loader, epoch)
            
            # Combine metrics
            epoch_metrics = {**{f'train_{k}': v for k, v in train_metrics.items()}, 
                           **{f'val_{k}': v for k, v in val_metrics.items()}}
            
            # Log metrics
            tracker.log_metrics(epoch_metrics, epoch)
            
            # Update progress
            progress_monitor.end_epoch(val_metrics)
            
            # Check for best model
            current_metric = val_metrics.get('auc_roc', 0)
            is_best = current_metric > best_metric
            if is_best:
                best_metric = current_metric
            
            # Save checkpoint
            if (epoch + 1) % args.save_interval == 0 or is_best:
                checkpoint_manager.save_checkpoint(
                    model, trainer, epoch, val_metrics, is_best
                )
            
            # Early stopping check (if available)
            if hasattr(trainer, 'early_stopping') and trainer.early_stopping:
                monitor_metric = val_metrics[trainer.early_stopping.monitor.replace('val_', '')]
                if trainer.early_stopping(monitor_metric, model):
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
        
        # Final evaluation on test set
        if test_loader:
            test_metrics = trainer.validate_epoch(test_loader, epoch)
            test_metrics_logged = {f'test_{k}': v for k, v in test_metrics.items()}
            tracker.log_metrics(test_metrics_logged, epoch)
            print("Final Test Results:")
            print_metrics(test_metrics)
        
        print(f"Training completed. Best model: {checkpoint_manager.best_model_path}")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
        # Save emergency checkpoint
        checkpoint_manager.save_checkpoint(
            model, trainer, epoch, val_metrics, False
        )
    
    finally:
        progress_monitor.close()
        tracker.finish()


if __name__ == "__main__":
    main()