"""Models package for steel defect prediction"""

from .baseline_model import BaselineXGBoostModel
from .model_trainer import ModelTrainer
from .model_config import ModelConfig
from .model_evaluator import ModelEvaluator
from .model_persistence import ModelPersistence
from .preprocessing import DataPreprocessor
from .hyperparameter_search import HyperparameterSearcher
from .training_config import TrainingPipelineConfig, ConfigurationManager
from .training_utils import TrainingUtils

# Optional PyTorch imports
try:
    from .lstm_model import LSTMModel, LSTMDataset
    from .model_trainer import LSTMTrainer
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    LSTMModel = None
    LSTMDataset = None
    LSTMTrainer = None

__all__ = [
    'BaselineXGBoostModel',
    'ModelTrainer',
    'ModelConfig',
    'ModelEvaluator',
    'ModelPersistence',
    'DataPreprocessor',
    'HyperparameterSearcher',
    'TrainingPipelineConfig',
    'ConfigurationManager',
    'TrainingUtils'
]

# Add PyTorch components if available
if PYTORCH_AVAILABLE:
    __all__.extend(['LSTMModel', 'LSTMDataset', 'LSTMTrainer'])
