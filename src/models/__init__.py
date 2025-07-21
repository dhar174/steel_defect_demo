"""Models package for steel defect prediction"""

# Optional imports with error handling
try:
    from .baseline_model import BaselineXGBoostModel
except ImportError:
    BaselineXGBoostModel = None

try:
    from .model_trainer import ModelTrainer
except ImportError:
    ModelTrainer = None

from .model_config import ModelConfig

try:
    from .model_evaluator import ModelEvaluator
except ImportError:
    ModelEvaluator = None

try:
    from .model_persistence import ModelPersistence
except ImportError:
    ModelPersistence = None

try:
    from .preprocessing import DataPreprocessor
except ImportError:
    DataPreprocessor = None

try:
    from .hyperparameter_search import HyperparameterSearcher
except ImportError:
    HyperparameterSearcher = None

try:
    from .training_config import TrainingPipelineConfig, ConfigurationManager
except ImportError:
    TrainingPipelineConfig = None
    ConfigurationManager = None

try:
    from .training_utils import TrainingUtils
except ImportError:
    TrainingUtils = None

# PyTorch imports
try:
    from .lstm_model import SteelDefectLSTM, CastingSequenceDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    SteelDefectLSTM = None
    CastingSequenceDataset = None

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
    __all__.extend(['SteelDefectLSTM', 'CastingSequenceDataset'])
