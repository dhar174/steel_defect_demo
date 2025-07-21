import yaml
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    name: str = "baseline_experiment"
    description: str = "Baseline model training"
    tags: List[str] = field(default_factory=lambda: ["baseline"])


@dataclass
class DataConfig:
    """Data configuration"""
    target_column: str = "defect"
    feature_columns: Optional[List[str]] = None
    test_size: float = 0.2
    validation_size: float = 0.2
    stratify: bool = True
    random_state: int = 42


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration"""
    handle_missing: bool = True
    missing_strategy: str = "median"
    categorical_missing_strategy: str = "most_frequent"
    
    # Scaling configuration
    scaling: Dict[str, Union[str, List]] = field(default_factory=lambda: {
        "method": "standard",
        "feature_range": [0, 1]
    })
    
    # Outlier handling
    outliers: Dict[str, Union[str, float]] = field(default_factory=lambda: {
        "method": "iqr",
        "threshold": 1.5
    })
    
    # Feature selection
    feature_selection: Dict[str, Union[bool, str, int]] = field(default_factory=lambda: {
        "enabled": False,
        "method": "variance",
        "k_best": 50
    })


@dataclass
class ModelConfig:
    """Model configuration"""
    type: str = "xgboost"
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1
    })


@dataclass
class CrossValidationConfig:
    """Cross-validation configuration"""
    enabled: bool = True
    cv_folds: int = 5
    stratify: bool = True
    shuffle: bool = True


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration"""
    enabled: bool = True
    monitor: str = "val_auc"
    patience: int = 10
    min_delta: float = 0.001
    restore_best_weights: bool = True


@dataclass
class ClassBalancingConfig:
    """Class balancing configuration"""
    method: str = "weights"  # weights, smote, undersampling, none
    ratio: str = "balanced"


@dataclass
class TrainingConfig:
    """Training configuration"""
    cross_validation: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    class_balancing: ClassBalancingConfig = field(default_factory=ClassBalancingConfig)


@dataclass
class HyperparameterSearchConfig:
    """Hyperparameter search configuration"""
    enabled: bool = True
    method: str = "grid"  # grid, random, bayesian
    cv_folds: int = 3
    scoring: str = "roc_auc"
    n_jobs: int = -1
    
    # Parameter grids
    param_grids: Dict[str, Dict[str, List]] = field(default_factory=lambda: {
        "coarse": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.05, 0.1, 0.2]
        }
    })
    
    # Random search parameters
    random_search: Dict[str, int] = field(default_factory=lambda: {
        "n_iter": 100
    })
    
    # Bayesian optimization parameters
    bayesian_search: Dict[str, Union[int, str]] = field(default_factory=lambda: {
        "n_calls": 50,
        "acq_func": "EI"
    })


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    metrics: List[str] = field(default_factory=lambda: [
        "roc_auc", "average_precision", "f1", "precision", "recall"
    ])
    threshold_optimization: bool = True
    plot_curves: bool = True
    confusion_matrix: bool = True


@dataclass
class VersioningConfig:
    """Model versioning configuration"""
    enabled: bool = True
    format: str = "{model_type}_{timestamp}_{performance}"


@dataclass
class SaveArtifactsConfig:
    """Save artifacts configuration"""
    model: bool = True
    preprocessor: bool = True
    training_history: bool = True
    feature_importance: bool = True
    evaluation_plots: bool = True


@dataclass
class OutputConfig:
    """Output configuration"""
    model_dir: str = "models/artifacts"
    results_dir: str = "results/training"
    plots_dir: str = "plots/training"
    logs_dir: str = "logs"
    
    versioning: VersioningConfig = field(default_factory=VersioningConfig)
    save_artifacts: SaveArtifactsConfig = field(default_factory=SaveArtifactsConfig)


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/training_pipeline.log"
    console: bool = True


@dataclass
class TrainingPipelineConfig:
    """Complete training pipeline configuration"""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hyperparameter_search: HyperparameterSearchConfig = field(default_factory=HyperparameterSearchConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingPipelineConfig':
        """
        Create configuration from dictionary
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            TrainingPipelineConfig instance
        """
        # Extract nested configurations
        config_data = {}
        
        if 'experiment' in config_dict:
            config_data['experiment'] = ExperimentConfig(**config_dict['experiment'])
        
        if 'data' in config_dict:
            config_data['data'] = DataConfig(**config_dict['data'])
        
        if 'preprocessing' in config_dict:
            config_data['preprocessing'] = PreprocessingConfig(**config_dict['preprocessing'])
        
        if 'model' in config_dict:
            config_data['model'] = ModelConfig(**config_dict['model'])
        
        if 'training' in config_dict:
            training_dict = config_dict['training']
            training_config = TrainingConfig()
            
            if 'cross_validation' in training_dict:
                training_config.cross_validation = CrossValidationConfig(**training_dict['cross_validation'])
            
            if 'early_stopping' in training_dict:
                training_config.early_stopping = EarlyStoppingConfig(**training_dict['early_stopping'])
            
            if 'class_balancing' in training_dict:
                training_config.class_balancing = ClassBalancingConfig(**training_dict['class_balancing'])
            
            config_data['training'] = training_config
        
        if 'hyperparameter_search' in config_dict:
            config_data['hyperparameter_search'] = HyperparameterSearchConfig(**config_dict['hyperparameter_search'])
        
        if 'evaluation' in config_dict:
            config_data['evaluation'] = EvaluationConfig(**config_dict['evaluation'])
        
        if 'output' in config_dict:
            output_dict = config_dict['output']
            output_config = OutputConfig()
            
            # Update basic fields
            for field_name in ['model_dir', 'results_dir', 'plots_dir', 'logs_dir']:
                if field_name in output_dict:
                    setattr(output_config, field_name, output_dict[field_name])
            
            if 'versioning' in output_dict:
                output_config.versioning = VersioningConfig(**output_dict['versioning'])
            
            if 'save_artifacts' in output_dict:
                output_config.save_artifacts = SaveArtifactsConfig(**output_dict['save_artifacts'])
            
            config_data['output'] = output_config
        
        if 'logging' in config_dict:
            config_data['logging'] = LoggingConfig(**config_dict['logging'])
        
        return cls(**config_data)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainingPipelineConfig':
        """
        Load configuration from YAML file
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            TrainingPipelineConfig instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Handle nested training_pipeline key
        if 'training_pipeline' in config_dict:
            config_dict = config_dict['training_pipeline']
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary
        
        Returns:
            Configuration dictionary
        """
        return asdict(self)
    
    def to_yaml(self, yaml_path: str) -> None:
        """
        Save configuration to YAML file
        
        Args:
            yaml_path: Path to save YAML file
        """
        config_dict = {'training_pipeline': self.to_dict()}
        
        # Create directory if it doesn't exist
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def to_json(self, json_path: str) -> None:
        """
        Save configuration to JSON file
        
        Args:
            json_path: Path to save JSON file
        """
        config_dict = self.to_dict()
        
        # Create directory if it doesn't exist
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def validate(self) -> List[str]:
        """
        Validate configuration parameters
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate data configuration
        if not (0 < self.data.test_size < 1):
            errors.append("test_size must be between 0 and 1")
        
        if not (0 < self.data.validation_size < 1):
            errors.append("validation_size must be between 0 and 1")
        
        if self.data.test_size + self.data.validation_size >= 1:
            errors.append("test_size + validation_size must be less than 1")
        
        # Validate model configuration
        if self.model.type not in ['xgboost', 'random_forest', 'logistic_regression']:
            errors.append(f"Unsupported model type: {self.model.type}")
        
        # Validate hyperparameter search
        if self.hyperparameter_search.method not in ['grid', 'random', 'bayesian']:
            errors.append(f"Unsupported search method: {self.hyperparameter_search.method}")
        
        # Validate preprocessing
        if self.preprocessing.scaling['method'] not in ['standard', 'robust', 'minmax', 'none']:
            errors.append(f"Unsupported scaling method: {self.preprocessing.scaling['method']}")
        
        if self.preprocessing.outliers['method'] not in ['iqr', 'zscore', 'isolation_forest', 'none']:
            errors.append(f"Unsupported outlier method: {self.preprocessing.outliers['method']}")
        
        # Validate early stopping
        if self.training.early_stopping.patience < 1:
            errors.append("early_stopping patience must be >= 1")
        
        # Validate cross-validation
        if self.training.cross_validation.cv_folds < 2:
            errors.append("cv_folds must be >= 2")
        
        return errors
    
    def get_model_params_for_search(self) -> Dict[str, Any]:
        """
        Get model parameters formatted for hyperparameter search
        
        Returns:
            Model parameters dictionary
        """
        base_params = self.model.parameters.copy()
        
        # Remove parameters that should not be searched
        search_exclude = ['random_state', 'n_jobs', 'objective', 'eval_metric']
        for param in search_exclude:
            base_params.pop(param, None)
        
        return base_params
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration from dictionary
        
        Args:
            updates: Dictionary with configuration updates
        """
        def update_nested_dict(target: Dict, source: Dict):
            for key, value in source.items():
                if isinstance(value, dict) and key in target:
                    update_nested_dict(target[key], value)
                else:
                    target[key] = value
        
        config_dict = self.to_dict()
        update_nested_dict(config_dict, updates)
        
        # Recreate from updated dictionary
        updated_config = self.from_dict(config_dict)
        
        # Update self
        for field_name, field_value in asdict(updated_config).items():
            setattr(self, field_name, field_value)


class ConfigurationManager:
    """Manage training pipeline configurations"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_default_config(self, config_path: str) -> TrainingPipelineConfig:
        """
        Create and save default configuration
        
        Args:
            config_path: Path to save configuration
            
        Returns:
            Default configuration
        """
        config = TrainingPipelineConfig()
        config.to_yaml(config_path)
        
        self.logger.info(f"Default configuration saved to {config_path}")
        return config
    
    def load_config(self, config_path: str) -> TrainingPipelineConfig:
        """
        Load configuration from file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded configuration
        """
        if not Path(config_path).exists():
            self.logger.warning(f"Configuration file {config_path} not found, creating default")
            return self.create_default_config(config_path)
        
        try:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = TrainingPipelineConfig.from_yaml(config_path)
            elif config_path.endswith('.json'):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = TrainingPipelineConfig.from_dict(config_dict)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path}")
            
            # Validate configuration
            errors = config.validate()
            if errors:
                raise ValueError(f"Configuration validation failed: {errors}")
            
            self.logger.info(f"Configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise
    
    def merge_configs(self, base_config: TrainingPipelineConfig, 
                     override_config: Dict[str, Any]) -> TrainingPipelineConfig:
        """
        Merge configuration with overrides
        
        Args:
            base_config: Base configuration
            override_config: Override dictionary
            
        Returns:
            Merged configuration
        """
        merged_config = TrainingPipelineConfig.from_dict(base_config.to_dict())
        merged_config.update_from_dict(override_config)
        
        return merged_config
    
    def create_experiment_config(self, 
                               base_config_path: str,
                               experiment_name: str,
                               overrides: Optional[Dict[str, Any]] = None) -> TrainingPipelineConfig:
        """
        Create experiment-specific configuration
        
        Args:
            base_config_path: Path to base configuration
            experiment_name: Name of the experiment
            overrides: Configuration overrides
            
        Returns:
            Experiment configuration
        """
        base_config = self.load_config(base_config_path)
        
        # Update experiment name
        base_config.experiment.name = experiment_name
        
        # Apply overrides if provided
        if overrides:
            base_config.update_from_dict(overrides)
        
        return base_config