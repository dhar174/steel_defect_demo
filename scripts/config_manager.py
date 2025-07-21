"""Configuration management for training pipeline"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
import jsonschema
from jsonschema import validate, ValidationError


@dataclass 
class DataConfig:
    """Data configuration parameters"""
    target_column: str = "defect"
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    handle_missing: bool = True
    
    def __post_init__(self):
        """Validate data configuration"""
        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {self.test_size}")
        if not 0 < self.validation_size < 1:
            raise ValueError(f"validation_size must be between 0 and 1, got {self.validation_size}")
        if self.test_size + self.validation_size >= 1:
            raise ValueError("Sum of test_size and validation_size must be < 1")


@dataclass
class ModelConfig:
    """Model configuration parameters"""
    type: str = "xgboost"
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate model configuration"""
        if self.type not in ['xgboost', 'random_forest', 'logistic_regression']:
            raise ValueError(f"Unsupported model type: {self.type}")


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    hyperparameter_search: bool = True
    search_method: str = "grid"
    cross_validation: bool = True
    cv_folds: int = 5
    early_stopping: bool = False
    early_stopping_rounds: int = 10
    
    def __post_init__(self):
        """Validate training configuration"""
        if self.search_method not in ['grid', 'random', 'bayesian']:
            raise ValueError(f"Unsupported search method: {self.search_method}")
        if self.cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {self.cv_folds}")


@dataclass
class OutputConfig:
    """Output configuration parameters"""
    save_artifacts: bool = True
    experiment_name_prefix: str = "baseline"
    output_dir: str = "results"
    
    
@dataclass
class ExecutionConfig:
    """Execution configuration parameters"""
    verbose: bool = True
    n_jobs: int = -1
    random_state: int = 42


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration parameters"""
    scaling_method: str = "standard"
    handle_missing: bool = True
    missing_strategy: str = "median"
    categorical_strategy: str = "most_frequent"
    
    def __post_init__(self):
        """Validate preprocessing configuration"""
        if self.scaling_method not in ['standard', 'robust', 'minmax', 'none']:
            raise ValueError(f"Unsupported scaling method: {self.scaling_method}")


@dataclass
class HyperparameterSearchConfig:
    """Hyperparameter search configuration"""
    enabled: bool = True
    method: str = "grid"
    n_iter: int = 50
    cv_folds: int = 5
    scoring: str = "roc_auc"
    param_grids: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate hyperparameter search configuration"""
        if self.method not in ['grid', 'random', 'bayesian']:
            raise ValueError(f"Unsupported search method: {self.method}")


@dataclass
class CompleteTrainingConfig:
    """Complete training configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    hyperparameter_search: HyperparameterSearchConfig = field(default_factory=HyperparameterSearchConfig)


class ConfigManager:
    """Handle configuration loading, validation, and management"""
    
    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            schema_path: Path to JSON schema file for validation
        """
        self.logger = logging.getLogger(__name__)
        self.schema = None
        
        if schema_path and Path(schema_path).exists():
            self.load_schema(schema_path)
        else:
            self.schema = self._get_default_schema()
            
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML or JSON file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
                    
            self.logger.info(f"Configuration loaded from {config_path}")
            return config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading config file: {e}")
    
    def save_config(self, config: Dict[str, Any], output_path: str, format: str = 'yaml') -> None:
        """
        Save configuration to file
        
        Args:
            config: Configuration dictionary
            output_path: Path to save configuration
            format: Output format ('yaml' or 'json')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                if format.lower() in ['yaml', 'yml']:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                elif format.lower() == 'json':
                    json.dump(config, f, indent=2, default=str)
                else:
                    raise ValueError(f"Unsupported output format: {format}")
                    
            self.logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            raise ValueError(f"Error saving config file: {e}")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration parameters
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            # Use JSON schema validation if available
            if self.schema:
                validate(instance=config, schema=self.schema)
            
            # Additional custom validation using dataclasses
            self._validate_with_dataclasses(config)
            
            self.logger.info("Configuration validation passed")
            return True
            
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e.message}")
        except Exception as e:
            raise ValueError(f"Configuration validation error: {e}")
    
    def _validate_with_dataclasses(self, config: Dict[str, Any]) -> None:
        """Validate configuration using dataclass validation"""
        try:
            # Extract each section and validate
            data_config = DataConfig(**config.get('data', {}))
            model_config = ModelConfig(**config.get('model', {}))
            training_config = TrainingConfig(**config.get('training', {}))
            output_config = OutputConfig(**config.get('output', {}))
            execution_config = ExecutionConfig(**config.get('execution', {}))
            preprocessing_config = PreprocessingConfig(**config.get('preprocessing', {}))
            hyperparameter_config = HyperparameterSearchConfig(**config.get('hyperparameter_search', {}))
            
            # Create complete config to ensure all validations run
            complete_config = CompleteTrainingConfig(
                data=data_config,
                model=model_config,
                training=training_config,
                output=output_config,
                execution=execution_config,
                preprocessing=preprocessing_config,
                hyperparameter_search=hyperparameter_config
            )
            
        except TypeError as e:
            raise ValueError(f"Invalid configuration parameter: {e}")
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge configuration dictionaries with override priority
        
        Args:
            base_config: Base configuration
            override_config: Override configuration
            
        Returns:
            Merged configuration dictionary
        """
        def deep_merge(base: Dict, override: Dict) -> Dict:
            """Recursively merge dictionaries"""
            result = base.copy()
            
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
                    
            return result
        
        merged = deep_merge(base_config, override_config)
        self.logger.info("Configurations merged successfully")
        return merged
    
    def create_default_config(self) -> Dict[str, Any]:
        """
        Create default configuration
        
        Returns:
            Default configuration dictionary
        """
        default_config = CompleteTrainingConfig()
        config_dict = asdict(default_config)
        
        # Add default parameter grids for hyperparameter search
        config_dict['hyperparameter_search']['param_grids'] = {
            'coarse': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'fine': {
                'n_estimators': [80, 100, 120],
                'max_depth': [5, 6, 7],
                'learning_rate': [0.08, 0.1, 0.12],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2]
            }
        }
        
        return config_dict
    
    def load_schema(self, schema_path: str) -> None:
        """
        Load JSON schema for configuration validation
        
        Args:
            schema_path: Path to JSON schema file
        """
        try:
            with open(schema_path, 'r') as f:
                self.schema = json.load(f)
            self.logger.info(f"Configuration schema loaded from {schema_path}")
        except Exception as e:
            self.logger.warning(f"Could not load schema from {schema_path}: {e}")
            self.schema = self._get_default_schema()
    
    def _get_default_schema(self) -> Dict[str, Any]:
        """Get default JSON schema for configuration validation"""
        return {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "target_column": {"type": "string"},
                        "test_size": {"type": "number", "minimum": 0, "maximum": 1},
                        "validation_size": {"type": "number", "minimum": 0, "maximum": 1},
                        "random_state": {"type": "integer"},
                        "handle_missing": {"type": "boolean"}
                    }
                },
                "model": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["xgboost", "random_forest", "logistic_regression"]},
                        "parameters": {"type": "object"}
                    }
                },
                "training": {
                    "type": "object",
                    "properties": {
                        "hyperparameter_search": {"type": "boolean"},
                        "search_method": {"type": "string", "enum": ["grid", "random", "bayesian"]},
                        "cross_validation": {"type": "boolean"},
                        "cv_folds": {"type": "integer", "minimum": 2}
                    }
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "save_artifacts": {"type": "boolean"},
                        "experiment_name_prefix": {"type": "string"},
                        "output_dir": {"type": "string"}
                    }
                },
                "execution": {
                    "type": "object",
                    "properties": {
                        "verbose": {"type": "boolean"},
                        "n_jobs": {"type": "integer"},
                        "random_state": {"type": "integer"}
                    }
                }
            }
        }
    
    def get_config_from_args(self, args) -> Dict[str, Any]:
        """
        Convert command line arguments to configuration dictionary
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Configuration dictionary
        """
        config = {}
        
        # Data configuration
        if hasattr(args, 'target_column') and args.target_column:
            config.setdefault('data', {})['target_column'] = args.target_column
        if hasattr(args, 'test_size') and args.test_size is not None:
            config.setdefault('data', {})['test_size'] = args.test_size
        if hasattr(args, 'validation_size') and args.validation_size is not None:
            config.setdefault('data', {})['validation_size'] = args.validation_size
        if hasattr(args, 'random_state') and args.random_state is not None:
            config.setdefault('data', {})['random_state'] = args.random_state
            
        # Model configuration
        if hasattr(args, 'model_type') and args.model_type:
            config.setdefault('model', {})['type'] = args.model_type
        if hasattr(args, 'model_params') and args.model_params:
            config.setdefault('model', {})['parameters'] = json.loads(args.model_params)
            
        # Training configuration
        if hasattr(args, 'hyperparameter_search') and args.hyperparameter_search is not None:
            config.setdefault('training', {})['hyperparameter_search'] = args.hyperparameter_search
        if hasattr(args, 'cross_validation') and args.cross_validation is not None:
            config.setdefault('training', {})['cross_validation'] = args.cross_validation
        if hasattr(args, 'cv_folds') and args.cv_folds is not None:
            config.setdefault('training', {})['cv_folds'] = args.cv_folds
        if hasattr(args, 'search_method') and args.search_method:
            config.setdefault('hyperparameter_search', {})['method'] = args.search_method
            
        # Output configuration
        if hasattr(args, 'output_dir') and args.output_dir:
            config.setdefault('output', {})['output_dir'] = args.output_dir
        if hasattr(args, 'experiment_name') and args.experiment_name:
            config.setdefault('output', {})['experiment_name_prefix'] = args.experiment_name
        if hasattr(args, 'save_artifacts') and args.save_artifacts is not None:
            config.setdefault('output', {})['save_artifacts'] = args.save_artifacts
            
        # Execution configuration
        if hasattr(args, 'verbose') and args.verbose is not None:
            config.setdefault('execution', {})['verbose'] = args.verbose
        if hasattr(args, 'n_jobs') and args.n_jobs is not None:
            config.setdefault('execution', {})['n_jobs'] = args.n_jobs
        if hasattr(args, 'random_state') and args.random_state is not None:
            config.setdefault('execution', {})['random_state'] = args.random_state
            
        return config
    
    def create_experiment_config(self, base_config: Dict[str, Any], experiment_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create configuration for a specific experiment
        
        Args:
            base_config: Base configuration
            experiment_params: Experiment-specific parameters
            
        Returns:
            Experiment configuration
        """
        experiment_config = self.merge_configs(base_config, experiment_params)
        
        # Validate the merged configuration
        self.validate_config(experiment_config)
        
        return experiment_config
    
    def get_quick_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create quick training configuration for development
        
        Args:
            base_config: Base configuration
            
        Returns:
            Quick training configuration
        """
        quick_overrides = {
            'training': {
                'cv_folds': 3,
                'hyperparameter_search': True
            },
            'hyperparameter_search': {
                'method': 'random',
                'n_iter': 10
            }
        }
        
        return self.merge_configs(base_config, quick_overrides)
    
    def get_production_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create production training configuration
        
        Args:
            base_config: Base configuration
            
        Returns:
            Production training configuration
        """
        production_overrides = {
            'training': {
                'cv_folds': 10,
                'hyperparameter_search': True
            },
            'hyperparameter_search': {
                'method': 'bayesian',
                'n_iter': 100
            },
            'execution': {
                'verbose': False
            }
        }
        
        return self.merge_configs(base_config, production_overrides)