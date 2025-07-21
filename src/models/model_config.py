"""
Model configuration management for steel casting defect prediction models.
"""

import yaml
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging


class ModelConfig:
    """Manage model configuration and hyperparameters"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = {}
        
        if config_path:
            self.config = self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            config_file = Path(config_path)
            
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif config_file.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_file.suffix}")
            
            if self.validate_config(config):
                self.config = config
                self.logger.info(f"Configuration loaded successfully from {config_path}")
                return config
            else:
                raise ValueError("Configuration validation failed")
                
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration parameters
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid
        """
        required_sections = ['baseline_model']
        
        # Check required sections
        for section in required_sections:
            if section not in config:
                self.logger.error(f"Missing required section: {section}")
                return False
        
        # Validate baseline model configuration
        baseline_config = config.get('baseline_model', {})
        
        # Check XGBoost parameters
        xgb_params = baseline_config.get('xgboost_params', {})
        required_xgb_params = ['n_estimators', 'max_depth', 'learning_rate']
        
        for param in required_xgb_params:
            if param not in xgb_params:
                self.logger.warning(f"Missing XGBoost parameter: {param}")
        
        # Validate parameter ranges
        if 'learning_rate' in xgb_params:
            lr = xgb_params['learning_rate']
            if not (0 < lr <= 1):
                self.logger.error(f"Invalid learning_rate: {lr}. Must be between 0 and 1")
                return False
        
        if 'max_depth' in xgb_params:
            depth = xgb_params['max_depth']
            if not (1 <= depth <= 20):
                self.logger.error(f"Invalid max_depth: {depth}. Must be between 1 and 20")
                return False
        
        return True
    
    def get_xgboost_params(self) -> Dict[str, Any]:
        """
        Extract XGBoost-specific parameters
        
        Returns:
            XGBoost parameter dictionary
        """
        baseline_config = self.config.get('baseline_model', {})
        return baseline_config.get('xgboost_params', {})
    
    def get_training_params(self) -> Dict[str, Any]:
        """
        Extract training-specific parameters
        
        Returns:
            Training parameter dictionary
        """
        baseline_config = self.config.get('baseline_model', {})
        return baseline_config.get('training', {})
    
    def get_evaluation_params(self) -> Dict[str, Any]:
        """
        Extract evaluation-specific parameters
        
        Returns:
            Evaluation parameter dictionary
        """
        return self.config.get('evaluation', {})
    
    def get_cross_validation_params(self) -> Dict[str, Any]:
        """
        Extract cross-validation parameters
        
        Returns:
            Cross-validation parameter dictionary
        """
        baseline_config = self.config.get('baseline_model', {})
        return baseline_config.get('cross_validation', {})
    
    def get_hyperparameter_search_params(self) -> Dict[str, Any]:
        """
        Extract hyperparameter search parameters
        
        Returns:
            Hyperparameter search parameter dictionary
        """
        baseline_config = self.config.get('baseline_model', {})
        return baseline_config.get('hyperparameter_search', {})
    
    def get_feature_selection_params(self) -> Dict[str, Any]:
        """
        Extract feature selection parameters
        
        Returns:
            Feature selection parameter dictionary
        """
        baseline_config = self.config.get('baseline_model', {})
        return baseline_config.get('feature_selection', {})
    
    def get_persistence_params(self) -> Dict[str, Any]:
        """
        Extract model persistence parameters
        
        Returns:
            Persistence parameter dictionary
        """
        baseline_config = self.config.get('baseline_model', {})
        return baseline_config.get('persistence', {})
    
    def get_logging_params(self) -> Dict[str, Any]:
        """
        Extract logging parameters
        
        Returns:
            Logging parameter dictionary
        """
        baseline_config = self.config.get('baseline_model', {})
        return baseline_config.get('logging', {})
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values
        
        Args:
            updates: Dictionary of configuration updates
        """
        def deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
            """Recursively update nested dictionaries"""
            for key, value in update_dict.items():
                if (key in base_dict and 
                    isinstance(base_dict[key], dict) and 
                    isinstance(value, dict)):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
            return base_dict
        
        deep_update(self.config, updates)
        self.logger.info("Configuration updated")
    
    def save_config(self, config_path: str) -> None:
        """
        Save current configuration to file
        
        Args:
            config_path: Path to save configuration
        """
        config_file = Path(config_path)
        
        try:
            with open(config_file, 'w') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                elif config_file.suffix.lower() == '.json':
                    json.dump(self.config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {config_file.suffix}")
            
            self.logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise
    
    def get_param_grid(self, search_type: str = 'coarse') -> Dict[str, Any]:
        """
        Get parameter grid for hyperparameter search
        
        Args:
            search_type: Type of search ('coarse', 'fine', 'extensive')
            
        Returns:
            Parameter grid dictionary
        """
        search_params = self.get_hyperparameter_search_params()
        param_grids = search_params.get('param_grids', {})
        
        if search_type in param_grids:
            return param_grids[search_type]
        else:
            self.logger.warning(f"Search type '{search_type}' not found in config, using coarse")
            return param_grids.get('coarse', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Return configuration as dictionary
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to configuration"""
        return self.config[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style assignment to configuration"""
        self.config[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator with configuration"""
        return key in self.config