import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import jsonschema
from jsonschema import validate, ValidationError
import logging
from copy import deepcopy

# Set up logging
logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors"""
    pass


class ConfigLoader:
    """Load and manage configuration files with validation and environment variable support"""
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize config loader.
        
        Args:
            config_dir (str): Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.schema_dir = self.config_dir / "schemas"
        self.configs = {}
        self.schemas = {}
        
        # Ensure directories exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.schema_dir.mkdir(parents=True, exist_ok=True)
        
        # Load schemas on initialization
        self._load_schemas()
    
    def _load_schemas(self) -> None:
        """Load all schema files from the schemas directory"""
        if not self.schema_dir.exists():
            logger.warning(f"Schema directory {self.schema_dir} does not exist")
            return
            
        for schema_file in self.schema_dir.glob("*.yaml"):
            schema_name = schema_file.stem
            try:
                with open(schema_file, 'r') as f:
                    self.schemas[schema_name] = yaml.safe_load(f)
                logger.debug(f"Loaded schema: {schema_name}")
            except Exception as e:
                logger.error(f"Failed to load schema {schema_file}: {e}")
    
    def _apply_env_overrides(self, config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.
        
        Args:
            config (Dict): Configuration to override
            prefix (str): Environment variable prefix
            
        Returns:
            Dict[str, Any]: Configuration with environment overrides applied
        """
        config_copy = deepcopy(config)
        
        def _override_recursive(obj: Any, path: str = "") -> Any:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}_{key}".upper() if path else key.upper()
                    env_var = f"{prefix}_{current_path}" if prefix else current_path
                    
                    # Check for environment variable
                    env_value = os.getenv(env_var)
                    if env_value is not None:
                        # Try to convert to appropriate type
                        obj[key] = self._convert_env_value(env_value, value)
                        logger.debug(f"Applied environment override: {env_var}={env_value}")
                    else:
                        # Recursively check nested dictionaries
                        obj[key] = _override_recursive(value, current_path)
                return obj
            else:
                return obj
        
        return _override_recursive(config_copy)
    
    def _convert_env_value(self, env_value: str, original_value: Any) -> Any:
        """
        Convert environment variable string to appropriate type based on original value.
        
        Args:
            env_value (str): Environment variable value
            original_value (Any): Original configuration value for type inference
            
        Returns:
            Any: Converted value
        """
        # Handle boolean values
        if isinstance(original_value, bool):
            return env_value.lower() in ('true', '1', 'yes', 'on')
        
        # Handle numeric values
        if isinstance(original_value, int):
            try:
                return int(env_value)
            except ValueError:
                logger.warning(f"Could not convert '{env_value}' to int, using string")
                return env_value
        
        if isinstance(original_value, float):
            try:
                return float(env_value)
            except ValueError:
                logger.warning(f"Could not convert '{env_value}' to float, using string")
                return env_value
        
        # Handle lists (comma-separated values)
        if isinstance(original_value, list):
            return [item.strip() for item in env_value.split(',')]
        
        # Default to string
        return env_value
    
    def load_yaml(self, filename: str, apply_env_overrides: bool = True, 
                  env_prefix: str = None) -> Dict[str, Any]:
        """
        Load YAML configuration file.
        
        Args:
            filename (str): Name of the YAML file
            apply_env_overrides (bool): Whether to apply environment variable overrides
            env_prefix (str): Prefix for environment variables
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        file_path = self.config_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if apply_env_overrides:
                prefix = env_prefix or filename.split('.')[0].upper()
                config = self._apply_env_overrides(config, prefix)
            
            # Cache the loaded config
            config_name = filename.split('.')[0]
            self.configs[config_name] = config
            
            logger.info(f"Successfully loaded configuration: {filename}")
            return config
            
        except yaml.YAMLError as e:
            raise ConfigValidationError(f"Invalid YAML in {filename}: {e}")
        except Exception as e:
            raise ConfigValidationError(f"Failed to load {filename}: {e}")
    
    def load_json(self, filename: str, apply_env_overrides: bool = True,
                  env_prefix: str = None) -> Dict[str, Any]:
        """
        Load JSON configuration file.
        
        Args:
            filename (str): Name of the JSON file
            apply_env_overrides (bool): Whether to apply environment variable overrides
            env_prefix (str): Prefix for environment variables
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        file_path = self.config_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            if apply_env_overrides:
                prefix = env_prefix or filename.split('.')[0].upper()
                config = self._apply_env_overrides(config, prefix)
            
            # Cache the loaded config
            config_name = filename.split('.')[0]
            self.configs[config_name] = config
            
            logger.info(f"Successfully loaded configuration: {filename}")
            return config
            
        except json.JSONDecodeError as e:
            raise ConfigValidationError(f"Invalid JSON in {filename}: {e}")
        except Exception as e:
            raise ConfigValidationError(f"Failed to load {filename}: {e}")
    
    def load_all_configs(self, apply_env_overrides: bool = True) -> Dict[str, Dict]:
        """
        Load all configuration files in the config directory.
        
        Args:
            apply_env_overrides (bool): Whether to apply environment variable overrides
        
        Returns:
            Dict[str, Dict]: All configurations keyed by filename (without extension)
        """
        configs = {}
        
        # Load YAML files
        for config_file in self.config_dir.glob("*.yaml"):
            if config_file.parent.name != "schemas":  # Skip schema files
                config_name = config_file.stem
                try:
                    configs[config_name] = self.load_yaml(
                        config_file.name, apply_env_overrides=apply_env_overrides
                    )
                except Exception as e:
                    logger.error(f"Failed to load {config_file.name}: {e}")
        
        # Load JSON files
        for config_file in self.config_dir.glob("*.json"):
            config_name = config_file.stem
            try:
                configs[config_name] = self.load_json(
                    config_file.name, apply_env_overrides=apply_env_overrides
                )
            except Exception as e:
                logger.error(f"Failed to load {config_file.name}: {e}")
        
        self.configs.update(configs)
        logger.info(f"Loaded {len(configs)} configuration files")
        return configs
    
    def get_config(self, config_name: str) -> Optional[Dict]:
        """
        Get a specific configuration.
        
        Args:
            config_name (str): Name of the configuration
            
        Returns:
            Optional[Dict]: Configuration dictionary if found
        """
        if config_name in self.configs:
            return self.configs[config_name]
        
        # Try to load if not already cached
        config_file = f"{config_name}.yaml"
        if (self.config_dir / config_file).exists():
            return self.load_yaml(config_file)
        
        config_file = f"{config_name}.json"
        if (self.config_dir / config_file).exists():
            return self.load_json(config_file)
        
        logger.warning(f"Configuration '{config_name}' not found")
        return None
    
    def merge_configs(self, *config_names: str) -> Dict[str, Any]:
        """
        Merge multiple configurations with deep merging for nested dictionaries.
        
        Args:
            *config_names (str): Names of configurations to merge
            
        Returns:
            Dict[str, Any]: Merged configuration
        """
        merged = {}
        
        def _deep_merge(base_dict: Dict, merge_dict: Dict) -> Dict:
            """Recursively merge dictionaries"""
            result = deepcopy(base_dict)
            for key, value in merge_dict.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = _deep_merge(result[key], value)
                else:
                    result[key] = deepcopy(value)
            return result
        
        for config_name in config_names:
            config = self.get_config(config_name)
            if config is None:
                logger.warning(f"Configuration '{config_name}' not found for merging")
                continue
            merged = _deep_merge(merged, config)
        
        return merged
    
    def validate_config(self, config: Dict, schema_name: str = None, 
                       schema: Dict = None) -> bool:
        """
        Validate configuration against a schema.
        
        Args:
            config (Dict): Configuration to validate
            schema_name (str): Name of schema to use (from schemas directory)
            schema (Dict): Schema dictionary to validate against
            
        Returns:
            bool: True if valid
            
        Raises:
            ConfigValidationError: If validation fails
        """
        if schema is None:
            if schema_name is None:
                raise ConfigValidationError("Either schema_name or schema must be provided")
            
            if schema_name not in self.schemas:
                raise ConfigValidationError(f"Schema '{schema_name}' not found")
            
            schema = self.schemas[schema_name]
        
        try:
            validate(instance=config, schema=schema)
            logger.debug(f"Configuration validation successful")
            return True
        except ValidationError as e:
            error_msg = f"Configuration validation failed: {e.message}"
            if e.path:
                error_msg += f" at path: {'.'.join(str(p) for p in e.path)}"
            logger.error(error_msg)
            raise ConfigValidationError(error_msg) from e
    
    def save_config(self, config: Dict, filename: str, 
                   format: str = "yaml") -> None:
        """
        Save configuration to file.
        
        Args:
            config (Dict): Configuration to save
            filename (str): Output filename
            format (str): Output format ('yaml' or 'json')
        """
        file_path = self.config_dir / filename
        
        try:
            if format.lower() == "yaml":
                with open(file_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            raise ConfigValidationError(f"Failed to save configuration: {e}")
    
    def update_config(self, config_name: str, updates: Dict) -> None:
        """
        Update an existing configuration with deep merging.
        
        Args:
            config_name (str): Name of configuration to update
            updates (Dict): Updates to apply
        """
        current_config = self.get_config(config_name)
        if current_config is None:
            raise ConfigValidationError(f"Configuration '{config_name}' not found")
        
        # Deep merge updates
        def _deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
            result = deepcopy(base_dict)
            for key, value in update_dict.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = _deep_update(result[key], value)
                else:
                    result[key] = deepcopy(value)
            return result
        
        updated_config = _deep_update(current_config, updates)
        self.configs[config_name] = updated_config
        
        # Save updated configuration
        config_file = f"{config_name}.yaml"
        self.save_config(updated_config, config_file)
        
        logger.info(f"Configuration '{config_name}' updated successfully")
    
    def get_schema(self, schema_name: str) -> Optional[Dict]:
        """
        Get a specific schema.
        
        Args:
            schema_name (str): Name of the schema
            
        Returns:
            Optional[Dict]: Schema dictionary if found
        """
        return self.schemas.get(schema_name)
    
    def list_configs(self) -> List[str]:
        """
        List all available configuration names.
        
        Returns:
            List[str]: List of configuration names
        """
        config_files = []
        for ext in ['*.yaml', '*.json']:
            config_files.extend([f.stem for f in self.config_dir.glob(ext) 
                               if f.parent.name != "schemas"])
        return sorted(set(config_files))
    
    def list_schemas(self) -> List[str]:
        """
        List all available schema names.
        
        Returns:
            List[str]: List of schema names
        """
        return list(self.schemas.keys())