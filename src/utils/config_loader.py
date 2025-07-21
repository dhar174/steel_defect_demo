import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, ValidationError

class ConfigLoader:
    """Load and manage configuration files with validation and environment variable overrides."""

    def __init__(self, config_dir: str = "configs", schema_dir: str = "configs/schemas"):
        """
        Initialize config loader.

        Args:
            config_dir (str): Directory containing configuration files.
            schema_dir (str): Directory containing Pydantic schema files.
        """
        self.config_dir = Path(config_dir)
        self.schema_dir = Path(schema_dir)
        self.configs = {}

    def load_config(self, config_name: str, schema: Optional[BaseModel] = None) -> Dict[str, Any]:
        """
        Load, validate, and process a single YAML configuration file.

        Args:
            config_name (str): The name of the configuration file (e.g., 'data_generation').
            schema (Optional[BaseModel]): The Pydantic schema for validation.

        Returns:
            Dict[str, Any]: The loaded and validated configuration dictionary.
        """
        if config_name in self.configs:
            return self.configs[config_name]

        file_path = self.config_dir / f"{config_name}.yaml"
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, 'r') as f:
            raw_config = yaml.safe_load(f)

        # Recursively override with environment variables
        self._override_with_env_vars(raw_config)

        # Validate with Pydantic schema if provided
        if schema:
            try:
                validated_config = schema.model_validate(raw_config)
                self.configs[config_name] = validated_config.model_dump()
            except ValidationError as e:
                raise ValueError(f"Configuration validation error in '{config_name}.yaml': {e}")
        else:
            self.configs[config_name] = raw_config

        return self.configs[config_name]

    def _override_with_env_vars(self, config: Dict[str, Any], prefix: str = "") -> None:
        """
        Recursively traverse the config dict and override values with environment variables.
        """
        for key, value in config.items():
            env_var_name = f"{prefix}{key}".upper()
            if isinstance(value, dict):
                self._override_with_env_vars(value, f"{env_var_name}_")
            else:
                env_value = os.environ.get(env_var_name)
                if env_value is not None:
                    config[key] = self._cast_env_var(env_value, type(value))

    @staticmethod
    def _cast_env_var(value: str, target_type: type) -> Any:
        """
        Cast environment variable string to the target type.
        """
        try:
            if target_type == bool:
                return value.lower() in ('true', '1', 'yes')
            return target_type(value)
        except (ValueError, TypeError):
            return value

    def get_config(self, config_name: str) -> Optional[Dict]:
        """
        Get a specific configuration.

        Args:
            config_name (str): Name of the configuration.

        Returns:
            Optional[Dict]: Configuration dictionary if found.
        """
        return self.configs.get(config_name)
