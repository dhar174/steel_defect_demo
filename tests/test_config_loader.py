import pytest
from pathlib import Path
from pydantic import BaseModel, ValidationError
from src.utils.config_loader import ConfigLoader

# Define a simple schema for testing
class TestSchema(BaseModel):
    key: str
    value: int

@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """Create a temporary config directory for testing."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    # Create a valid config file
    with open(config_dir / "valid_config.yaml", "w") as f:
        f.write("key: test\nvalue: 123")

    # Create an invalid config file
    with open(config_dir / "invalid_config.yaml", "w") as f:
        f.write("key: test\nvalue: abc")

    # Create a config file for environment variable override test
    with open(config_dir / "env_override_config.yaml", "w") as f:
        f.write("key: default\nvalue: 456")

    return config_dir

def test_load_config_success(config_dir: Path):
    """Test successful loading and validation of a configuration file."""
    loader = ConfigLoader(config_dir=str(config_dir))
    config = loader.load_config("valid_config", schema=TestSchema)
    assert config["key"] == "test"
    assert config["value"] == 123

def test_load_config_not_found(config_dir: Path):
    """Test that loading a non-existent config file raises FileNotFoundError."""
    loader = ConfigLoader(config_dir=str(config_dir))
    with pytest.raises(FileNotFoundError):
        loader.load_config("non_existent_config")

def test_load_config_invalid_schema(config_dir: Path):
    """Test that loading a config with invalid data raises a validation error."""
    loader = ConfigLoader(config_dir=str(config_dir))
    with pytest.raises(ValueError):
        loader.load_config("invalid_config", schema=TestSchema)

def test_env_variable_override(config_dir: Path, monkeypatch):
    """Test that environment variables override config values."""
    monkeypatch.setenv("KEY", "overridden")
    monkeypatch.setenv("VALUE", "789")

    loader = ConfigLoader(config_dir=str(config_dir))
    config = loader.load_config("env_override_config", schema=TestSchema)

    assert config["key"] == "overridden"
    assert config["value"] == 789

def test_get_config(config_dir: Path):
    """Test retrieving a loaded configuration."""
    loader = ConfigLoader(config_dir=str(config_dir))
    loader.load_config("valid_config", schema=TestSchema)
    config = loader.get_config("valid_config")
    assert config is not None
    assert config["key"] == "test"

def test_get_config_not_loaded(config_dir: Path):
    """Test that get_config returns None for a config that hasn't been loaded."""
    loader = ConfigLoader(config_dir=str(config_dir))
    config = loader.get_config("valid_config")
    assert config is None
