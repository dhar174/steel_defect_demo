# Configuration Management System

This document describes the comprehensive configuration management system for the steel defect prediction system.

## Overview

The configuration management system provides a centralized way to manage all configuration aspects for data generation, model training, and inference operations. It includes:

- YAML-based configuration files with validation
- Environment variable override support
- Schema validation
- Configuration merging and caching
- Type-safe configuration loading

## Configuration Files

### 1. Data Generation Configuration (`configs/data_generation.yaml`)

Controls synthetic data generation parameters:

```yaml
data_generation:
  # Basic parameters
  num_casts: 1200                    # Number of casting sequences to generate
  cast_duration_minutes: 120         # Duration of each cast
  sampling_rate_hz: 1               # Data sampling rate
  random_seed: 42                   # For reproducibility
  
  # Sensor configuration
  sensors:
    casting_speed:
      base_value: 1.2               # m/min
      noise_std: 0.05               # Standard deviation of noise
      min_value: 0.8                # Minimum allowed value
      max_value: 1.8                # Maximum allowed value
    
    mold_temperature:
      base_value: 1520              # Celsius
      noise_std: 10
      min_value: 1480
      max_value: 1580
    
    # ... other sensors ...
    
    # Operating ranges for defect detection
    mold_level_normal_range: [130, 170]  # Normal operating range
  
  # Defect simulation
  defect_simulation:
    defect_probability: 0.15          # Base defect probability (15%)
    max_defect_probability: 0.8       # Maximum when triggers present
    trigger_probability_factor: 0.2   # Factor for trigger-based probability
    defect_triggers:
      prolonged_mold_level_deviation: 30    # seconds
      rapid_temperature_drop: 50            # Celsius in 60 seconds
      high_speed_with_low_superheat: true
  
  # Output format
  output:
    raw_data_format: "parquet"
    metadata_format: "json"
    train_test_split: 0.8
```

### 2. Model Configuration (`configs/model_config.yaml`)

Defines model training parameters:

```yaml
baseline_model:
  algorithm: "xgboost"
  parameters:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    # ... other parameters ...
  
  feature_engineering:
    statistical_features: true
    stability_features: true
    duration_features: true
    interaction_features: true
  
  validation:
    cv_folds: 5
    early_stopping_rounds: 10
    eval_metric: "auc"

lstm_model:
  architecture:
    input_size: 5              # Number of sensors
    hidden_size: 64
    num_layers: 2
    dropout: 0.2
  
  training:
    batch_size: 32
    learning_rate: 0.001
    num_epochs: 100
    early_stopping_patience: 15
    weight_decay: 0.0001
  
  data_processing:
    sequence_length: 300       # 5 minutes at 1Hz
    normalization: "z_score"
    padding: "zero"

evaluation:
  metrics:
    - "auc_roc"
    - "auc_pr"
    - "f1_score"
    - "precision"
    - "recall"
    - "accuracy"
  test_size: 0.2
  stratify: true
  random_state: 42
```

### 3. Inference Configuration (`configs/inference_config.yaml`)

Controls real-time inference settings:

```yaml
inference:
  model_types:
    - "baseline"
    - "lstm"
  
  real_time_simulation:
    playback_speed_multiplier: 10    # 10x real time
    update_interval_seconds: 30
    buffer_size_seconds: 300
  
  thresholds:
    defect_probability: 0.5
    high_risk_threshold: 0.7
    alert_threshold: 0.8
  
  output:
    log_predictions: true
    save_trajectories: true
    dashboard_enabled: true
    dashboard_port: 8050

monitoring:
  metrics_logging: true
  performance_tracking: true
  data_drift_detection: true
```

## Configuration Loader Usage

### Basic Usage

```python
from utils.config_loader import ConfigLoader

# Initialize loader
loader = ConfigLoader()

# Load specific configuration
data_config = loader.load_yaml('data_generation.yaml')

# Load all configurations
all_configs = loader.load_all_configs()

# Get cached configuration
config = loader.get_config('model_config')
```

### Environment Variable Overrides

Environment variables can override configuration values:

```bash
# Override data generation parameters
export DATA_GENERATION_DATA_GENERATION_NUM_CASTS=500
export DATA_GENERATION_DATA_GENERATION_CAST_DURATION_MINUTES=60

# Override model parameters
export MODEL_CONFIG_BASELINE_MODEL_PARAMETERS_LEARNING_RATE=0.05
export MODEL_CONFIG_LSTM_MODEL_TRAINING_BATCH_SIZE=64
```

Variable naming convention: `{CONFIG_NAME}_{NESTED_PATH}={VALUE}`

### Configuration Validation

All configurations are validated against JSON schemas:

```python
# Validate configuration
try:
    loader.validate_config(config, 'data_generation_schema')
    print("Configuration is valid")
except ConfigValidationError as e:
    print(f"Validation failed: {e}")
```

### Configuration Merging

Merge multiple configurations:

```python
# Merge configurations with deep merging
merged_config = loader.merge_configs(
    'data_generation', 
    'model_config', 
    'inference_config'
)
```

## Schema Validation

Configuration schemas are located in `configs/schemas/` and define:

- Required and optional parameters
- Data types and value ranges
- Parameter descriptions
- Validation rules

Example schema structure:
```yaml
$schema: "http://json-schema.org/draft-07/schema#"
title: "Configuration Schema"
type: object
required: ["section1", "section2"]
properties:
  section1:
    type: object
    required: ["param1"]
    properties:
      param1:
        type: number
        minimum: 0
        maximum: 1
        description: "Parameter description"
```

## Environment-Specific Configurations

### Development Environment
```bash
export ENVIRONMENT=development
export DATA_GENERATION_DATA_GENERATION_NUM_CASTS=10
export MODEL_CONFIG_BASELINE_MODEL_PARAMETERS_N_ESTIMATORS=10
```

### Production Environment
```bash
export ENVIRONMENT=production
export INFERENCE_INFERENCE_OUTPUT_DASHBOARD_ENABLED=true
export INFERENCE_MONITORING_METRICS_LOGGING=true
```

## Integration with Components

### Data Generator
```python
from utils.config_loader import ConfigLoader
from data.data_generator import SteelCastingDataGenerator

loader = ConfigLoader()
config_path = loader.config_dir / 'data_generation.yaml'
generator = SteelCastingDataGenerator(str(config_path))
```

### Model Training
```python
from utils.config_loader import ConfigLoader

loader = ConfigLoader()
model_config = loader.get_config('model_config')

# XGBoost configuration
xgb_params = model_config['baseline_model']['parameters']

# LSTM configuration
lstm_config = model_config['lstm_model']
```

### Inference Engine
```python
from utils.config_loader import ConfigLoader

loader = ConfigLoader()
inference_config = loader.get_config('inference_config')

# Threshold configuration
thresholds = inference_config['inference']['thresholds']
defect_threshold = thresholds['defect_probability']
```

## Best Practices

### 1. Configuration Organization
- Keep related parameters grouped together
- Use descriptive parameter names
- Include units in parameter names or comments
- Provide sensible defaults

### 2. Environment Variables
- Use environment variables for environment-specific settings
- Don't put secrets in configuration files
- Use the hierarchical naming convention

### 3. Validation
- Always validate configurations before use
- Update schemas when adding new parameters
- Test configuration changes thoroughly

### 4. Version Control
- Version configuration files with your code
- Document configuration changes in commit messages
- Use separate configuration files for different environments

## Error Handling

The configuration system provides detailed error messages:

```python
try:
    config = loader.load_yaml('missing_config.yaml')
except FileNotFoundError:
    print("Configuration file not found")

try:
    loader.validate_config(config, 'schema_name')
except ConfigValidationError as e:
    print(f"Validation error: {e}")
    # Error includes path to invalid parameter
```

## Extending the Configuration System

### Adding New Configuration Files
1. Create the YAML configuration file
2. Create corresponding schema file
3. Add validation tests
4. Update documentation

### Adding New Parameters
1. Add parameter to configuration file
2. Update schema with validation rules
3. Update consuming code
4. Add tests for new parameter

### Custom Validation
```python
def custom_validator(config):
    """Custom validation logic"""
    if config['param1'] > config['param2']:
        raise ConfigValidationError("param1 must be <= param2")

# Apply custom validation
loader.validate_config(config, 'schema_name')
custom_validator(config)
```

## Troubleshooting

### Common Issues

1. **Schema validation fails**: Check parameter types and ranges
2. **Environment variables not applied**: Verify naming convention
3. **Configuration not found**: Check file paths and names
4. **Merge conflicts**: Review configuration structure and overlapping keys

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug logging for configuration loader
loader = ConfigLoader()
# Debug messages will show parameter loading and validation
```