# Steel Defect Training Pipeline

This document describes how to use the comprehensive command-line training script for steel casting defect prediction.

## Overview

The training pipeline provides a complete solution for training machine learning models with:
- ✅ Comprehensive command-line interface 
- ✅ Configuration management (YAML/JSON)
- ✅ Progress tracking and logging
- ✅ Artifact management and versioning
- ✅ Multiple training modes (development, production, quick)
- ✅ Error handling and validation
- ✅ Performance profiling and debugging

## Quick Start

### Basic Training
```bash
# Train with a CSV data file
python scripts/train_baseline_model.py --data data/processed/sample_data.csv

# Quick development training (reduced search space)
python scripts/train_baseline_model.py --data data/processed/sample_data.csv --quick --experiment-name dev_test
```

### Configuration-Based Training
```bash
# Use default configuration
python scripts/train_baseline_model.py --config configs/default_training.yaml --data data/processed/sample_data.csv

# Quick development mode
python scripts/train_baseline_model.py --config configs/quick_training.yaml --data data/processed/sample_data.csv

# Production training
python scripts/train_baseline_model.py --config configs/production_training.yaml --data data/processed/sample_data.csv
```

### Validation and Testing
```bash
# Dry run to validate configuration
python scripts/train_baseline_model.py --config configs/production_training.yaml --data data/processed/sample_data.csv --dry-run

# Debug mode with detailed logging
python scripts/train_baseline_model.py --data data/processed/sample_data.csv --debug --experiment-name debug_run
```

## Command-Line Arguments

### Required Arguments
- `--data PATH` - Path to training data CSV file
- `--config PATH` - Path to configuration file (YAML/JSON)

*Note: Either `--data` or `--config` must be specified. You can use both to override config with CLI arguments.*

### Data Arguments
- `--target-column NAME` - Target column name (default: 'defect')
- `--test-size FLOAT` - Test set fraction (default: 0.2)
- `--validation-size FLOAT` - Validation set fraction (default: 0.2)

### Model Arguments
- `--model-type TYPE` - Model type: xgboost, random_forest, logistic_regression (default: xgboost)
- `--model-params JSON` - Model parameters as JSON string

### Training Arguments
- `--hyperparameter-search` / `--no-hyperparameter-search` - Enable/disable hyperparameter search
- `--cross-validation` / `--no-cross-validation` - Enable/disable cross-validation
- `--cv-folds N` - Number of CV folds (default: 5)
- `--search-method METHOD` - Search method: grid, random, bayesian (default: grid)

### Output Arguments
- `--output-dir DIR` - Output directory (default: results)
- `--experiment-name NAME` - Experiment name (default: auto-generated)
- `--no-save-artifacts` - Disable artifact saving

### Execution Arguments
- `--random-state SEED` - Random seed (default: 42)
- `--n-jobs N` - Parallel jobs (default: -1)
- `--verbose` / `--quiet` - Control output verbosity
- `--quick` - Quick training mode (reduced search space)
- `--debug` - Enable debug logging

### Advanced Arguments
- `--resume-from PATH` - Resume from checkpoint
- `--profile` - Enable performance profiling
- `--dry-run` - Validate configuration without training

## Configuration Files

### Default Configuration (`configs/default_training.yaml`)
Comprehensive configuration with standard settings for most use cases.

### Quick Configuration (`configs/quick_training.yaml`)
Reduced settings for fast development and testing:
- Fewer hyperparameter search iterations
- Reduced CV folds
- Smaller models

### Production Configuration (`configs/production_training.yaml`)
Optimized settings for final production models:
- Extensive hyperparameter search
- Bayesian optimization
- More CV folds
- Deeper models

## Configuration Structure

```yaml
# Data Configuration
data:
  target_column: "defect"
  test_size: 0.2
  validation_size: 0.2
  random_state: 42

# Model Configuration  
model:
  type: "xgboost"
  parameters:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    # ... other XGBoost parameters

# Training Configuration
training:
  hyperparameter_search: true
  search_method: "grid"
  cross_validation: true
  cv_folds: 5

# Hyperparameter Search
hyperparameter_search:
  enabled: true
  method: "grid"
  param_grids:
    coarse:
      n_estimators: [50, 100, 200]
      max_depth: [3, 6, 9]
      # ... parameter ranges

# Output Configuration
output:
  save_artifacts: true
  experiment_name_prefix: "baseline"
  output_dir: "results"

# Execution Configuration  
execution:
  verbose: true
  n_jobs: -1
  random_state: 42
```

## Output Artifacts

The training pipeline generates comprehensive outputs:

```
results/
└── experiment_name_YYYYMMDD_HHMMSS/
    ├── models/
    │   ├── model.pkl                    # Trained model
    │   └── model_metadata.json         # Model metadata
    ├── feature_engineers/
    │   └── feature_engineer.joblib     # Feature engineering pipeline
    ├── results/
    │   └── results.json                 # Training results and metrics
    ├── configs/
    │   └── config.yaml                  # Training configuration
    ├── plots/                           # Generated plots (if any)
    ├── logs/
    │   └── training.log                 # Detailed logs
    ├── metadata/                        # Artifact metadata
    └── artifact_index.json             # Artifact registry
```

## Training Results

The results file contains comprehensive metrics:

```json
{
  "training": {
    "training_time": 0.188,
    "train_auc": 1.0,
    "n_samples": 1000,
    "n_features": 20
  },
  "evaluation": {
    "metrics": {
      "roc_auc": 1.0,
      "average_precision": 1.0,
      "f1_score": 1.0,
      "precision": 1.0,
      "recall": 1.0,
      "accuracy": 1.0
    },
    "confusion_matrix": [[160, 0], [0, 40]],
    "threshold_analysis": { ... },
    "calibration_metrics": { ... }
  },
  "cross_validation": {
    "roc_auc_mean": 0.9635,
    "roc_auc_std": 0.0138
  },
  "hyperparameter_search": {
    "best_params": { ... },
    "best_score": 0.965,
    "search_method": "grid"
  },
  "artifacts": {
    "model_path": "results/.../model.pkl",
    "config_path": "results/.../config.yaml"
  }
}
```

## Examples

### 1. Basic Development Training
```bash
python scripts/train_baseline_model.py \
  --data data/processed/sample_data.csv \
  --experiment-name development \
  --quick \
  --cv-folds 3
```

### 2. Production Training with Custom Parameters
```bash
python scripts/train_baseline_model.py \
  --config configs/production_training.yaml \
  --data data/processed/production_data.csv \
  --experiment-name production_v1 \
  --output-dir models/production \
  --model-params '{"n_estimators": 200, "max_depth": 8}'
```

### 3. Hyperparameter Search with Random Search
```bash
python scripts/train_baseline_model.py \
  --data data/processed/sample_data.csv \
  --search-method random \
  --cv-folds 10 \
  --experiment-name hp_search_random
```

### 4. Configuration Validation
```bash
python scripts/train_baseline_model.py \
  --config configs/production_training.yaml \
  --data data/processed/sample_data.csv \
  --dry-run
```

## Progress Tracking

The training script provides real-time progress tracking:

```
Training Pipeline:  62%|████████████████████▎              | 5/8 [00:02<00:01,  1.50it/s]
Model Training: 100%|█████████████████████████████████████| 100% [00:00, 513.34it/s]
```

Step details are logged with timing information:
- Data Loading
- Feature Engineering  
- Model Initialization
- Training Setup
- Model Training
- Model Evaluation
- Cross-Validation (if enabled)
- Artifact Saving (if enabled)

## Error Handling

The script provides comprehensive error handling:
- Argument validation
- File existence checks
- Configuration validation
- Data quality checks
- Graceful failure with detailed error messages

## Performance Considerations

### Memory Usage
- The script estimates memory requirements
- Use `--profile` for detailed performance analysis
- Consider `--quick` mode for limited resources

### Training Time
- Quick mode: ~30 seconds - 2 minutes
- Default mode: 2-10 minutes  
- Production mode: 10-60 minutes (depending on search space)

### Parallel Processing
- Use `--n-jobs -1` to utilize all CPU cores
- Cross-validation and hyperparameter search are parallelized
- Memory usage scales with number of parallel jobs

## Troubleshooting

### Common Issues

1. **Data file not found**
   ```bash
   # Check file path and permissions
   ls -la data/processed/sample_data.csv
   ```

2. **Configuration validation errors**
   ```bash
   # Use dry-run to validate config
   python scripts/train_baseline_model.py --config configs/default_training.yaml --dry-run
   ```

3. **Memory errors**
   ```bash
   # Use quick mode or reduce parallel jobs
   python scripts/train_baseline_model.py --data data.csv --quick --n-jobs 4
   ```

4. **Permission errors**
   ```bash
   # Check output directory permissions
   mkdir -p results && chmod 755 results
   ```

### Debug Mode
Use `--debug` for detailed logging and error diagnosis:
```bash
python scripts/train_baseline_model.py --data data.csv --debug --experiment-name debug
```

## Advanced Usage

### Resume Training
```bash
python scripts/train_baseline_model.py \
  --resume-from results/experiment/checkpoints/checkpoint.pkl \
  --data data/processed/sample_data.csv
```

### Performance Profiling
```bash
python scripts/train_baseline_model.py \
  --data data/processed/sample_data.csv \
  --profile \
  --experiment-name profiling_test
```

### Custom Model Parameters
```bash
python scripts/train_baseline_model.py \
  --data data/processed/sample_data.csv \
  --model-params '{"n_estimators": 500, "learning_rate": 0.05, "max_depth": 10}'
```

## Integration

The training script integrates seamlessly with the existing codebase:
- Uses existing `BaselineXGBoostModel`
- Compatible with `CastingFeatureEngineer`
- Leverages `ModelTrainer` and `ModelEvaluator`
- Supports all current data formats and configurations