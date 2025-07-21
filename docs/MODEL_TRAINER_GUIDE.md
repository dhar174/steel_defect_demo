# Automated Model Training Pipeline

This document describes the new automated model training pipeline implemented in the `ModelTrainer` class.

## Overview

The `ModelTrainer` provides a comprehensive automated training pipeline that orchestrates the entire machine learning workflow from data preprocessing to model evaluation. It features:

- **Automated data preprocessing** with configurable scaling and outlier handling
- **Hyperparameter optimization** using Grid Search, Random Search, or Bayesian Optimization
- **Cross-validation** with stratified sampling
- **Early stopping** and overfitting detection
- **Experiment tracking** and artifact management
- **Configuration-driven** approach for easy experimentation

## Quick Start

### Basic Usage

```python
from src.models import ModelTrainer

# Initialize trainer with default configuration
trainer = ModelTrainer(
    model_type='xgboost',
    experiment_name='my_experiment'
)

# Run complete training pipeline
results = trainer.train_pipeline(
    data_path='data/my_dataset.csv',
    target_column='target'
)

print(f"Test AUC: {results['test_evaluation']['roc_auc']:.3f}")
```

### Configuration-Based Training

```python
from src.models import ModelTrainer, ConfigurationManager

# Create and customize configuration
config_manager = ConfigurationManager()
config = config_manager.load_config('configs/training_pipeline.yaml')

# Customize for your experiment
config.experiment.name = "steel_defect_baseline"
config.hyperparameter_search.enabled = True
config.hyperparameter_search.method = "random"

# Train with configuration
trainer = ModelTrainer(config_path='configs/training_pipeline.yaml')
results = trainer.train_pipeline(data_path='data/processed.csv', target_column='defect')
```

## Key Components

### 1. ModelTrainer
The main class that orchestrates the training pipeline.

**Key Methods:**
- `train_pipeline()` - Complete automated pipeline
- `train_model()` - Train with specific parameters  
- `hyperparameter_search()` - Optimize hyperparameters
- `train_with_cross_validation()` - Cross-validation training

### 2. DataPreprocessor
Handles data preprocessing operations.

**Features:**
- Automatic feature type detection
- Missing value imputation
- Outlier detection and handling (IQR, Z-score, Isolation Forest)
- Feature scaling (Standard, Robust, MinMax)
- Categorical encoding

### 3. HyperparameterSearcher  
Performs hyperparameter optimization.

**Methods:**
- Grid Search
- Random Search  
- Bayesian Optimization (with scikit-optimize)

### 4. TrainingPipelineConfig
Configuration management system.

**Features:**
- YAML/JSON configuration files
- Parameter validation
- Default configurations
- Nested configuration structure

## Configuration

The training pipeline is configured via YAML files. Here's the structure:

```yaml
training_pipeline:
  experiment:
    name: "my_experiment"
    description: "Description of the experiment"
    
  data:
    target_column: "target"
    test_size: 0.2
    validation_size: 0.2
    stratify: true
    
  preprocessing:
    handle_missing: true
    missing_strategy: "median"
    scaling:
      method: "standard"
    outliers:
      method: "iqr"
      threshold: 1.5
      
  model:
    type: "xgboost"
    parameters:
      n_estimators: 100
      max_depth: 6
      learning_rate: 0.1
      
  hyperparameter_search:
    enabled: true
    method: "grid"
    param_grids:
      coarse:
        n_estimators: [50, 100, 200]
        max_depth: [3, 6, 9]
        learning_rate: [0.05, 0.1, 0.2]
        
  training:
    cross_validation:
      enabled: true
      cv_folds: 5
    early_stopping:
      enabled: true
      patience: 10
```

## Advanced Features

### Hyperparameter Optimization

The pipeline supports three hyperparameter optimization methods:

1. **Grid Search** - Exhaustive search over parameter grid
2. **Random Search** - Random sampling of parameter space
3. **Bayesian Optimization** - Intelligent search using Gaussian processes

Example:
```python
# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.05, 0.1, 0.2]
}

# Run hyperparameter search
search_results = trainer.hyperparameter_search(
    X_train, y_train,
    param_grid=param_grid,
    search_method='bayesian',
    n_calls=50
)
```

### Cross-Validation

Stratified k-fold cross-validation is built-in:

```python
cv_results = trainer.train_with_cross_validation(
    X, y, 
    cv_folds=5
)
print(f"CV AUC: {cv_results['roc_auc_mean']:.3f} ± {cv_results['roc_auc_std']:.3f}")
```

### Early Stopping and Overfitting Detection

The pipeline includes overfitting detection:

```python
# Configure early stopping
trainer.setup_early_stopping(
    monitor_metric='val_auc',
    patience=10,
    min_delta=0.001
)

# Train with early stopping
results = trainer.train_model(X_train, y_train, X_val, y_val)
```

### Experiment Tracking

All experiments are automatically tracked with:
- Timestamped directories
- Model artifacts (trained model, preprocessor)
- Training results and metrics
- Configuration snapshots
- Logs

```
experiments/
├── my_experiment/
│   ├── models/
│   │   ├── model_20230721_143022.joblib
│   │   └── preprocessing_pipeline_20230721_143022.joblib
│   ├── results/
│   │   ├── training_results_20230721_143022.json
│   │   └── training_history_20230721_143022.json
│   └── plots/
└── logs/
```

## Examples

### Complete Example
See `examples/model_trainer_example.py` for a complete working example.

### Training with Real Data
```python
from src.models import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(
    config_path='configs/training_pipeline.yaml',
    model_type='xgboost',
    experiment_name='steel_defect_production'
)

# Run pipeline with your data
results = trainer.train_pipeline(
    data_path='data/processed/steel_defect_features.csv',
    target_column='defect'
)

# Access results
print(f"Test Performance:")
print(f"  AUC: {results['test_evaluation']['roc_auc']:.3f}")
print(f"  F1: {results['test_evaluation']['f1_score']:.3f}")
print(f"  Precision: {results['test_evaluation']['precision']:.3f}")
print(f"  Recall: {results['test_evaluation']['recall']:.3f}")
```

## Integration with Existing Code

The `ModelTrainer` integrates seamlessly with existing models:

```python
# Use with existing BaselineXGBoostModel
from src.models import BaselineXGBoostModel, ModelTrainer

# Create trainer that wraps existing model
trainer = ModelTrainer(model_type='xgboost')

# Access underlying model
baseline_model = trainer.model  # This is a BaselineXGBoostModel instance

# Use all existing BaselineXGBoostModel functionality
feature_importance = baseline_model.get_feature_importance()
baseline_model.plot_feature_importance()
```

## Testing

Comprehensive tests are available in `tests/test_model_trainer_comprehensive.py`:

```bash
# Run all ModelTrainer tests
python -m pytest tests/test_model_trainer_comprehensive.py -v

# Run specific test
python -m pytest tests/test_model_trainer_comprehensive.py::TestModelTrainer::test_full_pipeline_execution -v
```

## Dependencies

The pipeline requires:
- `scikit-learn>=1.1.0`
- `pandas>=1.5.0` 
- `numpy>=1.23.0`
- `xgboost>=1.7.0`
- `pyyaml>=6.0`
- `joblib>=1.2.0`
- `matplotlib>=3.6.0`
- `seaborn>=0.12.0`
- `scikit-optimize>=0.9.0` (for Bayesian optimization)

Optional for PyTorch models:
- `torch>=1.12.0`

## Extending the Pipeline

### Adding New Model Types

```python
# Extend ModelTrainer for new model types
class CustomModelTrainer(ModelTrainer):
    def _initialize_model(self):
        if self.model_type == 'custom_model':
            self.model = CustomModel(**self.config['model']['parameters'])
        else:
            super()._initialize_model()
```

### Custom Preprocessing

```python
# Custom preprocessing pipeline
from src.models.preprocessing import DataPreprocessor

class CustomPreprocessor(DataPreprocessor):
    def custom_feature_engineering(self, X):
        # Add custom features
        X['interaction_feature'] = X['feature1'] * X['feature2']
        return X
```

For more examples and advanced usage, see the `examples/` directory.