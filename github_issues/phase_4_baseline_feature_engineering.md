# Phase 4: Baseline Feature Engineering & Training

## Description

Develop and train a baseline machine learning model using gradient boosted decision trees (GBDT) on engineered features from the continuous casting data. This establishes a strong baseline for comparison with sequence-based approaches and provides interpretable insights into defect prediction.

## Context

Based on the Technical Design specification, this phase implements the "Stage 1 Baseline" using GBDT on engineered per-cast features. This approach provides rapid benchmarking, interpretability through feature importance analysis, and serves as a foundation for understanding which engineered features capture defect-related patterns most effectively.

## Objectives

- Design comprehensive feature engineering pipeline for continuous casting data
- Implement robust baseline classification model using gradient boosting
- Establish model training, validation, and evaluation workflows
- Create interpretable model outputs for domain understanding
- Set performance benchmarks for sequence model comparison

## Acceptance Criteria

### Feature Engineering Pipeline (`src/modeling/features.py`)
- [ ] **Statistical Features**: Per-cast aggregations (mean, median, std, min, max, percentiles)
- [ ] **Dynamic/Stability Features**: Spike counts, excursion frequencies, variance measures
- [ ] **Duration-based Features**: Time above/below thresholds, stability periods
- [ ] **Cross-interaction Features**: Ratios, products, derived physics-based metrics
- [ ] **Anomaly Features**: Outlier scores, deviation metrics (optional)
- [ ] **Feature scaling and normalization**: Robust preprocessing pipeline

### Model Development and Training
- [ ] **Data splitting**: Stratified train/validation/test splits (70/15/15)
- [ ] **Model selection**: XGBoost or LightGBM implementation
- [ ] **Hyperparameter optimization**: Grid search or Bayesian optimization
- [ ] **Early stopping**: Validation-based training termination
- [ ] **Cross-validation**: K-fold validation for robust performance estimation

### Model Evaluation and Metrics
- [ ] **Primary metrics**: ROC-AUC, Precision-Recall AUC for imbalanced data
- [ ] **Classification metrics**: Precision, Recall, F1-score, Specificity
- [ ] **Confusion matrix analysis**: Detailed error analysis
- [ ] **Feature importance analysis**: SHAP values and built-in importance scores
- [ ] **Performance by subgroups**: Analysis by steel grade, operating conditions

### Model Artifacts and Persistence
- [ ] **Model serialization**: Trained model and preprocessing pipeline
- [ ] **Feature metadata**: Feature definitions, importance rankings
- [ ] **Performance reports**: Comprehensive evaluation documentation
- [ ] **Model configuration**: Hyperparameters and training settings

## Implementation Tasks

### Feature Engineering Implementation

#### Statistical Aggregation Features
```python
def compute_statistical_features(ts_data, cast_id):
    """Compute per-cast statistical aggregations for each sensor"""
    features = {}
    for sensor in sensors:
        sensor_data = ts_data[ts_data['sensor'] == sensor]['value']
        features.update({
            f'{sensor}_mean': sensor_data.mean(),
            f'{sensor}_median': sensor_data.median(),
            f'{sensor}_std': sensor_data.std(),
            f'{sensor}_min': sensor_data.min(),
            f'{sensor}_max': sensor_data.max(),
            f'{sensor}_q25': sensor_data.quantile(0.25),
            f'{sensor}_q75': sensor_data.quantile(0.75),
            f'{sensor}_range': sensor_data.max() - sensor_data.min(),
            f'{sensor}_iqr': sensor_data.quantile(0.75) - sensor_data.quantile(0.25)
        })
    return features
```

#### Dynamic and Stability Features
```python
def compute_stability_features(ts_data, cast_id):
    """Compute process stability and dynamic behavior features"""
    features = {}
    
    # Mold level stability
    mold_level = get_sensor_data(ts_data, 'mold_level')
    features['mold_level_excursions'] = count_excursions(mold_level, threshold=20)
    features['mold_level_stability_time'] = stable_operation_time(mold_level, tolerance=5)
    
    # Temperature change rates
    tundish_temp = get_sensor_data(ts_data, 'tundish_temperature')
    features['temp_change_rate_max'] = max_change_rate(tundish_temp, window=30)
    
    # Oscillation stability
    osc_freq = get_sensor_data(ts_data, 'oscillation_frequency')
    features['oscillation_cv'] = osc_freq.std() / osc_freq.mean()
    
    # Casting speed variability
    speed = get_sensor_data(ts_data, 'casting_speed')
    features['speed_outside_nominal'] = time_outside_range(speed, 0.8, 1.2)
    
    return features
```

#### Cross-Interaction and Physics-Based Features
```python
def compute_interaction_features(ts_data, cast_id):
    """Compute cross-sensor interactions and physics-based features"""
    features = {}
    
    # Speed-superheat interaction
    speed = get_sensor_data(ts_data, 'casting_speed').mean()
    superheat = get_sensor_metadata(cast_id, 'superheat')
    features['speed_superheat_product'] = speed * superheat
    
    # Cooling efficiency
    inlet_temp = get_sensor_data(ts_data, 'cooling_water_inlet_temp').mean()
    outlet_temp = get_sensor_data(ts_data, 'cooling_water_outlet_temp').mean()
    flow_rate = get_sensor_data(ts_data, 'cooling_water_flow').mean()
    features['cooling_efficiency'] = (outlet_temp - inlet_temp) * flow_rate
    
    # Heat flux proxy
    mold_temp = get_sensor_data(ts_data, 'mold_temperature').mean()
    water_temp = get_sensor_data(ts_data, 'cooling_water_outlet_temp').mean()
    features['heat_flux_proxy'] = mold_temp - water_temp
    
    return features
```

### Model Training Framework

#### Data Preparation Pipeline
```python
class FeatureEngineeringPipeline:
    def __init__(self):
        self.scalers = {}
        self.feature_names = []
        
    def fit_transform(self, ts_data, metadata):
        # Extract features for all casts
        # Apply scaling/normalization
        # Handle missing values
        # Return feature matrix and labels
        
    def transform(self, ts_data, metadata):
        # Apply fitted transformations to new data
        # Ensure consistent feature ordering
```

#### Model Training Implementation
```python
def train_baseline_model(X_train, y_train, X_val, y_val):
    # Hyperparameter optimization
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    }
    
    # Grid search with cross-validation
    model = xgboost.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        early_stopping_rounds=50
    )
    
    # Train best model
    # Return trained model, best params, validation scores
```

### Feature Engineering Categories

#### Core Statistical Features (per sensor)
- **Central tendency**: mean, median, mode
- **Variability**: standard deviation, variance, coefficient of variation
- **Distribution shape**: skewness, kurtosis
- **Percentiles**: 5th, 25th, 75th, 95th percentiles
- **Range metrics**: min, max, range, interquartile range

#### Process Stability Indicators
- **Mold level**: Excursion count/duration, stability percentage, fluctuation rate
- **Temperature**: Maximum change rates, gradient measures, stability periods
- **Casting speed**: Time outside nominal range, acceleration events
- **Oscillation**: Frequency variance, stroke consistency, harmonic analysis
- **Cooling water**: Flow balance, temperature differential stability

#### Critical Event Detection
- **Threshold exceedances**: Duration above/below critical values
- **Spike detection**: Count and severity of anomalous readings
- **Trend detection**: Monotonic increase/decrease periods
- **Pattern matching**: Known problematic signatures

#### Physics-Based Derived Features
- **Heat transfer**: Cooling efficiency proxies, heat flux estimates
- **Fluid dynamics**: Flow ratios, Reynolds number proxies
- **Mechanical stress**: Speed-oscillation interactions, force balance
- **Material properties**: Grade-specific parameter normalization

### Model Configuration and Tuning

#### Handling Class Imbalance
- **Class weights**: Balanced class weighting or custom ratios
- **Sampling strategies**: SMOTE for feature augmentation
- **Threshold tuning**: Optimize decision threshold for desired precision/recall balance
- **Cost-sensitive learning**: Custom loss functions emphasizing defect detection

#### Hyperparameter Optimization
- **Search strategy**: Grid search or Bayesian optimization (Optuna)
- **Validation strategy**: Stratified K-fold cross-validation
- **Early stopping**: Monitor validation AUC with patience
- **Regularization**: L1/L2 penalties to prevent overfitting

## Dependencies

- **Prerequisite**: Phase 2 (Synthetic Data Generation) and Phase 3 (EDA) complete
- **Required data**: Training dataset with engineered features
- **EDA insights**: Feature engineering informed by exploratory analysis

## Expected Deliverables

1. **Feature Engineering Pipeline**: `src/modeling/features.py`
   - Comprehensive feature extraction functions
   - Preprocessing and scaling utilities
   - Feature metadata and documentation

2. **Model Training Code**: `src/modeling/baseline_model.py`
   - Training pipeline implementation
   - Hyperparameter optimization
   - Model evaluation utilities

3. **Trained Model Artifacts**: `models/baseline/`
   - Serialized trained model (`.pkl` or `.joblib`)
   - Feature preprocessing pipeline
   - Model configuration and hyperparameters

4. **Evaluation Reports**: `models/baseline/evaluation/`
   - Performance metrics summary
   - Feature importance analysis
   - SHAP value explanations
   - Confusion matrix and classification reports

5. **Documentation**: 
   - Feature engineering methodology
   - Model training procedures
   - Performance interpretation guide

## Technical Considerations

### Feature Engineering Best Practices
- **Domain knowledge integration**: Leverage continuous casting physics
- **Feature selection**: Remove redundant and low-importance features
- **Scaling considerations**: Robust scaling for outlier resistance
- **Temporal aggregation**: Appropriate time windows for feature computation

### Model Robustness
- **Cross-validation**: Ensure stable performance across data splits
- **Feature stability**: Consistent importance across training runs
- **Generalization**: Performance on held-out test set
- **Interpretability**: Clear connection between features and domain knowledge

### Performance Optimization
- **Feature computation efficiency**: Vectorized operations for large datasets
- **Model training speed**: Parallel processing and early stopping
- **Memory management**: Efficient data structures for large feature matrices
- **Inference speed**: Fast prediction for real-time deployment

## Success Metrics

- [ ] **Model Performance**: ROC-AUC > 0.80, Precision-Recall AUC > 0.70
- [ ] **Feature Quality**: Top features align with domain knowledge
- [ ] **Training Efficiency**: Complete training pipeline < 30 minutes
- [ ] **Interpretability**: Clear feature importance explanations
- [ ] **Robustness**: Stable performance across cross-validation folds
- [ ] **Defect Detection**: Recall > 0.75 for defect class

## Model Interpretability Requirements

- [ ] **Feature importance rankings**: Global feature importance scores
- [ ] **SHAP analysis**: Local and global explanations
- [ ] **Decision tree visualization**: Sample tree inspection
- [ ] **Feature interaction analysis**: Two-way interaction effects
- [ ] **Threshold analysis**: Decision boundary interpretation

## Notes

This baseline model serves multiple purposes:
1. **Performance benchmark** for sequence models
2. **Feature validation** to understand most predictive signals
3. **Domain insight** through interpretable model outputs
4. **Rapid deployment option** for immediate value

Focus on creating a robust, well-documented baseline that provides both strong performance and interpretable insights. The feature engineering pipeline developed here will also benefit the sequence modeling phase.

Pay special attention to:
- Class imbalance handling (defects are minority class)
- Feature engineering quality over quantity
- Model interpretability for domain validation
- Robust evaluation methodology

## Labels
`enhancement`, `phase-4`, `baseline-model`, `feature-engineering`, `machine-learning`

## Priority
**High** - Establishes critical baseline performance and feature understanding