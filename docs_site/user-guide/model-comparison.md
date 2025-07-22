# Model Comparison

The Steel Defect Prediction System supports multiple machine learning models, allowing you to compare their performance
and choose the best model for your specific casting conditions.

## Available Models

### 1. LSTM Neural Network

**Best for**:  Time-series prediction with temporal dependencies

```python
from src.models.lstm_model import LSTMModel

# Load LSTM model

lstm_model = LSTMModel(
    input_size=10,
    hidden_size=64,
    num_layers=2,
    output_size=1
)
```

#### Advantages: 

- Captures temporal patterns in sensor data
- Good for sequential defect prediction
- Handles variable-length sequences

#### Performance: 

- Accuracy:  89.5%
- Precision:  87.2%
- Recall:  91.8%
- F1-Score:  89.4%

### 2. Random Forest

**Best for**:  Feature importance analysis and interpretability

```python
from src.models.random_forest_model import RandomForestModel

# Load Random Forest model

rf_model = RandomForestModel(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
```

#### Advantages: 

- Fast training and inference
- Feature importance ranking
- Robust to outliers
- No overfitting tendency

#### Performance: 

- Accuracy:  85.3%
- Precision:  84.1%
- Recall:  86.7%
- F1-Score:  85.4%

### 3. Gradient Boosting

**Best for**:  High accuracy with tabular data

```python
from src.models.gradient_boosting_model import GradientBoostingModel

# Load Gradient Boosting model

gb_model = GradientBoostingModel(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6
)
```

#### Advantages: 

- High prediction accuracy
- Good generalization
- Handles missing values
- Feature interaction capture

#### Performance: 

- Accuracy:  91.2%
- Precision:  90.8%
- Recall:  91.6%
- F1-Score:  91.2%

## Model Comparison Dashboard

### Access Comparison View

Navigate to the model comparison dashboard: 

```http
http: //localhost: 8000/models/comparison
```

### Performance Metrics

The dashboard displays comprehensive metrics for each model: 

#### Classification Metrics

- **Accuracy**:  Overall correct predictions
- **Precision**:  True positives / (True positives + False positives)
- **Recall**:  True positives / (True positives + False negatives)
- **F1-Score**:  Harmonic mean of precision and recall
- **AUC-ROC**:  Area under ROC curve

#### Operational Metrics

- **Inference Time**:  Average prediction time
- **Memory Usage**:  Model memory footprint
- **Training Time**:  Time to train the model
- **Model Size**:  File size of saved model

### Model Selection Criteria

#### For Real-time Applications

```python

# Prioritize inference speed

model_ranking = {
    'random_forest':  {'speed':  9, 'accuracy':  7, 'memory':  8},
    'gradient_boosting':  {'speed':  7, 'accuracy':  9, 'memory':  6},
    'lstm':  {'speed':  5, 'accuracy':  8, 'memory':  4}
}
```

#### For High Accuracy Requirements

```python

# Prioritize prediction accuracy

model_ranking = {
    'gradient_boosting':  {'accuracy':  9, 'stability':  8, 'interpretability':  7},
    'lstm':  {'accuracy':  8, 'stability':  7, 'interpretability':  5},
    'random_forest':  {'accuracy':  7, 'stability':  9, 'interpretability':  9}
}
```

## Running Model Comparisons

### Automated Comparison

```python
from src.evaluation.model_comparator import ModelComparator

# Initialize comparator

comparator = ModelComparator(
    models=['lstm', 'random_forest', 'gradient_boosting'],
    test_data_path='data/test_dataset.csv'
)

# Run comparison

results = comparator.compare_models()

# Display results

comparator.generate_report(output_path='reports/model_comparison.html')
```

### Custom Evaluation

```python
from src.evaluation.custom_evaluator import CustomEvaluator

# Define custom metrics

custom_metrics = {
    'defect_detection_rate':  lambda y_true, y_pred:  custom_defect_rate(y_true, y_pred),
    'false_alarm_rate':  lambda y_true, y_pred:  custom_false_alarm_rate(y_true, y_pred),
    'cost_savings':  lambda y_true, y_pred:  calculate_cost_savings(y_true, y_pred)
}

evaluator = CustomEvaluator(custom_metrics)
results = evaluator.evaluate_all_models(test_data)
```

## Model Ensemble

### Ensemble Configuration

Combine multiple models for improved performance: 

```python
from src.models.ensemble_model import EnsembleModel

# Create ensemble

ensemble = EnsembleModel(
    models=[lstm_model, rf_model, gb_model],
    weights=[0.4, 0.3, 0.3],  # Model weights
    method='weighted_average'  # or 'voting', 'stacking'
)

# Make predictions

prediction = ensemble.predict(sensor_data)
```

### Voting Ensemble

```python

# Majority voting ensemble

voting_ensemble = EnsembleModel(
    models=[lstm_model, rf_model, gb_model],
    method='majority_voting',
    threshold=0.5
)
```

### Stacking Ensemble

```python

# Stacking with meta-learner

stacking_ensemble = EnsembleModel(
    base_models=[lstm_model, rf_model, gb_model],
    meta_model=LogisticRegression(),
    method='stacking'
)
```

## Performance Analysis

### Cross-Validation Results

```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation

models = {
    'LSTM':  lstm_model,
    'Random Forest':  rf_model,
    'Gradient Boosting':  gb_model
}

cv_results = {}
for name, model in models.items(): 
    scores = cross_val_score(model, X, y, cv=5, scoring='f1')
    cv_results[name] = {
        'mean':  scores.mean(),
        'std':  scores.std(),
        'scores':  scores.tolist()
    }
```

### Learning Curves

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Generate learning curves

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
)

# Plot learning curves

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training Score')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('F1 Score')
plt.legend()
plt.show()
```

### Feature Importance Comparison

```python

# Compare feature importance across models

feature_importance = {}

# Random Forest importance

feature_importance['random_forest'] = rf_model.feature_importances_

# Gradient Boosting importance

feature_importance['gradient_boosting'] = gb_model.feature_importances_

# LSTM attention weights (if available)

feature_importance['lstm'] = lstm_model.get_attention_weights()

# Visualize feature importance

import pandas as pd
import seaborn as sns

importance_df = pd.DataFrame(feature_importance, index=feature_names)
plt.figure(figsize=(12, 8))
sns.heatmap(importance_df, annot=True, cmap='viridis')
plt.title('Feature Importance Comparison')
plt.show()
```

## Model Selection Wizard

### Interactive Selection Tool

```python
from src.tools.model_selector import ModelSelector

# Launch interactive selector

selector = ModelSelector()
selector.launch_wizard()

# Answer questions about your requirements

# - Prediction speed requirements

# - Accuracy requirements

# - Interpretability needs

# - Resource constraints

# Get recommendation

recommended_model = selector.get_recommendation()
print(f"Recommended model:  {recommended_model}")
```

### Automated Selection

```python

# Automated model selection based on data characteristics

from src.tools.auto_selector import AutoModelSelector

auto_selector = AutoModelSelector()
best_model = auto_selector.select_best_model(
    X_train, y_train, X_val, y_val,
    constraints={
        'max_inference_time':  100,  # milliseconds
        'max_memory_usage':  512,    # MB
        'min_accuracy':  0.85
    }
)
```

## Production Deployment

### A/B Testing

```python
from src.deployment.ab_testing import ABTestManager

# Set up A/B test

ab_test = ABTestManager()
ab_test.setup_test(
    model_a='gradient_boosting_v1',
    model_b='lstm_v2',
    traffic_split=0.5  # 50/50 split
)

# Monitor test results

results = ab_test.get_results()
print(f"Model A accuracy:  {results['model_a']['accuracy']}")
print(f"Model B accuracy:  {results['model_b']['accuracy']}")
```

### Gradual Rollout

```python

# Gradually increase traffic to new model

rollout_schedule = [
    {'model':  'new_model', 'percentage':  10, 'duration':  '1 day'},
    {'model':  'new_model', 'percentage':  25, 'duration':  '2 days'},
    {'model':  'new_model', 'percentage':  50, 'duration':  '3 days'},
    {'model':  'new_model', 'percentage':  100, 'duration':  'ongoing'}
]

ab_test.gradual_rollout(rollout_schedule)
```

## Model Monitoring

### Performance Drift Detection

```python
from src.monitoring.model_monitor import ModelMonitor

# Monitor model performance over time

monitor = ModelMonitor(model_name='production_model')

# Track prediction quality

monitor.log_prediction(y_true, y_pred, timestamp)

# Check for performance drift

drift_detected = monitor.detect_drift(
    window_size=1000,
    threshold=0.05  # 5% accuracy drop
)

if drift_detected: 
    print("Model performance drift detected - consider retraining")
```

### Data Drift Detection

```python
from src.monitoring.data_drift import DataDriftDetector

# Monitor input data distribution

drift_detector = DataDriftDetector(reference_data=X_train)

# Check for data drift

drift_score = drift_detector.detect_drift(new_data=X_recent)

if drift_score > 0.1: 
    print(f"Data drift detected (score:  {drift_score: .3f})")
    print("Consider updating model with recent data")
```

This comprehensive model comparison framework helps you select and maintain the optimal model for your steel defect prediction needs.
