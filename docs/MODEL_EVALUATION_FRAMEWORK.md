# Comprehensive Model Evaluation Framework

This document provides detailed documentation for the comprehensive model evaluation framework implemented for steel defect detection.

## Overview

The evaluation framework provides a complete suite of tools for evaluating, analyzing, and reporting on machine learning model performance, specifically designed for steel defect prediction scenarios.

## Core Components

### 1. ModelEvaluator (`src/models/model_evaluator.py`)

The main evaluation class providing comprehensive model assessment capabilities.

**Key Features:**
- Comprehensive metric calculation (25+ metrics)
- ROC and Precision-Recall curve generation
- Confusion matrix analysis
- Threshold optimization
- Cross-validation analysis
- Model comparison
- Calibration analysis

**Basic Usage:**
```python
from src.models.model_evaluator import ModelEvaluator

evaluator = ModelEvaluator(
    model=trained_model,
    model_name="SteelDefect_XGBoost",
    output_dir="results/evaluation"
)

results = evaluator.evaluate_model(X_test, y_test)
```

### 2. Custom Metrics (`src/models/evaluation_metrics.py`)

Steel defect-specific evaluation metrics designed for manufacturing scenarios.

**Available Metrics:**
- `defect_detection_rate()` - Recall for defect class
- `false_alarm_rate()` - False positive rate
- `production_impact_score()` - Cost-based impact assessment
- `quality_efficiency_score()` - Balanced quality-efficiency metric
- `manufacturing_kpi_suite()` - Complete manufacturing KPI set

**Usage:**
```python
from src.models.evaluation_metrics import CustomMetrics

steel_metrics = CustomMetrics.manufacturing_kpi_suite(y_true, y_pred)
```

### 3. Advanced Plotting (`src/models/evaluation_plots.py`)

Publication-quality visualization utilities for model evaluation.

**Features:**
- Interactive Plotly dashboards
- Publication-quality matplotlib plots
- Comprehensive evaluation dashboards
- Model comparison visualizations
- Threshold analysis plots

**Usage:**
```python
from src.models.evaluation_plots import EvaluationPlots

plotter = EvaluationPlots(style='seaborn')
dashboard = plotter.create_dashboard(evaluation_results)
```

### 4. Report Generation (`src/models/evaluation_reports.py`)

Multi-format report generation for comprehensive documentation.

**Supported Formats:**
- HTML (interactive reports)
- Markdown (documentation)
- JSON (data interchange)
- Excel (spreadsheet analysis)

**Usage:**
```python
from src.models.evaluation_reports import EvaluationReports

reporter = EvaluationReports(output_dir="results/reports")
report_paths = reporter.generate_comprehensive_report(results, model_info)
```

### 5. Evaluation Utilities (`src/models/evaluation_utils.py`)

Statistical analysis and utility functions for robust evaluation.

**Features:**
- Bootstrap confidence intervals
- Statistical significance testing
- Input validation
- Prediction issue detection
- Sample size calculations

**Usage:**
```python
from src.models.evaluation_utils import EvaluationUtils

# Bootstrap confidence intervals
bootstrap_results = EvaluationUtils.bootstrap_metric(y_true, y_pred, f1_score)

# Statistical significance testing
sig_test = EvaluationUtils.statistical_significance_test(scores1, scores2)
```

## Configuration

The framework uses YAML-based configuration in `configs/evaluation.yaml`:

```yaml
evaluation:
  settings:
    default_threshold: 0.5
    output_directory: "results/evaluation"
    save_plots: true
    
  metrics:
    basic: ["accuracy", "precision", "recall", "f1_score"]
    probabilistic: ["roc_auc", "average_precision"]
    custom: ["defect_detection_rate", "false_alarm_rate"]
    
  cross_validation:
    cv_folds: 5
    stratify: true
    scoring_metrics: ["roc_auc", "average_precision", "f1"]
```

## Complete Workflow Example

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.baseline_model import BaselineXGBoostModel
from src.models.model_evaluator import ModelEvaluator
from src.models.evaluation_metrics import CustomMetrics
from src.models.evaluation_reports import EvaluationReports

# 1. Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 2. Train model
model = BaselineXGBoostModel()
model.fit(X_train, y_train)

# 3. Initialize evaluator
evaluator = ModelEvaluator(
    model=model,
    model_name="SteelDefect_Production",
    output_dir="results/production_evaluation"
)

# 4. Comprehensive evaluation
results = evaluator.evaluate_model(X_test, y_test)

# 5. Steel-specific metrics
y_pred = model.predict(X_test)
steel_metrics = CustomMetrics.manufacturing_kpi_suite(y_test, y_pred)

# 6. Generate visualizations
y_proba = model.predict_proba(X_test)
roc_fig = evaluator.plot_roc_curve(y_test, y_proba)
pr_fig = evaluator.plot_precision_recall_curve(y_test, y_proba)

# 7. Cross-validation analysis
cv_results = evaluator.cross_validation_analysis(
    model.model, X_train, y_train, cv_folds=5
)

# 8. Threshold optimization
optimal_thresh, optimal_f1 = evaluator.find_optimal_threshold(
    y_test, y_proba, 'f1'
)

# 9. Model comparison
models = {
    'XGBoost': model.model,
    'RandomForest': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression()
}
comparison_df = evaluator.compare_models(models, X_train, y_train)

# 10. Generate comprehensive reports
reporter = EvaluationReports()
model_info = {
    'model_type': 'XGBoost',
    'training_samples': len(X_train),
    'features': X.shape[1]
}
report_paths = reporter.generate_comprehensive_report(results, model_info)

# 11. Export artifacts
from src.models.evaluation_utils import EvaluationUtils
artifacts = EvaluationUtils.save_evaluation_artifacts(
    results, "results/artifacts", "SteelDefect_Production"
)

print(f"Evaluation completed - ROC AUC: {results['metrics']['roc_auc']:.3f}")
```

## Key Metrics for Steel Defect Detection

### Standard Classification Metrics
- **ROC AUC**: Overall discriminative ability
- **Precision**: Proportion of defect predictions that are correct
- **Recall**: Proportion of actual defects detected
- **F1 Score**: Harmonic mean of precision and recall
- **Specificity**: Proportion of normal products correctly identified

### Steel Manufacturing Metrics
- **Defect Detection Rate**: Critical for quality assurance
- **False Alarm Rate**: Important for production efficiency
- **Production Impact Score**: Cost-based performance assessment
- **Quality-Efficiency Score**: Balanced manufacturing performance

## Visualization Outputs

The framework generates comprehensive visualizations:

1. **ROC Curve**: Shows true positive vs false positive rates
2. **Precision-Recall Curve**: Shows precision vs recall trade-offs
3. **Confusion Matrix**: Detailed breakdown of predictions
4. **Calibration Plot**: Reliability of probability predictions
5. **Threshold Analysis**: Performance across different thresholds
6. **Feature Importance**: Most influential model features
7. **Evaluation Dashboard**: Combined overview of all metrics

## Report Formats

Generated reports include:

### HTML Report
- Interactive visualizations
- Executive summary
- Detailed metric analysis
- Recommendations

### Markdown Report
- Documentation-friendly format
- Version control compatible
- Easy integration with documentation systems

### JSON Report
- Machine-readable format
- API integration ready
- Programmatic analysis support

### Excel Report
- Multiple worksheets
- Business-friendly format
- Further analysis capability

## Testing

Comprehensive test suite with 20+ unit tests:

```bash
# Run all evaluation framework tests
python -m pytest tests/test_model_evaluation.py -v

# Run specific test category
python -m pytest tests/test_model_evaluation.py::TestModelEvaluator -v
```

## Integration with Existing Models

The framework seamlessly integrates with the existing `BaselineXGBoostModel`:

```python
# Train baseline model
baseline_model = BaselineXGBoostModel()
baseline_model.fit(X_train, y_train)

# Evaluate with comprehensive framework
evaluator = ModelEvaluator(model=baseline_model)
results = evaluator.evaluate_model(X_test, y_test)
```

## Best Practices

1. **Always use cross-validation** for robust performance estimates
2. **Include steel-specific metrics** for manufacturing relevance
3. **Generate confidence intervals** for metric reliability
4. **Compare multiple models** for baseline establishment
5. **Optimize thresholds** based on business objectives
6. **Validate inputs** before evaluation
7. **Save all artifacts** for reproducibility

## Performance Considerations

- Framework supports datasets up to 100K+ samples
- Visualization generation scales with sample size
- Bootstrap analysis may be time-intensive for large datasets
- Cross-validation time scales with model complexity

## Troubleshooting

### Common Issues

**Import Errors**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

**Memory Issues**: Reduce bootstrap samples or CV folds for large datasets

**Plotting Errors**: Check matplotlib backend configuration

**Model Compatibility**: Ensure model has `predict` and `predict_proba` methods

## Future Extensions

Potential framework enhancements:
- Multi-class classification support
- Regression model evaluation
- Time series model assessment
- Deep learning model integration
- Real-time monitoring capabilities
- A/B testing framework

## Dependencies

Core dependencies:
- scikit-learn >= 1.1.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- plotly >= 5.15.0
- pandas >= 1.5.0
- numpy >= 1.23.0
- scipy >= 1.9.0
- statsmodels >= 0.13.0
- jinja2 >= 3.1.0
- openpyxl >= 3.1.0