"""
Comprehensive usage example for the BaselineXGBoostModel.

This example demonstrates all the key features and capabilities of the
implemented XGBoost baseline model for steel casting defect prediction.

Note: Run 'pip install -e .' from the repository root to install the package in development mode.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.baseline_model import BaselineXGBoostModel
from models.model_config import ModelConfig
from models.model_utils import optimize_threshold, calculate_cost_sensitive_metrics
from models.model_persistence import ModelPersistence


def generate_synthetic_steel_defect_data(n_samples=5000, n_features=25, random_state=42):
    """Generate synthetic data that mimics steel casting defect patterns."""
    np.random.seed(random_state)
    
    # Generate base features (sensor readings, process parameters, etc.)
    X = np.random.normal(0, 1, (n_samples, n_features))
    
    # Create meaningful feature names
    feature_names = (
        ['temperature_avg', 'temperature_std', 'pressure_avg', 'pressure_std'] +
        ['flow_rate_avg', 'flow_rate_std', 'vibration_level', 'chemical_composition_1'] +
        ['chemical_composition_2', 'cooling_rate', 'casting_speed', 'mold_temperature'] +
        [f'sensor_{i}' for i in range(12, n_features)]
    )
    
    # Create realistic defect signal based on domain knowledge
    # Higher temperatures and pressure variations increase defect probability
    temperature_effect = X[:, 0] * 0.4  # temperature_avg
    pressure_effect = X[:, 2] * 0.3     # pressure_avg  
    flow_effect = X[:, 4] * 0.2         # flow_rate_avg
    interaction_effect = X[:, 0] * X[:, 2] * 0.1  # temp-pressure interaction
    
    # Add some random noise
    noise = np.random.normal(0, 0.5, n_samples)
    
    # Combine effects
    defect_logits = temperature_effect + pressure_effect + flow_effect + interaction_effect + noise
    defect_proba = 1 / (1 + np.exp(-defect_logits))
    
    # Generate binary labels with realistic defect rate (5-15%)
    y = np.random.binomial(1, defect_proba * 0.12 + 0.03)
    
    # Create DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    
    return X_df, y


def main():
    """Main example demonstrating all model capabilities."""
    
    print("=" * 60)
    print("STEEL CASTING DEFECT PREDICTION - BASELINE XGBOOST MODEL")
    print("=" * 60)
    
    # 1. Generate synthetic data
    print("\n1. Generating synthetic steel casting data...")
    X, y = generate_synthetic_steel_defect_data(n_samples=3000, n_features=20)
    
    print(f"   Dataset shape: {X.shape}")
    print(f"   Features: {list(X.columns[:5])}... (showing first 5)")
    print(f"   Defect rate: {y.mean():.3f} ({np.sum(y)}/{len(y)} samples)")
    
    # 2. Load configuration
    print("\n2. Loading model configuration...")
    config_path = Path(__file__).parent.parent / 'configs' / 'baseline_model.yaml'
    
    if config_path.exists():
        config = ModelConfig(str(config_path))
        xgb_params = config.get_xgboost_params()
        print(f"   Loaded config from: {config_path.name}")
        print(f"   XGBoost params: n_estimators={xgb_params.get('n_estimators')}, "
              f"max_depth={xgb_params.get('max_depth')}")
    else:
        print("   Using default configuration")
        config = None
    
    # 3. Initialize model
    print("\n3. Initializing BaselineXGBoostModel...")
    if config:
        model = BaselineXGBoostModel(
            model_params=config.get_xgboost_params(),
            verbose=True
        )
    else:
        model = BaselineXGBoostModel(verbose=True)
    
    # 4. Split data
    print("\n4. Splitting data...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # 5. Cross-validation
    print("\n5. Performing cross-validation...")
    cv_results = model.cross_validate(
        X_train, y_train,
        cv_folds=5,
        scoring=['roc_auc', 'average_precision', 'f1']
    )
    
    for metric in ['roc_auc', 'average_precision', 'f1']:
        mean_score = cv_results[f'{metric}_mean']
        std_score = cv_results[f'{metric}_std']
        print(f"   CV {metric}: {mean_score:.4f} ± {std_score:.4f}")
    
    # 6. Hyperparameter optimization (optional, quick version)
    print("\n6. Hyperparameter optimization...")
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [4, 6],
        'learning_rate': [0.1, 0.15]
    }
    
    search_results = model.hyperparameter_search(
        X_train, y_train,
        param_grid=param_grid,
        cv_folds=3,
        search_method='grid'
    )
    
    print(f"   Best parameters: {search_results['best_params']}")
    print(f"   Best CV score: {search_results['best_score']:.4f}")
    
    # 7. Train final model
    print("\n7. Training final model...")
    training_history = model.fit(X_train, y_train)
    print(f"   Training completed in {training_history['training_time']:.2f} seconds")
    print(f"   Training AUC: {training_history['train_auc']:.4f}")
    
    # 8. Feature importance analysis
    print("\n8. Analyzing feature importance...")
    importance_df = model.get_feature_importance(max_features=10)
    print("   Top 10 most important features:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"     {row['feature']}: {row['importance']:.4f}")
    
    # 9. Model evaluation
    print("\n9. Evaluating model performance...")
    eval_results = model.evaluate(X_test, y_test, plot_curves=False)
    
    print("   Test set performance:")
    metrics_of_interest = ['roc_auc', 'average_precision', 'f1_score', 'precision', 'recall']
    for metric in metrics_of_interest:
        if metric in eval_results:
            print(f"     {metric}: {eval_results[metric]:.4f}")
    
    # 10. Threshold optimization
    print("\n10. Optimizing classification threshold...")
    y_test_proba = model.predict_proba(X_test)
    
    optimal_threshold, optimal_f1 = optimize_threshold(y_test, y_test_proba, metric='f1')
    print(f"    Optimal threshold (F1): {optimal_threshold:.3f} (F1: {optimal_f1:.4f})")
    
    # 11. Cost-sensitive evaluation
    print("\n11. Cost-sensitive evaluation...")
    y_test_pred_optimal = (y_test_proba >= optimal_threshold).astype(int)
    cost_metrics = calculate_cost_sensitive_metrics(
        y_test, y_test_pred_optimal,
        false_positive_cost=1.0,
        false_negative_cost=10.0  # False negatives are more costly
    )
    
    print(f"    Total cost: ${cost_metrics['total_cost']:.0f}")
    print(f"    False positives: {cost_metrics['false_positives']} (cost: ${cost_metrics['false_positive_cost']:.0f})")
    print(f"    False negatives: {cost_metrics['false_negatives']} (cost: ${cost_metrics['false_negative_cost']:.0f})")
    print(f"    Cost savings ratio: {cost_metrics['savings_ratio']:.3f}")
    
    # 12. Model persistence
    print("\n12. Demonstrating model persistence...")
    persistence = ModelPersistence(base_dir="models/artifacts")
    
    # Save model with metadata
    metadata = {
        'dataset_size': len(X),
        'features': list(X.columns),
        'defect_rate': float(y.mean()),
        'cv_auc': cv_results['roc_auc_mean'],
        'test_auc': eval_results['roc_auc'],
        'optimal_threshold': optimal_threshold
    }
    
    model_path = persistence.save_model(
        model.model,
        model_name="steel_defect_baseline",
        metadata=metadata,
        version="v1.0.0"
    )
    print(f"    Model saved to: {model_path}")
    
    # List available models
    models_info = persistence.list_models()
    print(f"    Available models: {len(models_info)}")
    
    # 13. Performance summary
    print("\n13. Performance Summary")
    print("    " + "="*40)
    
    # Check against targets
    targets = {
        'AUC-ROC': (eval_results.get('roc_auc', 0), 0.85),
        'AUC-PR': (eval_results.get('average_precision', 0), 0.70),
        'F1-Score': (eval_results.get('f1_score', 0), 0.75),
        'Precision': (eval_results.get('precision', 0), 0.80),
        'Recall': (eval_results.get('recall', 0), 0.70)
    }
    
    for metric_name, (actual, target) in targets.items():
        status = "✓" if actual >= target else "✗"
        print(f"    {metric_name:12}: {actual:.4f} (target: {target:.2f}) {status}")
    
    # Performance characteristics
    training_time = training_history['training_time']
    n_test_samples = len(X_test)
    
    print(f"\n    Training time: {training_time:.2f}s (target: < 300s) {'✓' if training_time < 300 else '✗'}")
    
    # Estimate inference time
    import time
    start_time = time.time()
    _ = model.predict_proba(X_test[:100])
    inference_time_per_sample = (time.time() - start_time) / 100 * 1000  # ms
    print(f"    Inference time: {inference_time_per_sample:.3f}ms/sample (target: < 100ms) {'✓' if inference_time_per_sample < 100 else '✗'}")
    
    print(f"\n    Model successfully trained and evaluated!")
    print(f"    Ready for production deployment with optimal threshold: {optimal_threshold:.3f}")
    
    return model, eval_results, optimal_threshold


if __name__ == "__main__":
    # Set up matplotlib for non-interactive use
    import matplotlib
    matplotlib.use('Agg')
    
    # Run the comprehensive example
    model, results, threshold = main()
    
    print(f"\nExample completed successfully!")
    print(f"Final model AUC-ROC: {results['roc_auc']:.4f}")
    print(f"Optimal threshold: {threshold:.3f}")