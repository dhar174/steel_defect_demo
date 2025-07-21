#!/usr/bin/env python3
"""
Example usage of the automated ModelTrainer pipeline for steel defect prediction.

This script demonstrates how to:
1. Initialize a ModelTrainer with configuration
2. Run a complete training pipeline with hyperparameter optimization
3. Evaluate the trained model and save artifacts

Usage:
    python examples/model_trainer_example.py
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import ModelTrainer, ConfigurationManager


def create_sample_data(n_samples: int = 1000, n_features: int = 15) -> pd.DataFrame:
    """Create synthetic steel casting defect data for demonstration"""
    np.random.seed(42)
    
    # Create features representing various sensor measurements
    feature_names = [
        'temperature_1', 'temperature_2', 'pressure_1', 'pressure_2',
        'flow_rate', 'casting_speed', 'steel_composition_c', 'steel_composition_si',
        'steel_composition_mn', 'steel_composition_p', 'steel_composition_s',
        'humidity', 'ambient_temp', 'vibration', 'acoustic_emission'
    ]
    
    # Generate features with some correlation structure
    X = np.random.randn(n_samples, n_features)
    
    # Add some realistic correlations
    X[:, 1] = X[:, 0] + np.random.randn(n_samples) * 0.3  # Temperature correlation
    X[:, 3] = X[:, 2] + np.random.randn(n_samples) * 0.2  # Pressure correlation
    
    # Create defect labels based on complex interactions
    defect_score = (
        0.3 * X[:, 0] +  # temperature_1
        0.2 * X[:, 2] +  # pressure_1
        0.15 * X[:, 4] + # flow_rate
        0.1 * X[:, 5] +  # casting_speed
        0.25 * (X[:, 6] + X[:, 7]) +  # steel composition
        0.1 * np.random.randn(n_samples)  # noise
    )
    
    # Convert to binary labels (defect/no defect)
    defect_threshold = np.percentile(defect_score, 80)  # 20% defect rate
    y = (defect_score > defect_threshold).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names[:n_features])
    df['defect'] = y
    
    return df


def main():
    """Run the model training example"""
    print("=== Steel Defect Prediction - Automated Training Pipeline Example ===\n")
    
    # Create example data
    print("1. Creating synthetic steel casting defect data...")
    data = create_sample_data(n_samples=2000, n_features=12)
    print(f"   Generated {len(data)} samples with {data.shape[1]-1} features")
    print(f"   Defect rate: {data['defect'].mean():.1%}")
    
    # Save data for pipeline
    data_dir = Path("data/examples")
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "steel_defect_sample.csv"
    data.to_csv(data_path, index=False)
    print(f"   Data saved to: {data_path}\n")
    
    # Create configuration
    print("2. Setting up training configuration...")
    config_manager = ConfigurationManager()
    config_path = "configs/training_pipeline.yaml"
    
    # Check if config exists, if not create default
    if not Path(config_path).exists():
        config = config_manager.create_default_config(config_path)
        print(f"   Created default configuration: {config_path}")
    else:
        config = config_manager.load_config(config_path)
        print(f"   Loaded configuration: {config_path}")
    
    # Customize config for this example
    config.experiment.name = "steel_defect_example"
    config.experiment.description = "Example training pipeline for steel casting defect prediction"
    config.data.target_column = "defect"
    
    # Enable hyperparameter search with small grid for speed
    config.hyperparameter_search.enabled = True
    config.hyperparameter_search.method = "grid"
    config.hyperparameter_search.param_grids = {
        'coarse': {
            'n_estimators': [50, 100],
            'max_depth': [4, 6],
            'learning_rate': [0.1, 0.15]
        }
    }
    
    # Enable cross-validation
    config.training.cross_validation.enabled = True
    config.training.cross_validation.cv_folds = 3  # Reduced for speed
    
    print(f"   Experiment: {config.experiment.name}")
    print(f"   Model type: {config.model.type}")
    print(f"   Hyperparameter search: {config.hyperparameter_search.enabled}")
    print(f"   Cross-validation: {config.training.cross_validation.enabled}\n")
    
    # Initialize trainer
    print("3. Initializing ModelTrainer...")
    trainer = ModelTrainer(
        config_path=config_path,
        model_type='xgboost',
        experiment_name=config.experiment.name,
        verbose=True
    )
    print("   ModelTrainer initialized successfully\n")
    
    # Run complete training pipeline
    print("4. Running automated training pipeline...")
    print("   This includes:")
    print("   - Data loading and preprocessing")
    print("   - Train/validation/test splits")
    print("   - Hyperparameter optimization")
    print("   - Model training with best parameters")
    print("   - Cross-validation")
    print("   - Model evaluation")
    print("   - Artifact saving\n")
    
    results = trainer.train_pipeline(
        data_path=str(data_path),
        target_column='defect'
    )
    
    # Display results
    print("\n5. Training Results:")
    print("=" * 50)
    
    # Data info
    data_info = results['data_info']
    print(f"Dataset: {data_info['total_samples']} samples, {data_info['n_features']} features")
    print(f"Train/Val/Test split: {data_info['train_samples']}/{data_info['val_samples']}/{data_info['test_samples']}")
    
    # Hyperparameter search results
    if 'hyperparameter_search' in results:
        hp_results = results['hyperparameter_search']
        print(f"\nHyperparameter Search:")
        print(f"  Method: {hp_results.get('search_method', 'N/A')}")
        print(f"  Best CV Score: {hp_results.get('best_score', 0):.4f}")
        print(f"  Best Parameters: {hp_results.get('best_params', {})}")
    
    # Training results
    training_results = results['training']
    print(f"\nTraining Performance:")
    print(f"  Training AUC: {training_results.get('train_auc', 0):.4f}")
    print(f"  Validation AUC: {training_results.get('val_auc', 0):.4f}")
    print(f"  Training Time: {training_results.get('training_time', 0):.2f}s")
    
    # Cross-validation results
    if 'cross_validation' in results:
        cv_results = results['cross_validation']
        print(f"\nCross-Validation:")
        print(f"  CV Folds: {cv_results.get('cv_folds', 0)}")
        print(f"  Mean ROC AUC: {cv_results.get('roc_auc_mean', 0):.4f} ± {cv_results.get('roc_auc_std', 0):.4f}")
        if 'average_precision_mean' in cv_results:
            print(f"  Mean Average Precision: {cv_results.get('average_precision_mean', 0):.4f} ± {cv_results.get('average_precision_std', 0):.4f}")
    
    # Test evaluation
    test_results = results['test_evaluation']
    print(f"\nTest Set Performance:")
    print(f"  ROC AUC: {test_results.get('roc_auc', 0):.4f}")
    print(f"  Average Precision: {test_results.get('average_precision', 0):.4f}")
    print(f"  F1 Score: {test_results.get('f1_score', 0):.4f}")
    print(f"  Precision: {test_results.get('precision', 0):.4f}")
    print(f"  Recall: {test_results.get('recall', 0):.4f}")
    
    # Feature importance
    print(f"\n6. Feature Analysis:")
    print("=" * 50)
    
    if trainer.model and trainer.model.is_trained:
        feature_importance = trainer.model.get_feature_importance(max_features=5)
        print("Top 5 Most Important Features:")
        for i, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Artifacts
    print(f"\n7. Saved Artifacts:")
    print("=" * 50)
    print(f"Experiment directory: {trainer.experiment_dir}")
    print("Artifacts saved:")
    print(f"  - Trained model: {trainer.model_dir}")
    print(f"  - Training results: {trainer.results_dir}")
    print(f"  - Plots and visualizations: {trainer.plots_dir}")
    print(f"  - Logs: logs/")
    
    print(f"\n8. Usage:")
    print("=" * 50)
    print("You can now use the trained model for predictions:")
    print(f"```python")
    print(f"from src.models import ModelTrainer")
    print(f"")
    print(f"# Load trained model")
    print(f"trainer = ModelTrainer.load('{trainer.experiment_dir}')")
    print(f"")
    print(f"# Make predictions on new data")
    print(f"predictions = trainer.predict(new_data)")
    print(f"```")
    
    print(f"\n=== Training Pipeline Complete ===")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)