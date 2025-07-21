"""Training utility functions for the training pipeline"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from datetime import datetime
from pathlib import Path
import json
import warnings
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from scipy import stats


class TrainingUtils:
    """Utility functions for training pipeline"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize training utilities
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        
    def detect_overfitting(self, 
                          train_scores: List[float], 
                          val_scores: List[float], 
                          patience: int = 5,
                          threshold: float = 0.05) -> bool:
        """
        Detect overfitting in training curves
        
        Args:
            train_scores: Training scores over epochs
            val_scores: Validation scores over epochs
            patience: Number of epochs to check
            threshold: Minimum gap to consider overfitting
            
        Returns:
            True if overfitting detected
        """
        if len(train_scores) < patience or len(val_scores) < patience:
            return False
        
        # Check if validation score is consistently lower than training score
        recent_train = np.mean(train_scores[-patience:])
        recent_val = np.mean(val_scores[-patience:])
        
        gap = recent_train - recent_val
        
        # Check if gap is increasing
        if len(train_scores) >= patience * 2:
            early_gap = np.mean(train_scores[-patience*2:-patience]) - np.mean(val_scores[-patience*2:-patience])
            gap_increasing = gap > early_gap + threshold
        else:
            gap_increasing = False
        
        overfitting = gap > threshold and gap_increasing
        
        if overfitting:
            self.logger.warning(f"Overfitting detected: train={recent_train:.3f}, val={recent_val:.3f}, gap={gap:.3f}")
        
        return overfitting
    
    def apply_regularization(self, 
                           model_params: Dict[str, Any], 
                           regularization_strength: float = 0.1) -> Dict[str, Any]:
        """
        Apply regularization techniques to model parameters
        
        Args:
            model_params: Base model parameters
            regularization_strength: Strength of regularization
            
        Returns:
            Regularized parameters
        """
        regularized_params = model_params.copy()
        
        # XGBoost regularization
        if 'reg_alpha' not in regularized_params:
            regularized_params['reg_alpha'] = regularization_strength
        if 'reg_lambda' not in regularized_params:
            regularized_params['reg_lambda'] = regularization_strength
        
        # Reduce learning rate for stronger regularization
        if regularization_strength > 0.1:
            current_lr = regularized_params.get('learning_rate', 0.1)
            regularized_params['learning_rate'] = current_lr * 0.8
        
        self.logger.info(f"Applied regularization with strength {regularization_strength}")
        return regularized_params
    
    def plot_training_history(self, 
                             history: Dict[str, List[float]], 
                             metrics: List[str] = ['loss', 'auc'],
                             figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
        """
        Plot training history curves
        
        Args:
            history: Training history dictionary
            metrics: Metrics to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(figsize[0] * n_metrics // 2, figsize[1]))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Plot training metric
            train_key = f'train_{metric}'
            val_key = f'val_{metric}'
            
            if train_key in history:
                ax.plot(history[train_key], label=f'Training {metric.upper()}', linewidth=2)
            
            if val_key in history:
                ax.plot(history[val_key], label=f'Validation {metric.upper()}', linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} History')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_learning_curve(self, 
                           estimator: Any, 
                           X: np.ndarray, 
                           y: np.ndarray,
                           cv: int = 5,
                           train_sizes: Optional[np.ndarray] = None,
                           scoring: str = 'roc_auc',
                           figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot learning curve to analyze model performance vs training size
        
        Args:
            estimator: Model estimator
            X: Training features
            y: Training labels
            cv: Cross-validation folds
            train_sizes: Training sizes to evaluate
            scoring: Scoring metric
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=cv, train_sizes=train_sizes, 
            scoring=scoring, n_jobs=-1, random_state=self.random_state
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
        ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                       alpha=0.2, color='blue')
        
        ax.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
        ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, 
                       alpha=0.2, color='red')
        
        ax.set_xlabel('Training Size')
        ax.set_ylabel(f'{scoring.upper()} Score')
        ax.set_title('Learning Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_validation_curve(self, 
                             estimator: Any,
                             X: np.ndarray, 
                             y: np.ndarray,
                             param_name: str,
                             param_range: List[Any],
                             cv: int = 5,
                             scoring: str = 'roc_auc',
                             figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot validation curve for hyperparameter analysis
        
        Args:
            estimator: Model estimator
            X: Training features
            y: Training labels
            param_name: Parameter name to vary
            param_range: Range of parameter values
            cv: Cross-validation folds
            scoring: Scoring metric
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        train_scores, val_scores = validation_curve(
            estimator, X, y, param_name=param_name, param_range=param_range,
            cv=cv, scoring=scoring, n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
        ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, 
                       alpha=0.2, color='blue')
        
        ax.plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
        ax.fill_between(param_range, val_mean - val_std, val_mean + val_std, 
                       alpha=0.2, color='red')
        
        ax.set_xlabel(param_name)
        ax.set_ylabel(f'{scoring.upper()} Score')
        ax.set_title(f'Validation Curve for {param_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def analyze_feature_stability(self, 
                                 feature_importance_history: List[Dict[str, float]],
                                 top_n: int = 20) -> Dict[str, Any]:
        """
        Analyze stability of feature importance across training iterations
        
        Args:
            feature_importance_history: List of feature importance dictionaries
            top_n: Number of top features to analyze
            
        Returns:
            Feature stability analysis results
        """
        if not feature_importance_history:
            return {}
        
        # Get all features that appear in any iteration
        all_features = set()
        for importance_dict in feature_importance_history:
            all_features.update(importance_dict.keys())
        
        # Create importance matrix
        importance_matrix = []
        for importance_dict in feature_importance_history:
            row = [importance_dict.get(feature, 0.0) for feature in all_features]
            importance_matrix.append(row)
        
        importance_matrix = np.array(importance_matrix)
        feature_names = list(all_features)
        
        # Calculate stability metrics
        mean_importance = np.mean(importance_matrix, axis=0)
        std_importance = np.std(importance_matrix, axis=0)
        cv_importance = std_importance / (mean_importance + 1e-8)  # Coefficient of variation
        
        # Create results dataframe
        stability_df = pd.DataFrame({
            'feature': feature_names,
            'mean_importance': mean_importance,
            'std_importance': std_importance,
            'cv_importance': cv_importance
        })
        
        # Sort by mean importance
        stability_df = stability_df.sort_values('mean_importance', ascending=False)
        
        # Get top features
        top_features = stability_df.head(top_n)
        
        # Calculate rank correlation between iterations
        rank_correlations = []
        for i in range(len(feature_importance_history)):
            for j in range(i + 1, len(feature_importance_history)):
                imp1 = [feature_importance_history[i].get(f, 0) for f in feature_names]
                imp2 = [feature_importance_history[j].get(f, 0) for f in feature_names]
                corr, _ = stats.spearmanr(imp1, imp2)
                rank_correlations.append(corr)
        
        return {
            'stability_df': stability_df,
            'top_features': top_features,
            'mean_rank_correlation': np.mean(rank_correlations) if rank_correlations else 0.0,
            'feature_count': len(all_features),
            'iterations': len(feature_importance_history)
        }
    
    def calculate_model_complexity(self, model: Any) -> Dict[str, Any]:
        """
        Calculate model complexity metrics
        
        Args:
            model: Trained model
            
        Returns:
            Model complexity metrics
        """
        complexity_metrics = {}
        
        # XGBoost specific metrics
        if hasattr(model, 'get_booster'):
            booster = model.get_booster()
            
            # Number of trees
            complexity_metrics['n_trees'] = model.n_estimators
            
            # Tree depth
            if hasattr(model, 'max_depth'):
                complexity_metrics['max_depth'] = model.max_depth
            
            # Number of features used
            feature_names = booster.feature_names
            if feature_names:
                complexity_metrics['n_features'] = len(feature_names)
            
            # Total number of nodes (approximate)
            tree_dump = booster.get_dump()
            total_nodes = sum(len(tree.split('\n')) for tree in tree_dump)
            complexity_metrics['total_nodes'] = total_nodes
            
        # General model parameters
        if hasattr(model, 'get_params'):
            params = model.get_params()
            complexity_metrics['parameters'] = len(params)
        
        return complexity_metrics
    
    def generate_training_report(self, 
                               results: Dict[str, Any], 
                               output_path: str,
                               include_plots: bool = True) -> None:
        """
        Generate comprehensive training report
        
        Args:
            results: Training results dictionary
            output_path: Path to save report
            include_plots: Whether to include plots in report
        """
        report_lines = []
        
        # Header
        report_lines.append("# Steel Casting Defect Prediction - Training Report")
        report_lines.append(f"Generated on: {datetime.now().isoformat()}")
        report_lines.append("")
        
        # Experiment Information
        if 'experiment_name' in results:
            report_lines.append(f"**Experiment**: {results['experiment_name']}")
        if 'training_time' in results:
            report_lines.append(f"**Training Time**: {results['training_time']:.2f} seconds")
        report_lines.append("")
        
        # Data Information
        if 'data_info' in results:
            data_info = results['data_info']
            report_lines.append("## Data Information")
            report_lines.append(f"- Total Samples: {data_info.get('total_samples', 'N/A')}")
            report_lines.append(f"- Features: {data_info.get('n_features', 'N/A')}")
            report_lines.append(f"- Training Samples: {data_info.get('train_samples', 'N/A')}")
            report_lines.append(f"- Validation Samples: {data_info.get('val_samples', 'N/A')}")
            report_lines.append(f"- Test Samples: {data_info.get('test_samples', 'N/A')}")
            report_lines.append("")
        
        # Model Performance
        if 'test_evaluation' in results and 'metrics' in results['test_evaluation']:
            metrics = results['test_evaluation']['metrics']
            report_lines.append("## Model Performance")
            report_lines.append(f"- **ROC AUC**: {metrics.get('roc_auc', 'N/A'):.4f}")
            report_lines.append(f"- **Average Precision**: {metrics.get('average_precision', 'N/A'):.4f}")
            report_lines.append(f"- **F1 Score**: {metrics.get('f1_score', 'N/A'):.4f}")
            report_lines.append(f"- **Precision**: {metrics.get('precision', 'N/A'):.4f}")
            report_lines.append(f"- **Recall**: {metrics.get('recall', 'N/A'):.4f}")
            report_lines.append(f"- **Accuracy**: {metrics.get('accuracy', 'N/A'):.4f}")
            report_lines.append("")
        
        # Cross-validation Results
        if 'cross_validation' in results:
            cv_results = results['cross_validation']
            report_lines.append("## Cross-Validation Results")
            if 'roc_auc_mean' in cv_results:
                report_lines.append(f"- **ROC AUC**: {cv_results['roc_auc_mean']:.4f} ± {cv_results.get('roc_auc_std', 0):.4f}")
            if 'average_precision_mean' in cv_results:
                report_lines.append(f"- **Average Precision**: {cv_results['average_precision_mean']:.4f} ± {cv_results.get('average_precision_std', 0):.4f}")
            report_lines.append(f"- **CV Folds**: {cv_results.get('cv_folds', 'N/A')}")
            report_lines.append("")
        
        # Hyperparameter Search
        if 'hyperparameter_search' in results:
            hp_results = results['hyperparameter_search']
            report_lines.append("## Hyperparameter Optimization")
            report_lines.append(f"- **Best Score**: {hp_results.get('best_score', 'N/A'):.4f}")
            report_lines.append(f"- **Search Method**: {hp_results.get('search_method', 'N/A')}")
            report_lines.append(f"- **Search Time**: {hp_results.get('search_time', 'N/A'):.2f} seconds")
            
            if 'best_params' in hp_results:
                report_lines.append("- **Best Parameters**:")
                for param, value in hp_results['best_params'].items():
                    report_lines.append(f"  - {param}: {value}")
            report_lines.append("")
        
        # Feature Importance
        if 'feature_importance' in results:
            feature_imp = results['feature_importance']
            if isinstance(feature_imp, pd.DataFrame):
                top_features = feature_imp.head(10)
                report_lines.append("## Top 10 Important Features")
                for _, row in top_features.iterrows():
                    report_lines.append(f"- {row['feature']}: {row['importance']:.4f}")
                report_lines.append("")
        
        # Artifacts
        if 'artifacts' in results:
            artifacts = results['artifacts']
            report_lines.append("## Saved Artifacts")
            for artifact_type, path in artifacts.items():
                report_lines.append(f"- {artifact_type}: {path}")
            report_lines.append("")
        
        # Write report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Training report saved to: {output_path}")
    
    def compare_models(self, 
                      model_results: Dict[str, Dict[str, Any]],
                      metrics: List[str] = ['roc_auc', 'average_precision', 'f1_score']) -> pd.DataFrame:
        """
        Compare multiple model results
        
        Args:
            model_results: Dictionary of model names to results
            metrics: Metrics to compare
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for model_name, results in model_results.items():
            row = {'model': model_name}
            
            # Extract test metrics
            if 'test_evaluation' in results and 'metrics' in results['test_evaluation']:
                test_metrics = results['test_evaluation']['metrics']
                for metric in metrics:
                    row[f'test_{metric}'] = test_metrics.get(metric, np.nan)
            
            # Extract CV metrics
            if 'cross_validation' in results:
                cv_results = results['cross_validation']
                for metric in metrics:
                    mean_key = f'{metric}_mean'
                    std_key = f'{metric}_std'
                    if mean_key in cv_results:
                        row[f'cv_{metric}_mean'] = cv_results[mean_key]
                        row[f'cv_{metric}_std'] = cv_results.get(std_key, np.nan)
            
            # Extract training time
            if 'training_time' in results:
                row['training_time'] = results['training_time']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df
    
    def plot_model_comparison(self, 
                             comparison_df: pd.DataFrame,
                             metric: str = 'test_roc_auc',
                             figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot model comparison chart
        
        Args:
            comparison_df: Model comparison DataFrame
            metric: Metric to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        models = comparison_df['model'].tolist()
        scores = comparison_df[metric].tolist()
        
        bars = ax.bar(models, scores, color='skyblue', alpha=0.7)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            if not np.isnan(score):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Model Comparison - {metric.replace("_", " ").title()}')
        ax.set_ylim(0, 1.1)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig
    
    def estimate_memory_usage(self, 
                             data_shape: Tuple[int, int],
                             model_type: str = 'xgboost') -> Dict[str, float]:
        """
        Estimate memory usage for training
        
        Args:
            data_shape: Shape of training data (n_samples, n_features)
            model_type: Type of model
            
        Returns:
            Memory usage estimates in MB
        """
        n_samples, n_features = data_shape
        
        # Data memory (assuming float64)
        data_memory = (n_samples * n_features * 8) / (1024 * 1024)  # MB
        
        # Feature engineering memory (estimated 2x data size)
        feature_memory = data_memory * 2
        
        # Model memory estimates
        if model_type == 'xgboost':
            # Rough estimate based on trees and features
            model_memory = (n_features * 100 * 8) / (1024 * 1024)  # MB
        else:
            model_memory = data_memory * 0.1  # Conservative estimate
        
        # Cross-validation memory (k-fold overhead)
        cv_memory = data_memory * 0.2
        
        total_memory = data_memory + feature_memory + model_memory + cv_memory
        
        return {
            'data_memory_mb': data_memory,
            'feature_memory_mb': feature_memory,
            'model_memory_mb': model_memory,
            'cv_memory_mb': cv_memory,
            'total_estimated_mb': total_memory,
            'total_estimated_gb': total_memory / 1024
        }
    
    def validate_data_quality(self, 
                             X: pd.DataFrame, 
                             y: pd.Series) -> Dict[str, Any]:
        """
        Validate data quality for training
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Data quality report
        """
        quality_report = {}
        
        # Basic statistics
        quality_report['n_samples'] = len(X)
        quality_report['n_features'] = len(X.columns)
        quality_report['target_balance'] = y.value_counts().to_dict()
        
        # Missing values
        missing_counts = X.isnull().sum()
        quality_report['missing_features'] = missing_counts[missing_counts > 0].to_dict()
        quality_report['missing_percentage'] = (missing_counts / len(X) * 100).to_dict()
        
        # Constant features
        constant_features = []
        for col in X.columns:
            if X[col].nunique() <= 1:
                constant_features.append(col)
        quality_report['constant_features'] = constant_features
        
        # High correlation features
        numeric_features = X.select_dtypes(include=[np.number]).columns
        if len(numeric_features) > 1:
            corr_matrix = X[numeric_features].corr().abs()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.9:
                        high_corr_pairs.append((
                            corr_matrix.columns[i], 
                            corr_matrix.columns[j], 
                            corr_matrix.iloc[i, j]
                        ))
            
            quality_report['high_correlation_pairs'] = high_corr_pairs
        
        # Outlier detection
        outlier_counts = {}
        for col in numeric_features:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
            outlier_counts[col] = outliers
        
        quality_report['outlier_counts'] = outlier_counts
        
        # Data type issues
        mixed_types = []
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    pd.to_numeric(X[col])
                    mixed_types.append(col)
                except (ValueError, TypeError):
                    pass
        quality_report['potential_numeric_columns'] = mixed_types
        
        return quality_report