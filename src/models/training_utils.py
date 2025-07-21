import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Any, Optional
import logging
import json
from datetime import datetime
from pathlib import Path
import warnings


class TrainingUtils:
    """Training utility functions"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize training utilities
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def detect_overfitting(self,
                          train_scores: List[float],
                          val_scores: List[float],
                          patience: int = 5,
                          min_delta: float = 0.001) -> bool:
        """
        Detect overfitting in training curves
        
        Args:
            train_scores: Training scores over time
            val_scores: Validation scores over time
            patience: Number of epochs to check
            min_delta: Minimum change threshold
            
        Returns:
            True if overfitting detected
        """
        if len(train_scores) < patience or len(val_scores) < patience:
            return False
        
        # Check if validation score has stopped improving
        recent_val_scores = val_scores[-patience:]
        val_improvement = max(recent_val_scores) - min(recent_val_scores)
        
        # Check if training score is still improving significantly
        recent_train_scores = train_scores[-patience:]
        train_improvement = max(recent_train_scores) - min(recent_train_scores)
        
        # Overfitting if validation plateaus but training continues improving
        val_plateaued = val_improvement < min_delta
        train_improving = train_improvement > min_delta
        
        overfitting_detected = val_plateaued and train_improving
        
        if overfitting_detected:
            self.logger.warning(f"Overfitting detected: val improvement {val_improvement:.4f}, train improvement {train_improvement:.4f}")
        
        return overfitting_detected
    
    def apply_regularization(self,
                            model_params: Dict[str, Any],
                            regularization_strength: float = 0.1,
                            model_type: str = 'xgboost') -> Dict[str, Any]:
        """
        Apply regularization techniques to model parameters
        
        Args:
            model_params: Base model parameters
            regularization_strength: Strength of regularization
            model_type: Type of model
            
        Returns:
            Regularized parameters
        """
        regularized_params = model_params.copy()
        
        if model_type == 'xgboost':
            # Apply L1 and L2 regularization
            regularized_params['reg_alpha'] = regularization_strength
            regularized_params['reg_lambda'] = regularization_strength
            
            # Reduce learning rate for stronger regularization
            if 'learning_rate' in regularized_params:
                regularized_params['learning_rate'] *= (1 - regularization_strength * 0.5)
            
            # Increase minimum child weight
            if 'min_child_weight' in regularized_params:
                regularized_params['min_child_weight'] = max(1, 
                    regularized_params['min_child_weight'] * (1 + regularization_strength))
        
        elif model_type == 'random_forest':
            # Reduce max features and increase min samples
            if 'max_features' in regularized_params:
                if isinstance(regularized_params['max_features'], str):
                    regularized_params['max_features'] = 'sqrt'
                else:
                    regularized_params['max_features'] *= (1 - regularization_strength * 0.3)
            
            if 'min_samples_split' in regularized_params:
                regularized_params['min_samples_split'] = max(2, 
                    int(regularized_params['min_samples_split'] * (1 + regularization_strength)))
        
        self.logger.info(f"Applied regularization (strength={regularization_strength}) to {model_type} parameters")
        
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
        fig, axes = plt.subplots(1, n_metrics, figsize=(figsize[0], figsize[1]))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            train_key = f'train_{metric}'
            val_key = f'val_{metric}'
            
            if train_key in history:
                epochs = range(1, len(history[train_key]) + 1)
                axes[i].plot(epochs, history[train_key], label=f'Training {metric}', marker='o')
            
            if val_key in history:
                epochs = range(1, len(history[val_key]) + 1)
                axes[i].plot(epochs, history[val_key], label=f'Validation {metric}', marker='s')
            
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_title(f'{metric.capitalize()} Over Time')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self,
                               feature_importance: pd.DataFrame,
                               top_k: int = 20,
                               figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot feature importance
        
        Args:
            feature_importance: DataFrame with feature names and importance scores
            top_k: Number of top features to show
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Select top features
        top_features = feature_importance.head(top_k)
        
        plt.figure(figsize=figsize)
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_k} Feature Importance')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_confusion_matrix(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             normalize: bool = True,
                             figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            normalize: Whether to normalize
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', square=True,
                   xticklabels=class_names or ['0', '1'],
                   yticklabels=class_names or ['0', '1'])
        
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_roc_pr_curves(self,
                          y_true: np.ndarray,
                          y_proba: np.ndarray,
                          figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        """
        Plot ROC and Precision-Recall curves
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        ax1.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', linewidth=2)
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        ax2.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})', linewidth=2)
        ax2.axhline(y=y_true.mean(), color='k', linestyle='--', alpha=0.5, label='Baseline')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_learning_curves(self,
                            train_sizes: np.ndarray,
                            train_scores: np.ndarray,
                            val_scores: np.ndarray,
                            figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot learning curves
        
        Args:
            train_sizes: Training set sizes
            train_scores: Training scores for each size
            val_scores: Validation scores for each size
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=figsize)
        
        plt.plot(train_sizes, train_mean, 'o-', label='Training Score', linewidth=2)
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        
        plt.plot(train_sizes, val_mean, 'o-', label='Validation Score', linewidth=2)
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
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
        report_path = Path(output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create HTML report
        html_content = self._create_html_report(results, include_plots)
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        # Save plots if requested
        if include_plots and 'plots' in results:
            plots_dir = report_path.parent / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            for plot_name, fig in results['plots'].items():
                plot_path = plots_dir / f"{plot_name}.png"
                fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        self.logger.info(f"Training report generated: {output_path}")
    
    def _create_html_report(self, results: Dict[str, Any], include_plots: bool = True) -> str:
        """Create HTML training report"""
        html_parts = [
            "<html><head><title>Training Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "h1, h2, h3 { color: #333; }",
            "table { border-collapse: collapse; width: 100%; margin: 10px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            ".metric { background-color: #e8f5e8; }",
            ".section { margin: 20px 0; }",
            "</style></head><body>"
        ]
        
        # Title and timestamp
        html_parts.extend([
            "<h1>Model Training Report</h1>",
            f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        ])
        
        # Data information
        if 'data_info' in results:
            data_info = results['data_info']
            html_parts.extend([
                "<div class='section'>",
                "<h2>Data Information</h2>",
                "<table>",
                f"<tr><th>Total Samples</th><td>{data_info.get('total_samples', 'N/A')}</td></tr>",
                f"<tr><th>Features</th><td>{data_info.get('n_features', 'N/A')}</td></tr>",
                f"<tr><th>Training Samples</th><td>{data_info.get('train_samples', 'N/A')}</td></tr>",
                f"<tr><th>Validation Samples</th><td>{data_info.get('val_samples', 'N/A')}</td></tr>",
                f"<tr><th>Test Samples</th><td>{data_info.get('test_samples', 'N/A')}</td></tr>",
                "</table>",
                "</div>"
            ])
        
        # Training results
        if 'training' in results:
            training = results['training']
            html_parts.extend([
                "<div class='section'>",
                "<h2>Training Results</h2>",
                "<table>",
                f"<tr><th>Training Time (s)</th><td>{training.get('training_time', 'N/A')}</td></tr>",
                f"<tr class='metric'><th>Training AUC</th><td>{training.get('train_auc', 'N/A'):.4f}</td></tr>",
                f"<tr class='metric'><th>Validation AUC</th><td>{training.get('val_auc', 'N/A'):.4f}</td></tr>",
                "</table>",
                "</div>"
            ])
        
        # Test evaluation
        if 'test_evaluation' in results:
            test_eval = results['test_evaluation']
            html_parts.extend([
                "<div class='section'>",
                "<h2>Test Evaluation</h2>",
                "<table>"
            ])
            
            for metric, value in test_eval.items():
                if isinstance(value, (int, float)) and metric != 'plots':
                    html_parts.append(f"<tr class='metric'><th>{metric.replace('_', ' ').title()}</th><td>{value:.4f}</td></tr>")
            
            html_parts.extend(["</table>", "</div>"])
        
        # Cross-validation results
        if 'cross_validation' in results:
            cv_results = results['cross_validation']
            html_parts.extend([
                "<div class='section'>",
                "<h2>Cross-Validation Results</h2>",
                "<table>",
                f"<tr><th>CV Folds</th><td>{cv_results.get('cv_folds', 'N/A')}</td></tr>",
                f"<tr class='metric'><th>Mean ROC AUC</th><td>{cv_results.get('roc_auc_mean', 'N/A'):.4f} ± {cv_results.get('roc_auc_std', 'N/A'):.4f}</td></tr>",
                f"<tr class='metric'><th>Mean Average Precision</th><td>{cv_results.get('average_precision_mean', 'N/A'):.4f} ± {cv_results.get('average_precision_std', 'N/A'):.4f}</td></tr>",
                "</table>",
                "</div>"
            ])
        
        # Hyperparameter search results
        if 'hyperparameter_search' in results:
            hp_search = results['hyperparameter_search']
            html_parts.extend([
                "<div class='section'>",
                "<h2>Hyperparameter Search</h2>",
                "<table>",
                f"<tr><th>Search Method</th><td>{hp_search.get('search_method', 'N/A')}</td></tr>",
                f"<tr><th>Best Score</th><td>{hp_search.get('best_score', 'N/A'):.4f}</td></tr>",
                f"<tr><th>Search Time (s)</th><td>{hp_search.get('search_time', 'N/A')}</td></tr>",
                "</table>",
                "<h3>Best Parameters</h3>",
                "<table>"
            ])
            
            best_params = hp_search.get('best_params', {})
            for param, value in best_params.items():
                html_parts.append(f"<tr><th>{param}</th><td>{value}</td></tr>")
            
            html_parts.extend(["</table>", "</div>"])
        
        html_parts.append("</body></html>")
        
        return "\n".join(html_parts)
    
    def calculate_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced datasets
        
        Args:
            y: Target labels
            
        Returns:
            Dictionary with class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        
        class_weights = dict(zip(classes, weights))
        
        self.logger.info(f"Calculated class weights: {class_weights}")
        
        return class_weights
    
    def create_sample_weights(self, y: np.ndarray, method: str = 'balanced') -> np.ndarray:
        """
        Create sample weights for training
        
        Args:
            y: Target labels
            method: Weighting method ('balanced', 'custom')
            
        Returns:
            Sample weights array
        """
        if method == 'balanced':
            class_weights = self.calculate_class_weights(y)
            sample_weights = np.array([class_weights[label] for label in y])
        else:
            # Default to equal weights
            sample_weights = np.ones(len(y))
        
        return sample_weights
    
    def optimize_threshold(self,
                          y_true: np.ndarray,
                          y_proba: np.ndarray,
                          metric: str = 'f1') -> Tuple[float, float]:
        """
        Optimize classification threshold
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            metric: Metric to optimize ('f1', 'precision', 'recall')
            
        Returns:
            Optimal threshold and corresponding metric value
        """
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        thresholds = np.linspace(0.01, 0.99, 99)
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            scores.append(score)
        
        best_idx = np.argmax(scores)
        best_threshold = thresholds[best_idx]
        best_score = scores[best_idx]
        
        self.logger.info(f"Optimal threshold for {metric}: {best_threshold:.3f} (score: {best_score:.4f})")
        
        return best_threshold, best_score
    
    def create_ensemble_predictions(self,
                                   predictions_list: List[np.ndarray],
                                   method: str = 'average') -> np.ndarray:
        """
        Create ensemble predictions from multiple models
        
        Args:
            predictions_list: List of prediction arrays
            method: Ensemble method ('average', 'weighted', 'voting')
            
        Returns:
            Ensemble predictions
        """
        predictions_array = np.array(predictions_list)
        
        if method == 'average':
            ensemble_pred = np.mean(predictions_array, axis=0)
        elif method == 'voting':
            ensemble_pred = np.mean(predictions_array > 0.5, axis=0)
        else:
            # Default to average
            ensemble_pred = np.mean(predictions_array, axis=0)
        
        self.logger.info(f"Created ensemble predictions using {method} method")
        
        return ensemble_pred