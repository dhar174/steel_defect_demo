import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    balanced_accuracy_score, matthews_corrcoef,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
    log_loss, brier_score_loss
)
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
import warnings
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

class ModelEvaluator:
    """Comprehensive model evaluation and comparison framework"""
    
    def __init__(self, 
                 model: Any = None,
                 model_name: str = "model",
                 threshold: float = 0.5,
                 output_dir: str = "results/evaluation",
                 save_plots: bool = True,
                 plot_style: str = "seaborn",
                 random_state: int = 42):
        """
        Initialize the model evaluation framework
        
        Args:
            model: Trained model object
            model_name: Name identifier for the model
            threshold: Classification threshold
            output_dir: Directory to save evaluation results
            save_plots: Whether to automatically save plots
            plot_style: Plotting style ('seaborn', 'plotly', 'matplotlib')
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.model_name = model_name
        self.threshold = threshold
        self.output_dir = Path(output_dir)
        self.save_plots = save_plots
        self.plot_style = plot_style
        self.random_state = random_state
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Setup plotting style
        self._setup_plotting_style()
        
        # Store evaluation results
        self.evaluation_results = {}
        
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _setup_plotting_style(self) -> None:
        """Setup plotting style"""
        if self.plot_style == "seaborn":
            sns.set_style("whitegrid")
            plt.rcParams['figure.facecolor'] = 'white'
        elif self.plot_style == "ggplot":
            plt.style.use('ggplot')
    
    def evaluate_model(self,
                      X: Union[pd.DataFrame, np.ndarray],
                      y: Union[pd.Series, np.ndarray],
                      y_pred: Optional[np.ndarray] = None,
                      y_proba: Optional[np.ndarray] = None,
                      sample_weight: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation
        
        Args:
            X: Input features
            y: True labels
            y_pred: Predicted labels (optional, will compute if not provided)
            y_proba: Prediction probabilities (optional, will compute if not provided)
            sample_weight: Sample weights for evaluation
            
        Returns:
            Dictionary containing all evaluation metrics and results
        """
        start_time = datetime.now()
        self.logger.info(f"Starting comprehensive evaluation for {self.model_name}")
        
        # Convert inputs to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Generate predictions if not provided
        if y_proba is None and self.model is not None:
            if hasattr(self.model, 'predict_proba'):
                y_proba_raw = self.model.predict_proba(X)
                # Handle different predict_proba output formats
                if y_proba_raw.ndim == 2:
                    if y_proba_raw.shape[1] == 2:  # Validate binary classification
                        y_proba = y_proba_raw[:, 1]  # Standard sklearn format
                    else:
                        raise ValueError(
                            f"Invalid predict_proba output: expected 2 classes for binary classification, "
                            f"but got {y_proba_raw.shape[1]} classes. Ensure the model is binary."
                        )
                else:
                    y_proba = y_proba_raw  # BaselineXGBoostModel format (already positive class probabilities)
            else:
                self.logger.warning("Model does not support probability predictions")
                
        if y_pred is None:
            if y_proba is not None:
                y_pred = (y_proba >= self.threshold).astype(int)
            elif self.model is not None:
                y_pred = self.model.predict(X)
            else:
                raise ValueError("Must provide either y_pred, y_proba, or a trained model")
                
        # Calculate all metrics
        metrics = self.calculate_all_metrics(y, y_pred, y_proba, sample_weight)
        
        # Generate confusion matrix
        cm, cm_analysis = self.generate_confusion_matrix(y, y_pred)
        
        # Generate classification report
        class_report = self.generate_classification_report(y, y_pred)
        
        # Threshold analysis if probabilities available
        threshold_analysis = None
        if y_proba is not None:
            threshold_analysis = self.evaluate_threshold_sensitivity(y, y_proba)
            
        # Calibration analysis if probabilities available
        calibration_metrics = None
        if y_proba is not None:
            calibration_metrics = self.calculate_calibration_metrics(y, y_proba)
            
        evaluation_time = (datetime.now() - start_time).total_seconds()
        
        # Compile results
        results = {
            'model_name': self.model_name,
            'evaluation_timestamp': datetime.now().isoformat(),
            'evaluation_time': evaluation_time,
            'threshold': self.threshold,
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_analysis': cm_analysis,
            'classification_report': class_report,
            'threshold_analysis': threshold_analysis,
            'calibration_metrics': calibration_metrics,
            'sample_size': len(y),
            'positive_class_ratio': y.mean()
        }
        
        self.evaluation_results = results
        self.logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
        
        return results
    
    def calculate_all_metrics(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             y_proba: Optional[np.ndarray] = None,
                             sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive set of classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            sample_weight: Sample weights
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Basic classification metrics
        basic_metrics = self.calculate_basic_metrics(y_true, y_pred, sample_weight)
        metrics.update(basic_metrics)
        
        # Probabilistic metrics
        if y_proba is not None:
            prob_metrics = self.calculate_probabilistic_metrics(y_true, y_proba, sample_weight)
            metrics.update(prob_metrics)
            
        # Class-specific metrics
        class_metrics = self.calculate_class_specific_metrics(y_true, y_pred)
        metrics.update(class_metrics)
        
        return metrics
    
    def calculate_basic_metrics(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate basic classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sample_weight: Sample weights
            
        Returns:
            Basic metrics (accuracy, precision, recall, f1, etc.)
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred, sample_weight=sample_weight),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight),
            'precision': precision_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0),
            'recall': recall_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
            'specificity': self._calculate_specificity(y_true, y_pred)
        }
        
    def calculate_probabilistic_metrics(self,
                                       y_true: np.ndarray,
                                       y_proba: np.ndarray,
                                       sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate probabilistic evaluation metrics
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            sample_weight: Sample weights
            
        Returns:
            Probabilistic metrics (AUC-ROC, AUC-PR, log loss, etc.)
        """
        metrics = {}
        
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba, sample_weight=sample_weight)
        except ValueError as e:
            self.logger.warning(f"Could not calculate ROC AUC: {e}")
            metrics['roc_auc'] = np.nan
            
        try:
            metrics['average_precision'] = average_precision_score(y_true, y_proba, sample_weight=sample_weight)
        except ValueError as e:
            self.logger.warning(f"Could not calculate Average Precision: {e}")
            metrics['average_precision'] = np.nan
            
        try:
            metrics['log_loss'] = log_loss(y_true, y_proba, sample_weight=sample_weight)
        except ValueError as e:
            self.logger.warning(f"Could not calculate Log Loss: {e}")
            metrics['log_loss'] = np.nan
            
        try:
            metrics['brier_score'] = brier_score_loss(y_true, y_proba, sample_weight=sample_weight)
        except ValueError as e:
            self.logger.warning(f"Could not calculate Brier Score: {e}")
            metrics['brier_score'] = np.nan
            
        return metrics
    
    def calculate_class_specific_metrics(self,
                                        y_true: np.ndarray,
                                        y_pred: np.ndarray,
                                        labels: Optional[List] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-class performance metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            
        Returns:
            Per-class metrics dictionary
        """
        if labels is None:
            labels = ['class_0', 'class_1']
            
        # Get classification report as dict
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        class_metrics = {}
        for i, label in enumerate(labels):
            if str(i) in report_dict:
                class_metrics[label] = report_dict[str(i)]
                
        return class_metrics
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (True Negative Rate)"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def generate_confusion_matrix(self,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 labels: Optional[List[str]] = None,
                                 normalize: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate confusion matrix with detailed analysis
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels for display
            normalize: Normalization method ('true', 'pred', 'all', None)
            
        Returns:
            Confusion matrix array and analysis dictionary
        """
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)
        
        # Calculate detailed analysis
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        analysis = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'total_samples': int(len(y_true)),
            'positive_samples': int(y_true.sum()),
            'negative_samples': int(len(y_true) - y_true.sum()),
            'predicted_positive': int(y_pred.sum()),
            'predicted_negative': int(len(y_pred) - y_pred.sum())
        }
        
        return cm, analysis
    
    def plot_confusion_matrix(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             labels: Optional[List[str]] = None,
                             normalize: Optional[str] = None,
                             figsize: Tuple[int, int] = (8, 6),
                             cmap: str = 'Blues') -> plt.Figure:
        """
        Create publication-quality confusion matrix plot
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels for display
            normalize: Normalization method
            figsize: Figure size
            cmap: Color map
            
        Returns:
            Matplotlib figure
        """
        cm, _ = self.generate_confusion_matrix(y_true, y_pred, labels, normalize)
        
        if labels is None:
            labels = ['Normal', 'Defect']
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='.3f' if normalize else 'd',
                   cmap=cmap, square=True, ax=ax,
                   xticklabels=labels, yticklabels=labels)
        
        # Customize plot
        title = 'Confusion Matrix'
        if normalize:
            title += f' (Normalized by {normalize})'
        ax.set_title(f'{title} - {self.model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        
        if self.save_plots:
            filepath = self.output_dir / f'{self.model_name}_confusion_matrix.png'
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            self.logger.info(f"Confusion matrix saved to {filepath}")
            
        return fig
    
    def generate_classification_report(self,
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      labels: Optional[List[str]] = None,
                                      output_dict: bool = True) -> Union[str, Dict]:
        """
        Generate detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            output_dict: Return as dictionary vs string
            
        Returns:
            Classification report
        """
        if labels is None:
            target_names = ['Normal', 'Defect']
        else:
            target_names = labels
            
        return classification_report(
            y_true, y_pred, 
            target_names=target_names,
            output_dict=output_dict,
            zero_division=0
        )
    
    def plot_roc_curve(self,
                      y_true: np.ndarray,
                      y_proba: np.ndarray,
                      figsize: Tuple[int, int] = (8, 6),
                      show_thresholds: bool = False,
                      optimal_threshold: bool = True) -> plt.Figure:
        """
        Plot ROC curve with AUC score and optimal threshold
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            figsize: Figure size
            show_thresholds: Show threshold values on curve
            optimal_threshold: Mark optimal threshold point
            
        Returns:
            Matplotlib figure
        """
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC Curve (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
               label='Random Classifier')
        
        # Mark optimal threshold if requested
        if optimal_threshold:
            optimal_thresh, _ = self.find_optimal_threshold(y_true, y_proba, 'youden')
            optimal_idx = np.argmin(np.abs(thresholds - optimal_thresh))
            ax.plot(fpr[optimal_idx], tpr[optimal_idx], marker='o', markersize=8, 
                   color='red', label=f'Optimal Threshold ({optimal_thresh:.3f})')
        
        # Show thresholds on curve
        if show_thresholds:
            # Show every 10th threshold to avoid clutter
            step = max(1, len(thresholds) // 10)
            for i in range(0, len(thresholds), step):
                ax.annotate(f'{thresholds[i]:.2f}', 
                          (fpr[i], tpr[i]), 
                          xytext=(5, 5), 
                          textcoords='offset points',
                          fontsize=8)
        
        # Customize plot
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - {self.model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            filepath = self.output_dir / f'{self.model_name}_roc_curve.png'
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            self.logger.info(f"ROC curve saved to {filepath}")
            
        return fig
    
    def plot_precision_recall_curve(self,
                                   y_true: np.ndarray,
                                   y_proba: np.ndarray,
                                   figsize: Tuple[int, int] = (8, 6),
                                   show_thresholds: bool = False,
                                   baseline: bool = True) -> plt.Figure:
        """
        Plot Precision-Recall curve with AUC-PR score
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            figsize: Figure size
            show_thresholds: Show threshold values on curve
            baseline: Show random classifier baseline
            
        Returns:
            Matplotlib figure
        """
        # Calculate PR curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot PR curve
        ax.plot(recall, precision, color='blue', lw=2,
               label=f'PR Curve (AP = {avg_precision:.3f})')
        
        # Plot baseline (random classifier)
        if baseline:
            baseline_precision = y_true.mean()
            ax.axhline(y=baseline_precision, color='navy', linestyle='--', lw=2,
                      label=f'Random Classifier (AP = {baseline_precision:.3f})')
        
        # Show thresholds on curve
        if show_thresholds and len(thresholds) > 0:
            # Show every 10th threshold to avoid clutter
            step = max(1, len(thresholds) // 10)
            for i in range(0, len(thresholds), step):
                if i < len(precision) - 1:  # Avoid index out of bounds
                    ax.annotate(f'{thresholds[i]:.2f}', 
                              (recall[i], precision[i]), 
                              xytext=(5, 5), 
                              textcoords='offset points',
                              fontsize=8)
        
        # Customize plot
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curve - {self.model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            filepath = self.output_dir / f'{self.model_name}_pr_curve.png'
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            self.logger.info(f"PR curve saved to {filepath}")
            
        return fig
    
    def plot_combined_curves(self,
                            y_true: np.ndarray,
                            y_proba: np.ndarray,
                            figsize: Tuple[int, int] = (15, 6)) -> plt.Figure:
        """
        Plot ROC and PR curves side by side
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            figsize: Figure size
            
        Returns:
            Matplotlib figure with subplots
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        ax1.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # PR Curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        baseline_precision = y_true.mean()
        
        ax2.plot(recall, precision, color='blue', lw=2,
                label=f'PR Curve (AP = {avg_precision:.3f})')
        ax2.axhline(y=baseline_precision, color='navy', linestyle='--', lw=2,
                   label=f'Random Classifier (AP = {baseline_precision:.3f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Performance Curves - {self.model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if self.save_plots:
            filepath = self.output_dir / f'{self.model_name}_combined_curves.png'
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            self.logger.info(f"Combined curves saved to {filepath}")
            
        return fig
    
    def find_optimal_threshold(self,
                              y_true: np.ndarray,
                              y_proba: np.ndarray,
                              metric: str = 'f1') -> Tuple[float, float]:
        """
        Find optimal classification threshold based on specified metric
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            metric: Optimization metric ('f1', 'youden', 'precision', 'recall')
            
        Returns:
            Optimal threshold and corresponding metric value
        """
        thresholds = np.linspace(0, 1, 100)
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'youden':
                # Youden's J statistic (sensitivity + specificity - 1)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = sensitivity + specificity - 1
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
                
            scores.append(score)
        
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = scores[optimal_idx]
        
        return optimal_threshold, optimal_score
    
    def evaluate_threshold_sensitivity(self,
                                      y_true: np.ndarray,
                                      y_proba: np.ndarray,
                                      thresholds: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate model performance across different classification thresholds
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            thresholds: Array of thresholds to evaluate
            
        Returns:
            Threshold analysis results
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 50)
            
        results = {
            'thresholds': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'accuracy': [],
            'specificity': []
        }
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            # Calculate metrics
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)
            specificity = self._calculate_specificity(y_true, y_pred)
            
            results['thresholds'].append(threshold)
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1_score'].append(f1)
            results['accuracy'].append(accuracy)
            results['specificity'].append(specificity)
        
        # Find optimal thresholds for different metrics
        results['optimal_thresholds'] = {
            'f1': self.find_optimal_threshold(y_true, y_proba, 'f1'),
            'youden': self.find_optimal_threshold(y_true, y_proba, 'youden'),
            'precision': self.find_optimal_threshold(y_true, y_proba, 'precision'),
            'recall': self.find_optimal_threshold(y_true, y_proba, 'recall')
        }
        
        return results
    
    def calculate_calibration_metrics(self,
                                     y_true: np.ndarray,
                                     y_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate calibration quality metrics
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            
        Returns:
            Calibration metrics (Brier score, calibration error, etc.)
        """
        metrics = {}
        
        # Brier Score
        metrics['brier_score'] = brier_score_loss(y_true, y_proba)
        
        # Calibration curve data for ECE calculation
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_proba, n_bins=10, strategy='uniform'
            )
            
            # Expected Calibration Error (ECE)
            bin_boundaries = np.linspace(0, 1, 11)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_proba[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    
            metrics['expected_calibration_error'] = ece
            
        except Exception as e:
            self.logger.warning(f"Could not calculate calibration error: {e}")
            metrics['expected_calibration_error'] = np.nan
            
        return metrics
    
    def plot_calibration_curve(self,
                              y_true: np.ndarray,
                              y_proba: np.ndarray,
                              n_bins: int = 10,
                              strategy: str = 'uniform',
                              figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot calibration curve for probability predictions
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            n_bins: Number of bins for calibration
            strategy: Binning strategy
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=n_bins, strategy=strategy
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot calibration curve
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", 
               label=f"{self.model_name}", linewidth=2, markersize=8)
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        
        # Customize plot
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(f'Calibration Plot - {self.model_name}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            filepath = self.output_dir / f'{self.model_name}_calibration_curve.png'
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            self.logger.info(f"Calibration curve saved to {filepath}")
            
        return fig
    
    def cross_validation_analysis(self,
                                 model: Any,
                                 X: Union[pd.DataFrame, np.ndarray],
                                 y: Union[pd.Series, np.ndarray],
                                 cv_folds: int = 5,
                                 scoring: Union[str, List[str]] = ['roc_auc', 'average_precision', 'f1'],
                                 stratify: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive cross-validation analysis
        
        Args:
            model: Model to evaluate
            X: Input features
            y: True labels
            cv_folds: Number of CV folds
            scoring: Metrics to evaluate
            stratify: Use stratified CV
            
        Returns:
            Cross-validation results and statistics
        """
        from sklearn.model_selection import StratifiedKFold, KFold
        
        # Convert inputs
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Setup CV strategy
        if stratify:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
        # Ensure scoring is a list
        if isinstance(scoring, str):
            scoring = [scoring]
            
        # Perform cross-validation
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, 
                                   return_train_score=True, n_jobs=-1)
        
        # Calculate statistics
        results = {
            'cv_folds': cv_folds,
            'stratified': stratify,
            'metrics': {}
        }
        
        for metric in scoring:
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            results['metrics'][metric] = {
                'test_scores': test_scores.tolist(),
                'train_scores': train_scores.tolist(),
                'test_mean': test_scores.mean(),
                'test_std': test_scores.std(),
                'train_mean': train_scores.mean(),
                'train_std': train_scores.std(),
                'overfitting': train_scores.mean() - test_scores.mean()
            }
            
        return results
    
    def compare_models(self,
                      models: Dict[str, Any],
                      X: Union[pd.DataFrame, np.ndarray],
                      y: Union[pd.Series, np.ndarray],
                      cv_folds: int = 5,
                      scoring: str = 'roc_auc') -> pd.DataFrame:
        """
        Compare multiple models using cross-validation
        
        Args:
            models: Dictionary of model names and objects
            X: Input features
            y: True labels
            cv_folds: Number of CV folds
            scoring: Evaluation metric
            
        Returns:
            Comparison results DataFrame
        """
        results = []
        
        for name, model in models.items():
            cv_results = self.cross_validation_analysis(
                model, X, y, cv_folds=cv_folds, scoring=[scoring], stratify=True
            )
            
            metric_results = cv_results['metrics'][scoring]
            results.append({
                'model': name,
                'mean_score': metric_results['test_mean'],
                'std_score': metric_results['test_std'],
                'min_score': min(metric_results['test_scores']),
                'max_score': max(metric_results['test_scores']),
                'train_score': metric_results['train_mean'],
                'overfitting': metric_results['overfitting']
            })
            
        return pd.DataFrame(results).sort_values('mean_score', ascending=False)
    
    def generate_evaluation_report(self,
                                  evaluation_results: Dict[str, Any],
                                  output_path: str,
                                  include_plots: bool = True) -> None:
        """
        Generate comprehensive evaluation report
        
        Args:
            evaluation_results: Complete evaluation results
            output_path: Path to save report
            include_plots: Include plots in report
        """
        report_lines = []
        
        # Header
        report_lines.append(f"# Model Evaluation Report: {evaluation_results['model_name']}")
        report_lines.append(f"Generated on: {evaluation_results['evaluation_timestamp']}")
        report_lines.append(f"Evaluation time: {evaluation_results['evaluation_time']:.2f} seconds")
        report_lines.append("")
        
        # Executive Summary
        metrics = evaluation_results['metrics']
        report_lines.append("## Executive Summary")
        report_lines.append(f"- **Model**: {evaluation_results['model_name']}")
        report_lines.append(f"- **Threshold**: {evaluation_results['threshold']}")
        report_lines.append(f"- **Sample Size**: {evaluation_results['sample_size']}")
        report_lines.append(f"- **Positive Class Ratio**: {evaluation_results['positive_class_ratio']:.3f}")
        report_lines.append("")
        
        # Key Metrics
        report_lines.append("## Key Performance Metrics")
        if 'roc_auc' in metrics:
            report_lines.append(f"- **ROC AUC**: {metrics['roc_auc']:.3f}")
        if 'average_precision' in metrics:
            report_lines.append(f"- **Average Precision**: {metrics['average_precision']:.3f}")
        report_lines.append(f"- **F1 Score**: {metrics['f1_score']:.3f}")
        report_lines.append(f"- **Precision**: {metrics['precision']:.3f}")
        report_lines.append(f"- **Recall**: {metrics['recall']:.3f}")
        report_lines.append(f"- **Accuracy**: {metrics['accuracy']:.3f}")
        report_lines.append("")
        
        # Confusion Matrix Analysis
        if 'confusion_matrix_analysis' in evaluation_results:
            cm_analysis = evaluation_results['confusion_matrix_analysis']
            report_lines.append("## Confusion Matrix Analysis")
            report_lines.append(f"- **True Positives**: {cm_analysis['true_positives']}")
            report_lines.append(f"- **True Negatives**: {cm_analysis['true_negatives']}")
            report_lines.append(f"- **False Positives**: {cm_analysis['false_positives']}")
            report_lines.append(f"- **False Negatives**: {cm_analysis['false_negatives']}")
            report_lines.append("")
        
        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
            
        self.logger.info(f"Evaluation report saved to {output_path}")
    
    def export_metrics_to_json(self,
                              metrics: Dict[str, Any],
                              filepath: str) -> None:
        """
        Export evaluation metrics to JSON file
        
        Args:
            metrics: Metrics dictionary
            filepath: Output file path
        """
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        serializable_metrics = convert_numpy_types(metrics)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_metrics, f, indent=2, default=str)
            
        self.logger.info(f"Metrics exported to {filepath}")
    
    # Backward compatibility methods
    def evaluate_binary_classification(self, y_true: np.ndarray, 
                                     y_pred_proba: np.ndarray,
                                     y_pred: np.ndarray = None,
                                     model_name: str = "model") -> Dict:
        """
        Evaluate binary classification model (backward compatibility method).
        
        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            y_pred (np.ndarray): Predicted classes (optional)
            model_name (str): Name of the model
            
        Returns:
            Dict: Evaluation metrics
        """
        # Use the new comprehensive evaluation method
        if y_pred is None:
            y_pred = (y_pred_proba >= self.threshold).astype(int)
            
        return self.evaluate_model(None, y_true, y_pred, y_pred_proba)
    
    def calculate_threshold_metrics(self, y_true: np.ndarray, 
                                  y_pred_proba: np.ndarray,
                                  thresholds: List[float] = None) -> pd.DataFrame:
        """
        Calculate metrics at different thresholds (backward compatibility method).
        
        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            thresholds (List[float]): Thresholds to evaluate
            
        Returns:
            pd.DataFrame: Metrics at different thresholds
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 50)
            
        threshold_results = self.evaluate_threshold_sensitivity(y_true, y_pred_proba, thresholds)
        
        # Convert to DataFrame for backward compatibility
        df = pd.DataFrame({
            'threshold': threshold_results['thresholds'],
            'precision': threshold_results['precision'],
            'recall': threshold_results['recall'],
            'f1_score': threshold_results['f1_score'],
            'accuracy': threshold_results['accuracy'],
            'specificity': threshold_results['specificity']
        })
        
        return df
    
    def generate_evaluation_report(self, results: Dict, 
                                 output_path: str = None) -> str:
        """
        Generate comprehensive evaluation report (backward compatibility method).
        
        Args:
            results (Dict): Evaluation results
            output_path (str): Path to save report (optional)
            
        Returns:
            str: Evaluation report
        """
        if output_path is None:
            output_path = self.output_dir / f"{self.model_name}_evaluation_report.md"
            
        self.generate_evaluation_report(results, output_path)
        
        # Return the report content
        with open(output_path, 'r') as f:
            return f.read()