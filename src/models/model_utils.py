"""
Utility functions for model training, evaluation, and analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_curve, precision_recall_curve, f1_score,
    precision_score, recall_score
)
from sklearn.utils.class_weight import compute_class_weight
import logging
from datetime import datetime


def optimize_threshold(y_true: np.ndarray, 
                      y_proba: np.ndarray,
                      metric: str = 'f1') -> Tuple[float, float]:
    """
    Find optimal classification threshold
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall', 'youden')
        
    Returns:
        Tuple of (optimal_threshold, optimal_score)
    """
    thresholds = np.linspace(0.1, 0.9, 81)  # More granular threshold search
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'youden':
            # Youden's J statistic = sensitivity + specificity - 1
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            # Find closest threshold
            threshold_idx = np.argmin(np.abs(fpr + tpr - 1))
            score = tpr[threshold_idx] - fpr[threshold_idx]
        else:
            raise ValueError(f"Unsupported metric: {metric}")
            
        scores.append(score)
    
    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_score = scores[optimal_idx]
    
    return optimal_threshold, optimal_score


def plot_learning_curves(train_scores: List[float],
                        val_scores: List[float],
                        metric_name: str,
                        figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot learning curves for training monitoring
    
    Args:
        train_scores: Training scores over epochs/iterations
        val_scores: Validation scores over epochs/iterations
        metric_name: Name of the metric being plotted
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = range(1, len(train_scores) + 1)
    
    ax.plot(epochs, train_scores, 'b-', label=f'Training {metric_name}', linewidth=2)
    ax.plot(epochs, val_scores, 'r-', label=f'Validation {metric_name}', linewidth=2)
    
    ax.set_xlabel('Epoch/Iteration')
    ax.set_ylabel(metric_name.title())
    ax.set_title(f'Learning Curves - {metric_name.title()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Highlight best validation score
    best_val_idx = np.argmax(val_scores)
    ax.scatter(best_val_idx + 1, val_scores[best_val_idx], 
              color='red', s=100, zorder=5, label=f'Best Val: {val_scores[best_val_idx]:.4f}')
    
    plt.tight_layout()
    return fig


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        y: Array of class labels
        
    Returns:
        Dictionary mapping class labels to weights
    """
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))


def create_stratified_splits(X: Union[pd.DataFrame, np.ndarray],
                           y: Union[pd.Series, np.ndarray],
                           n_splits: int = 5,
                           random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified train/validation splits
    
    Args:
        X: Features
        y: Labels
        n_splits: Number of splits
        random_state: Random seed
        
    Returns:
        List of (train_idx, val_idx) tuples
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
        
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(X, y))


def log_model_performance(metrics: Dict[str, float],
                         model_name: str,
                         timestamp: str,
                         log_file: Optional[str] = None) -> None:
    """
    Log model performance to tracking system
    
    Args:
        metrics: Dictionary of performance metrics
        model_name: Name of the model
        timestamp: Timestamp of evaluation
        log_file: Optional log file path
    """
    logger = logging.getLogger('ModelPerformance')
    
    # Format metrics for logging
    metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))])
    log_message = f"Model: {model_name} | Time: {timestamp} | Metrics: {metrics_str}"
    
    logger.info(log_message)
    
    # Also log to file if specified
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()} - {log_message}\n")


def plot_threshold_analysis(y_true: np.ndarray, 
                          y_proba: np.ndarray,
                          figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """
    Plot threshold analysis showing precision, recall, and F1 vs threshold
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    precisions, recalls, f1s = [], [], []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Precision vs Threshold
    ax1.plot(thresholds, precisions, 'b-', linewidth=2)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision vs Threshold')
    ax1.grid(True, alpha=0.3)
    
    # Recall vs Threshold
    ax2.plot(thresholds, recalls, 'r-', linewidth=2)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Recall')
    ax2.set_title('Recall vs Threshold')
    ax2.grid(True, alpha=0.3)
    
    # F1 vs Threshold
    ax3.plot(thresholds, f1s, 'g-', linewidth=2)
    best_f1_idx = np.argmax(f1s)
    ax3.scatter(thresholds[best_f1_idx], f1s[best_f1_idx], 
               color='red', s=100, zorder=5)
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('F1 Score')
    ax3.set_title(f'F1 vs Threshold (Best: {thresholds[best_f1_idx]:.3f})')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def analyze_prediction_distribution(y_true: np.ndarray, 
                                  y_proba: np.ndarray,
                                  figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """
    Analyze distribution of predicted probabilities by class
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram of probabilities by class
    ax1.hist(y_proba[y_true == 0], bins=50, alpha=0.7, label='No Defect', color='blue')
    ax1.hist(y_proba[y_true == 1], bins=50, alpha=0.7, label='Defect', color='red')
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Predicted Probabilities')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot of probabilities by class
    data_for_box = [y_proba[y_true == 0], y_proba[y_true == 1]]
    labels = ['No Defect', 'Defect']
    ax2.boxplot(data_for_box, labels=labels)
    ax2.set_ylabel('Predicted Probability')
    ax2.set_title('Probability Distribution by Class')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def calculate_cost_sensitive_metrics(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   false_positive_cost: float = 1.0,
                                   false_negative_cost: float = 10.0) -> Dict[str, float]:
    """
    Calculate cost-sensitive evaluation metrics
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        false_positive_cost: Cost of false positive
        false_negative_cost: Cost of false negative
        
    Returns:
        Dictionary of cost-sensitive metrics
    """
    # Calculate confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Cost calculations
    total_cost = fp * false_positive_cost + fn * false_negative_cost
    cost_per_sample = total_cost / len(y_true)
    
    # Cost-normalized metrics
    savings_ratio = 1 - (total_cost / (len(y_true) * false_negative_cost))
    
    return {
        'total_cost': total_cost,
        'cost_per_sample': cost_per_sample,
        'false_positive_cost': fp * false_positive_cost,
        'false_negative_cost': fn * false_negative_cost,
        'savings_ratio': max(0, savings_ratio),  # Ensure non-negative
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }


def create_model_comparison_plot(model_results: Dict[str, Dict[str, float]],
                               metrics: List[str] = ['roc_auc', 'average_precision', 'f1_score'],
                               figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Create comparison plot for multiple models
    
    Args:
        model_results: Dictionary mapping model names to their metrics
        metrics: List of metrics to compare
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    model_names = list(model_results.keys())
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        scores = [model_results[model].get(metric, 0) for model in model_names]
        
        bars = axes[i].bar(model_names, scores, alpha=0.7)
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig


def detect_overfitting(train_scores: List[float],
                      val_scores: List[float],
                      patience: int = 5,
                      min_delta: float = 0.001) -> Dict[str, Any]:
    """
    Detect overfitting in training curves
    
    Args:
        train_scores: Training scores over epochs
        val_scores: Validation scores over epochs
        patience: Number of epochs with no improvement to flag overfitting
        min_delta: Minimum change to qualify as improvement
        
    Returns:
        Dictionary with overfitting analysis
    """
    if len(train_scores) != len(val_scores):
        raise ValueError("Training and validation scores must have same length")
    
    # Find best validation score
    best_val_epoch = np.argmax(val_scores)
    best_val_score = val_scores[best_val_epoch]
    
    # Check for deterioration after best epoch
    epochs_since_best = len(val_scores) - best_val_epoch - 1
    
    # Calculate gap between train and validation
    final_gap = train_scores[-1] - val_scores[-1] if len(train_scores) > 0 else 0
    
    # Detect if validation stopped improving
    overfitting_detected = epochs_since_best >= patience
    
    # Check if training continues to improve while validation stagnates
    if len(train_scores) > patience:
        recent_train_improvement = train_scores[-1] - train_scores[-patience]
        recent_val_improvement = val_scores[-1] - val_scores[-patience]
        
        train_val_divergence = (recent_train_improvement > min_delta and 
                               recent_val_improvement < min_delta)
    else:
        train_val_divergence = False
    
    return {
        'overfitting_detected': overfitting_detected,
        'train_val_divergence': train_val_divergence,
        'best_val_epoch': best_val_epoch,
        'best_val_score': best_val_score,
        'epochs_since_best': epochs_since_best,
        'final_train_val_gap': final_gap,
        'recommendation': 'early_stopping' if overfitting_detected else 'continue_training'
    }


def memory_usage_tracker(func):
    """
    Decorator to track memory usage of functions
    
    Args:
        func: Function to track
        
    Returns:
        Wrapped function with memory tracking
    """
    import psutil
    import os
    
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        logger = logging.getLogger('MemoryTracker')
        logger.info(f"Function {func.__name__} used {memory_used:.2f} MB "
                   f"(Before: {memory_before:.2f} MB, After: {memory_after:.2f} MB)")
        
        return result
    
    return wrapper