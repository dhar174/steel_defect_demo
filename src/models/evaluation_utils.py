"""
Utility functions for model evaluation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import warnings
from pathlib import Path
import joblib
import json


class EvaluationUtils:
    """Utility functions for model evaluation"""
    
    @staticmethod
    def bootstrap_metric(y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        metric_func: Callable,
                        n_bootstrap: int = 1000,
                        confidence_level: float = 0.95,
                        random_state: int = 42) -> Dict[str, float]:
        """
        Calculate bootstrap confidence intervals for a metric
        
        Args:
            y_true: True labels
            y_pred: Predicted labels or probabilities
            metric_func: Metric function to bootstrap
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            random_state: Random seed
            
        Returns:
            Dictionary with metric statistics
        """
        np.random.seed(random_state)
        n_samples = len(y_true)
        bootstrap_scores = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            try:
                score = metric_func(y_true_boot, y_pred_boot)
                bootstrap_scores.append(score)
            except Exception:
                # Skip if metric calculation fails for this bootstrap sample
                continue
        
        bootstrap_scores = np.array(bootstrap_scores)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        return {
            'mean': bootstrap_scores.mean(),
            'std': bootstrap_scores.std(),
            'confidence_interval_lower': np.percentile(bootstrap_scores, lower_percentile),
            'confidence_interval_upper': np.percentile(bootstrap_scores, upper_percentile),
            'confidence_level': confidence_level,
            'n_bootstrap': len(bootstrap_scores)
        }
    
    @staticmethod
    def statistical_significance_test(scores1: np.ndarray,
                                     scores2: np.ndarray,
                                     test_type: str = 'paired_ttest',
                                     alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform statistical significance test between two sets of CV scores
        
        Args:
            scores1: First set of scores
            scores2: Second set of scores
            test_type: Type of test ('paired_ttest', 'wilcoxon', 'mannwhitneyu')
            alpha: Significance level
            
        Returns:
            Test results
        """
        if test_type == 'paired_ttest':
            statistic, p_value = stats.ttest_rel(scores1, scores2)
            test_name = "Paired t-test"
        elif test_type == 'wilcoxon':
            statistic, p_value = stats.wilcoxon(scores1, scores2)
            test_name = "Wilcoxon signed-rank test"
        elif test_type == 'mannwhitneyu':
            statistic, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
            test_name = "Mann-Whitney U test"
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
        
        # Effect size (Cohen's d for t-test)
        effect_size = None
        if test_type == 'paired_ttest':
            pooled_std = np.sqrt(((len(scores1) - 1) * np.var(scores1, ddof=1) + 
                                 (len(scores2) - 1) * np.var(scores2, ddof=1)) / 
                                (len(scores1) + len(scores2) - 2))
            effect_size = (np.mean(scores1) - np.mean(scores2)) / pooled_std
        
        return {
            'test_name': test_name,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'alpha': alpha,
            'effect_size': float(effect_size) if effect_size is not None else None,
            'interpretation': EvaluationUtils._interpret_significance_test(p_value, alpha, effect_size)
        }
    
    @staticmethod
    def _interpret_significance_test(p_value: float, alpha: float, effect_size: Optional[float]) -> str:
        """Interpret statistical significance test results"""
        interpretation = []
        
        if p_value < alpha:
            interpretation.append("Statistically significant difference detected")
        else:
            interpretation.append("No statistically significant difference detected")
        
        if effect_size is not None:
            if abs(effect_size) < 0.2:
                interpretation.append("Small effect size")
            elif abs(effect_size) < 0.5:
                interpretation.append("Medium effect size")
            elif abs(effect_size) < 0.8:
                interpretation.append("Large effect size")
            else:
                interpretation.append("Very large effect size")
        
        return ". ".join(interpretation)
    
    @staticmethod
    def mcnemar_test(y_true: np.ndarray, 
                    y_pred1: np.ndarray, 
                    y_pred2: np.ndarray) -> Dict[str, Any]:
        """
        Perform McNemar's test for comparing two classifiers
        
        Args:
            y_true: True labels
            y_pred1: Predictions from first classifier
            y_pred2: Predictions from second classifier
            
        Returns:
            McNemar test results
        """
        from statsmodels.stats.contingency_tables import mcnemar
        
        # Create contingency table
        correct1 = (y_pred1 == y_true)
        correct2 = (y_pred2 == y_true)
        
        # McNemar table: [both wrong, 1 right 2 wrong], [1 wrong 2 right, both right]
        both_wrong = np.sum(~correct1 & ~correct2)
        model1_right_model2_wrong = np.sum(correct1 & ~correct2)
        model1_wrong_model2_right = np.sum(~correct1 & correct2)
        both_right = np.sum(correct1 & correct2)
        
        contingency_table = np.array([
            [both_wrong, model1_right_model2_wrong],
            [model1_wrong_model2_right, both_right]
        ])
        
        # Perform McNemar test
        result = mcnemar(contingency_table, exact=False, correction=True)
        
        return {
            'contingency_table': contingency_table.tolist(),
            'statistic': float(result.statistic),
            'p_value': float(result.pvalue),
            'significant': result.pvalue < 0.05,
            'interpretation': "Model 1 significantly different from Model 2" if result.pvalue < 0.05 
                           else "No significant difference between models"
        }
    
    @staticmethod
    def cross_validation_with_confidence(model: Any,
                                       X: Union[pd.DataFrame, np.ndarray],
                                       y: Union[pd.Series, np.ndarray],
                                       scoring: str = 'roc_auc',
                                       cv: int = 5,
                                       confidence_level: float = 0.95,
                                       random_state: int = 42) -> Dict[str, Any]:
        """
        Perform cross-validation with confidence intervals
        
        Args:
            model: Model to evaluate
            X: Features
            y: Labels
            scoring: Scoring metric
            cv: Number of CV folds
            confidence_level: Confidence level for intervals
            random_state: Random seed
            
        Returns:
            CV results with confidence intervals
        """
        # Convert inputs
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X, y, 
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
            scoring=scoring,
            n_jobs=-1
        )
        
        # Calculate confidence intervals
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        
        # t-distribution confidence interval
        from scipy.stats import t
        alpha = 1 - confidence_level
        df = len(cv_scores) - 1
        t_critical = t.ppf(1 - alpha/2, df)
        margin_error = t_critical * (std_score / np.sqrt(len(cv_scores)))
        
        return {
            'scores': cv_scores.tolist(),
            'mean': float(mean_score),
            'std': float(std_score),
            'confidence_interval_lower': float(mean_score - margin_error),
            'confidence_interval_upper': float(mean_score + margin_error),
            'confidence_level': confidence_level,
            'cv_folds': cv
        }
    
    @staticmethod
    def validate_inputs(y_true: np.ndarray, 
                       y_pred: Optional[np.ndarray] = None,
                       y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Validate evaluation inputs and return statistics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (optional)
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Validation results and input statistics
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check y_true
        if not isinstance(y_true, np.ndarray):
            y_true = np.array(y_true)
        
        # Basic validation
        if len(y_true) == 0:
            validation_results['valid'] = False
            validation_results['errors'].append("Empty y_true array")
            return validation_results
        
        # Check for valid binary classification
        unique_labels = np.unique(y_true)
        if len(unique_labels) != 2:
            validation_results['errors'].append(f"Expected binary classification, found {len(unique_labels)} unique labels: {unique_labels}")
        
        if not all(label in [0, 1] for label in unique_labels):
            validation_results['warnings'].append("Labels are not 0/1, assuming binary classification")
        
        # Class balance statistics
        validation_results['statistics']['sample_size'] = len(y_true)
        validation_results['statistics']['positive_count'] = int(np.sum(y_true))
        validation_results['statistics']['negative_count'] = int(len(y_true) - np.sum(y_true))
        validation_results['statistics']['positive_ratio'] = float(np.mean(y_true))
        
        # Check class imbalance
        pos_ratio = validation_results['statistics']['positive_ratio']
        if pos_ratio < 0.05 or pos_ratio > 0.95:
            validation_results['warnings'].append(f"Severe class imbalance detected: {pos_ratio:.1%} positive samples")
        elif pos_ratio < 0.1 or pos_ratio > 0.9:
            validation_results['warnings'].append(f"Class imbalance detected: {pos_ratio:.1%} positive samples")
        
        # Validate y_pred if provided
        if y_pred is not None:
            if len(y_pred) != len(y_true):
                validation_results['valid'] = False
                validation_results['errors'].append(f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})")
        
        # Validate y_proba if provided
        if y_proba is not None:
            if len(y_proba) != len(y_true):
                validation_results['valid'] = False
                validation_results['errors'].append(f"Length mismatch: y_true ({len(y_true)}) vs y_proba ({len(y_proba)})")
            
            if np.any(y_proba < 0) or np.any(y_proba > 1):
                validation_results['warnings'].append("y_proba contains values outside [0, 1] range")
            
            # Check for degenerate probabilities
            if np.all(y_proba == y_proba[0]):
                validation_results['warnings'].append("All probability predictions are identical")
        
        return validation_results
    
    @staticmethod
    def detect_prediction_issues(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Detect common issues with model predictions
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary of detected issues and recommendations
        """
        issues = {
            'warnings': [],
            'recommendations': []
        }
        
        # Check for perfect predictions (may indicate data leakage)
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_true, y_pred)
        if accuracy == 1.0:
            issues['warnings'].append("Perfect accuracy detected - check for data leakage")
            issues['recommendations'].append("Verify that future information is not leaking into features")
        
        # Check for very poor predictions
        if accuracy < 0.5:
            issues['warnings'].append(f"Very low accuracy ({accuracy:.3f}) - worse than random")
            issues['recommendations'].append("Check model training, feature engineering, or data quality")
        
        # Check prediction distribution
        pred_positive_rate = np.mean(y_pred)
        true_positive_rate = np.mean(y_true)
        
        if abs(pred_positive_rate - true_positive_rate) > 0.2:
            issues['warnings'].append(
                f"Large discrepancy between predicted ({pred_positive_rate:.3f}) "
                f"and actual ({true_positive_rate:.3f}) positive rates"
            )
            issues['recommendations'].append("Consider adjusting classification threshold or rebalancing training data")
        
        # Check for constant predictions
        if len(np.unique(y_pred)) == 1:
            issues['warnings'].append("Model makes constant predictions")
            issues['recommendations'].append("Check model training convergence and feature informativeness")
        
        # Probability-specific checks
        if y_proba is not None:
            # Check for extreme probabilities
            extreme_count = np.sum((y_proba < 0.01) | (y_proba > 0.99))
            if extreme_count > len(y_proba) * 0.5:
                issues['warnings'].append(f"High proportion ({extreme_count/len(y_proba):.1%}) of extreme probabilities")
                issues['recommendations'].append("Consider probability calibration techniques")
            
            # Check probability distribution
            prob_std = np.std(y_proba)
            if prob_std < 0.1:
                issues['warnings'].append(f"Low probability standard deviation ({prob_std:.3f})")
                issues['recommendations'].append("Model may be underconfident - check calibration")
        
        return issues
    
    @staticmethod
    def calculate_sample_size_requirements(expected_effect_size: float = 0.1,
                                         power: float = 0.8,
                                         alpha: float = 0.05,
                                         baseline_rate: float = 0.5) -> Dict[str, int]:
        """
        Calculate required sample sizes for evaluation
        
        Args:
            expected_effect_size: Expected difference in performance metrics
            power: Statistical power (1 - β)
            alpha: Type I error rate
            baseline_rate: Baseline positive class rate
            
        Returns:
            Required sample sizes for different scenarios
        """
        from scipy.stats import norm
        
        # Z-scores for power and alpha
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        
        # Sample size for comparing two proportions
        p1 = baseline_rate
        p2 = baseline_rate + expected_effect_size
        p_pooled = (p1 + p2) / 2
        
        n_comparison = (2 * p_pooled * (1 - p_pooled) * (z_alpha + z_beta)**2) / (expected_effect_size**2)
        
        # Sample size for single proportion confidence interval
        n_single = (z_alpha**2 * baseline_rate * (1 - baseline_rate)) / (expected_effect_size**2)
        
        # Sample size for AUC comparison (rough approximation)
        n_auc = (z_alpha + z_beta)**2 * 2 / (expected_effect_size**2)
        
        return {
            'single_model_evaluation': int(np.ceil(n_single)),
            'model_comparison': int(np.ceil(n_comparison)),
            'auc_comparison': int(np.ceil(n_auc)),
            'recommended_minimum': int(np.ceil(max(n_single, n_comparison, n_auc)))
        }
    
    @staticmethod
    def save_evaluation_artifacts(evaluation_results: Dict[str, Any],
                                 output_dir: str,
                                 model_name: str) -> Dict[str, str]:
        """
        Save evaluation artifacts to disk
        
        Args:
            evaluation_results: Complete evaluation results
            output_dir: Output directory
            model_name: Model name for file naming
            
        Returns:
            Dictionary of saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save main results as JSON
        results_path = output_path / f"{model_name}_evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        saved_files['results_json'] = str(results_path)
        
        # Save metrics as CSV
        if 'metrics' in evaluation_results:
            metrics_df = pd.DataFrame([evaluation_results['metrics']])
            metrics_path = output_path / f"{model_name}_metrics.csv"
            metrics_df.to_csv(metrics_path, index=False)
            saved_files['metrics_csv'] = str(metrics_path)
        
        # Save threshold analysis if available
        if 'threshold_analysis' in evaluation_results and evaluation_results['threshold_analysis']:
            threshold_data = evaluation_results['threshold_analysis']
            threshold_df = pd.DataFrame({
                'threshold': threshold_data.get('thresholds', []),
                'precision': threshold_data.get('precision', []),
                'recall': threshold_data.get('recall', []),
                'f1_score': threshold_data.get('f1_score', []),
                'accuracy': threshold_data.get('accuracy', []),
                'specificity': threshold_data.get('specificity', [])
            })
            threshold_path = output_path / f"{model_name}_threshold_analysis.csv"
            threshold_df.to_csv(threshold_path, index=False)
            saved_files['threshold_csv'] = str(threshold_path)
        
        return saved_files
    
    @staticmethod
    def load_evaluation_results(filepath: str) -> Dict[str, Any]:
        """
        Load evaluation results from disk
        
        Args:
            filepath: Path to evaluation results file
            
        Returns:
            Evaluation results dictionary
        """
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def merge_evaluation_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple evaluation results for comparison
        
        Args:
            results_list: List of evaluation result dictionaries
            
        Returns:
            Merged results for comparison
        """
        merged = {
            'models': {},
            'comparison_timestamp': datetime.now().isoformat(),
            'model_count': len(results_list)
        }
        
        for result in results_list:
            model_name = result.get('model_name', f"model_{len(merged['models'])}")
            merged['models'][model_name] = result
        
        # Create comparison summary
        comparison_metrics = {}
        for model_name, result in merged['models'].items():
            if 'metrics' in result:
                comparison_metrics[model_name] = result['metrics']
        
        merged['comparison_metrics'] = comparison_metrics
        
        return merged