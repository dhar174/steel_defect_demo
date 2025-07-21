"""
Custom evaluation metrics for steel defect prediction
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Dict, Optional


class CustomMetrics:
    """Custom evaluation metrics for steel defect prediction"""
    
    @staticmethod
    def defect_detection_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate defect detection rate (recall for defect class)
        
        Args:
            y_true: True labels (0=normal, 1=defect)
            y_pred: Predicted labels
            
        Returns:
            Defect detection rate (sensitivity/recall)
        """
        from sklearn.metrics import recall_score
        return recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    @staticmethod
    def false_alarm_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate false alarm rate (FPR)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            False alarm rate (false positive rate)
        """
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return fp / (fp + tn) if (fp + tn) > 0 else 0.0
        except ValueError as e:
            logging.error(f"ValueError in false_alarm_rate: {e}")
            return 0.0
    
    @staticmethod
    def production_impact_score(y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               cost_matrix: np.ndarray) -> float:
        """
        Calculate production impact score based on cost matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            cost_matrix: 2x2 cost matrix [[TN_cost, FP_cost], [FN_cost, TP_cost]]
            
        Returns:
            Total production cost impact
        """
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Default cost matrix if not provided
            if cost_matrix is None:
                cost_matrix = np.array([[0, 10], [100, 0]])  # Missing defects cost 10x more
            
            total_cost = (
                tn * cost_matrix[0, 0] +  # True Negative cost
                fp * cost_matrix[0, 1] +  # False Positive cost
                fn * cost_matrix[1, 0] +  # False Negative cost
                tp * cost_matrix[1, 1]    # True Positive cost
            )
            
            return float(total_cost)
            
        except ValueError:
            return 0.0
    
    @staticmethod
    def quality_efficiency_score(y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                efficiency_weight: float = 0.3) -> float:
        """
        Balanced score considering both quality (defect detection) and efficiency (false alarms)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            efficiency_weight: Weight for efficiency component (0-1)
            
        Returns:
            Combined quality-efficiency score
        """
        defect_detection = CustomMetrics.defect_detection_rate(y_true, y_pred)
        false_alarm = CustomMetrics.false_alarm_rate(y_true, y_pred)
        
        # Quality score (higher is better)
        quality_score = defect_detection
        
        # Efficiency score (lower false alarm rate is better)
        efficiency_score = 1.0 - false_alarm
        
        # Combined score
        combined_score = (
            (1 - efficiency_weight) * quality_score + 
            efficiency_weight * efficiency_score
        )
        
        return combined_score
    
    @staticmethod
    def missed_defect_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate rate of missed defects (false negative rate)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Missed defect rate (false negative rate)
        """
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return fn / (fn + tp) if (fn + tp) > 0 else 0.0
        except ValueError:
            return 0.0
    
    @staticmethod
    def process_efficiency_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate process efficiency based on correct predictions
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Process efficiency score
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_true, y_pred)
    
    @staticmethod
    def critical_defect_score(y_true: np.ndarray, 
                             y_pred: np.ndarray, 
                             defect_severity: Optional[np.ndarray] = None) -> float:
        """
        Calculate score focusing on critical defects
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            defect_severity: Severity weights for defects (optional)
            
        Returns:
            Critical defect detection score
        """
        if defect_severity is None:
            # Default: all defects have equal severity
            defect_severity = np.ones_like(y_true)
        
        # Only consider defective samples
        defect_mask = y_true == 1
        
        if not np.any(defect_mask):
            return 1.0  # No defects to detect
        
        # Calculate weighted detection rate
        detected_severity = defect_severity[defect_mask & (y_pred == 1)].sum()
        total_severity = defect_severity[defect_mask].sum()
        
        return detected_severity / total_severity if total_severity > 0 else 0.0
    
    @staticmethod
    def manufacturing_kpi_suite(y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               cost_matrix: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive suite of manufacturing KPIs
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            cost_matrix: Cost matrix for production impact
            
        Returns:
            Dictionary of manufacturing KPIs
        """
        return {
            'defect_detection_rate': CustomMetrics.defect_detection_rate(y_true, y_pred),
            'false_alarm_rate': CustomMetrics.false_alarm_rate(y_true, y_pred),
            'missed_defect_rate': CustomMetrics.missed_defect_rate(y_true, y_pred),
            'production_impact_score': CustomMetrics.production_impact_score(y_true, y_pred, cost_matrix),
            'quality_efficiency_score': CustomMetrics.quality_efficiency_score(y_true, y_pred),
            'process_efficiency_score': CustomMetrics.process_efficiency_score(y_true, y_pred),
            'critical_defect_score': CustomMetrics.critical_defect_score(y_true, y_pred)
        }


class ThresholdOptimizer:
    """Utility class for threshold optimization"""
    
    @staticmethod
    def optimize_for_cost(y_true: np.ndarray, 
                         y_proba: np.ndarray,
                         cost_matrix: np.ndarray,
                         thresholds: Optional[np.ndarray] = None) -> tuple:
        """
        Find threshold that minimizes total cost
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            cost_matrix: Cost matrix
            thresholds: Thresholds to evaluate
            
        Returns:
            Tuple of (optimal_threshold, min_cost)
        """
        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 100)
        
        min_cost = float('inf')
        optimal_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            cost = CustomMetrics.production_impact_score(y_true, y_pred, cost_matrix)
            
            if cost < min_cost:
                min_cost = cost
                optimal_threshold = threshold
        
        return optimal_threshold, min_cost
    
    @staticmethod
    def optimize_for_quality_efficiency(y_true: np.ndarray, 
                                       y_proba: np.ndarray,
                                       efficiency_weight: float = 0.3,
                                       thresholds: Optional[np.ndarray] = None) -> tuple:
        """
        Find threshold that maximizes quality-efficiency score
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            efficiency_weight: Weight for efficiency component
            thresholds: Thresholds to evaluate
            
        Returns:
            Tuple of (optimal_threshold, max_score)
        """
        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 100)
        
        max_score = -1
        optimal_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            score = CustomMetrics.quality_efficiency_score(y_true, y_pred, efficiency_weight)
            
            if score > max_score:
                max_score = score
                optimal_threshold = threshold
        
        return optimal_threshold, max_score