"""
Unit tests for the model evaluation framework
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import tempfile
import shutil
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_evaluator import ModelEvaluator
from src.models.evaluation_metrics import CustomMetrics, ThresholdOptimizer
from src.models.evaluation_plots import EvaluationPlots
from src.models.evaluation_reports import EvaluationReports
from src.models.evaluation_utils import EvaluationUtils


class TestModelEvaluator(unittest.TestCase):
    """Test the ModelEvaluator class"""
    
    def setUp(self):
        """Set up test data"""
        # Generate test data
        self.X, self.y = make_classification(
            n_samples=200, n_features=10, n_classes=2,
            weights=[0.7, 0.3], random_state=42
        )
        
        # Train a simple model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X, self.y)
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(
            model=self.model,
            model_name="test_model",
            output_dir=self.temp_dir,
            save_plots=False  # Don't save plots in tests
        )
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)
    
    def test_evaluate_model(self):
        """Test comprehensive model evaluation"""
        results = self.evaluator.evaluate_model(self.X, self.y)
        
        # Check required keys
        required_keys = ['model_name', 'metrics', 'confusion_matrix', 'threshold']
        for key in required_keys:
            self.assertIn(key, results)
        
        # Check metrics
        metrics = results['metrics']
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            self.assertGreaterEqual(metrics[metric], 0)
            self.assertLessEqual(metrics[metric], 1)
    
    def test_calculate_basic_metrics(self):
        """Test basic metrics calculation"""
        y_pred = self.model.predict(self.X)
        metrics = self.evaluator.calculate_basic_metrics(self.y, y_pred)
        
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
    
    def test_confusion_matrix_generation(self):
        """Test confusion matrix generation"""
        y_pred = self.model.predict(self.X)
        cm, analysis = self.evaluator.generate_confusion_matrix(self.y, y_pred)
        
        # Check confusion matrix shape
        self.assertEqual(cm.shape, (2, 2))
        
        # Check analysis components
        required_components = ['true_positives', 'true_negatives', 'false_positives', 'false_negatives']
        for component in required_components:
            self.assertIn(component, analysis)
    
    def test_threshold_optimization(self):
        """Test threshold optimization"""
        y_proba = self.model.predict_proba(self.X)[:, 1]
        
        optimal_thresh, optimal_score = self.evaluator.find_optimal_threshold(
            self.y, y_proba, 'f1'
        )
        
        self.assertGreaterEqual(optimal_thresh, 0)
        self.assertLessEqual(optimal_thresh, 1)
        self.assertGreaterEqual(optimal_score, 0)
        self.assertLessEqual(optimal_score, 1)
    
    def test_cross_validation_analysis(self):
        """Test cross-validation analysis"""
        cv_results = self.evaluator.cross_validation_analysis(
            self.model, self.X, self.y, cv_folds=3, scoring=['roc_auc']
        )
        
        self.assertIn('metrics', cv_results)
        self.assertIn('roc_auc', cv_results['metrics'])
        
        roc_results = cv_results['metrics']['roc_auc']
        required_keys = ['test_scores', 'test_mean', 'test_std']
        for key in required_keys:
            self.assertIn(key, roc_results)


class TestCustomMetrics(unittest.TestCase):
    """Test custom metrics for steel defect prediction"""
    
    def setUp(self):
        """Set up test data"""
        self.y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        self.y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])
    
    def test_defect_detection_rate(self):
        """Test defect detection rate calculation"""
        ddr = CustomMetrics.defect_detection_rate(self.y_true, self.y_pred)
        
        # Manual calculation: TP=3, FN=1, so recall = 3/4 = 0.75
        expected = 3/4
        self.assertAlmostEqual(ddr, expected, places=3)
    
    def test_false_alarm_rate(self):
        """Test false alarm rate calculation"""
        far = CustomMetrics.false_alarm_rate(self.y_true, self.y_pred)
        
        # Manual calculation: FP=1, TN=3, so FPR = 1/4 = 0.25
        expected = 1/4
        self.assertAlmostEqual(far, expected, places=3)
    
    def test_production_impact_score(self):
        """Test production impact score calculation"""
        cost_matrix = np.array([[0, 10], [100, 0]])
        impact = CustomMetrics.production_impact_score(
            self.y_true, self.y_pred, cost_matrix
        )
        
        # Manual calculation: TN=3*0=0, FP=1*10=10, FN=1*100=100, TP=3*0=0
        # Total = 110
        expected = 110
        self.assertEqual(impact, expected)
    
    def test_quality_efficiency_score(self):
        """Test quality efficiency score calculation"""
        score = CustomMetrics.quality_efficiency_score(
            self.y_true, self.y_pred, efficiency_weight=0.3
        )
        
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
    
    def test_manufacturing_kpi_suite(self):
        """Test comprehensive manufacturing KPI calculation"""
        kpis = CustomMetrics.manufacturing_kpi_suite(self.y_true, self.y_pred)
        
        required_kpis = [
            'defect_detection_rate', 'false_alarm_rate', 'missed_defect_rate',
            'production_impact_score', 'quality_efficiency_score',
            'process_efficiency_score', 'critical_defect_score'
        ]
        
        for kpi in required_kpis:
            self.assertIn(kpi, kpis)


class TestThresholdOptimizer(unittest.TestCase):
    """Test threshold optimization utilities"""
    
    def setUp(self):
        """Set up test data"""
        self.y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 0])
        self.y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.4, 0.6, 0.9, 0.1])
    
    def test_optimize_for_cost(self):
        """Test cost-based threshold optimization"""
        cost_matrix = np.array([[0, 10], [100, 0]])
        
        optimal_thresh, min_cost = ThresholdOptimizer.optimize_for_cost(
            self.y_true, self.y_proba, cost_matrix
        )
        
        self.assertGreaterEqual(optimal_thresh, 0)
        self.assertLessEqual(optimal_thresh, 1)
        self.assertGreaterEqual(min_cost, 0)
    
    def test_optimize_for_quality_efficiency(self):
        """Test quality-efficiency threshold optimization"""
        optimal_thresh, max_score = ThresholdOptimizer.optimize_for_quality_efficiency(
            self.y_true, self.y_proba, efficiency_weight=0.3
        )
        
        self.assertGreaterEqual(optimal_thresh, 0)
        self.assertLessEqual(optimal_thresh, 1)
        self.assertGreaterEqual(max_score, 0)
        self.assertLessEqual(max_score, 1)


class TestEvaluationUtils(unittest.TestCase):
    """Test evaluation utility functions"""
    
    def setUp(self):
        """Set up test data"""
        self.y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        self.y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])
        self.y_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.8, 0.3, 0.6, 0.7])
    
    def test_validate_inputs(self):
        """Test input validation"""
        validation = EvaluationUtils.validate_inputs(
            self.y_true, self.y_pred, self.y_proba
        )
        
        self.assertIn('valid', validation)
        self.assertIn('statistics', validation)
        
        stats = validation['statistics']
        self.assertIn('sample_size', stats)
        self.assertIn('positive_ratio', stats)
    
    def test_bootstrap_metric(self):
        """Test bootstrap confidence intervals"""
        from sklearn.metrics import accuracy_score
        
        bootstrap_results = EvaluationUtils.bootstrap_metric(
            self.y_true, self.y_pred, accuracy_score,
            n_bootstrap=100, random_state=42
        )
        
        required_keys = ['mean', 'std', 'confidence_interval_lower', 'confidence_interval_upper']
        for key in required_keys:
            self.assertIn(key, bootstrap_results)
    
    def test_statistical_significance_test(self):
        """Test statistical significance testing"""
        scores1 = np.array([0.8, 0.85, 0.82, 0.87, 0.83])
        scores2 = np.array([0.75, 0.78, 0.76, 0.80, 0.77])
        
        result = EvaluationUtils.statistical_significance_test(
            scores1, scores2, 'paired_ttest'
        )
        
        required_keys = ['test_name', 'statistic', 'p_value', 'significant']
        for key in required_keys:
            self.assertIn(key, result)
    
    def test_detect_prediction_issues(self):
        """Test prediction issue detection"""
        issues = EvaluationUtils.detect_prediction_issues(
            self.y_true, self.y_pred, self.y_proba
        )
        
        self.assertIn('warnings', issues)
        self.assertIn('recommendations', issues)


class TestEvaluationPlots(unittest.TestCase):
    """Test evaluation plotting utilities"""
    
    def setUp(self):
        """Set up test data"""
        self.temp_dir = tempfile.mkdtemp()
        self.plotter = EvaluationPlots(output_dir=self.temp_dir)
        
        # Mock evaluation results
        self.evaluation_results = {
            'model_name': 'test_model',
            'metrics': {
                'accuracy': 0.85,
                'precision': 0.80,
                'recall': 0.75,
                'f1_score': 0.77,
                'roc_auc': 0.88
            },
            'confusion_matrix': [[80, 5], [10, 15]],
            'threshold': 0.5,
            'sample_size': 110,
            'positive_class_ratio': 0.23
        }
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)
    
    def test_create_dashboard(self):
        """Test dashboard creation"""
        import matplotlib.pyplot as plt
        
        fig = self.plotter.create_dashboard(self.evaluation_results)
        self.assertIsNotNone(fig)
        plt.close(fig)
    
    def test_plot_metric_comparison(self):
        """Test metric comparison plotting"""
        import matplotlib.pyplot as plt
        
        metrics_dict = {
            'Model1': {'accuracy': 0.85, 'precision': 0.80, 'recall': 0.75, 'f1_score': 0.77},
            'Model2': {'accuracy': 0.82, 'precision': 0.78, 'recall': 0.80, 'f1_score': 0.79}
        }
        
        fig = self.plotter.plot_metric_comparison(metrics_dict)
        self.assertIsNotNone(fig)
        plt.close(fig)


class TestEvaluationReports(unittest.TestCase):
    """Test evaluation report generation"""
    
    def setUp(self):
        """Set up test data"""
        self.temp_dir = tempfile.mkdtemp()
        self.reporter = EvaluationReports(output_dir=self.temp_dir)
        
        # Mock evaluation results
        self.evaluation_results = {
            'model_name': 'test_model',
            'evaluation_timestamp': '2023-01-01T12:00:00',
            'evaluation_time': 1.5,
            'threshold': 0.5,
            'sample_size': 100,
            'positive_class_ratio': 0.2,
            'metrics': {
                'accuracy': 0.85,
                'precision': 0.80,
                'recall': 0.75,
                'f1_score': 0.77,
                'roc_auc': 0.88
            }
        }
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)
    
    def test_generate_markdown_report(self):
        """Test Markdown report generation"""
        report_path = self.reporter.generate_markdown_report(self.evaluation_results)
        
        self.assertTrue(Path(report_path).exists())
        
        # Check report content
        with open(report_path, 'r') as f:
            content = f.read()
            self.assertIn('Model Evaluation Report', content)
            self.assertIn('test_model', content)
    
    def test_generate_json_report(self):
        """Test JSON report generation"""
        report_path = self.reporter.generate_json_report(self.evaluation_results)
        
        self.assertTrue(Path(report_path).exists())
        
        # Check report content
        import json
        with open(report_path, 'r') as f:
            data = json.load(f)
            self.assertIn('evaluation_results', data)
            self.assertIn('recommendations', data)


if __name__ == '__main__':
    unittest.main()