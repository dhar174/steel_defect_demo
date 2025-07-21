"""
Tests for the ModelComparison component.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

# Import the component
from src.visualization.components.model_comparison import ModelComparison, create_sample_model_results


class TestModelComparison:
    """Test suite for ModelComparison component."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.comparison = ModelComparison()
        self.sample_results = create_sample_model_results()
    
    def test_initialization(self):
        """Test ModelComparison initialization."""
        comparison = ModelComparison()
        assert comparison.theme == "plotly_white"
        assert len(comparison.color_palette) == 10
        
        # Test custom theme
        comparison_custom = ModelComparison(theme="plotly_dark")
        assert comparison_custom.theme == "plotly_dark"
    
    def test_create_sample_model_results(self):
        """Test sample model results creation."""
        results = create_sample_model_results()
        
        # Check structure
        assert isinstance(results, dict)
        assert 'XGBoost' in results
        assert 'LSTM' in results
        
        # Check XGBoost results
        xgb_results = results['XGBoost']
        assert 'y_true' in xgb_results
        assert 'y_pred' in xgb_results
        assert 'y_pred_proba' in xgb_results
        assert 'feature_importance' in xgb_results
        
        # Check LSTM results
        lstm_results = results['LSTM']
        assert 'y_true' in lstm_results
        assert 'y_pred' in lstm_results
        assert 'y_pred_proba' in lstm_results
        assert 'attention_weights' in lstm_results
        
        # Check data consistency
        assert len(xgb_results['y_true']) == len(lstm_results['y_true'])
        assert np.array_equal(xgb_results['y_true'], lstm_results['y_true'])
    
    def test_create_roc_pr_comparison(self):
        """Test ROC and PR curve comparison."""
        fig = self.comparison.create_roc_pr_comparison(self.sample_results)
        
        # Check figure structure
        assert fig is not None
        assert len(fig.data) >= 2  # At least ROC and PR curves
        assert fig.layout.title.text == "ROC and Precision-Recall Curve Comparison"
    
    def test_create_feature_importance_chart(self):
        """Test feature importance chart creation."""
        fig = self.comparison.create_feature_importance_chart(self.sample_results)
        
        # Check figure structure
        assert fig is not None
        assert fig.layout.title.text == "Feature Importance Comparison"
        
        # Test with empty feature importance
        empty_results = {
            'Model1': {'y_true': [0, 1], 'y_pred': [0, 1], 'y_pred_proba': [0.3, 0.7]}
        }
        fig_empty = self.comparison.create_feature_importance_chart(empty_results)
        assert fig_empty is not None
    
    def test_create_attention_visualization(self):
        """Test attention weight visualization."""
        fig = self.comparison.create_attention_visualization(self.sample_results)
        
        # Check figure structure
        assert fig is not None
        assert fig.layout.title.text == "LSTM Attention Weight Visualization"
        
        # Test with no attention weights
        no_attention_results = {
            'Model1': {'y_true': [0, 1], 'y_pred': [0, 1], 'y_pred_proba': [0.3, 0.7]}
        }
        fig_no_att = self.comparison.create_attention_visualization(no_attention_results)
        assert fig_no_att is not None
    
    def test_create_prediction_correlation_analysis(self):
        """Test prediction correlation analysis."""
        fig = self.comparison.create_prediction_correlation_analysis(self.sample_results)
        
        # Check figure structure
        assert fig is not None
        assert fig.layout.title.text == "Model Prediction Correlation Analysis"
        
        # Test with insufficient models
        single_model = {'Model1': self.sample_results['XGBoost']}
        fig_single = self.comparison.create_prediction_correlation_analysis(single_model)
        assert fig_single is not None
    
    def test_create_performance_metrics_table(self):
        """Test performance metrics table creation."""
        table = self.comparison.create_performance_metrics_table(self.sample_results)
        
        # Check table structure
        assert table is not None
        assert hasattr(table, 'data')
        assert hasattr(table, 'columns')
        
        # Check that data contains both models
        data = table.data
        assert len(data) == 2
        model_names = [row['Model'] for row in data]
        assert 'XGBoost' in model_names
        assert 'LSTM' in model_names
    
    def test_create_side_by_side_charts(self):
        """Test side-by-side charts creation."""
        fig = self.comparison.create_side_by_side_charts(self.sample_results)
        
        # Check figure structure
        assert fig is not None
        assert fig.layout.title.text == "Model Performance Comparison"
        
        # Test with custom chart types
        custom_charts = ['roc', 'pr']
        fig_custom = self.comparison.create_side_by_side_charts(
            self.sample_results, chart_types=custom_charts
        )
        assert fig_custom is not None
    
    def test_get_dashboard_layout(self):
        """Test dashboard layout creation."""
        layout = self.comparison.get_dashboard_layout(self.sample_results)
        
        # Check layout structure
        assert layout is not None
        assert hasattr(layout, 'children')
    
    def test_empty_results_handling(self):
        """Test handling of empty or invalid results."""
        empty_results = {}
        
        # These should not crash
        fig_roc = self.comparison.create_roc_pr_comparison(empty_results)
        fig_feature = self.comparison.create_feature_importance_chart(empty_results)
        fig_attention = self.comparison.create_attention_visualization(empty_results)
        fig_correlation = self.comparison.create_prediction_correlation_analysis(empty_results)
        table = self.comparison.create_performance_metrics_table(empty_results)
        fig_side = self.comparison.create_side_by_side_charts(empty_results)
        
        assert all(fig is not None for fig in [fig_roc, fig_feature, fig_attention, fig_correlation, fig_side])
        assert table is not None
    
    def test_malformed_data_handling(self):
        """Test handling of malformed data."""
        malformed_results = {
            'BadModel': {
                'y_true': [],  # Empty arrays
                'y_pred': [],
                'y_pred_proba': []
            }
        }
        
        # These should handle empty data gracefully
        table = self.comparison.create_performance_metrics_table(malformed_results)
        assert table is not None
        
        fig = self.comparison.create_roc_pr_comparison(malformed_results)
        assert fig is not None


def test_model_comparison_integration():
    """Integration test for ModelComparison with real-like data."""
    # Create more realistic test data
    np.random.seed(123)
    n_samples = 100
    
    y_true = np.random.binomial(1, 0.2, n_samples)
    
    # Model 1: Good performance
    model1_proba = np.random.beta(1, 4, n_samples)
    model1_proba[y_true == 1] += 0.5
    model1_proba = np.clip(model1_proba, 0, 1)
    
    # Model 2: Moderate performance
    model2_proba = np.random.beta(2, 3, n_samples)
    model2_proba[y_true == 1] += 0.3
    model2_proba = np.clip(model2_proba, 0, 1)
    
    results = {
        'Model_A': {
            'y_true': y_true,
            'y_pred': (model1_proba > 0.5).astype(int),
            'y_pred_proba': model1_proba,
            'feature_importance': {'feature_1': 0.3, 'feature_2': 0.7}
        },
        'Model_B': {
            'y_true': y_true,
            'y_pred': (model2_proba > 0.5).astype(int),
            'y_pred_proba': model2_proba,
            'attention_weights': np.random.random((1, 20, 5))
        }
    }
    
    comparison = ModelComparison()
    
    # Test all major functionality
    roc_fig = comparison.create_roc_pr_comparison(results)
    feature_fig = comparison.create_feature_importance_chart(results)
    attention_fig = comparison.create_attention_visualization(results)
    correlation_fig = comparison.create_prediction_correlation_analysis(results)
    metrics_table = comparison.create_performance_metrics_table(results)
    layout = comparison.get_dashboard_layout(results)
    
    # Basic assertions
    assert all(fig is not None for fig in [roc_fig, feature_fig, attention_fig, correlation_fig])
    assert metrics_table is not None
    assert layout is not None
    
    print("✓ Integration test passed")


if __name__ == "__main__":
    # Run basic tests
    test_model_comparison_integration()
    
    # Run pytest if available
    try:
        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not available, running basic integration test only")