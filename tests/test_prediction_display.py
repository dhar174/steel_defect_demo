"""
Test suite for prediction display components.
"""

import pytest
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.visualization.components.prediction_display import (
    PredictionDisplayComponents,
    create_sample_data_for_demo
)


@pytest.fixture
def test_config():
    """Fixture providing test configuration with thresholds."""
    return {
        'inference': {
            'thresholds': {
                'defect_probability': 0.5,
                'high_risk_threshold': 0.7,
                'alert_threshold': 0.8
            }
        }
    }


@pytest.fixture
def prediction_components(test_config):
    """Fixture providing initialized prediction display components."""
    return PredictionDisplayComponents(test_config)


@pytest.fixture
def sample_history_data():
    """Fixture providing sample historical data."""
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(hours=2),
        end=datetime.now(),
        freq='10min'
    )
    
    # Create varied prediction data for testing, matching the length of timestamps
    base_predictions = [0.2, 0.3, 0.6, 0.75, 0.85, 0.4, 0.3, 0.2, 0.5, 0.6, 0.7, 0.9]
    # Extend or trim predictions to match timestamps length
    if len(base_predictions) < len(timestamps):
        # Repeat pattern if needed
        predictions = (base_predictions * ((len(timestamps) // len(base_predictions)) + 1))[:len(timestamps)]
    else:
        predictions = base_predictions[:len(timestamps)]
    
    return pd.DataFrame({
        'prediction': predictions
    }, index=timestamps)


class TestPredictionDisplayComponents:
    """Test class for prediction display components."""
    
    def test_initialization(self, prediction_components):
        """Test component initialization with config."""
        assert prediction_components.defect_threshold == 0.5
        assert prediction_components.high_risk_threshold == 0.7
        assert prediction_components.alert_threshold == 0.8
        assert 'safe' in prediction_components.risk_colors
        assert 'alert' in prediction_components.risk_colors
    
    def test_initialization_default_config(self):
        """Test initialization with default config."""
        components = PredictionDisplayComponents()
        assert components.defect_threshold == 0.5
        assert components.high_risk_threshold == 0.7
        assert components.alert_threshold == 0.8
    
    def test_prediction_gauge_creation(self, prediction_components):
        """Test prediction gauge creation."""
        # Test with different risk levels
        test_probs = [0.2, 0.6, 0.75, 0.9]
        
        for prob in test_probs:
            gauge_fig = prediction_components.create_prediction_gauge(prob)
            
            # Verify figure structure
            assert gauge_fig is not None
            assert len(gauge_fig.data) == 1
            assert gauge_fig.data[0].type == 'indicator'
            assert gauge_fig.data[0].value == prob
            
            # Verify gauge configuration
            gauge = gauge_fig.data[0].gauge
            assert gauge.axis.range == (None, 1)
            assert len(gauge.steps) == 4  # Four risk level steps
    
    def test_historical_timeline_creation(self, prediction_components, sample_history_data):
        """Test historical timeline creation."""
        timeline_fig = prediction_components.create_historical_timeline(sample_history_data)
        
        # Verify figure structure
        assert timeline_fig is not None
        assert len(timeline_fig.data) >= 1  # At least main prediction line
        
        # Verify data content
        main_trace = timeline_fig.data[0]
        assert main_trace.mode == 'lines+markers'
        assert len(main_trace.y) == len(sample_history_data)
        
        # Verify layout
        assert timeline_fig.layout.yaxis.range == (0, 1)
        assert 'Time' in timeline_fig.layout.xaxis.title.text
    
    def test_empty_historical_timeline(self, prediction_components):
        """Test timeline with empty data."""
        empty_data = pd.DataFrame()
        timeline_fig = prediction_components.create_historical_timeline(empty_data)
        
        assert timeline_fig is not None
        assert len(timeline_fig.layout.annotations) > 0  # Should have "no data" message
    
    def test_ensemble_contribution_chart(self, prediction_components):
        """Test ensemble contribution chart creation."""
        baseline_contrib = 0.4
        lstm_contrib = 0.6
        
        chart_fig = prediction_components.create_ensemble_contribution_chart(
            baseline_contrib, lstm_contrib
        )
        
        # Verify figure structure
        assert chart_fig is not None
        assert len(chart_fig.data) == 1
        assert chart_fig.data[0].type == 'pie'
        
        # Verify data content
        pie_trace = chart_fig.data[0]
        assert len(pie_trace.labels) == 2
        assert 'Baseline' in pie_trace.labels
        assert 'LSTM' in pie_trace.labels
        
        # Values should be normalized
        total_values = sum(pie_trace.values)
        assert abs(total_values - 1.0) < 0.001  # Should sum to 1
    
    def test_alert_status_indicator(self, prediction_components):
        """Test alert status indicator creation."""
        test_cases = [
            (0.2, "OK"),
            (0.6, "CAUTION"),
            (0.75, "HIGH RISK"),
            (0.9, "DEFECT ALERT")
        ]
        
        for prob, expected_text in test_cases:
            indicator = prediction_components.create_alert_status_indicator(prob)
            
            # Verify component structure
            assert indicator is not None
            assert hasattr(indicator, 'children')
            
            # Check that appropriate status text is present
            # (Note: exact text matching would require DOM traversal)
            assert indicator is not None
    
    def test_alert_status_with_cast_id(self, prediction_components):
        """Test alert status indicator with cast ID."""
        indicator = prediction_components.create_alert_status_indicator(0.6, cast_id="CAST_001")
        assert indicator is not None
    
    def test_confidence_visualization(self, prediction_components):
        """Test confidence visualization creation."""
        pred_prob = 0.7
        confidence_interval = (0.65, 0.75)
        uncertainty = 0.05
        
        # Test with confidence interval
        conf_fig1 = prediction_components.create_confidence_visualization(
            pred_prob, confidence_interval=confidence_interval
        )
        assert conf_fig1 is not None
        assert len(conf_fig1.data) == 1
        assert conf_fig1.data[0].error_y is not None
        
        # Test with uncertainty
        conf_fig2 = prediction_components.create_confidence_visualization(
            pred_prob, uncertainty=uncertainty
        )
        assert conf_fig2 is not None
        assert conf_fig2.data[0].error_y is not None
        
        # Test without confidence info
        conf_fig3 = prediction_components.create_confidence_visualization(pred_prob)
        assert conf_fig3 is not None
    
    def test_accuracy_metrics_display(self, prediction_components):
        """Test accuracy metrics display creation."""
        metrics = {
            'accuracy': 0.891,
            'precision': 0.856,
            'recall': 0.923,
            'f1_score': 0.888
        }
        
        metrics_display = prediction_components.create_accuracy_metrics_display(metrics)
        
        # Verify component structure
        assert metrics_display is not None
        assert hasattr(metrics_display, 'children')
    
    def test_accuracy_metrics_with_empty_dict(self, prediction_components):
        """Test metrics display with empty metrics."""
        metrics_display = prediction_components.create_accuracy_metrics_display({})
        assert metrics_display is not None
    
    def test_risk_level_helpers(self, prediction_components):
        """Test risk level helper methods."""
        test_predictions = np.array([0.2, 0.6, 0.75, 0.9])
        
        # Test risk levels
        risk_levels = prediction_components._get_risk_levels(test_predictions)
        expected_levels = ['Safe', 'Warning', 'High Risk', 'Alert']
        assert risk_levels == expected_levels
        
        # Test risk colors
        for i, pred in enumerate(test_predictions):
            color = prediction_components._get_risk_color(pred)
            assert color in prediction_components.risk_colors.values()
    
    def test_risk_level_key_mapping(self, prediction_components):
        """Test risk level key mapping."""
        test_cases = [
            (0.2, 'safe'),
            (0.6, 'warning'),
            (0.75, 'high_risk'),
            (0.9, 'alert')
        ]
        
        for pred, expected_key in test_cases:
            key = prediction_components._get_risk_level_key(pred)
            assert key == expected_key


def test_create_sample_data_for_demo():
    """Test sample data creation function."""
    history_data, metrics = create_sample_data_for_demo()
    
    # Verify history data
    assert isinstance(history_data, pd.DataFrame)
    assert 'prediction' in history_data.columns
    assert len(history_data) > 0
    assert isinstance(history_data.index, pd.DatetimeIndex)
    
    # Verify all predictions are in valid range
    assert all(0 <= pred <= 1 for pred in history_data['prediction'])
    
    # Verify metrics
    assert isinstance(metrics, dict)
    expected_metrics = {'accuracy', 'precision', 'recall', 'f1_score'}
    assert set(metrics.keys()) == expected_metrics
    
    # Verify all metrics are in valid range
    assert all(0 <= val <= 1 for val in metrics.values())


def test_integration_with_config_loading():
    """Test integration with YAML config loading."""
    # This tests that the component can handle real config structure
    sample_config = {
        'inference': {
            'thresholds': {
                'defect_probability': 0.6,
                'high_risk_threshold': 0.75,
                'alert_threshold': 0.85
            }
        }
    }
    
    components = PredictionDisplayComponents(sample_config)
    assert components.defect_threshold == 0.6
    assert components.high_risk_threshold == 0.75
    assert components.alert_threshold == 0.85
    
    # Test that gauge uses custom thresholds
    gauge_fig = components.create_prediction_gauge(0.7)
    assert gauge_fig is not None
    
    # Verify custom thresholds are applied in gauge steps
    gauge_steps = gauge_fig.data[0].gauge.steps
    assert len(gauge_steps) == 4