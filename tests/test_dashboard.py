"""
Test suite for the DefectMonitoringDashboard
"""

import pytest
import sys
import os
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.visualization.dashboard import DefectMonitoringDashboard

@pytest.fixture
def test_config():
    """Fixture providing test configuration."""
    return {
        'inference': {
            'dashboard_port': 8050
        },
        'refresh_interval': 5000
    }

@pytest.fixture 
def dashboard(test_config):
    """Fixture providing initialized dashboard."""
    return DefectMonitoringDashboard(test_config)

def test_dashboard_initialization(dashboard):
    """Test dashboard initialization."""
    assert dashboard.app is not None
    assert dashboard.default_theme == "plotly_white"
    assert dashboard.refresh_interval == 5000

def test_page_layouts(dashboard):
    """Test page layout creation."""
    pages = ['real-time-monitoring', 'model-comparison', 'historical-analysis', 'system-status']
    
    for page in pages:
        layout = dashboard.create_page_layout(page)
        assert layout is not None

def test_visualization_methods(dashboard):
    """Test visualization methods."""
    # Test sample sensor plot
    sensor_fig = dashboard.create_sample_sensor_plot()
    assert sensor_fig is not None
    assert 'data' in sensor_fig
    
    # Test prediction gauge
    gauge_fig = dashboard.create_prediction_gauge(0.7)
    assert gauge_fig is not None
    assert 'data' in gauge_fig
    
    # Test prediction history
    history_fig = dashboard.create_sample_prediction_history()
    assert history_fig is not None
    assert 'data' in history_fig
    
    # Test model comparison
    comparison_fig = dashboard.create_model_comparison_chart()
    assert comparison_fig is not None
    assert 'data' in comparison_fig
    
    # Test historical trends
    trends_fig = dashboard.create_historical_trends_chart()
    assert trends_fig is not None
    assert 'data' in trends_fig

def test_system_status(dashboard):
    """Test system status content creation."""
    status_content = dashboard.create_system_status_content(10)
    assert status_content is not None

def test_theme_support(dashboard):
    """Test theme support in visualizations."""
    themes = ['plotly_white', 'plotly_dark']
    
    for theme in themes:
        sensor_fig = dashboard.create_sample_sensor_plot(theme)
        # Check that template was applied - the figure should have template info
        assert sensor_fig.layout.template is not None
        
        gauge_fig = dashboard.create_prediction_gauge(0.5, theme)
        assert gauge_fig.layout.template is not None

def test_error_handling(dashboard):
    """Test error handling scenarios."""
    # Test with edge case values
    gauge_fig = dashboard.create_prediction_gauge(0.0)  # Minimum valid value
    assert gauge_fig is not None
    
    gauge_fig = dashboard.create_prediction_gauge(1.0)  # Maximum valid value
    assert gauge_fig is not None

def test_configuration_loading():
    """Test configuration loading from file."""
    config_path = Path('configs/inference_config.yaml')
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        dashboard = DefectMonitoringDashboard(config)
        assert dashboard is not None
        assert 'inference' in config

if __name__ == "__main__":
    pytest.main([__file__, "-v"])