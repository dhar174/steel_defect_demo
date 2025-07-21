"""
Visualization components for the steel defect detection dashboard.

This package contains modular dashboard components for different aspects
of the steel defect detection system visualization.

Prediction Visualization Components for Steel Defect Detection Dashboard.

This module provides components for visualizing defect prediction outputs.
"""

from .prediction_display import PredictionDisplayComponents, create_sample_data_for_demo
from .model_comparison import ModelComparison

__all__ = ['PredictionDisplayComponents', 'create_sample_data_for_demo','ModelComparison']