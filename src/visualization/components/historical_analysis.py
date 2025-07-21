"""
Historical Data Analysis Components for Steel Defect Detection

This module provides comprehensive tools for offline analysis of historical casting data,
enabling users to explore past data, identify long-term trends, and understand defect root causes.

Features:
- Interactive data exploration with filtering and aggregation
- Statistical Process Control (SPC) charts for process monitoring
- Defect pattern analysis using clustering and dimensionality reduction
- Time-based correlation analysis between sensor readings
- Batch analysis for comparing multiple historical casts
- Export functionality for data and visualizations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import logging

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc

from scipy import stats
from scipy.cluster.vq import kmeans2, whiten
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import base64
import io
import os

logger = logging.getLogger(__name__)


class HistoricalAnalysisComponents:
    """
    Comprehensive historical data analysis tools for steel casting process.
    
    Provides statistical analysis, pattern recognition, and visualization tools
    for understanding long-term trends and defect patterns in historical data.
    """
    
    def __init__(self, 
                 component_id: str = "historical-analysis",
                 data_path: str = "data/processed/",
                 config: Optional[Dict] = None):
        """
        Initialize historical analysis components.
        
        Args:
            component_id: Unique identifier for the component
            data_path: Path to processed historical data directory
            config: Configuration dictionary for analysis parameters
        """
        self.component_id = component_id
        self.data_path = data_path
        self.config = config or {}
        
        # Analysis configuration
        self.spc_control_limits = self.config.get('spc_control_limits', 3)  # 3-sigma limits
        self.clustering_n_clusters = self.config.get('clustering_n_clusters', 5)
        self.pca_n_components = self.config.get('pca_n_components', 3)
        
        # Sensor columns (excluding defect and timestamp)
        self.sensor_columns = [
            'temperature_1', 'temperature_2', 'pressure_1', 'pressure_2',
            'flow_rate', 'casting_speed', 'steel_composition_c', 
            'steel_composition_si', 'steel_composition_mn', 'steel_composition_p',
            'steel_composition_s', 'humidity'
        ]
        
        # Color schemes for different analysis types
        self.defect_colors = {0: '#2E8B57', 1: '#DC143C'}  # Green for normal, Red for defect
        self.cluster_colors = px.colors.qualitative.Set3
        
        # Data cache for performance
        self._data_cache = {}
        self._last_cache_update = None
        
    def create_layout(self) -> html.Div:
        """
        Create the complete historical analysis dashboard layout.
        
        Returns:
            html.Div: Complete dashboard layout with all analysis tools
        """
        return html.Div([
            # Control Panel
            self._create_control_panel(),
            
            # Main Analysis Tabs
            dbc.Tabs([
                # Data Exploration Tab
                dbc.Tab(
                    label="Data Exploration",
                    tab_id="data-exploration",
                    children=[self._create_data_exploration_layout()]
                ),
                
                # SPC Charts Tab
                dbc.Tab(
                    label="SPC Charts",
                    tab_id="spc-charts", 
                    children=[self._create_spc_charts_layout()]
                ),
                
                # Pattern Analysis Tab
                dbc.Tab(
                    label="Pattern Analysis",
                    tab_id="pattern-analysis",
                    children=[self._create_pattern_analysis_layout()]
                ),
                
                # Correlation Analysis Tab
                dbc.Tab(
                    label="Correlation Analysis",
                    tab_id="correlation-analysis",
                    children=[self._create_correlation_analysis_layout()]
                ),
                
                # Batch Comparison Tab
                dbc.Tab(
                    label="Batch Comparison",
                    tab_id="batch-comparison",
                    children=[self._create_batch_comparison_layout()]
                )
            ], 
            id=f"{self.component_id}-main-tabs",
            active_tab="data-exploration",
            className="mt-3"
            ),
            
            # Hidden components for data storage and processing
            dcc.Store(id=f"{self.component_id}-data-store", data={}),
            dcc.Store(id=f"{self.component_id}-filtered-data-store", data={}),
            dcc.Store(id=f"{self.component_id}-analysis-cache", data={}),
            
            # Export components
            dcc.Download(id=f"{self.component_id}-download-data"),
            dcc.Download(id=f"{self.component_id}-download-chart"),
            
        ], id=f"{self.component_id}-container")
    
    def _create_control_panel(self) -> dbc.Card:
        """Create the main control panel for data loading and filtering."""
        return dbc.Card([
            dbc.CardHeader([
                html.H4("Historical Data Analysis Controls", className="mb-0"),
                html.Small("Load and filter historical casting data for analysis", className="text-muted")
            ]),
            dbc.CardBody([
                dbc.Row([
                    # Data Loading Controls
                    dbc.Col([
                        html.Label("Data Source", className="fw-bold"),
                        dbc.ButtonGroup([
                            dbc.Button("Load Sample Data", 
                                     id=f"{self.component_id}-load-sample",
                                     size="sm", color="primary"),
                            dbc.Button("Load Processed Data", 
                                     id=f"{self.component_id}-load-processed",
                                     size="sm", color="secondary"),
                            dbc.Button("Refresh", 
                                     id=f"{self.component_id}-refresh",
                                     size="sm", color="info")
                        ], size="sm")
                    ], md=3),
                    
                    # Date Range Filtering
                    dbc.Col([
                        html.Label("Date Range Filter", className="fw-bold"),
                        dcc.DatePickerRange(
                            id=f"{self.component_id}-date-range",
                            start_date=None,
                            end_date=None,
                            display_format='YYYY-MM-DD',
                            style={'width': '100%'}
                        )
                    ], md=3),
                    
                    # Cast ID Filtering
                    dbc.Col([
                        html.Label("Cast ID Filter", className="fw-bold"),
                        dcc.Dropdown(
                            id=f"{self.component_id}-cast-filter",
                            options=[],
                            value=None,
                            multi=True,
                            placeholder="Select cast IDs..."
                        )
                    ], md=3),
                    
                    # Defect Status Filtering
                    dbc.Col([
                        html.Label("Defect Status", className="fw-bold"),
                        dcc.Dropdown(
                            id=f"{self.component_id}-defect-filter",
                            options=[
                                {'label': 'All Data', 'value': 'all'},
                                {'label': 'Normal Only', 'value': 0},
                                {'label': 'Defects Only', 'value': 1}
                            ],
                            value='all'
                        )
                    ], md=3)
                ], className="mb-3"),
                
                dbc.Row([
                    # Aggregation Controls
                    dbc.Col([
                        html.Label("Data Aggregation", className="fw-bold"),
                        dbc.ButtonGroup([
                            dbc.Button("Raw Data", 
                                     id=f"{self.component_id}-agg-raw",
                                     size="sm", outline=True, color="primary", active=True),
                            dbc.Button("Hourly Avg", 
                                     id=f"{self.component_id}-agg-hourly",
                                     size="sm", outline=True, color="primary"),
                            dbc.Button("Daily Avg", 
                                     id=f"{self.component_id}-agg-daily",
                                     size="sm", outline=True, color="primary"),
                            dbc.Button("By Cast", 
                                     id=f"{self.component_id}-agg-cast",
                                     size="sm", outline=True, color="primary")
                        ], size="sm")
                    ], md=4),
                    
                    # Data Info Display
                    dbc.Col([
                        html.Div(id=f"{self.component_id}-data-info", 
                                className="small text-muted mt-2")
                    ], md=4),
                    
                    # Export Controls
                    dbc.Col([
                        html.Label("Export Data", className="fw-bold"),
                        dbc.ButtonGroup([
                            dbc.Button("📊 Export Chart", 
                                     id=f"{self.component_id}-export-chart",
                                     size="sm", color="success"),
                            dbc.Button("📋 Export Data", 
                                     id=f"{self.component_id}-export-data",
                                     size="sm", color="success")
                        ], size="sm")
                    ], md=4)
                ])
            ])
        ], className="mb-3")
    
    def _create_data_exploration_layout(self) -> html.Div:
        """Create the data exploration tab layout."""
        return html.Div([
            dbc.Row([
                # Data Overview Cards
                dbc.Col([
                    html.H5("Data Overview", className="mb-3"),
                    html.Div(id=f"{self.component_id}-data-overview-cards")
                ], md=4),
                
                # Data Distribution Plot
                dbc.Col([
                    html.H5("Feature Distributions", className="mb-3"),
                    dcc.Dropdown(
                        id=f"{self.component_id}-feature-select",
                        options=[{'label': col.replace('_', ' ').title(), 'value': col} 
                                for col in self.sensor_columns],
                        value=self.sensor_columns[0],
                        className="mb-3"
                    ),
                    dcc.Graph(
                        id=f"{self.component_id}-distribution-plot",
                        style={'height': '400px'}
                    )
                ], md=8)
            ], className="mb-4"),
            
            dbc.Row([
                # Time Series View
                dbc.Col([
                    html.H5("Time Series View", className="mb-3"),
                    dcc.Graph(
                        id=f"{self.component_id}-timeseries-plot",
                        style={'height': '500px'}
                    )
                ], width=12)
            ])
        ])
    
    def _create_spc_charts_layout(self) -> html.Div:
        """Create the SPC charts tab layout."""
        return html.Div([
            dbc.Row([
                # SPC Chart Controls
                dbc.Col([
                    html.H5("SPC Chart Configuration", className="mb-3"),
                    html.Label("Select Sensor", className="fw-bold"),
                    dcc.Dropdown(
                        id=f"{self.component_id}-spc-sensor-select",
                        options=[{'label': col.replace('_', ' ').title(), 'value': col} 
                                for col in self.sensor_columns],
                        value=self.sensor_columns[0],
                        className="mb-3"
                    ),
                    html.Label("Chart Type", className="fw-bold"),
                    dcc.Dropdown(
                        id=f"{self.component_id}-spc-chart-type",
                        options=[
                            {'label': 'Individual & Moving Range (I-MR)', 'value': 'i-mr'},
                            {'label': 'X-bar & Range (X-R)', 'value': 'x-r'},
                            {'label': 'Individual Chart Only', 'value': 'individual'},
                            {'label': 'Control Limits Summary', 'value': 'summary'}
                        ],
                        value='i-mr',
                        className="mb-3"
                    ),
                    html.Label("Subgroup Size (for X-R charts)", className="fw-bold"),
                    dbc.Input(
                        id=f"{self.component_id}-subgroup-size",
                        type="number",
                        value=5,
                        min=2,
                        max=10,
                        className="mb-3"
                    ),
                    html.Div(id=f"{self.component_id}-spc-statistics", className="mt-3")
                ], md=3),
                
                # SPC Charts Display
                dbc.Col([
                    html.H5("Statistical Process Control Charts", className="mb-3"),
                    dcc.Graph(
                        id=f"{self.component_id}-spc-charts",
                        style={'height': '600px'}
                    )
                ], md=9)
            ])
        ])
    
    def _create_pattern_analysis_layout(self) -> html.Div:
        """Create the pattern analysis tab layout."""
        return html.Div([
            dbc.Row([
                # Pattern Analysis Controls
                dbc.Col([
                    html.H5("Pattern Analysis Configuration", className="mb-3"),
                    html.Label("Number of Clusters", className="fw-bold"),
                    dcc.Slider(
                        id=f"{self.component_id}-n-clusters",
                        min=2,
                        max=10,
                        step=1,
                        value=self.clustering_n_clusters,
                        marks={i: str(i) for i in range(2, 11)},
                        className="mb-3"
                    ),
                    html.Label("PCA Components", className="fw-bold"),
                    dcc.Slider(
                        id=f"{self.component_id}-pca-components",
                        min=2,
                        max=min(len(self.sensor_columns), 5),
                        step=1,
                        value=self.pca_n_components,
                        marks={i: str(i) for i in range(2, min(len(self.sensor_columns), 5) + 1)},
                        className="mb-3"
                    ),
                    dbc.Button(
                        "Run Clustering Analysis",
                        id=f"{self.component_id}-run-clustering",
                        color="primary",
                        className="mb-3"
                    ),
                    html.Div(id=f"{self.component_id}-clustering-stats", className="mt-3")
                ], md=3),
                
                # Pattern Visualization
                dbc.Col([
                    html.H5("Defect Pattern Analysis", className="mb-3"),
                    dbc.Tabs([
                        dbc.Tab(
                            label="Cluster Visualization",
                            children=[
                                dcc.Graph(
                                    id=f"{self.component_id}-cluster-plot",
                                    style={'height': '500px'}
                                )
                            ]
                        ),
                        dbc.Tab(
                            label="Feature Importance",
                            children=[
                                dcc.Graph(
                                    id=f"{self.component_id}-feature-importance",
                                    style={'height': '500px'}
                                )
                            ]
                        )
                    ])
                ], md=9)
            ])
        ])
    
    def _create_correlation_analysis_layout(self) -> html.Div:
        """Create the correlation analysis tab layout."""
        return html.Div([
            dbc.Row([
                # Correlation Controls
                dbc.Col([
                    html.H5("Correlation Analysis", className="mb-3"),
                    html.Label("Analysis Type", className="fw-bold"),
                    dcc.Dropdown(
                        id=f"{self.component_id}-correlation-type",
                        options=[
                            {'label': 'Pearson Correlation', 'value': 'pearson'},
                            {'label': 'Spearman Correlation', 'value': 'spearman'},
                            {'label': 'Kendall Correlation', 'value': 'kendall'}
                        ],
                        value='pearson',
                        className="mb-3"
                    ),
                    html.Label("Time Window Analysis", className="fw-bold"),
                    dcc.Dropdown(
                        id=f"{self.component_id}-time-window",
                        options=[
                            {'label': 'Overall Correlation', 'value': 'overall'},
                            {'label': 'Rolling Window', 'value': 'rolling'},
                            {'label': 'Before/During Defects', 'value': 'defect_analysis'}
                        ],
                        value='overall',
                        className="mb-3"
                    ),
                    html.Div(id=f"{self.component_id}-correlation-stats", className="mt-3")
                ], md=3),
                
                # Correlation Visualization
                dbc.Col([
                    dbc.Tabs([
                        dbc.Tab(
                            label="Correlation Matrix",
                            children=[
                                dcc.Graph(
                                    id=f"{self.component_id}-correlation-heatmap",
                                    style={'height': '500px'}
                                )
                            ]
                        ),
                        dbc.Tab(
                            label="Scatter Plot Matrix",
                            children=[
                                html.Label("Select Features (max 6)", className="fw-bold mb-2"),
                                dcc.Dropdown(
                                    id=f"{self.component_id}-scatter-features",
                                    options=[{'label': col.replace('_', ' ').title(), 'value': col} 
                                            for col in self.sensor_columns],
                                    value=self.sensor_columns[:4],
                                    multi=True,
                                    className="mb-3"
                                ),
                                dcc.Graph(
                                    id=f"{self.component_id}-scatter-matrix",
                                    style={'height': '600px'}
                                )
                            ]
                        )
                    ])
                ], md=9)
            ])
        ])
    
    def _create_batch_comparison_layout(self) -> html.Div:
        """Create the batch comparison tab layout."""
        return html.Div([
            dbc.Row([
                # Batch Selection Controls
                dbc.Col([
                    html.H5("Batch Comparison", className="mb-3"),
                    html.Label("Select Batches to Compare", className="fw-bold"),
                    dcc.Dropdown(
                        id=f"{self.component_id}-batch-select",
                        options=[],
                        value=[],
                        multi=True,
                        placeholder="Select up to 4 batches...",
                        className="mb-3"
                    ),
                    html.Label("Comparison Metric", className="fw-bold"),
                    dcc.Dropdown(
                        id=f"{self.component_id}-comparison-metric",
                        options=[
                            {'label': 'All Sensors Overview', 'value': 'overview'},
                            {'label': 'Individual Sensor Trends', 'value': 'individual'},
                            {'label': 'Statistical Summary', 'value': 'statistics'},
                            {'label': 'Defect Prediction Timeline', 'value': 'prediction'}
                        ],
                        value='overview',
                        className="mb-3"
                    ),
                    html.Div(id=f"{self.component_id}-batch-stats", className="mt-3")
                ], md=3),
                
                # Batch Comparison Visualization
                dbc.Col([
                    html.H5("Batch Comparison Analysis", className="mb-3"),
                    dcc.Graph(
                        id=f"{self.component_id}-batch-comparison",
                        style={'height': '600px'}
                    )
                ], md=9)
            ])
        ])
    
    def load_sample_data(self) -> pd.DataFrame:
        """
        Load sample data for analysis.
        
        Returns:
            pd.DataFrame: Loaded and preprocessed sample data
        """
        try:
            # Load the sample data
            data_file = "data/examples/steel_defect_sample.csv"
            if os.path.exists(data_file):
                df = pd.read_csv(data_file)
                
                # Add synthetic timestamp and cast_id for demonstration
                start_time = datetime.now() - timedelta(days=30)
                timestamps = pd.date_range(
                    start=start_time, 
                    periods=len(df), 
                    freq='5min'
                )
                df['timestamp'] = timestamps
                
                # Create synthetic cast IDs (assuming ~100 data points per cast)
                cast_ids = []
                for i in range(len(df)):
                    cast_id = f"CAST_{(i // 100) + 1:03d}"
                    cast_ids.append(cast_id)
                df['cast_id'] = cast_ids
                
                logger.info(f"Loaded sample data: {len(df)} records, {len(df.cast_id.unique())} casts")
                return df
            else:
                logger.warning(f"Sample data file not found: {data_file}")
                return self._generate_synthetic_data()
                
        except Exception as e:
            logger.error(f"Error loading sample data: {e}")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self, n_samples: int = 2000) -> pd.DataFrame:
        """Generate synthetic data for demonstration purposes."""
        np.random.seed(42)  # For reproducible results
        
        # Generate timestamps over 30 days
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)
        timestamps = pd.date_range(start=start_time, periods=n_samples, freq='5min')
        
        # Generate synthetic sensor data
        data = {}
        
        # Temperature sensors (correlated, with daily cycles)
        base_temp1 = 1530 + 20 * np.sin(np.linspace(0, 30 * 2 * np.pi, n_samples))
        data['temperature_1'] = base_temp1 + np.random.normal(0, 10, n_samples)
        data['temperature_2'] = base_temp1 + np.random.normal(0, 8, n_samples) + 5
        
        # Pressure sensors (correlated)
        base_pressure = 10 + 2 * np.sin(np.linspace(0, 15 * 2 * np.pi, n_samples))
        data['pressure_1'] = base_pressure + np.random.normal(0, 1, n_samples)
        data['pressure_2'] = base_pressure + np.random.normal(0, 1.2, n_samples) + 0.5
        
        # Flow rate and casting speed (process parameters)
        data['flow_rate'] = 100 + np.random.normal(0, 5, n_samples)
        data['casting_speed'] = 1.2 + np.random.normal(0, 0.1, n_samples)
        
        # Steel composition (relatively stable)
        data['steel_composition_c'] = 0.4 + np.random.normal(0, 0.02, n_samples)
        data['steel_composition_si'] = 0.25 + np.random.normal(0, 0.01, n_samples)
        data['steel_composition_mn'] = 1.5 + np.random.normal(0, 0.05, n_samples)
        data['steel_composition_p'] = 0.02 + np.random.normal(0, 0.002, n_samples)
        data['steel_composition_s'] = 0.015 + np.random.normal(0, 0.002, n_samples)
        
        # Humidity (environmental factor)
        data['humidity'] = 60 + 20 * np.sin(np.linspace(0, 30 * 2 * np.pi, n_samples)) + np.random.normal(0, 5, n_samples)
        
        # Generate defects based on threshold conditions
        defects = np.zeros(n_samples)
        for i in range(n_samples):
            # Higher probability of defect if multiple conditions are met
            defect_score = 0
            if data['temperature_1'][i] > 1550 or data['temperature_1'][i] < 1510:
                defect_score += 1
            if data['pressure_1'][i] > 12 or data['pressure_1'][i] < 8:
                defect_score += 1
            if data['casting_speed'][i] > 1.4 or data['casting_speed'][i] < 1.0:
                defect_score += 1
            
            # Base 5% chance, increased based on conditions
            defect_prob = 0.05 + defect_score * 0.15
            defects[i] = 1 if np.random.random() < defect_prob else 0
        
        data['defect'] = defects
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = timestamps
        
        # Generate cast IDs
        cast_ids = []
        for i in range(len(df)):
            cast_id = f"CAST_{(i // 100) + 1:03d}"
            cast_ids.append(cast_id)
        df['cast_id'] = cast_ids
        
        logger.info(f"Generated synthetic data: {len(df)} records, {len(df.cast_id.unique())} casts")
        return df


# Utility functions for SPC calculations
def calculate_control_limits(data: np.ndarray, method: str = 'individual') -> Dict[str, float]:
    """
    Calculate control limits for SPC charts.
    
    Args:
        data: Array of measurement values
        method: Type of control chart ('individual', 'xbar', 'range')
        
    Returns:
        Dictionary containing control limits and statistics
    """
    if len(data) == 0:
        return {}
    
    if method == 'individual':
        # Individual chart (I-chart)
        mean_val = np.mean(data)
        moving_ranges = np.abs(np.diff(data))
        mean_mr = np.mean(moving_ranges)
        
        # Control limits for individual chart
        ucl = mean_val + 2.66 * mean_mr  # A2 constant for n=2
        lcl = mean_val - 2.66 * mean_mr
        
        return {
            'center_line': mean_val,
            'ucl': ucl,
            'lcl': lcl,
            'mean_mr': mean_mr,
            'sigma_est': mean_mr / 1.128  # d2 constant for n=2
        }
    
    elif method == 'moving_range':
        # Moving Range chart (MR-chart)
        moving_ranges = np.abs(np.diff(data))
        mean_mr = np.mean(moving_ranges)
        
        ucl_mr = 3.27 * mean_mr  # D4 constant for n=2
        lcl_mr = 0  # D3 constant for n=2
        
        return {
            'center_line': mean_mr,
            'ucl': ucl_mr,
            'lcl': lcl_mr
        }
    
    elif method == 'xbar':
        # X-bar chart for subgroups
        subgroup_size = 5  # Default subgroup size
        n_subgroups = len(data) // subgroup_size
        
        if n_subgroups == 0:
            return calculate_control_limits(data, 'individual')
        
        # Calculate subgroup means
        subgroup_means = []
        for i in range(n_subgroups):
            start_idx = i * subgroup_size
            end_idx = start_idx + subgroup_size
            subgroup_means.append(np.mean(data[start_idx:end_idx]))
        
        grand_mean = np.mean(subgroup_means)
        
        # Estimate sigma from subgroup ranges
        subgroup_ranges = []
        for i in range(n_subgroups):
            start_idx = i * subgroup_size
            end_idx = start_idx + subgroup_size
            subgroup_ranges.append(np.max(data[start_idx:end_idx]) - np.min(data[start_idx:end_idx]))
        
        mean_range = np.mean(subgroup_ranges)
        
        # Control chart constants for n=5
        A2 = 0.577
        D3 = 0
        D4 = 2.115
        
        ucl_xbar = grand_mean + A2 * mean_range
        lcl_xbar = grand_mean - A2 * mean_range
        
        return {
            'center_line': grand_mean,
            'ucl': ucl_xbar,
            'lcl': lcl_xbar,
            'mean_range': mean_range,
            'subgroup_means': subgroup_means,
            'subgroup_ranges': subgroup_ranges
        }
    
    return {}


def detect_spc_violations(data: np.ndarray, control_limits: Dict[str, float]) -> List[int]:
    """
    Detect SPC rule violations in data.
    
    Args:
        data: Array of measurement values
        control_limits: Control limits from calculate_control_limits
        
    Returns:
        List of indices where violations occur
    """
    violations = []
    
    if 'ucl' not in control_limits or 'lcl' not in control_limits:
        return violations
    
    ucl = control_limits['ucl']
    lcl = control_limits['lcl']
    center_line = control_limits.get('center_line', np.mean(data))
    
    for i, value in enumerate(data):
        # Rule 1: Point beyond control limits
        if value > ucl or value < lcl:
            violations.append(i)
        
        # Rule 2: 7 consecutive points on same side of center line
        if i >= 6:
            recent_points = data[i-6:i+1]
            if all(p > center_line for p in recent_points) or all(p < center_line for p in recent_points):
                violations.extend(range(i-6, i+1))
        
        # Rule 3: 2 out of 3 consecutive points beyond 2-sigma
        if i >= 2:
            sigma_est = control_limits.get('sigma_est', (ucl - lcl) / 6)
            upper_2sigma = center_line + 2 * sigma_est
            lower_2sigma = center_line - 2 * sigma_est
            
            recent_points = data[i-2:i+1]
            beyond_2sigma = sum(1 for p in recent_points if p > upper_2sigma or p < lower_2sigma)
            
            if beyond_2sigma >= 2:
                violations.extend(range(i-2, i+1))
    
    return list(set(violations))  # Remove duplicates


# Visualization methods for HistoricalAnalysisComponents class
def create_data_overview_cards(self, df: pd.DataFrame) -> html.Div:
    """Create overview cards showing data summary statistics."""
    if df.empty:
        return html.Div("No data available", className="text-muted")
    
    total_records = len(df)
    defect_rate = (df['defect'].sum() / len(df)) * 100 if 'defect' in df.columns else 0
    date_range = f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}" if 'timestamp' in df.columns else "N/A"
    unique_casts = df['cast_id'].nunique() if 'cast_id' in df.columns else 0
    
    cards = [
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{total_records:,}", className="text-primary"),
                html.P("Total Records", className="mb-0")
            ])
        ], className="mb-2"),
        
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{defect_rate:.1f}%", className="text-danger"),
                html.P("Defect Rate", className="mb-0")
            ])
        ], className="mb-2"),
        
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{unique_casts}", className="text-info"),
                html.P("Unique Casts", className="mb-0")
            ])
        ], className="mb-2"),
        
        dbc.Card([
            dbc.CardBody([
                html.P("Date Range", className="mb-1 font-weight-bold"),
                html.Small(date_range, className="text-muted")
            ])
        ], className="mb-2")
    ]
    
    return html.Div(cards)


def create_distribution_plot(self, df: pd.DataFrame, feature: str) -> go.Figure:
    """Create distribution plot for selected feature."""
    if df.empty or feature not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected feature",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Distribution by Defect Status", "Box Plot Comparison"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Histogram by defect status
    for defect_status in [0, 1]:
        data_subset = df[df['defect'] == defect_status][feature]
        fig.add_trace(
            go.Histogram(
                x=data_subset,
                name=f"Normal" if defect_status == 0 else "Defect",
                opacity=0.7,
                marker_color=self.defect_colors[defect_status],
                nbinsx=30
            ),
            row=1, col=1
        )
    
    # Box plot comparison
    fig.add_trace(
        go.Box(
            y=df[df['defect'] == 0][feature],
            name="Normal",
            marker_color=self.defect_colors[0]
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Box(
            y=df[df['defect'] == 1][feature],
            name="Defect", 
            marker_color=self.defect_colors[1]
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f"Distribution Analysis: {feature.replace('_', ' ').title()}",
        showlegend=True,
        template="plotly_white",
        height=400
    )
    
    return fig


def create_timeseries_plot(self, df: pd.DataFrame, features: List[str] = None) -> go.Figure:
    """Create time series plot for selected features."""
    if df.empty or 'timestamp' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No time series data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    if features is None:
        features = self.sensor_columns[:4]  # Default to first 4 sensors
    
    fig = make_subplots(
        rows=len(features), cols=1,
        shared_xaxes=True,
        subplot_titles=[f.replace('_', ' ').title() for f in features],
        vertical_spacing=0.05
    )
    
    for i, feature in enumerate(features):
        if feature in df.columns:
            # Normal data points
            normal_data = df[df['defect'] == 0]
            fig.add_trace(
                go.Scatter(
                    x=normal_data['timestamp'],
                    y=normal_data[feature],
                    mode='lines+markers',
                    name=f"{feature} (Normal)",
                    line=dict(color=self.defect_colors[0], width=1),
                    marker=dict(size=3),
                    showlegend=(i == 0)
                ),
                row=i+1, col=1
            )
            
            # Defect data points
            defect_data = df[df['defect'] == 1]
            if not defect_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=defect_data['timestamp'],
                        y=defect_data[feature],
                        mode='markers',
                        name=f"{feature} (Defect)",
                        marker=dict(color=self.defect_colors[1], size=6, symbol='x'),
                        showlegend=(i == 0)
                    ),
                    row=i+1, col=1
                )
    
    fig.update_layout(
        title="Time Series Analysis - Sensor Readings",
        template="plotly_white",
        height=500,
        hovermode='x unified'
    )
    
    return fig


def create_spc_charts(self, df: pd.DataFrame, sensor: str, chart_type: str, subgroup_size: int = 5) -> go.Figure:
    """Create Statistical Process Control charts."""
    if df.empty or sensor not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for SPC analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    data = df[sensor].values
    timestamps = df['timestamp'].tolist() if 'timestamp' in df.columns else list(range(len(data)))
    
    if chart_type == 'i-mr':
        # Individual and Moving Range charts
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Individual Chart", "Moving Range Chart"),
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        # Individual chart
        control_limits_i = calculate_control_limits(data, 'individual')
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=data,
                mode='lines+markers',
                name="Individual Values",
                line=dict(color='blue'),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # Control limits for individual chart
        if control_limits_i:
            fig.add_hline(y=control_limits_i['center_line'], line_dash="solid", 
                         line_color="green", row=1, col=1)
            fig.add_hline(y=control_limits_i['ucl'], line_dash="dash", 
                         line_color="red", row=1, col=1)
            fig.add_hline(y=control_limits_i['lcl'], line_dash="dash", 
                         line_color="red", row=1, col=1)
            
            # Highlight violations
            violations = detect_spc_violations(data, control_limits_i)
            if violations:
                fig.add_trace(
                    go.Scatter(
                        x=[timestamps[i] for i in violations],
                        y=[data[i] for i in violations],
                        mode='markers',
                        name="Violations",
                        marker=dict(color='red', size=8, symbol='x')
                    ),
                    row=1, col=1
                )
        
        # Moving Range chart
        moving_ranges = np.abs(np.diff(data))
        control_limits_mr = calculate_control_limits(data, 'moving_range')
        
        fig.add_trace(
            go.Scatter(
                x=timestamps[1:],  # One less point for moving range
                y=moving_ranges,
                mode='lines+markers',
                name="Moving Range",
                line=dict(color='orange'),
                marker=dict(size=4)
            ),
            row=2, col=1
        )
        
        if control_limits_mr:
            fig.add_hline(y=control_limits_mr['center_line'], line_dash="solid", 
                         line_color="green", row=2, col=1)
            fig.add_hline(y=control_limits_mr['ucl'], line_dash="dash", 
                         line_color="red", row=2, col=1)
    
    elif chart_type == 'x-r':
        # X-bar and Range charts
        control_limits_xbar = calculate_control_limits(data, 'xbar')
        
        if not control_limits_xbar:
            return self.create_spc_charts(df, sensor, 'individual')  # Fallback
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("X-bar Chart", "Range Chart"),
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        subgroup_means = control_limits_xbar['subgroup_means']
        subgroup_ranges = control_limits_xbar['subgroup_ranges']
        subgroup_indices = range(len(subgroup_means))
        
        # X-bar chart
        fig.add_trace(
            go.Scatter(
                x=subgroup_indices,
                y=subgroup_means,
                mode='lines+markers',
                name="Subgroup Means",
                line=dict(color='blue'),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        fig.add_hline(y=control_limits_xbar['center_line'], line_dash="solid", 
                     line_color="green", row=1, col=1)
        fig.add_hline(y=control_limits_xbar['ucl'], line_dash="dash", 
                     line_color="red", row=1, col=1)
        fig.add_hline(y=control_limits_xbar['lcl'], line_dash="dash", 
                     line_color="red", row=1, col=1)
        
        # Range chart
        fig.add_trace(
            go.Scatter(
                x=subgroup_indices,
                y=subgroup_ranges,
                mode='lines+markers',
                name="Subgroup Ranges",
                line=dict(color='orange'),
                marker=dict(size=6)
            ),
            row=2, col=1
        )
        
        fig.add_hline(y=control_limits_xbar['mean_range'], line_dash="solid", 
                     line_color="green", row=2, col=1)
        
    elif chart_type == 'individual':
        # Individual chart only
        fig = go.Figure()
        
        control_limits = calculate_control_limits(data, 'individual')
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=data,
                mode='lines+markers',
                name="Individual Values",
                line=dict(color='blue'),
                marker=dict(size=4)
            )
        )
        
        if control_limits:
            fig.add_hline(y=control_limits['center_line'], line_dash="solid", 
                         line_color="green", annotation_text="Center Line")
            fig.add_hline(y=control_limits['ucl'], line_dash="dash", 
                         line_color="red", annotation_text="UCL")
            fig.add_hline(y=control_limits['lcl'], line_dash="dash", 
                         line_color="red", annotation_text="LCL")
            
            # Highlight violations
            violations = detect_spc_violations(data, control_limits)
            if violations:
                fig.add_trace(
                    go.Scatter(
                        x=[timestamps[i] for i in violations],
                        y=[data[i] for i in violations],
                        mode='markers',
                        name="Violations",
                        marker=dict(color='red', size=8, symbol='x')
                    )
                )
    
    elif chart_type == 'summary':
        # Control limits summary table
        control_limits = calculate_control_limits(data, 'individual')
        
        if control_limits:
            summary_data = [
                ["Center Line", f"{control_limits['center_line']:.3f}"],
                ["Upper Control Limit", f"{control_limits['ucl']:.3f}"],
                ["Lower Control Limit", f"{control_limits['lcl']:.3f}"],
                ["Estimated Sigma", f"{control_limits['sigma_est']:.3f}"],
                ["Mean Moving Range", f"{control_limits['mean_mr']:.3f}"]
            ]
            
            fig = go.Figure(data=[go.Table(
                header=dict(values=['Statistic', 'Value'],
                           fill_color='lightblue',
                           align='center'),
                cells=dict(values=list(zip(*summary_data)),
                          fill_color='white',
                          align='center'))
            ])
        else:
            fig = go.Figure()
            fig.add_annotation(
                text="Unable to calculate control limits",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
    
    fig.update_layout(
        title=f"SPC Analysis: {sensor.replace('_', ' ').title()}",
        template="plotly_white",
        height=600
    )
    
    return fig


def create_clustering_analysis(self, df: pd.DataFrame, n_clusters: int, n_components: int) -> Tuple[go.Figure, Dict]:
    """Create clustering analysis visualization."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for clustering analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig, {}
    
    # Prepare data for clustering
    feature_data = df[self.sensor_columns].fillna(df[self.sensor_columns].mean())
    
    # Standardize features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(feature_data)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=min(n_components, len(self.sensor_columns)))
    pca_data = pca.fit_transform(scaled_data)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # Create 3D scatter plot if n_components >= 3, otherwise 2D
    if n_components >= 3:
        fig = go.Figure(data=[
            go.Scatter3d(
                x=pca_data[cluster_labels == i, 0],
                y=pca_data[cluster_labels == i, 1],
                z=pca_data[cluster_labels == i, 2],
                mode='markers',
                name=f'Cluster {i}',
                marker=dict(
                    size=5,
                    color=self.cluster_colors[i % len(self.cluster_colors)]
                )
            ) for i in range(n_clusters)
        ])
        
        # Add defect information
        defect_mask = df['defect'] == 1
        if defect_mask.any():
            defect_pca = pca_data[defect_mask]
            fig.add_trace(
                go.Scatter3d(
                    x=defect_pca[:, 0],
                    y=defect_pca[:, 1],
                    z=defect_pca[:, 2],
                    mode='markers',
                    name='Defects',
                    marker=dict(
                        size=8,
                        color='red',
                        symbol='x',
                        line=dict(width=2, color='darkred')
                    )
                )
            )
        
        fig.update_layout(
            scene=dict(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)',
                zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.1%} var)'
            )
        )
    
    else:
        fig = go.Figure()
        
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            fig.add_trace(
                go.Scatter(
                    x=pca_data[cluster_mask, 0],
                    y=pca_data[cluster_mask, 1],
                    mode='markers',
                    name=f'Cluster {i}',
                    marker=dict(
                        size=8,
                        color=self.cluster_colors[i % len(self.cluster_colors)]
                    )
                )
            )
        
        # Add defect information
        defect_mask = df['defect'] == 1
        if defect_mask.any():
            defect_pca = pca_data[defect_mask]
            fig.add_trace(
                go.Scatter(
                    x=defect_pca[:, 0],
                    y=defect_pca[:, 1],
                    mode='markers',
                    name='Defects',
                    marker=dict(
                        size=12,
                        color='red',
                        symbol='x',
                        line=dict(width=2, color='darkred')
                    )
                )
            )
        
        fig.update_layout(
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)'
        )
    
    fig.update_layout(
        title="Defect Pattern Clustering Analysis",
        template="plotly_white",
        height=500
    )
    
    # Calculate clustering statistics
    stats = {
        'n_clusters': n_clusters,
        'silhouette_score': None,  # Could add sklearn.metrics.silhouette_score
        'explained_variance': pca.explained_variance_ratio_[:n_components].sum(),
        'cluster_sizes': [np.sum(cluster_labels == i) for i in range(n_clusters)],
        'defects_per_cluster': [np.sum((cluster_labels == i) & (df['defect'] == 1)) for i in range(n_clusters)]
    }
    
    return fig, stats


def create_feature_importance_plot(self, df: pd.DataFrame, n_components: int) -> go.Figure:
    """Create feature importance plot from PCA analysis."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for feature importance analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Prepare data
    feature_data = df[self.sensor_columns].fillna(df[self.sensor_columns].mean())
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(feature_data)
    
    # PCA analysis
    pca = PCA(n_components=min(n_components, len(self.sensor_columns)))
    pca.fit(scaled_data)
    
    # Create heatmap of feature loadings
    components_df = pd.DataFrame(
        pca.components_[:n_components],
        columns=[col.replace('_', ' ').title() for col in self.sensor_columns],
        index=[f'PC{i+1}' for i in range(n_components)]
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=components_df.values,
        x=components_df.columns,
        y=components_df.index,
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title="Loading"),
        hovertemplate='Feature: %{x}<br>Component: %{y}<br>Loading: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Feature Importance in Principal Components",
        xaxis_title="Features",
        yaxis_title="Principal Components",
        template="plotly_white",
        height=500
    )
    
    return fig


def create_correlation_heatmap(self, df: pd.DataFrame, method: str = 'pearson') -> go.Figure:
    """Create correlation heatmap for sensor features."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for correlation analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Calculate correlation matrix
    corr_matrix = df[self.sensor_columns].corr(method=method)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[col.replace('_', ' ').title() for col in corr_matrix.columns],
        y=[col.replace('_', ' ').title() for col in corr_matrix.index],
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(title=f"{method.title()} Correlation"),
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    # Add correlation values as text
    annotations = []
    for i, row in enumerate(corr_matrix.values):
        for j, value in enumerate(row):
            annotations.append(
                dict(
                    x=j, y=i,
                    text=f"{value:.2f}",
                    showarrow=False,
                    font=dict(color="white" if abs(value) > 0.5 else "black")
                )
            )
    
    fig.update_layout(
        title=f"{method.title()} Correlation Matrix - Sensor Features",
        annotations=annotations,
        template="plotly_white",
        height=500
    )
    
    return fig


def create_scatter_matrix(self, df: pd.DataFrame, features: List[str]) -> go.Figure:
    """Create scatter plot matrix for selected features."""
    if df.empty or not features:
        fig = go.Figure()
        fig.add_annotation(
            text="No data or features selected for scatter matrix",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Limit features to avoid overcrowding
    features = features[:6]
    
    # Create scatter matrix using plotly express
    fig = px.scatter_matrix(
        df,
        dimensions=features,
        color='defect',
        color_discrete_map={0: self.defect_colors[0], 1: self.defect_colors[1]},
        title="Scatter Plot Matrix - Feature Relationships",
        labels={col: col.replace('_', ' ').title() for col in features}
    )
    
    fig.update_layout(
        template="plotly_white",
        height=600,
        showlegend=True
    )
    
    return fig


def create_batch_comparison(self, df: pd.DataFrame, selected_batches: List[str], 
                          comparison_type: str) -> go.Figure:
    """Create batch comparison visualization."""
    if df.empty or not selected_batches or 'cast_id' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No batches selected or cast_id column missing",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Filter data for selected batches
    batch_data = df[df['cast_id'].isin(selected_batches)]
    
    if comparison_type == 'overview':
        # Multi-sensor overview for selected batches
        n_sensors = min(6, len(self.sensor_columns))
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[col.replace('_', ' ').title() for col in self.sensor_columns[:n_sensors]],
            vertical_spacing=0.1,
            horizontal_spacing=0.08
        )
        
        colors = px.colors.qualitative.Set1[:len(selected_batches)]
        
        for i, sensor in enumerate(self.sensor_columns[:n_sensors]):
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            for j, batch_id in enumerate(selected_batches):
                batch_subset = batch_data[batch_data['cast_id'] == batch_id]
                if not batch_subset.empty and sensor in batch_subset.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=list(batch_subset.index),  # Convert to list
                            y=batch_subset[sensor],
                            mode='lines',
                            name=batch_id if i == 0 else None,  # Show legend only for first subplot
                            line=dict(color=colors[j % len(colors)]),
                            showlegend=(i == 0),
                            hovertemplate=f'{batch_id}<br>{sensor}: %{{y:.2f}}<extra></extra>'
                        ),
                        row=row, col=col
                    )
        
        fig.update_layout(
            title=f"Batch Overview Comparison: {', '.join(selected_batches)}",
            template="plotly_white",
            height=600
        )
    
    elif comparison_type == 'individual':
        # Individual sensor detailed comparison
        sensor = self.sensor_columns[0]  # Default to first sensor, could be made configurable
        
        fig = go.Figure()
        colors = px.colors.qualitative.Set1[:len(selected_batches)]
        
        for i, batch_id in enumerate(selected_batches):
            batch_subset = batch_data[batch_data['cast_id'] == batch_id]
            if not batch_subset.empty and sensor in batch_subset.columns:
                fig.add_trace(
                    go.Scatter(
                        x=batch_subset['timestamp'].tolist() if 'timestamp' in batch_subset.columns else list(batch_subset.index),
                        y=batch_subset[sensor],
                        mode='lines+markers',
                        name=batch_id,
                        line=dict(color=colors[i % len(colors)]),
                        marker=dict(size=4)
                    )
                )
                
                # Add defect markers
                defect_points = batch_subset[batch_subset['defect'] == 1]
                if not defect_points.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=defect_points['timestamp'].tolist() if 'timestamp' in defect_points.columns else list(defect_points.index),
                            y=defect_points[sensor],
                            mode='markers',
                            name=f'{batch_id} Defects',
                            marker=dict(color=colors[i % len(colors)], size=10, symbol='x'),
                            showlegend=False
                        )
                    )
        
        fig.update_layout(
            title=f"Individual Sensor Comparison: {sensor.replace('_', ' ').title()}",
            xaxis_title="Time",
            yaxis_title=sensor.replace('_', ' ').title(),
            template="plotly_white",
            height=600
        )
    
    elif comparison_type == 'statistics':
        # Statistical summary comparison
        stats_data = []
        
        for batch_id in selected_batches:
            batch_subset = batch_data[batch_data['cast_id'] == batch_id]
            if not batch_subset.empty:
                batch_stats = {
                    'Batch ID': batch_id,
                    'Records': len(batch_subset),
                    'Defect Rate (%)': (batch_subset['defect'].sum() / len(batch_subset)) * 100,
                    'Avg Temperature 1': batch_subset['temperature_1'].mean() if 'temperature_1' in batch_subset.columns else 0,
                    'Avg Pressure 1': batch_subset['pressure_1'].mean() if 'pressure_1' in batch_subset.columns else 0,
                    'Avg Casting Speed': batch_subset['casting_speed'].mean() if 'casting_speed' in batch_subset.columns else 0
                }
                stats_data.append(batch_stats)
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=list(stats_df.columns),
                    fill_color='lightblue',
                    align='center',
                    font=dict(size=12)
                ),
                cells=dict(
                    values=[stats_df[col] for col in stats_df.columns],
                    fill_color='white',
                    align='center',
                    format=[None, None, '.1f', '.2f', '.2f', '.3f']
                )
            )])
            
            fig.update_layout(
                title="Batch Statistical Summary Comparison",
                height=400
            )
        else:
            fig = go.Figure()
            fig.add_annotation(
                text="No statistics available for selected batches",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
    
    else:  # comparison_type == 'prediction'
        # Simulated prediction timeline comparison
        fig = go.Figure()
        colors = px.colors.qualitative.Set1[:len(selected_batches)]
        
        for i, batch_id in enumerate(selected_batches):
            batch_subset = batch_data[batch_data['cast_id'] == batch_id]
            if not batch_subset.empty:
                # Generate synthetic prediction probabilities based on defect status
                predictions = []
                for _, row in batch_subset.iterrows():
                    if row['defect'] == 1:
                        pred = np.random.beta(8, 2)  # Higher probability for defects
                    else:
                        pred = np.random.beta(2, 8)  # Lower probability for normal
                    predictions.append(pred)
                
                fig.add_trace(
                    go.Scatter(
                        x=batch_subset['timestamp'].tolist() if 'timestamp' in batch_subset.columns else list(batch_subset.index),
                        y=predictions,
                        mode='lines',
                        name=f'{batch_id} Prediction',
                        line=dict(color=colors[i % len(colors)])
                    )
                )
        
        # Add defect threshold line
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                     annotation_text="Defect Threshold")
        
        fig.update_layout(
            title="Prediction Timeline Comparison",
            xaxis_title="Time",
            yaxis_title="Defect Probability",
            yaxis=dict(range=[0, 1]),
            template="plotly_white",
            height=600
        )
    
    return fig


# Add these methods to the HistoricalAnalysisComponents class
HistoricalAnalysisComponents.create_data_overview_cards = create_data_overview_cards
HistoricalAnalysisComponents.create_distribution_plot = create_distribution_plot
HistoricalAnalysisComponents.create_timeseries_plot = create_timeseries_plot
HistoricalAnalysisComponents.create_spc_charts = create_spc_charts
HistoricalAnalysisComponents.create_clustering_analysis = create_clustering_analysis
HistoricalAnalysisComponents.create_feature_importance_plot = create_feature_importance_plot
HistoricalAnalysisComponents.create_correlation_heatmap = create_correlation_heatmap
HistoricalAnalysisComponents.create_scatter_matrix = create_scatter_matrix
HistoricalAnalysisComponents.create_batch_comparison = create_batch_comparison


# Export functionality methods
def export_data_to_csv(self, df: pd.DataFrame, filename: str = None) -> Dict[str, str]:
    """
    Export filtered data to CSV format.
    
    Args:
        df: DataFrame to export
        filename: Optional filename override
        
    Returns:
        Dictionary with content and filename for download
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"historical_analysis_data_{timestamp}.csv"
    
    # Create CSV content
    output = io.StringIO()
    df.to_csv(output, index=False)
    csv_content = output.getvalue()
    output.close()
    
    return {
        'content': csv_content,
        'filename': filename,
        'type': 'text/csv'
    }


def export_chart_to_image(self, fig: go.Figure, filename: str = None) -> Dict[str, str]:
    """
    Export chart to PNG format.
    
    Args:
        fig: Plotly figure to export
        filename: Optional filename override
        
    Returns:
        Dictionary with content and filename for download
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"historical_analysis_chart_{timestamp}.png"
    
    try:
        # Try to convert figure to image using kaleido
        img_bytes = fig.to_image(format="png", width=1200, height=800)
        
        # Encode as base64 for download
        img_b64 = base64.b64encode(img_bytes).decode()
        
        return {
            'content': img_b64,
            'filename': filename,
            'type': 'image/png'
        }
    except Exception as e:
        # Fallback: export as HTML
        logger.warning(f"Image export failed ({e}), falling back to HTML export")
        html_content = fig.to_html(include_plotlyjs='inline')
        
        return {
            'content': html_content,
            'filename': filename.replace('.png', '.html'),
            'type': 'text/html'
        }


def filter_data(self, df: pd.DataFrame, 
                date_range: Optional[List[str]] = None,
                cast_ids: Optional[List[str]] = None,
                defect_filter: Union[str, int] = 'all',
                aggregation: str = 'raw') -> pd.DataFrame:
    """
    Apply filters and aggregation to the data.
    
    Args:
        df: Source DataFrame
        date_range: List of [start_date, end_date] strings
        cast_ids: List of cast IDs to include
        defect_filter: 'all', 0 (normal), or 1 (defects)
        aggregation: 'raw', 'hourly', 'daily', or 'cast'
        
    Returns:
        Filtered and potentially aggregated DataFrame
    """
    filtered_df = df.copy()
    
    # Apply date range filter
    if date_range and len(date_range) == 2 and 'timestamp' in filtered_df.columns:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        filtered_df = filtered_df[
            (filtered_df['timestamp'] >= start_date) & 
            (filtered_df['timestamp'] <= end_date)
        ]
    
    # Apply cast ID filter
    if cast_ids and 'cast_id' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['cast_id'].isin(cast_ids)]
    
    # Apply defect status filter
    if defect_filter != 'all' and 'defect' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['defect'] == defect_filter]
    
    # Apply aggregation
    if aggregation != 'raw' and not filtered_df.empty:
        if aggregation == 'hourly' and 'timestamp' in filtered_df.columns:
            filtered_df = filtered_df.groupby([
                filtered_df['timestamp'].dt.floor('H'),
                'cast_id'
            ]).agg({
                **{col: 'mean' for col in self.sensor_columns if col in filtered_df.columns},
                'defect': 'max'  # Max defect status in the hour
            }).reset_index()
            
        elif aggregation == 'daily' and 'timestamp' in filtered_df.columns:
            filtered_df = filtered_df.groupby([
                filtered_df['timestamp'].dt.date,
                'cast_id'
            ]).agg({
                **{col: 'mean' for col in self.sensor_columns if col in filtered_df.columns},
                'defect': 'max'  # Max defect status in the day
            }).reset_index()
            
        elif aggregation == 'cast' and 'cast_id' in filtered_df.columns:
            filtered_df = filtered_df.groupby('cast_id').agg({
                **{col: 'mean' for col in self.sensor_columns if col in filtered_df.columns},
                'defect': 'max',  # Max defect status for the cast
                'timestamp': 'first'  # Keep first timestamp
            }).reset_index()
    
    return filtered_df


def get_data_info_summary(self, df: pd.DataFrame) -> str:
    """
    Generate a summary string of current data state.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Summary string for display
    """
    if df.empty:
        return "No data loaded"
    
    info_parts = []
    info_parts.append(f"{len(df):,} records")
    
    if 'defect' in df.columns:
        defect_count = df['defect'].sum()
        defect_rate = (defect_count / len(df)) * 100
        info_parts.append(f"{defect_count} defects ({defect_rate:.1f}%)")
    
    if 'cast_id' in df.columns:
        unique_casts = df['cast_id'].nunique()
        info_parts.append(f"{unique_casts} casts")
    
    if 'timestamp' in df.columns:
        date_range = df['timestamp'].max() - df['timestamp'].min()
        info_parts.append(f"{date_range.days} days range")
    
    return " | ".join(info_parts)


def create_spc_statistics_summary(self, df: pd.DataFrame, sensor: str) -> html.Div:
    """
    Create SPC statistics summary display.
    
    Args:
        df: Source DataFrame
        sensor: Sensor name to analyze
        
    Returns:
        html.Div with statistics summary
    """
    if df.empty or sensor not in df.columns:
        return html.Div("No data available for SPC statistics", className="text-muted")
    
    data = df[sensor].values
    control_limits = calculate_control_limits(data, 'individual')
    violations = detect_spc_violations(data, control_limits)
    
    if not control_limits:
        return html.Div("Unable to calculate SPC statistics", className="text-muted")
    
    stats_cards = [
        dbc.Card([
            dbc.CardBody([
                html.H6("Center Line", className="card-title"),
                html.H5(f"{control_limits['center_line']:.3f}", className="text-primary mb-0")
            ])
        ], className="mb-2"),
        
        dbc.Card([
            dbc.CardBody([
                html.H6("Control Limits", className="card-title"),
                html.P([
                    f"UCL: {control_limits['ucl']:.3f}",
                    html.Br(),
                    f"LCL: {control_limits['lcl']:.3f}"
                ], className="mb-0 small")
            ])
        ], className="mb-2"),
        
        dbc.Card([
            dbc.CardBody([
                html.H6("Process Sigma", className="card-title"),
                html.H5(f"{control_limits['sigma_est']:.3f}", className="text-info mb-0")
            ])
        ], className="mb-2"),
        
        dbc.Card([
            dbc.CardBody([
                html.H6("Violations", className="card-title"),
                html.H5(f"{len(violations)}", className="text-danger mb-0" if violations else "text-success mb-0")
            ])
        ], className="mb-2")
    ]
    
    return html.Div([
        html.H6(f"SPC Statistics: {sensor.replace('_', ' ').title()}", className="mb-3"),
        html.Div(stats_cards)
    ])


def create_clustering_statistics_summary(self, stats: Dict) -> html.Div:
    """
    Create clustering statistics summary display.
    
    Args:
        stats: Statistics dictionary from clustering analysis
        
    Returns:
        html.Div with statistics summary
    """
    if not stats:
        return html.Div("No clustering statistics available", className="text-muted")
    
    stats_cards = [
        dbc.Card([
            dbc.CardBody([
                html.H6("Number of Clusters", className="card-title"),
                html.H5(f"{stats.get('n_clusters', 0)}", className="text-primary mb-0")
            ])
        ], className="mb-2"),
        
        dbc.Card([
            dbc.CardBody([
                html.H6("Explained Variance", className="card-title"),
                html.H5(f"{stats.get('explained_variance', 0):.1%}", className="text-info mb-0")
            ])
        ], className="mb-2"),
        
        dbc.Card([
            dbc.CardBody([
                html.H6("Cluster Sizes", className="card-title"),
                html.Div([
                    html.Small(f"Cluster {i}: {size} samples", className="d-block")
                    for i, size in enumerate(stats.get('cluster_sizes', []))
                ])
            ])
        ], className="mb-2"),
        
        dbc.Card([
            dbc.CardBody([
                html.H6("Defects per Cluster", className="card-title"),
                html.Div([
                    html.Small(f"Cluster {i}: {defects} defects", className="d-block")
                    for i, defects in enumerate(stats.get('defects_per_cluster', []))
                ])
            ])
        ], className="mb-2")
    ]
    
    return html.Div([
        html.H6("Clustering Analysis Results", className="mb-3"),
        html.Div(stats_cards)
    ])


def create_correlation_statistics_summary(self, df: pd.DataFrame, method: str = 'pearson') -> html.Div:
    """
    Create correlation statistics summary display.
    
    Args:
        df: Source DataFrame
        method: Correlation method
        
    Returns:
        html.Div with statistics summary
    """
    if df.empty:
        return html.Div("No correlation statistics available", className="text-muted")
    
    # Calculate correlation matrix
    corr_matrix = df[self.sensor_columns].corr(method=method)
    
    # Find strongest correlations (excluding self-correlations)
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            corr_pairs.append((col1, col2, abs(corr_val), corr_val))
    
    # Sort by absolute correlation value
    corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Top correlations
    top_correlations = corr_pairs[:5]  # Top 5
    
    stats_content = [
        html.H6(f"{method.title()} Correlation Analysis", className="mb-3"),
        html.P("Strongest Feature Correlations:", className="fw-bold"),
    ]
    
    for col1, col2, abs_corr, corr in top_correlations:
        color_class = "text-danger" if abs_corr > 0.7 else "text-warning" if abs_corr > 0.5 else "text-info"
        stats_content.append(
            html.P([
                f"{col1.replace('_', ' ').title()} ↔ {col2.replace('_', ' ').title()}: ",
                html.Span(f"{corr:.3f}", className=color_class)
            ], className="mb-1 small")
        )
    
    # Correlation with defect status
    if 'defect' in df.columns:
        defect_correlations = []
        for col in self.sensor_columns:
            if col in df.columns:
                corr_val = df[col].corr(df['defect'], method=method)
                if not np.isnan(corr_val):
                    defect_correlations.append((col, abs(corr_val), corr_val))
        
        defect_correlations.sort(key=lambda x: x[1], reverse=True)
        
        stats_content.extend([
            html.Hr(),
            html.P("Correlation with Defect Status:", className="fw-bold"),
        ])
        
        for col, abs_corr, corr in defect_correlations[:5]:
            color_class = "text-danger" if abs_corr > 0.3 else "text-warning" if abs_corr > 0.1 else "text-muted"
            stats_content.append(
                html.P([
                    f"{col.replace('_', ' ').title()}: ",
                    html.Span(f"{corr:.3f}", className=color_class)
                ], className="mb-1 small")
            )
    
    return html.Div(stats_content)


def create_batch_statistics_summary(self, df: pd.DataFrame, selected_batches: List[str]) -> html.Div:
    """
    Create batch statistics summary display.
    
    Args:
        df: Source DataFrame
        selected_batches: List of selected batch IDs
        
    Returns:
        html.Div with statistics summary
    """
    if df.empty or not selected_batches or 'cast_id' not in df.columns:
        return html.Div("No batch statistics available", className="text-muted")
    
    # Filter data for selected batches
    batch_data = df[df['cast_id'].isin(selected_batches)]
    
    if batch_data.empty:
        return html.Div("No data found for selected batches", className="text-muted")
    
    # Calculate batch statistics
    batch_stats = []
    for batch_id in selected_batches:
        batch_subset = batch_data[batch_data['cast_id'] == batch_id]
        if not batch_subset.empty:
            stats = {
                'id': batch_id,
                'records': len(batch_subset),
                'defect_rate': (batch_subset['defect'].sum() / len(batch_subset)) * 100 if 'defect' in batch_subset.columns else 0,
                'duration': (batch_subset['timestamp'].max() - batch_subset['timestamp'].min()).total_seconds() / 3600 if 'timestamp' in batch_subset.columns else 0
            }
            batch_stats.append(stats)
    
    stats_content = [
        html.H6("Batch Comparison Summary", className="mb-3"),
    ]
    
    for stats in batch_stats:
        color_class = "text-danger" if stats['defect_rate'] > 10 else "text-warning" if stats['defect_rate'] > 5 else "text-success"
        stats_content.append(
            dbc.Card([
                dbc.CardBody([
                    html.H6(stats['id'], className="card-title"),
                    html.P([
                        f"Records: {stats['records']:,}",
                        html.Br(),
                        f"Defect Rate: ",
                        html.Span(f"{stats['defect_rate']:.1f}%", className=color_class),
                        html.Br(),
                        f"Duration: {stats['duration']:.1f}h"
                    ], className="mb-0 small")
                ])
            ], className="mb-2")
        )
    
    return html.Div(stats_content)


# Add new methods to the class
HistoricalAnalysisComponents.export_data_to_csv = export_data_to_csv
HistoricalAnalysisComponents.export_chart_to_image = export_chart_to_image
HistoricalAnalysisComponents.filter_data = filter_data
HistoricalAnalysisComponents.get_data_info_summary = get_data_info_summary
HistoricalAnalysisComponents.create_spc_statistics_summary = create_spc_statistics_summary
HistoricalAnalysisComponents.create_clustering_statistics_summary = create_clustering_statistics_summary
HistoricalAnalysisComponents.create_correlation_statistics_summary = create_correlation_statistics_summary
HistoricalAnalysisComponents.create_batch_statistics_summary = create_batch_statistics_summary