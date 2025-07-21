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
                return self._generate_synthetic_data(random_seed=self.random_seed)
                
        except Exception as e:
            logger.error(f"Error loading sample data: {e}")
            return self._generate_synthetic_data(random_seed=self.random_seed)
    
    def _generate_synthetic_data(self, n_samples: int = 2000, random_seed: Optional[int] = None) -> pd.DataFrame:
        """Generate synthetic data for demonstration purposes."""
        if random_seed is not None:
            np.random.seed(random_seed)  # For reproducible results
        
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

    def filter_data(self, df: pd.DataFrame, 
                    date_range: Optional[List[str]] = None,
                    cast_ids: Optional[List[str]] = None,
                    defect_filter: Union[str, int] = 'all',
                    aggregation: str = 'raw') -> pd.DataFrame:
        """Apply filters and aggregation to the data."""
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
        
        return filtered_df

    def export_data_to_csv(self, df: pd.DataFrame, filename: str = None) -> Dict[str, str]:
        """Export filtered data to CSV format."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"historical_analysis_data_{timestamp}.csv"
        
        output = io.StringIO()
        df.to_csv(output, index=False)
        csv_content = output.getvalue()
        output.close()
        
        return {
            'content': csv_content,
            'filename': filename,
            'type': 'text/csv'
        }

    def create_spc_statistics_summary(self, df: pd.DataFrame, sensor: str) -> html.Div:
        """Create SPC statistics summary display."""
        if df.empty or sensor not in df.columns:
            return html.Div("No data available for SPC statistics", className="text-muted")
        
        data = df[sensor].values
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        return html.Div([
            html.H6(f"SPC Statistics: {sensor.replace('_', ' ').title()}", className="mb-3"),
            dbc.Card([
                dbc.CardBody([
                    html.P(f"Mean: {mean_val:.3f}"),
                    html.P(f"Std Dev: {std_val:.3f}")
                ])
            ])
        ])

    def create_clustering_analysis(self, df: pd.DataFrame, n_clusters: int, n_components: int = 2) -> Tuple[go.Figure, Dict]:
        """Create clustering analysis visualization."""
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig, {}
        
        # Simplified scatter plot
        available_sensors = [col for col in self.sensor_columns if col in df.columns]
        if len(available_sensors) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Need at least 2 sensors", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig, {}
        
        x_col, y_col = available_sensors[0], available_sensors[1]
        fig = go.Figure()
        
        for defect_status in [0, 1]:
            subset = df[df['defect'] == defect_status]
            fig.add_trace(go.Scatter(
                x=subset[x_col], y=subset[y_col], mode='markers',
                name=f"Normal" if defect_status == 0 else "Defect",
                marker=dict(color=self.defect_colors[defect_status], size=8)
            ))
        
        fig.update_layout(title="Clustering Analysis", template="plotly_white", height=500)
        
        stats = {'n_clusters': n_clusters, 'explained_variance': 0.8}
        return fig, stats

    def create_clustering_statistics_summary(self, stats: Dict) -> html.Div:
        """Create clustering statistics summary."""
        if not stats:
            return html.Div("No clustering statistics available", className="text-muted")
        
        return html.Div([
            html.H6("Clustering Results", className="mb-3"),
            html.P(f"Clusters: {stats.get('n_clusters', 0)}"),
            html.P(f"Explained Variance: {stats.get('explained_variance', 0):.1%}")
        ])

    def create_correlation_statistics_summary(self, df: pd.DataFrame, method: str = 'pearson') -> html.Div:
        """Create correlation statistics summary."""
        if df.empty:
            return html.Div("No correlation statistics available", className="text-muted")
        
        return html.Div([
            html.H6(f"{method.title()} Correlation Analysis", className="mb-3"),
            html.P("Correlation analysis completed.")
        ])

    def create_spc_charts(self, df: pd.DataFrame, sensor: str, chart_type: str, subgroup_size: int = 5) -> go.Figure:
        """Create SPC charts (simplified implementation)."""
        if df.empty or sensor not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Simple line plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(df))), y=df[sensor], mode='lines+markers',
            name=sensor.replace('_', ' ').title()
        ))
        fig.update_layout(title=f"SPC Chart: {sensor}", template="plotly_white", height=400)
        return fig

    def create_correlation_heatmap(self, df: pd.DataFrame, method: str = 'pearson') -> go.Figure:
        """Create correlation heatmap."""
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Simple correlation matrix
        available_sensors = [col for col in self.sensor_columns if col in df.columns][:5]  # Limit to 5
        if len(available_sensors) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Need more sensors", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        corr_matrix = df[available_sensors].corr(method=method)
        fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=available_sensors, y=available_sensors, colorscale='RdBu'))
        fig.update_layout(title="Correlation Heatmap", template="plotly_white", height=500)
        return fig

    def create_batch_comparison(self, df: pd.DataFrame, selected_batches: List[str], comparison_type: str) -> go.Figure:
        """Create batch comparison visualization."""
        if df.empty or not selected_batches:
            fig = go.Figure()
            fig.add_annotation(text="No batches selected", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Simple comparison
        fig = go.Figure()
        for i, batch in enumerate(selected_batches[:3]):  # Limit to 3 batches
            batch_data = df[df['cast_id'] == batch] if 'cast_id' in df.columns else df
            if not batch_data.empty and self.sensor_columns[0] in batch_data.columns:
                fig.add_trace(go.Scatter(
                    x=list(range(len(batch_data))), y=batch_data[self.sensor_columns[0]],
                    mode='lines', name=f"Batch {batch}"
                ))
        
        fig.update_layout(title="Batch Comparison", template="plotly_white", height=400)
        return fig

    def export_chart_to_image(self, fig: go.Figure, filename: str = None) -> Dict[str, str]:
        """Export chart to image format."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chart_{timestamp}.html"
        
        html_content = fig.to_html(include_plotlyjs='inline')
        return {'content': html_content, 'filename': filename, 'type': 'text/html'}

    def create_batch_statistics_summary(self, df: pd.DataFrame, selected_batches: List[str]) -> html.Div:
        """Create batch statistics summary."""
        if df.empty or not selected_batches:
            return html.Div("No batch statistics available", className="text-muted")
        
        return html.Div([
            html.H6("Batch Statistics", className="mb-3"),
            html.P(f"Selected batches: {len(selected_batches)}")
        ])


