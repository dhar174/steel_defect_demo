"""
Alert Management Interface for steel casting defect prediction system.

This module provides a comprehensive alert management interface that serves as 
the central hub for operators to view, manage, and configure alerts generated 
by the defect prediction system.

Key Features:
- Real-time alert feed with sortable DataTable
- Alert history and trend analysis
- Configurable alert thresholds
- Alert acknowledgment and resolution
- Alert performance analytics (MTTA, active vs resolved counts)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import yaml
import os
from collections import deque

logger = logging.getLogger(__name__)


class AlertManagementComponent:
    """
    Alert Management Interface for steel casting defect prediction system.
    
    Provides comprehensive alert management capabilities including real-time
    monitoring, configuration, acknowledgment, and performance analytics.
    """
    
    def __init__(self, 
                 component_id: str = "alert-management",
                 config_file: str = None,
                 base_dir: str = None,
                 max_alerts: int = 1000,
                 update_interval: int = 5000,
                 initialize_sample_data: bool = True):
        """
        Initialize the alert management component.
        
        Args:
            component_id: Unique identifier for the component
            config_file: Path to inference_config.yaml
            base_dir: Base directory for resolving relative paths (defaults to project root)
            max_alerts: Maximum number of alerts to store in memory
            update_interval: Update interval in milliseconds
            initialize_sample_data: Whether to initialize with sample data
        """
        self.component_id = component_id
        self.max_alerts = max_alerts
        self.update_interval = update_interval
        
        # Set base directory for consistent file path resolution
        if base_dir is None:
            # Default to the project root directory (3 levels up from this file)
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_dir)))
        else:
            self.base_dir = os.path.abspath(base_dir)
        
        # Load configuration
        self.config_file = config_file or "configs/inference_config.yaml"
        self.config = self._load_config()
        
        # Alert storage - using deque for efficient FIFO operations
        self.alerts_buffer = deque(maxlen=max_alerts)
        
        # Alert counter for consistent ID generation
        self._alert_counter = 0
        
        # Alert statuses
        self.alert_statuses = ['New', 'Acknowledged', 'Resolved']
        self.severity_levels = ['Low', 'Medium', 'High', 'Critical']
        
        # Alert performance tracking
        self.performance_metrics = {
            'total_alerts': 0,
            'active_alerts': 0,
            'resolved_alerts': 0,
            'acknowledged_alerts': 0,
            'mean_time_to_acknowledge': 0.0,
            'mean_time_to_resolve': 0.0
        }
        
        # Initialize with some sample data for demonstration
        if initialize_sample_data:
            self._initialize_sample_data()
    
    def _load_config(self) -> Dict:
        """Load configuration from inference_config.yaml."""
        try:
            config_path = os.path.join(self.base_dir, self.config_file)
            if os.path.exists(config_path):
                with open(config_path, 'r') as file:
                    config = yaml.safe_load(file)
                    return config
            else:
                logger.warning(f"Config file not found: {config_path}")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration if config file is not available."""
        return {
            'inference': {
                'thresholds': {
                    'defect_probability': 0.5,
                    'high_risk_threshold': 0.7,
                    'alert_threshold': 0.8
                }
            },
            'monitoring': {
                'alerts': {
                    'alert_suppression_minutes': 5
                }
            }
        }
    
    def _initialize_sample_data(self):
        """Initialize with sample alert data for demonstration."""
        current_time = datetime.now()
        sample_alerts = []
        
        for i in range(20):
            alert_id = f'ALT-{self._alert_counter:04d}'
            self._alert_counter += 1
            
            alert = {
                'id': alert_id,
                'timestamp': current_time - timedelta(minutes=i*5),
                'severity': np.random.choice(self.severity_levels, p=[0.4, 0.3, 0.2, 0.1]),
                'description': f'Defect probability threshold exceeded: {0.5 + np.random.random() * 0.4:.3f}',
                'status': np.random.choice(self.alert_statuses, p=[0.3, 0.4, 0.3]),
                'source': np.random.choice(['LSTM Model', 'Baseline Model', 'Ensemble']),
                'acknowledged_at': None,
                'resolved_at': None,
                'acknowledged_by': None,
                'resolved_by': None
            }
            sample_alerts.append(alert)
        
        # Set acknowledgment and resolution times for non-new alerts
        for alert in sample_alerts:
            if alert['status'] in ['Acknowledged', 'Resolved']:
                alert['acknowledged_at'] = alert['timestamp'] + timedelta(minutes=np.random.randint(1, 15))
                alert['acknowledged_by'] = np.random.choice(['operator1', 'operator2', 'supervisor'])
                
            if alert['status'] == 'Resolved':
                alert['resolved_at'] = alert['acknowledged_at'] + timedelta(minutes=np.random.randint(5, 30))
                alert['resolved_by'] = alert['acknowledged_by']
        
        self.alerts_buffer.extend(sample_alerts)
        self._update_performance_metrics()
    
    def create_layout(self) -> html.Div:
        """
        Create the complete alert management dashboard layout.
        
        Returns:
            html.Div: Complete dashboard layout
        """
        return html.Div([
            # Header with title and key metrics
            self._create_header(),
            
            # Control Panel
            self._create_control_panel(),
            
            # Main content area with tabs
            dbc.Tabs([
                # Real-time Alert Feed Tab
                dbc.Tab(
                    label="Real-time Alerts",
                    tab_id="realtime-tab",
                    children=[
                        html.Div([
                            self._create_alert_feed(),
                        ], className="p-3")
                    ]
                ),
                
                # Alert History and Trends Tab
                dbc.Tab(
                    label="History & Trends",
                    tab_id="history-tab", 
                    children=[
                        html.Div([
                            self._create_history_analysis(),
                        ], className="p-3")
                    ]
                ),
                
                # Configuration Tab
                dbc.Tab(
                    label="Configuration",
                    tab_id="config-tab",
                    children=[
                        html.Div([
                            self._create_threshold_configuration(),
                        ], className="p-3")
                    ]
                ),
                
                # Performance Analytics Tab
                dbc.Tab(
                    label="Analytics",
                    tab_id="analytics-tab",
                    children=[
                        html.Div([
                            self._create_performance_analytics(),
                        ], className="p-3")
                    ]
                )
            ], id=f'{self.component_id}-tabs', active_tab="realtime-tab"),
            
            # Hidden storage components
            dcc.Store(id=f'{self.component_id}-alerts-store', data=[]),
            dcc.Store(id=f'{self.component_id}-config-store', data=self.config),
            dcc.Store(id=f'{self.component_id}-performance-store', data=self.performance_metrics),
            
            # Real-time update interval
            dcc.Interval(
                id=f'{self.component_id}-interval',
                interval=self.update_interval,
                n_intervals=0
            )
        ], id=f'{self.component_id}-container')
    
    def _create_header(self) -> dbc.Card:
        """Create the header with title and key metrics."""
        return dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H2("Alert Management Center", className="mb-0"),
                        html.P("Real-time monitoring and management of steel defect prediction alerts", 
                               className="text-muted mb-0")
                    ], md=6),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.H4("0", id=f'{self.component_id}-total-alerts', className="mb-0"),
                                    html.Small("Total Alerts", className="text-muted")
                                ], className="text-center")
                            ], md=3),
                            dbc.Col([
                                html.Div([
                                    html.H4("0", id=f'{self.component_id}-active-alerts', className="mb-0 text-warning"),
                                    html.Small("Active", className="text-muted")
                                ], className="text-center")
                            ], md=3),
                            dbc.Col([
                                html.Div([
                                    html.H4("0", id=f'{self.component_id}-acknowledged-alerts', className="mb-0 text-info"),
                                    html.Small("Acknowledged", className="text-muted")
                                ], className="text-center")
                            ], md=3),
                            dbc.Col([
                                html.Div([
                                    html.H4("0", id=f'{self.component_id}-resolved-alerts', className="mb-0 text-success"),
                                    html.Small("Resolved", className="text-muted")
                                ], className="text-center")
                            ], md=3)
                        ])
                    ], md=6)
                ])
            ])
        ], className="mb-3")
    
    def _create_control_panel(self) -> dbc.Card:
        """Create the control panel with filters and actions."""
        return dbc.Card([
            dbc.CardHeader(
                html.H5("Controls & Filters", className="mb-0")
            ),
            dbc.CardBody([
                dbc.Row([
                    # Severity Filter
                    dbc.Col([
                        html.Label("Filter by Severity", className="fw-bold"),
                        dcc.Dropdown(
                            id=f'{self.component_id}-severity-filter',
                            options=[{'label': 'All', 'value': 'all'}] + 
                                   [{'label': s, 'value': s} for s in self.severity_levels],
                            value='all',
                            clearable=False
                        )
                    ], md=2),
                    
                    # Status Filter
                    dbc.Col([
                        html.Label("Filter by Status", className="fw-bold"),
                        dcc.Dropdown(
                            id=f'{self.component_id}-status-filter',
                            options=[{'label': 'All', 'value': 'all'}] + 
                                   [{'label': s, 'value': s} for s in self.alert_statuses],
                            value='all',
                            clearable=False
                        )
                    ], md=2),
                    
                    # Time Range Filter
                    dbc.Col([
                        html.Label("Time Range", className="fw-bold"),
                        dcc.Dropdown(
                            id=f'{self.component_id}-time-filter',
                            options=[
                                {'label': 'Last Hour', 'value': 1},
                                {'label': 'Last 6 Hours', 'value': 6},
                                {'label': 'Last 24 Hours', 'value': 24},
                                {'label': 'Last Week', 'value': 168},
                                {'label': 'All Time', 'value': 'all'}
                            ],
                            value=24,
                            clearable=False
                        )
                    ], md=2),
                    
                    # Auto-refresh Toggle
                    dbc.Col([
                        html.Label("Auto-refresh", className="fw-bold"),
                        dbc.Switch(
                            id=f'{self.component_id}-auto-refresh',
                            value=True,
                            label="Enabled"
                        )
                    ], md=2),
                    
                    # Action Buttons
                    dbc.Col([
                        html.Label("Actions", className="fw-bold"),
                        dbc.ButtonGroup([
                            dbc.Button("🔄 Refresh", id=f'{self.component_id}-refresh-btn', 
                                     size="sm", color="primary"),
                            dbc.Button("🗑️ Clear", id=f'{self.component_id}-clear-btn', 
                                     size="sm", color="danger")
                        ])
                    ], md=4)
                ])
            ])
        ], className="mb-3")
    
    def _create_alert_feed(self) -> html.Div:
        """Create the real-time alert feed with DataTable."""
        return html.Div([
            html.H4("Real-time Alert Feed", className="mb-3"),
            
            # Alert action buttons
            dbc.Row([
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button("✓ Acknowledge Selected", 
                                 id=f'{self.component_id}-acknowledge-btn',
                                 color="info", size="sm"),
                        dbc.Button("✅ Resolve Selected", 
                                 id=f'{self.component_id}-resolve-btn',
                                 color="success", size="sm"),
                        dbc.Button("🔔 Generate Test Alert", 
                                 id=f'{self.component_id}-test-alert-btn',
                                 color="warning", size="sm")
                    ])
                ], md=12, className="mb-3")
            ]),
            
            # Alert DataTable
            dash_table.DataTable(
                id=f'{self.component_id}-alert-table',
                columns=[
                    {"name": "Select", "id": "select", "type": "text", "presentation": "dropdown"},
                    {"name": "ID", "id": "id", "type": "text"},
                    {"name": "Timestamp", "id": "timestamp", "type": "datetime"},
                    {"name": "Severity", "id": "severity", "type": "text"},
                    {"name": "Description", "id": "description", "type": "text"},
                    {"name": "Status", "id": "status", "type": "text"},
                    {"name": "Source", "id": "source", "type": "text"},
                    {"name": "Ack. By", "id": "acknowledged_by", "type": "text"},
                    {"name": "Resolved By", "id": "resolved_by", "type": "text"}
                ],
                data=[],
                editable=False,
                row_selectable="multi",
                selected_rows=[],
                sort_action="native",
                sort_mode="multi",
                filter_action="native",
                page_action="native",
                page_current=0,
                page_size=10,
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px',
                    'fontFamily': 'Arial'
                },
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{severity} = Critical'},
                        'backgroundColor': '#ffebee',
                        'color': 'black',
                    },
                    {
                        'if': {'filter_query': '{severity} = High'},
                        'backgroundColor': '#fff3e0',
                        'color': 'black',
                    },
                    {
                        'if': {'filter_query': '{status} = New'},
                        'fontWeight': 'bold',
                    }
                ],
                style_table={'overflowX': 'auto'},
                css=[{
                    'selector': '.dash-spreadsheet td div',
                    'rule': '''
                        line-height: 15px;
                        max-height: 30px; min-height: 30px; height: 30px;
                        display: block;
                        overflow-y: hidden;
                    '''
                }]
            )
        ])
    
    def _create_history_analysis(self) -> html.Div:
        """Create alert history and trend analysis."""
        return html.Div([
            html.H4("Alert History & Trend Analysis", className="mb-3"),
            
            dbc.Row([
                # Alert frequency trend chart
                dbc.Col([
                    dcc.Graph(
                        id=f'{self.component_id}-frequency-chart',
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ], md=8),
                
                # Alert distribution by severity
                dbc.Col([
                    dcc.Graph(
                        id=f'{self.component_id}-severity-distribution',
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ], md=4)
            ], className="mb-4"),
            
            dbc.Row([
                # Resolution time analysis
                dbc.Col([
                    dcc.Graph(
                        id=f'{self.component_id}-resolution-time-chart',
                        config={'displayModeBar': False},
                        style={'height': '350px'}
                    )
                ], md=12)
            ])
        ])
    
    def _create_threshold_configuration(self) -> html.Div:
        """Create configurable alert threshold interface."""
        return html.Div([
            html.H4("Alert Threshold Configuration", className="mb-3"),
            
            dbc.Card([
                dbc.CardHeader(
                    html.H5("Prediction Thresholds", className="mb-0")
                ),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Defect Probability Threshold", className="fw-bold"),
                            dcc.Slider(
                                id=f'{self.component_id}-defect-threshold',
                                min=0.1,
                                max=0.9,
                                step=0.05,
                                value=self.config.get('inference', {}).get('thresholds', {}).get('defect_probability', 0.5),
                                marks={i/10: f'{i/10:.1f}' for i in range(1, 10, 2)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            html.P("Minimum probability to consider a defect prediction", 
                                   className="text-muted small")
                        ], md=4),
                        
                        dbc.Col([
                            html.Label("High Risk Threshold", className="fw-bold"),
                            dcc.Slider(
                                id=f'{self.component_id}-high-risk-threshold',
                                min=0.5,
                                max=0.95,
                                step=0.05,
                                value=self.config.get('inference', {}).get('thresholds', {}).get('high_risk_threshold', 0.7),
                                marks={i/100: f'{i/100:.2f}' for i in range(50, 100, 10)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            html.P("Threshold for high-risk defect predictions", 
                                   className="text-muted small")
                        ], md=4),
                        
                        dbc.Col([
                            html.Label("Alert Threshold", className="fw-bold"),
                            dcc.Slider(
                                id=f'{self.component_id}-alert-threshold',
                                min=0.6,
                                max=1.0,
                                step=0.05,
                                value=self.config.get('inference', {}).get('thresholds', {}).get('alert_threshold', 0.8),
                                marks={i/100: f'{i/100:.2f}' for i in range(60, 105, 10)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            html.P("Threshold to trigger immediate alerts", 
                                   className="text-muted small")
                        ], md=4)
                    ], className="mb-4"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Alert Suppression Period (minutes)", className="fw-bold"),
                            dcc.Input(
                                id=f'{self.component_id}-suppression-period',
                                type='number',
                                value=self.config.get('monitoring', {}).get('alerts', {}).get('alert_suppression_minutes', 5),
                                min=1,
                                max=60,
                                step=1,
                                className="form-control"
                            ),
                            html.P("Time to suppress duplicate alerts", className="text-muted small")
                        ], md=3),
                        
                        dbc.Col([
                            html.Div([
                                dbc.Button("💾 Save Configuration", 
                                         id=f'{self.component_id}-save-config-btn',
                                         color="success", className="me-2"),
                                dbc.Button("🔄 Reset to Defaults", 
                                         id=f'{self.component_id}-reset-config-btn',
                                         color="secondary"),
                            ], className="mt-4")
                        ], md=3)
                    ])
                ])
            ], className="mb-4"),
            
            # Configuration preview
            dbc.Alert(
                id=f'{self.component_id}-config-alert',
                is_open=False,
                duration=3000
            )
        ])
    
    def _create_performance_analytics(self) -> html.Div:
        """Create alert performance analytics dashboard."""
        return html.Div([
            html.H4("Alert Performance Analytics", className="mb-3"),
            
            # Key performance metrics cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("0 min", id=f'{self.component_id}-mtta-metric', className="text-primary"),
                            html.P("Mean Time to Acknowledge", className="mb-0")
                        ])
                    ])
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("0 min", id=f'{self.component_id}-mttr-metric', className="text-success"),
                            html.P("Mean Time to Resolve", className="mb-0")
                        ])
                    ])
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("0%", id=f'{self.component_id}-resolution-rate', className="text-info"),
                            html.P("Resolution Rate", className="mb-0")
                        ])
                    ])
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("0", id=f'{self.component_id}-avg-daily-alerts', className="text-warning"),
                            html.P("Avg Daily Alerts", className="mb-0")
                        ])
                    ])
                ], md=3)
            ], className="mb-4"),
            
            # Performance charts
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id=f'{self.component_id}-performance-timeline',
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ], md=12)
            ])
        ])
    
    def get_alerts_data(self) -> List[Dict]:
        """
        Get current alerts data for the DataTable.
        
        Returns:
            List of alert dictionaries
        """
        alerts = []
        for alert in self.alerts_buffer:
            alerts.append({
                'select': '',  # For row selection
                'id': alert['id'],
                'timestamp': alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'severity': alert['severity'],
                'description': alert['description'],
                'status': alert['status'],
                'source': alert['source'],
                'acknowledged_by': alert.get('acknowledged_by', ''),
                'resolved_by': alert.get('resolved_by', '')
            })
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x['timestamp'], reverse=True)
        return alerts
    
    def add_alert(self, severity: str, description: str, source: str = "System") -> str:
        """
        Add a new alert to the system.
        
        Args:
            severity: Alert severity level
            description: Alert description
            source: Source of the alert
            
        Returns:
            Alert ID
        """
        alert_id = f'ALT-{self._alert_counter:04d}'
        self._alert_counter += 1
        
        alert = {
            'id': alert_id,
            'timestamp': datetime.now(),
            'severity': severity,
            'description': description,
            'status': 'New',
            'source': source,
            'acknowledged_at': None,
            'resolved_at': None,
            'acknowledged_by': None,
            'resolved_by': None
        }
        
        self.alerts_buffer.append(alert)
        self._update_performance_metrics()
        return alert_id
    
    def acknowledge_alert(self, alert_id: str, user: str = "operator") -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: ID of the alert to acknowledge
            user: User acknowledging the alert
            
        Returns:
            True if successful, False otherwise
        """
        for alert in self.alerts_buffer:
            if alert['id'] == alert_id and alert['status'] == 'New':
                alert['status'] = 'Acknowledged'
                alert['acknowledged_at'] = datetime.now()
                alert['acknowledged_by'] = user
                self._update_performance_metrics()
                return True
        return False
    
    def resolve_alert(self, alert_id: str, user: str = "operator") -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: ID of the alert to resolve
            user: User resolving the alert
            
        Returns:
            True if successful, False otherwise
        """
        for alert in self.alerts_buffer:
            if alert['id'] == alert_id and alert['status'] in ['New', 'Acknowledged']:
                alert['status'] = 'Resolved'
                alert['resolved_at'] = datetime.now()
                alert['resolved_by'] = user
                
                # Auto-acknowledge if not already acknowledged
                if alert['acknowledged_at'] is None:
                    alert['acknowledged_at'] = alert['resolved_at']
                    alert['acknowledged_by'] = user
                
                self._update_performance_metrics()
                return True
        return False
    
    def _update_performance_metrics(self):
        """Update performance metrics based on current alerts."""
        if not self.alerts_buffer:
            return
        
        alerts = list(self.alerts_buffer)
        total = len(alerts)
        new_count = sum(1 for a in alerts if a['status'] == 'New')
        acknowledged_count = sum(1 for a in alerts if a['status'] == 'Acknowledged')
        resolved_count = sum(1 for a in alerts if a['status'] == 'Resolved')
        
        # Calculate mean times
        acknowledge_times = []
        resolve_times = []
        
        for alert in alerts:
            if alert['acknowledged_at']:
                ack_time = (alert['acknowledged_at'] - alert['timestamp']).total_seconds() / 60
                acknowledge_times.append(ack_time)
            
            if alert['resolved_at']:
                resolve_time = (alert['resolved_at'] - alert['timestamp']).total_seconds() / 60
                resolve_times.append(resolve_time)
        
        self.performance_metrics.update({
            'total_alerts': total,
            'active_alerts': new_count,
            'acknowledged_alerts': acknowledged_count,
            'resolved_alerts': resolved_count,
            'mean_time_to_acknowledge': np.mean(acknowledge_times) if acknowledge_times else 0.0,
            'mean_time_to_resolve': np.mean(resolve_times) if resolve_times else 0.0
        })
    
    def create_frequency_chart(self) -> go.Figure:
        """Create alert frequency trend chart."""
        if not self.alerts_buffer:
            return go.Figure().add_annotation(text="No data available", showarrow=False)
        
        # Group alerts by hour
        alerts_df = pd.DataFrame([
            {
                'timestamp': alert['timestamp'],
                'severity': alert['severity'],
                'status': alert['status']
            }
            for alert in self.alerts_buffer
        ])
        
        alerts_df['hour'] = alerts_df['timestamp'].dt.floor('h')
        hourly_counts = alerts_df.groupby(['hour', 'severity']).size().reset_index(name='count')
        
        fig = px.bar(
            hourly_counts,
            x='hour',
            y='count',
            color='severity',
            title='Alert Frequency Over Time',
            color_discrete_map={
                'Critical': '#d32f2f',
                'High': '#ff9800',
                'Medium': '#2196f3',
                'Low': '#4caf50'
            }
        )
        
        fig.update_layout(
            template='plotly_white',
            xaxis_title='Time',
            yaxis_title='Number of Alerts',
            legend_title='Severity'
        )
        
        return fig
    
    def create_severity_distribution(self) -> go.Figure:
        """Create severity distribution pie chart."""
        if not self.alerts_buffer:
            return go.Figure().add_annotation(text="No data available", showarrow=False)
        
        severity_counts = {}
        for alert in self.alerts_buffer:
            severity = alert['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        fig = go.Figure(data=[go.Pie(
            labels=list(severity_counts.keys()),
            values=list(severity_counts.values()),
            hole=0.4,
            marker_colors=['#4caf50', '#2196f3', '#ff9800', '#d32f2f']
        )])
        
        fig.update_layout(
            title='Alert Distribution by Severity',
            template='plotly_white'
        )
        
        return fig
    
    def update_config_file(self, new_config: Dict) -> bool:
        """
        Update the configuration file with new threshold values.
        
        Args:
            new_config: New configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Helper function for deep merging configuration dictionaries
            def deep_update(target: Dict, source: Dict) -> None:
                """Recursively update target dict with source dict values."""
                for key, value in source.items():
                    if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                        deep_update(target[key], value)
                    else:
                        target[key] = value
            
            # Update internal config with deep merge
            deep_update(self.config, new_config)
            
            # Write updated configuration to file
            config_path = os.path.join(self.base_dir, self.config_file)
            
            # Ensure the directory exists
            config_dir = os.path.dirname(config_path)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)
            
            # Write the updated configuration to the YAML file
            with open(config_path, 'w', encoding='utf-8') as file:
                yaml.safe_dump(self.config, file, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Configuration updated successfully and written to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False