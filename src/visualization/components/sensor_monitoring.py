"""
Real-time sensor monitoring components for steel casting process visualization.

This module provides components for displaying live sensor data with features like:
- Multi-sensor time series visualization with synchronized cross-filtering
- Real-time streaming with rolling window buffer
- Configurable time ranges and auto-scaling
- Anomaly and threshold highlighting
- Sensor health and data quality indicators
- Interactive tooltips with detailed metadata
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class SensorMonitoringComponent:
    """
    Real-time sensor monitoring component for steel casting process.
    
    Provides multi-sensor visualization with real-time updates, synchronized
    cross-filtering, threshold monitoring, and data quality indicators.
    """
    
    def __init__(self, 
                 component_id: str = "sensor-monitoring",
                 buffer_size: int = 1000,
                 sensors: List[str] = None,
                 thresholds: Dict[str, Dict[str, float]] = None,
                 update_interval: int = 1000):
        """
        Initialize the sensor monitoring component.
        
        Args:
            component_id: Unique identifier for the component
            buffer_size: Maximum number of data points to keep in rolling window
            sensors: List of sensor names to monitor
            thresholds: Dictionary of sensor thresholds {sensor: {min: val, max: val}}
            update_interval: Update interval in milliseconds for real-time updates
        """
        self.component_id = component_id
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        
        # Default sensors for steel casting process
        self.sensors = sensors or [
            'casting_speed', 'mold_temperature', 'mold_level', 
            'strand_temperature', 'cooling_water_flow', 'tundish_temperature'
        ]
        
        # Default operational thresholds
        self.thresholds = thresholds or {
            'casting_speed': {'min': 0.8, 'max': 1.5, 'warning_min': 0.9, 'warning_max': 1.4},
            'mold_temperature': {'min': 1480, 'max': 1580, 'warning_min': 1500, 'warning_max': 1560},
            'mold_level': {'min': 140, 'max': 160, 'warning_min': 145, 'warning_max': 155},
            'strand_temperature': {'min': 1200, 'max': 1400, 'warning_min': 1220, 'warning_max': 1380},
            'cooling_water_flow': {'min': 80, 'max': 120, 'warning_min': 85, 'warning_max': 115},
            'tundish_temperature': {'min': 1530, 'max': 1580, 'warning_min': 1540, 'warning_max': 1570}
        }
        
        # Rolling window buffers for each sensor
        self.data_buffers = {sensor: deque(maxlen=buffer_size) for sensor in self.sensors}
        self.timestamp_buffer = deque(maxlen=buffer_size)
        
        # Sensor health tracking
        self.sensor_health = {sensor: 'good' for sensor in self.sensors}
        self.last_update_times = {sensor: None for sensor in self.sensors}
        
        # Configuration state
        self.time_range_minutes = 30  # Default time range
        self.auto_scale = True
        self.show_thresholds = True
        self.show_anomalies = True
        
    def create_layout(self) -> html.Div:
        """
        Create the complete sensor monitoring dashboard layout.
        
        Returns:
            html.Div: Complete dashboard layout with controls and plots
        """
        return html.Div([
            # Control Panel
            self._create_control_panel(),
            
            # Sensor Health Status Bar
            self._create_health_status_bar(),
            
            # Main Sensor Plots Container
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id=f'{self.component_id}-main-plot',
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                        },
                        style={'height': '600px'}
                    )
                ], width=12)
            ], className="mb-3"),
            
            # Individual Sensor Detail Plots
            dbc.Row([
                dbc.Col([
                    html.H5("Individual Sensor Details", className="mb-3"),
                    dcc.Graph(
                        id=f'{self.component_id}-detail-plots',
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                        },
                        style={'height': '800px'}
                    )
                ], width=12)
            ]),
            
            # Hidden components for data storage
            dcc.Store(id=f'{self.component_id}-data-store', data={}),
            dcc.Store(id=f'{self.component_id}-config-store', data={
                'time_range_minutes': self.time_range_minutes,
                'auto_scale': self.auto_scale,
                'show_thresholds': self.show_thresholds,
                'show_anomalies': self.show_anomalies
            }),
            
            # Real-time update interval
            dcc.Interval(
                id=f'{self.component_id}-interval',
                interval=self.update_interval,
                n_intervals=0
            )
        ], id=f'{self.component_id}-container')
    
    def _create_control_panel(self) -> dbc.Card:
        """Create the control panel with user configuration options."""
        return dbc.Card([
            dbc.CardHeader(
                html.H4("Sensor Monitoring Controls", className="mb-0")
            ),
            dbc.CardBody([
                dbc.Row([
                    # Time Range Controls
                    dbc.Col([
                        html.Label("Time Range", className="fw-bold"),
                        dbc.ButtonGroup([
                            dbc.Button("5min", id=f'{self.component_id}-time-5min', 
                                     size="sm", outline=True, color="primary"),
                            dbc.Button("15min", id=f'{self.component_id}-time-15min', 
                                     size="sm", outline=True, color="primary"),
                            dbc.Button("30min", id=f'{self.component_id}-time-30min', 
                                     size="sm", outline=True, color="primary", active=True),
                            dbc.Button("1hr", id=f'{self.component_id}-time-1hr', 
                                     size="sm", outline=True, color="primary"),
                        ], size="sm")
                    ], md=3),
                    
                    # Display Options
                    dbc.Col([
                        html.Label("Display Options", className="fw-bold"),
                        dbc.Checklist(
                            options=[
                                {"label": "Auto-scale Y-axis", "value": "auto_scale"},
                                {"label": "Show Thresholds", "value": "show_thresholds"},
                                {"label": "Highlight Anomalies", "value": "show_anomalies"},
                            ],
                            value=["auto_scale", "show_thresholds", "show_anomalies"],
                            id=f'{self.component_id}-display-options',
                            inline=True
                        )
                    ], md=4),
                    
                    # Update Interval Controls
                    dbc.Col([
                        html.Label("Update Rate", className="fw-bold"),
                        dbc.ButtonGroup([
                            dbc.Button("1s", id=f'{self.component_id}-update-1s', 
                                     size="sm", outline=True, color="secondary", active=True),
                            dbc.Button("5s", id=f'{self.component_id}-update-5s', 
                                     size="sm", outline=True, color="secondary"),
                            dbc.Button("10s", id=f'{self.component_id}-update-10s', 
                                     size="sm", outline=True, color="secondary"),
                        ], size="sm")
                    ], md=2),
                    
                    # Stream Controls
                    dbc.Col([
                        html.Label("Stream Control", className="fw-bold"),
                        dbc.ButtonGroup([
                            dbc.Button("⏸️ Pause", id=f'{self.component_id}-pause', 
                                     size="sm", color="warning"),
                            dbc.Button("🔄 Reset", id=f'{self.component_id}-reset', 
                                     size="sm", color="danger"),
                        ], size="sm")
                    ], md=3)
                ])
            ])
        ], className="mb-3")
    
    def _create_health_status_bar(self) -> dbc.Card:
        """Create sensor health status indicators."""
        return dbc.Card([
            dbc.CardHeader(
                html.H5("Sensor Health Status", className="mb-0")
            ),
            dbc.CardBody([
                html.Div(id=f'{self.component_id}-health-indicators')
            ])
        ], className="mb-3")
    
    def create_multi_sensor_plot(self, 
                                data: Dict[str, List], 
                                timestamps: List[datetime],
                                config: Dict[str, Any]) -> go.Figure:
        """
        Create synchronized multi-sensor time series plot.
        
        Args:
            data: Dictionary of sensor data {sensor_name: [values]}
            timestamps: List of timestamp values
            config: Configuration dictionary with display options
            
        Returns:
            go.Figure: Multi-sensor plot with synchronized axes
        """
        # Create subplots with secondary y-axes for different sensor scales
        fig = make_subplots(
            rows=len(self.sensors), 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=[sensor.replace('_', ' ').title() for sensor in self.sensors]
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, sensor in enumerate(self.sensors):
            if sensor in data and data[sensor]:
                sensor_data = data[sensor]
                color = colors[i % len(colors)]
                
                # Main sensor line
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=sensor_data,
                        mode='lines',
                        name=sensor.replace('_', ' ').title(),
                        line=dict(color=color, width=2),
                        hovertemplate=(
                            f"<b>{sensor.replace('_', ' ').title()}</b><br>"
                            "Time: %{x}<br>"
                            "Value: %{y:.2f}<br>"
                            f"Status: {self.sensor_health.get(sensor, 'unknown')}"
                            "<extra></extra>"
                        )
                    ),
                    row=i+1, col=1
                )
                
                # Add threshold lines if enabled
                if config.get('show_thresholds', True) and sensor in self.thresholds:
                    thresholds = self.thresholds[sensor]
                    
                    # Critical thresholds (red)
                    if 'min' in thresholds:
                        fig.add_hline(
                            y=thresholds['min'], 
                            line_dash="dash", 
                            line_color="red",
                            opacity=0.7,
                            row=i+1, col=1
                        )
                    if 'max' in thresholds:
                        fig.add_hline(
                            y=thresholds['max'], 
                            line_dash="dash", 
                            line_color="red",
                            opacity=0.7,
                            row=i+1, col=1
                        )
                    
                    # Warning thresholds (orange)
                    if 'warning_min' in thresholds:
                        fig.add_hline(
                            y=thresholds['warning_min'], 
                            line_dash="dot", 
                            line_color="orange",
                            opacity=0.5,
                            row=i+1, col=1
                        )
                    if 'warning_max' in thresholds:
                        fig.add_hline(
                            y=thresholds['warning_max'], 
                            line_dash="dot", 
                            line_color="orange",
                            opacity=0.5,
                            row=i+1, col=1
                        )
                
                # Highlight anomalies if enabled
                if config.get('show_anomalies', True):
                    anomalies = self._detect_anomalies(sensor_data, sensor)
                    if anomalies:
                        anomaly_times = [timestamps[j] for j in anomalies]
                        anomaly_values = [sensor_data[j] for j in anomalies]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=anomaly_times,
                                y=anomaly_values,
                                mode='markers',
                                name=f'{sensor} Anomalies',
                                marker=dict(
                                    color='red',
                                    size=8,
                                    symbol='x',
                                    line=dict(width=2, color='darkred')
                                ),
                                showlegend=False,
                                hovertemplate=(
                                    f"<b>ANOMALY</b><br>"
                                    f"{sensor.replace('_', ' ').title()}<br>"
                                    "Time: %{x}<br>"
                                    "Value: %{y:.2f}<br>"
                                    "<extra></extra>"
                                )
                            ),
                            row=i+1, col=1
                        )
                
                # Configure y-axis
                y_axis_name = f'yaxis{i+1}' if i > 0 else 'yaxis'
                if config.get('auto_scale', True):
                    # Auto-scale with some padding
                    if sensor_data:
                        y_min = min(sensor_data)
                        y_max = max(sensor_data)
                        y_range = y_max - y_min
                        padding = y_range * 0.1 if y_range > 0 else 1
                        fig.update_layout(**{
                            y_axis_name: dict(
                                title=sensor.replace('_', ' ').title(),
                                range=[y_min - padding, y_max + padding]
                            )
                        })
                else:
                    # Use threshold-based scaling
                    if sensor in self.thresholds:
                        thresholds = self.thresholds[sensor]
                        y_min = thresholds.get('min', min(sensor_data) if sensor_data else 0)
                        y_max = thresholds.get('max', max(sensor_data) if sensor_data else 100)
                        fig.update_layout(**{
                            y_axis_name: dict(
                                title=sensor.replace('_', ' ').title(),
                                range=[y_min * 0.9, y_max * 1.1]
                            )
                        })
        
        # Update layout for synchronized zooming and panning
        fig.update_layout(
            title="Real-time Sensor Monitoring - Steel Casting Process",
            height=600,
            showlegend=False,
            hovermode='x unified',
            template='plotly_white',
            margin=dict(l=60, r=20, t=60, b=40)
        )
        
        # Configure x-axis for time range
        if timestamps:
            time_range_minutes = config.get('time_range_minutes', 30)
            end_time = timestamps[-1] if timestamps else datetime.now()
            start_time = end_time - timedelta(minutes=time_range_minutes)
            
            fig.update_layout(
                xaxis=dict(
                    title="Time",
                    range=[start_time, end_time],
                    type='date'
                )
            )
        
        return fig
    
    def create_detail_plots(self, 
                           data: Dict[str, List], 
                           timestamps: List[datetime],
                           config: Dict[str, Any]) -> go.Figure:
        """
        Create detailed individual sensor plots with enhanced features.
        
        Args:
            data: Dictionary of sensor data
            timestamps: List of timestamp values  
            config: Configuration dictionary
            
        Returns:
            go.Figure: Detailed sensor plots
        """
        # Create a 2x3 grid for individual sensor details
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[sensor.replace('_', ' ').title() for sensor in self.sensors[:6]],
            vertical_spacing=0.1,
            horizontal_spacing=0.08
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, sensor in enumerate(self.sensors[:6]):  # Limit to 6 sensors for 2x3 grid
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            if sensor in data and data[sensor]:
                sensor_data = data[sensor]
                color = colors[i % len(colors)]
                
                # Main line with enhanced styling
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=sensor_data,
                        mode='lines+markers',
                        name=sensor,
                        line=dict(color=color, width=2),
                        marker=dict(size=3, color=color),
                        hovertemplate=(
                            f"<b>{sensor.replace('_', ' ').title()}</b><br>"
                            "Time: %{x}<br>"
                            "Value: %{y:.3f}<br>"
                            f"Health: {self.sensor_health.get(sensor, 'unknown')}<br>"
                            f"Last Update: {self.last_update_times.get(sensor, 'N/A')}"
                            "<extra></extra>"
                        ),
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # Add statistical indicators
                if sensor_data:
                    mean_val = np.mean(sensor_data)
                    std_val = np.std(sensor_data)
                    
                    # Mean line
                    fig.add_hline(
                        y=mean_val,
                        line_dash="dot",
                        line_color="gray",
                        opacity=0.7,
                        row=row, col=col
                    )
                    
                    # Standard deviation bands
                    fig.add_hrect(
                        y0=mean_val - std_val,
                        y1=mean_val + std_val,
                        fillcolor="lightblue",
                        opacity=0.2,
                        layer="below",
                        line_width=0,
                        row=row, col=col
                    )
                
                # Add threshold indicators
                if sensor in self.thresholds:
                    thresholds = self.thresholds[sensor]
                    
                    # Critical thresholds
                    for thresh_type, color_thresh in [('min', 'red'), ('max', 'red')]:
                        if thresh_type in thresholds:
                            fig.add_hline(
                                y=thresholds[thresh_type],
                                line_dash="solid",
                                line_color=color_thresh,
                                line_width=1,
                                opacity=0.8,
                                row=row, col=col
                            )
                    
                    # Warning thresholds
                    for thresh_type, color_thresh in [('warning_min', 'orange'), ('warning_max', 'orange')]:
                        if thresh_type in thresholds:
                            fig.add_hline(
                                y=thresholds[thresh_type],
                                line_dash="dash",
                                line_color=color_thresh,
                                line_width=1,
                                opacity=0.6,
                                row=row, col=col
                            )
        
        fig.update_layout(
            title="Individual Sensor Detail Analysis",
            height=800,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def _detect_anomalies(self, sensor_data: List[float], sensor_name: str) -> List[int]:
        """
        Detect anomalies in sensor data using multiple methods.
        
        Args:
            sensor_data: List of sensor values
            sensor_name: Name of the sensor
            
        Returns:
            List of indices where anomalies are detected
        """
        if not sensor_data or len(sensor_data) < 5:
            return []
        
        anomalies = []
        data_array = np.array(sensor_data)
        
        # Method 1: Threshold-based detection
        if sensor_name in self.thresholds:
            thresholds = self.thresholds[sensor_name]
            min_thresh = thresholds.get('min', float('-inf'))
            max_thresh = thresholds.get('max', float('inf'))
            
            threshold_anomalies = np.where(
                (data_array < min_thresh) | (data_array > max_thresh)
            )[0].tolist()
            anomalies.extend(threshold_anomalies)
        
        # Method 2: Statistical outliers (IQR method)
        if len(data_array) >= 10:
            Q1 = np.percentile(data_array, 25)
            Q3 = np.percentile(data_array, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            statistical_anomalies = np.where(
                (data_array < lower_bound) | (data_array > upper_bound)
            )[0].tolist()
            anomalies.extend(statistical_anomalies)
        
        # Method 3: Rate of change detection (for recent data)
        if len(data_array) >= 3:
            # Check last few points for sudden changes
            recent_data = data_array[-10:]  # Last 10 points
            if len(recent_data) >= 3:
                gradients = np.gradient(recent_data)
                mean_grad = np.mean(np.abs(gradients))
                std_grad = np.std(gradients)
                
                # Flag points with unusually high rate of change
                threshold_grad = mean_grad + 2 * std_grad
                rapid_change_indices = np.where(np.abs(gradients) > threshold_grad)[0]
                
                # Convert to global indices
                global_indices = [len(data_array) - len(recent_data) + idx for idx in rapid_change_indices]
                anomalies.extend(global_indices)
        
        # Remove duplicates and return
        return list(set(anomalies))
    
    def update_sensor_health(self, data: Dict[str, List], timestamps: List[datetime]) -> Dict[str, str]:
        """
        Update sensor health status based on data quality and recency.
        
        Args:
            data: Dictionary of sensor data
            timestamps: List of timestamps
            
        Returns:
            Dictionary of sensor health statuses
        """
        current_time = datetime.now()
        health_status = {}
        
        for sensor in self.sensors:
            if sensor not in data or not data[sensor]:
                health_status[sensor] = 'offline'
                continue
            
            sensor_data = data[sensor]
            
            # Check data recency
            if timestamps and len(timestamps) > 0:
                last_update = timestamps[-1]
                time_since_update = (current_time - last_update).total_seconds()
                
                if time_since_update > 30:  # No data for 30 seconds
                    health_status[sensor] = 'stale'
                    continue
            
            # Check for threshold violations
            anomalies = self._detect_anomalies(sensor_data, sensor)
            recent_anomalies = [idx for idx in anomalies if idx >= len(sensor_data) - 10]
            
            if len(recent_anomalies) > 3:  # More than 3 anomalies in last 10 points
                health_status[sensor] = 'critical'
            elif len(recent_anomalies) > 0:  # Some anomalies but not critical
                health_status[sensor] = 'warning'
            else:
                # Check data quality (variance, missing values, etc.)
                if len(sensor_data) > 5:
                    recent_data = sensor_data[-10:]
                    variance = np.var(recent_data)
                    
                    # If variance is too low, sensor might be stuck
                    if variance < 0.001:
                        health_status[sensor] = 'warning'
                    else:
                        health_status[sensor] = 'good'
                else:
                    health_status[sensor] = 'unknown'
        
        self.sensor_health = health_status
        return health_status
    
    def create_health_indicators(self, health_status: Dict[str, str]) -> html.Div:
        """
        Create visual health status indicators for all sensors.
        
        Args:
            health_status: Dictionary of sensor health statuses
            
        Returns:
            html.Div: Health indicator layout
        """
        status_colors = {
            'good': 'success',
            'warning': 'warning', 
            'critical': 'danger',
            'offline': 'secondary',
            'stale': 'info',
            'unknown': 'light'
        }
        
        status_icons = {
            'good': '✅',
            'warning': '⚠️',
            'critical': '🚨',
            'offline': '⭕',
            'stale': '⏳',
            'unknown': '❓'
        }
        
        indicators = []
        
        for sensor in self.sensors:
            status = health_status.get(sensor, 'unknown')
            color = status_colors.get(status, 'light')
            icon = status_icons.get(status, '❓')
            
            indicators.append(
                dbc.Col([
                    dbc.Badge([
                        html.Span(icon, className="me-1"),
                        html.Span(sensor.replace('_', ' ').title()),
                        html.Br(),
                        html.Small(status.upper(), className="text-muted")
                    ], 
                    color=color, 
                    className="p-2 w-100 text-center",
                    style={"minHeight": "60px", "display": "flex", "flexDirection": "column", "justifyContent": "center"})
                ], md=2, className="mb-2")
            )
        
        return dbc.Row(indicators)
    
    def add_data_point(self, sensor_data: Dict[str, float], timestamp: datetime = None):
        """
        Add a new data point to the rolling window buffers.
        
        Args:
            sensor_data: Dictionary of sensor values {sensor_name: value}
            timestamp: Timestamp for the data point (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Add timestamp to buffer
        self.timestamp_buffer.append(timestamp)
        
        # Add sensor data to respective buffers
        for sensor in self.sensors:
            if sensor in sensor_data:
                self.data_buffers[sensor].append(sensor_data[sensor])
                self.last_update_times[sensor] = timestamp
            else:
                # Add NaN for missing sensors to maintain buffer alignment
                self.data_buffers[sensor].append(float('nan'))
    
    def get_current_data(self) -> Tuple[Dict[str, List], List[datetime]]:
        """
        Get current data from all buffers.
        
        Returns:
            Tuple of (sensor_data_dict, timestamps)
        """
        sensor_data = {}
        for sensor in self.sensors:
            # Filter out None values but maintain index alignment
            sensor_data[sensor] = [val for val in self.data_buffers[sensor] if val is not None]
        
        timestamps = list(self.timestamp_buffer)
        return sensor_data, timestamps
    
    def generate_mock_data_point(self) -> Dict[str, float]:
        """
        Generate mock sensor data for testing purposes.
        
        Returns:
            Dictionary of mock sensor values
        """
        # Base values for different sensors
        base_values = {
            'casting_speed': 1.2,
            'mold_temperature': 1530,
            'mold_level': 150,
            'strand_temperature': 1300,
            'cooling_water_flow': 100,
            'tundish_temperature': 1555
        }
        
        mock_data = {}
        current_time = datetime.now()
        
        for sensor in self.sensors:
            if sensor in base_values:
                base_value = base_values[sensor]
                
                # Add some realistic variation
                if 'temperature' in sensor:
                    # Temperature sensors have slower, smaller variations
                    noise = np.random.normal(0, base_value * 0.01)  # 1% noise
                    trend = 5 * np.sin(current_time.timestamp() / 300)  # 5-minute cycle
                else:
                    # Other sensors have more variation
                    noise = np.random.normal(0, base_value * 0.02)  # 2% noise
                    trend = base_value * 0.05 * np.sin(current_time.timestamp() / 180)  # 3-minute cycle
                
                # Occasionally introduce anomalies (5% chance)
                if np.random.random() < 0.05:
                    anomaly_factor = np.random.choice([-1, 1]) * np.random.uniform(0.1, 0.3)
                    noise += base_value * anomaly_factor
                
                mock_data[sensor] = base_value + trend + noise
            else:
                # Default value for unknown sensors
                mock_data[sensor] = np.random.uniform(0, 100)
        
        return mock_data