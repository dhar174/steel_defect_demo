"""
System Monitoring Dashboard Components for Steel Defect Detection System.

This module provides components for monitoring the operational health and performance
of the entire defect prediction system, including:
- Real-time system performance metrics (CPU, memory, disk)
- Model inference latency and throughput tracking
- Data pipeline health monitoring
- Error rate and system availability metrics
- Resource utilization trends (historical charts)
- Integration status indicators for external systems
"""

import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import deque
import time
import logging

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import html, dcc
import dash_bootstrap_components as dbc

logger = logging.getLogger(__name__)


class SystemMonitoringComponent:
    """
    System monitoring component for tracking operational health and performance
    of the steel defect prediction system.
    
    Provides real-time monitoring of system resources, model performance,
    data pipeline health, and integration status.
    """
    
    def __init__(self, 
                 component_id: str = "system-monitoring",
                 buffer_size: int = 1000,
                 update_interval: int = 5000):
        """
        Initialize the system monitoring component.
        
        Args:
            component_id: Unique identifier for the component
            buffer_size: Maximum number of data points to keep in history
            update_interval: Update interval in milliseconds for real-time updates
        """
        self.component_id = component_id
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        
        # System metrics buffers
        self.system_metrics_buffer = deque(maxlen=buffer_size)
        self.timestamp_buffer = deque(maxlen=buffer_size)
        
        # Model performance tracking
        self.model_latency_buffer = deque(maxlen=buffer_size)
        self.model_throughput_buffer = deque(maxlen=buffer_size)
        
        # Data pipeline metrics
        self.pipeline_metrics_buffer = deque(maxlen=buffer_size)
        
        # Error tracking
        self.error_count_buffer = deque(maxlen=buffer_size)
        self.system_availability_buffer = deque(maxlen=buffer_size)
        
        # Integration status tracking
        self.integration_status = {
            'data_source': 'unknown',
            'model_service': 'unknown',
            'alert_service': 'unknown',
            'database': 'unknown',
            'notification_service': 'unknown'
        }
        
        # Performance thresholds
        self.thresholds = {
            'cpu_usage': {'warning': 70, 'critical': 85},
            'memory_usage': {'warning': 80, 'critical': 90},
            'disk_usage': {'warning': 85, 'critical': 95},
            'model_latency': {'warning': 1000, 'critical': 2000},  # milliseconds
            'error_rate': {'warning': 0.05, 'critical': 0.10},  # 5% and 10%
            'availability': {'warning': 0.95, 'critical': 0.90}  # 95% and 90%
        }
        
        # Initialize with current system state
        self._collect_initial_metrics()
    
    def _collect_initial_metrics(self):
        """Collect initial system metrics to populate buffers."""
        current_time = datetime.now()
        
        # Collect system metrics
        system_metrics = self._get_current_system_metrics()
        self.system_metrics_buffer.append(system_metrics)
        self.timestamp_buffer.append(current_time)
        
        # Initialize other metrics with default values
        self.model_latency_buffer.append(100)  # Default 100ms
        self.model_throughput_buffer.append(10)  # Default 10 predictions/sec
        
        pipeline_metrics = {
            'data_ingestion_rate': 50,
            'processing_queue_size': 5,
            'failed_processes': 0
        }
        self.pipeline_metrics_buffer.append(pipeline_metrics)
        
        self.error_count_buffer.append(0)
        self.system_availability_buffer.append(1.0)
    
    def create_layout(self) -> html.Div:
        """
        Create the complete system monitoring dashboard layout.
        
        Returns:
            html.Div: Complete dashboard layout with all monitoring components
        """
        return html.Div([
            # System Status Overview Cards
            html.H3("System Health Overview", className="mb-3"),
            self._create_status_overview_cards(),
            
            # Real-time System Performance Gauges
            html.H4("Real-time System Performance", className="mt-4 mb-3"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id=f'{self.component_id}-cpu-gauge',
                        config={'displayModeBar': False},
                        style={'height': '300px'}
                    )
                ], width=4),
                dbc.Col([
                    dcc.Graph(
                        id=f'{self.component_id}-memory-gauge',
                        config={'displayModeBar': False},
                        style={'height': '300px'}
                    )
                ], width=4),
                dbc.Col([
                    dcc.Graph(
                        id=f'{self.component_id}-disk-gauge',
                        config={'displayModeBar': False},
                        style={'height': '300px'}
                    )
                ], width=4)
            ], className="mb-4"),
            
            # Model Performance Metrics
            html.H4("Model Performance Metrics", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id=f'{self.component_id}-latency-chart',
                        config={'displayModeBar': True},
                        style={'height': '300px'}
                    )
                ], width=6),
                dbc.Col([
                    dcc.Graph(
                        id=f'{self.component_id}-throughput-chart',
                        config={'displayModeBar': True},
                        style={'height': '300px'}
                    )
                ], width=6)
            ], className="mb-4"),
            
            # Data Pipeline Health and Error Tracking
            html.H4("Data Pipeline & Error Monitoring", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id=f'{self.component_id}-pipeline-health',
                        config={'displayModeBar': True},
                        style={'height': '350px'}
                    )
                ], width=6),
                dbc.Col([
                    dcc.Graph(
                        id=f'{self.component_id}-error-availability',
                        config={'displayModeBar': True},
                        style={'height': '350px'}
                    )
                ], width=6)
            ], className="mb-4"),
            
            # Resource Utilization Trends
            html.H4("Resource Utilization Trends", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id=f'{self.component_id}-resource-trends',
                        config={'displayModeBar': True},
                        style={'height': '400px'}
                    )
                ], width=12)
            ], className="mb-4"),
            
            # Integration Status Indicators
            html.H4("Integration Status", className="mb-3"),
            html.Div(id=f'{self.component_id}-integration-status'),
            
            # Hidden components for data storage
            dcc.Store(id=f'{self.component_id}-data-store', data={}),
            
            # Real-time update interval
            dcc.Interval(
                id=f'{self.component_id}-interval',
                interval=self.update_interval,
                n_intervals=0
            )
        ], id=f'{self.component_id}-container')
    
    def _create_status_overview_cards(self) -> dbc.Row:
        """Create overview status cards for key metrics."""
        # Get current metrics
        current_metrics = self._get_current_system_metrics()
        current_time = datetime.now()
        
        # Calculate current values
        cpu_usage = current_metrics.get('cpu_percent', 0)
        memory_usage = current_metrics.get('memory_percent', 0)
        
        # Model performance (use latest from buffer or defaults)
        model_latency = self.model_latency_buffer[-1] if self.model_latency_buffer else 100
        
        # System availability (use latest or default)
        availability = self.system_availability_buffer[-1] if self.system_availability_buffer else 1.0
        
        cards = [
            self._create_metric_card(
                "CPU Usage", f"{cpu_usage:.1f}%", "🖥️",
                self._get_status_color(cpu_usage, self.thresholds['cpu_usage'])
            ),
            self._create_metric_card(
                "Memory Usage", f"{memory_usage:.1f}%", "💾",
                self._get_status_color(memory_usage, self.thresholds['memory_usage'])
            ),
            self._create_metric_card(
                "Model Latency", f"{model_latency:.0f}ms", "⚡",
                self._get_status_color(model_latency, self.thresholds['model_latency'])
            ),
            self._create_metric_card(
                "System Availability", f"{availability*100:.1f}%", "🛡️",
                self._get_status_color(availability*100, {'warning': 95, 'critical': 90}, reverse=True)
            )
        ]
        
        return dbc.Row([
            dbc.Col(card, width=3) for card in cards
        ], className="mb-4")
    
    def _create_metric_card(self, title: str, value: str, icon: str, color: str) -> dbc.Card:
        """Create a metric overview card."""
        return dbc.Card([
            dbc.CardBody([
                html.H6([
                    html.Span(icon, style={'marginRight': '8px'}),
                    title
                ], className="card-title text-center mb-1"),
                html.H4(
                    value,
                    className=f"text-{color} text-center mb-0 fw-bold"
                )
            ])
        ], color=color, outline=True, className="text-center")
    
    def _get_status_color(self, value: float, thresholds: Dict[str, float], reverse: bool = False) -> str:
        """Get color based on threshold levels."""
        warning = thresholds['warning']
        critical = thresholds['critical']
        
        if not reverse:
            if value >= critical:
                return 'danger'
            elif value >= warning:
                return 'warning'
            else:
                return 'success'
        else:
            if value <= critical:
                return 'danger'
            elif value <= warning:
                return 'warning'
            else:
                return 'success'
    
    def create_cpu_gauge(self, cpu_percent: float) -> go.Figure:
        """Create CPU usage gauge."""
        return self._create_performance_gauge(
            value=cpu_percent,
            title="CPU Usage",
            unit="%",
            max_value=100,
            thresholds=self.thresholds['cpu_usage']
        )
    
    def create_memory_gauge(self, memory_percent: float) -> go.Figure:
        """Create memory usage gauge."""
        return self._create_performance_gauge(
            value=memory_percent,
            title="Memory Usage",
            unit="%",
            max_value=100,
            thresholds=self.thresholds['memory_usage']
        )
    
    def create_disk_gauge(self, disk_percent: float) -> go.Figure:
        """Create disk usage gauge."""
        return self._create_performance_gauge(
            value=disk_percent,
            title="Disk Usage",
            unit="%",
            max_value=100,
            thresholds=self.thresholds['disk_usage']
        )
    
    def _create_performance_gauge(self, 
                                value: float, 
                                title: str, 
                                unit: str,
                                max_value: float = 100,
                                thresholds: Dict[str, float] = None) -> go.Figure:
        """Create a performance gauge using plotly.indicator."""
        if thresholds is None:
            thresholds = {'warning': 70, 'critical': 85}
        
        # Determine gauge color based on thresholds
        if value >= thresholds['critical']:
            gauge_color = '#DC143C'  # Red
        elif value >= thresholds['warning']:
            gauge_color = '#FFD700'  # Yellow
        else:
            gauge_color = '#2E8B57'  # Green
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 16}},
            number={'font': {'size': 20}, 'suffix': unit},
            delta={
                'reference': thresholds['warning'],
                'increasing': {'color': '#DC143C'},
                'decreasing': {'color': '#2E8B57'}
            },
            gauge={
                'axis': {
                    'range': [None, max_value],
                    'tickwidth': 1,
                    'tickcolor': "darkblue"
                },
                'bar': {'color': gauge_color, 'thickness': 0.75},
                'steps': [
                    {'range': [0, thresholds['warning']], 'color': '#E8F5E8'},
                    {'range': [thresholds['warning'], thresholds['critical']], 'color': '#FFF8DC'},
                    {'range': [thresholds['critical'], max_value], 'color': '#FFE4E1'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': thresholds['critical']
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    def create_model_latency_chart(self, timestamps: List[datetime], latencies: List[float]) -> go.Figure:
        """Create model inference latency chart."""
        fig = go.Figure()
        
        if timestamps and latencies:
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=latencies,
                mode='lines+markers',
                name='Inference Latency',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=4),
                hovertemplate='Time: %{x}<br>Latency: %{y:.0f}ms<extra></extra>'
            ))
            
            # Add threshold lines
            fig.add_hline(
                y=self.thresholds['model_latency']['warning'],
                line_dash="dash",
                line_color="orange",
                annotation_text="Warning Threshold"
            )
            
            fig.add_hline(
                y=self.thresholds['model_latency']['critical'],
                line_dash="solid",
                line_color="red",
                annotation_text="Critical Threshold"
            )
        
        fig.update_layout(
            title="Model Inference Latency",
            xaxis_title="Time",
            yaxis_title="Latency (ms)",
            height=300,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def create_model_throughput_chart(self, timestamps: List[datetime], throughputs: List[float]) -> go.Figure:
        """Create model throughput chart."""
        fig = go.Figure()
        
        if timestamps and throughputs:
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=throughputs,
                mode='lines+markers',
                name='Throughput',
                line=dict(color='#2ca02c', width=2),
                marker=dict(size=4),
                fill='tonexty',
                hovertemplate='Time: %{x}<br>Throughput: %{y:.1f} pred/sec<extra></extra>'
            ))
        
        fig.update_layout(
            title="Model Inference Throughput",
            xaxis_title="Time",
            yaxis_title="Predictions/Second",
            height=300,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def create_pipeline_health_chart(self, timestamps: List[datetime], 
                                   pipeline_metrics: List[Dict[str, Any]]) -> go.Figure:
        """Create data pipeline health visualization."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Data Ingestion Rate', 'Processing Queue Size'],
            vertical_spacing=0.15,
            shared_xaxes=True
        )
        
        if timestamps and pipeline_metrics:
            # Extract metrics
            ingestion_rates = [m.get('data_ingestion_rate', 0) for m in pipeline_metrics]
            queue_sizes = [m.get('processing_queue_size', 0) for m in pipeline_metrics]
            
            # Data ingestion rate
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=ingestion_rates,
                    mode='lines+markers',
                    name='Ingestion Rate',
                    line=dict(color='#ff7f0e', width=2),
                    marker=dict(size=3),
                    hovertemplate='Time: %{x}<br>Rate: %{y:.1f} records/sec<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Processing queue size
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=queue_sizes,
                    mode='lines+markers',
                    name='Queue Size',
                    line=dict(color='#d62728', width=2),
                    marker=dict(size=3),
                    fill='tozeroy',
                    hovertemplate='Time: %{x}<br>Queue Size: %{y}<extra></extra>'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title="Data Pipeline Health",
            height=350,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def create_error_availability_chart(self, timestamps: List[datetime], 
                                      error_counts: List[int],
                                      availability: List[float]) -> go.Figure:
        """Create error rate and system availability chart."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Error Count', 'System Availability'],
            vertical_spacing=0.15,
            shared_xaxes=True
        )
        
        if timestamps and error_counts and availability:
            # Error count
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=error_counts,
                    mode='lines+markers',
                    name='Error Count',
                    line=dict(color='#d62728', width=2),
                    marker=dict(size=3),
                    fill='tozeroy',
                    hovertemplate='Time: %{x}<br>Errors: %{y}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # System availability
            availability_percent = [a * 100 for a in availability]
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=availability_percent,
                    mode='lines+markers',
                    name='Availability',
                    line=dict(color='#2ca02c', width=2),
                    marker=dict(size=3),
                    hovertemplate='Time: %{x}<br>Availability: %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add availability threshold line
            fig.add_hline(
                y=95,
                line_dash="dash",
                line_color="orange",
                row=2, col=1
            )
        
        fig.update_layout(
            title="Error Rate & System Availability",
            height=350,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def create_resource_trends_chart(self, timestamps: List[datetime],
                                   system_metrics: List[Dict[str, Any]]) -> go.Figure:
        """Create resource utilization trends chart."""
        fig = go.Figure()
        
        if timestamps and system_metrics:
            # Extract metrics
            cpu_usage = [m.get('cpu_percent', 0) for m in system_metrics]
            memory_usage = [m.get('memory_percent', 0) for m in system_metrics]
            disk_usage = [m.get('disk_percent', 0) for m in system_metrics]
            
            # CPU usage
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=cpu_usage,
                mode='lines',
                name='CPU Usage (%)',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='Time: %{x}<br>CPU: %{y:.1f}%<extra></extra>'
            ))
            
            # Memory usage
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=memory_usage,
                mode='lines',
                name='Memory Usage (%)',
                line=dict(color='#ff7f0e', width=2),
                hovertemplate='Time: %{x}<br>Memory: %{y:.1f}%<extra></extra>'
            ))
            
            # Disk usage
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=disk_usage,
                mode='lines',
                name='Disk Usage (%)',
                line=dict(color='#2ca02c', width=2),
                hovertemplate='Time: %{x}<br>Disk: %{y:.1f}%<extra></extra>'
            ))
        
        fig.update_layout(
            title="Resource Utilization Trends",
            xaxis_title="Time",
            yaxis_title="Usage (%)",
            yaxis=dict(range=[0, 100]),
            height=400,
            template='plotly_white',
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def create_integration_status_indicators(self) -> html.Div:
        """Create integration status indicators for external systems."""
        status_colors = {
            'healthy': 'success',
            'warning': 'warning',
            'error': 'danger',
            'unknown': 'secondary'
        }
        
        status_icons = {
            'healthy': '✅',
            'warning': '⚠️',
            'error': '❌',
            'unknown': '❓'
        }
        
        integration_info = {
            'data_source': {'label': 'Data Source', 'description': 'Sensor data input'},
            'model_service': {'label': 'Model Service', 'description': 'ML inference service'},
            'alert_service': {'label': 'Alert Service', 'description': 'Notification system'},
            'database': {'label': 'Database', 'description': 'Data storage'},
            'notification_service': {'label': 'Notifications', 'description': 'Alert delivery'}
        }
        
        indicators = []
        
        for service, status in self.integration_status.items():
            if service in integration_info:
                info = integration_info[service]
                color = status_colors.get(status, 'secondary')
                icon = status_icons.get(status, '❓')
                
                indicators.append(
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6([
                                    html.Span(icon, style={'marginRight': '8px'}),
                                    info['label']
                                ], className="card-title text-center mb-1"),
                                html.P(
                                    info['description'],
                                    className="card-text text-center small text-muted mb-1"
                                ),
                                dbc.Badge(
                                    status.upper(),
                                    color=color,
                                    className="w-100"
                                )
                            ], className="p-2")
                        ], outline=True, color=color)
                    ], width=2, className="mb-2")
                )
        
        return dbc.Row(indicators)
    
    def _get_current_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics using psutil."""
        try:
            # CPU usage (instantaneous, non-blocking)
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage for root partition
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Additional metrics
            network_io = psutil.net_io_counters()
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk_percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_total_gb': disk.total / (1024**3),
                'network_bytes_sent': network_io.bytes_sent,
                'network_bytes_recv': network_io.bytes_recv
            }
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            # Return default values on error
            return {
                'cpu_percent': 0,
                'memory_percent': 0,
                'memory_used_gb': 0,
                'memory_total_gb': 8,
                'disk_percent': 0,
                'disk_used_gb': 0,
                'disk_total_gb': 100,
                'network_bytes_sent': 0,
                'network_bytes_recv': 0
            }
    
    def add_model_performance_data(self, latency_ms: float, throughput_per_sec: float):
        """Add model performance data point."""
        self.model_latency_buffer.append(latency_ms)
        self.model_throughput_buffer.append(throughput_per_sec)
    
    def add_pipeline_metrics(self, ingestion_rate: float, queue_size: int, failed_processes: int):
        """Add data pipeline metrics."""
        pipeline_metrics = {
            'data_ingestion_rate': ingestion_rate,
            'processing_queue_size': queue_size,
            'failed_processes': failed_processes
        }
        self.pipeline_metrics_buffer.append(pipeline_metrics)
    
    def add_error_data(self, error_count: int, availability: float):
        """Add error and availability data."""
        self.error_count_buffer.append(error_count)
        self.system_availability_buffer.append(availability)
    
    def update_integration_status(self, service: str, status: str):
        """Update integration status for a service."""
        if service in self.integration_status:
            self.integration_status[service] = status
    
    def collect_and_store_metrics(self):
        """Collect current system metrics and store in buffers."""
        current_time = datetime.now()
        
        # Collect system metrics
        system_metrics = self._get_current_system_metrics()
        self.system_metrics_buffer.append(system_metrics)
        self.timestamp_buffer.append(current_time)
        
        # Simulate model performance metrics (in real implementation, these would come from actual monitoring)
        base_latency = 100
        latency_variation = np.random.normal(0, self.LATENCY_VARIATION_STD_DEV)
        simulated_latency = max(50, base_latency + latency_variation)
        
        base_throughput = 10
        throughput_variation = np.random.normal(0, self.THROUGHPUT_VARIATION_STD_DEV)
        simulated_throughput = max(1, base_throughput + throughput_variation)
        
        self.add_model_performance_data(simulated_latency, simulated_throughput)
        
        # Simulate pipeline metrics
        base_ingestion = 50
        ingestion_variation = np.random.normal(0, 10)
        simulated_ingestion = max(0, base_ingestion + ingestion_variation)
        
        queue_size = max(0, int(np.random.normal(5, 2)))
        failed_processes = int(np.random.poisson(0.1))  # Low failure rate
        
        self.add_pipeline_metrics(simulated_ingestion, queue_size, failed_processes)
        
        # Simulate error and availability data
        error_count = int(np.random.poisson(0.5))  # Low error rate
        availability = min(1.0, max(0.95, np.random.normal(0.99, 0.01)))
        
        self.add_error_data(error_count, availability)
        
        # Update integration status (simulate occasional status changes)
        if np.random.random() < self.STATUS_UPDATE_PROBABILITY:  # 10% chance to update status
            services = list(self.integration_status.keys())
            random_service = np.random.choice(services)
            statuses = ['healthy', 'warning', 'error']
            weights = [0.8, 0.15, 0.05]  # Mostly healthy
            new_status = np.random.choice(statuses, p=weights)
            self.update_integration_status(random_service, new_status)
    
    def get_current_data(self) -> Dict[str, Any]:
        """Get current data from all buffers for dashboard updates."""
        timestamps = list(self.timestamp_buffer)
        
        return {
            'timestamps': timestamps,
            'system_metrics': list(self.system_metrics_buffer),
            'model_latencies': list(self.model_latency_buffer),
            'model_throughputs': list(self.model_throughput_buffer),
            'pipeline_metrics': list(self.pipeline_metrics_buffer),
            'error_counts': list(self.error_count_buffer),
            'availability': list(self.system_availability_buffer),
            'integration_status': self.integration_status.copy()
        }


def create_sample_system_data() -> Dict[str, Any]:
    """
    Create sample system monitoring data for demonstration purposes.
    
    Returns:
        Dict[str, Any]: Sample system data for testing
    """
    # Generate sample data for the last hour
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    timestamps = pd.date_range(start=start_time, end=end_time, freq='30s')
    
    sample_data = {
        'timestamps': timestamps.tolist(),
        'system_metrics': [],
        'model_latencies': [],
        'model_throughputs': [],
        'pipeline_metrics': [],
        'error_counts': [],
        'availability': [],
        'integration_status': {
            'data_source': 'healthy',
            'model_service': 'healthy',
            'alert_service': 'warning',
            'database': 'healthy',
            'notification_service': 'healthy'
        }
    }
    
    # Generate realistic sample metrics
    for i, timestamp in enumerate(timestamps):
        # System metrics with some variation
        cpu_base = 45 + 20 * np.sin(i * 0.1) + np.random.normal(0, 5)
        memory_base = 60 + 15 * np.sin(i * 0.05) + np.random.normal(0, 3)
        disk_base = 75 + np.random.normal(0, 2)
        
        sample_data['system_metrics'].append({
            'cpu_percent': max(0, min(100, cpu_base)),
            'memory_percent': max(0, min(100, memory_base)),
            'disk_percent': max(0, min(100, disk_base)),
            'memory_used_gb': 4.8,
            'memory_total_gb': 8.0,
            'disk_used_gb': 150.0,
            'disk_total_gb': 200.0
        })
        
        # Model performance metrics
        latency_base = 120 + 30 * np.sin(i * 0.08) + np.random.normal(0, 10)
        throughput_base = 8 + 3 * np.sin(i * 0.06) + np.random.normal(0, 1)
        
        sample_data['model_latencies'].append(max(50, latency_base))
        sample_data['model_throughputs'].append(max(1, throughput_base))
        
        # Pipeline metrics
        ingestion_base = 50 + 10 * np.sin(i * 0.04) + np.random.normal(0, 5)
        queue_size = max(0, int(np.random.normal(5, 2)))
        failed_processes = int(np.random.poisson(0.1))
        
        sample_data['pipeline_metrics'].append({
            'data_ingestion_rate': max(0, ingestion_base),
            'processing_queue_size': queue_size,
            'failed_processes': failed_processes
        })
        
        # Error and availability data
        error_count = int(np.random.poisson(0.5))
        availability = min(1.0, max(0.95, np.random.normal(0.99, 0.01)))
        
        sample_data['error_counts'].append(error_count)
        sample_data['availability'].append(availability)
    
    return sample_data