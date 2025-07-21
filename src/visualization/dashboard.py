import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
from typing import Dict
import numpy as np

class DefectMonitoringDashboard:
    """Real-time monitoring dashboard for steel defect prediction"""
    
    def __init__(self, config: Dict):
        """
        Initialize dashboard.
        
        Args:
            config (Dict): Dashboard configuration
        """
        self.app = dash.Dash(__name__)
        self.config = config
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self) -> None:
        """Setup dashboard layout with all components."""
        self.app.layout = html.Div([
            html.H1("Steel Defect Prediction Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': '30px'}),
            
            # Control panel
            html.Div([
                html.H3("Control Panel"),
                html.Button("Start Stream", id="start-button"),
                html.Button("Stop Stream", id="stop-button"),
                html.Button("Reset Stream", id="reset-button"),
            ], style={'marginBottom': '20px'}),
            
            # Real-time sensor plots
            html.Div([
                html.H3("Real-time Sensor Data"),
                dcc.Graph(id='sensor-timeseries'),
            ]),
            
            # Prediction probability gauge
            html.Div([
                html.H3("Defect Prediction"),
                dcc.Graph(id='prediction-gauge'),
            ]),
            
            # Historical predictions
            html.Div([
                html.H3("Prediction History"),
                dcc.Graph(id='prediction-history'),
            ]),
            
            # Model comparison
            html.Div([
                html.H3("Model Comparison"),
                dcc.Graph(id='model-comparison'),
            ]),
            
            # System status
            html.Div([
                html.H3("System Status"),
                html.Div(id='system-status'),
            ]),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=5000,  # 5 seconds
                n_intervals=0
            ),
            
            # Data storage for callbacks
            dcc.Store(id='sensor-data-store'),
            dcc.Store(id='prediction-data-store'),
        ])
    
    def setup_callbacks(self) -> None:
        """Setup interactive callbacks for dashboard updates."""
        
        @self.app.callback(
            [Output('sensor-timeseries', 'figure'),
             Output('prediction-gauge', 'figure'),
             Output('prediction-history', 'figure'),
             Output('system-status', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            """Update all dashboard components with latest data."""
            
            # Create sample sensor plot
            sensor_fig = self.create_sample_sensor_plot()
            
            # Create sample prediction gauge
            # TODO: Replace with actual predictive model or data source integration.
            # For demonstration purposes, using a random value for prediction probability.
            import random
            prediction_prob = random.uniform(0.1, 0.9)
            gauge_fig = self.create_prediction_gauge(prediction_prob)
            
            # Create sample prediction history
            history_fig = self.create_sample_prediction_history()
            
            # System status
            status_text = f"Dashboard updated at interval {n}. System operational."
            
            return sensor_fig, gauge_fig, history_fig, status_text
    
    def create_sensor_plot(self, sensor_data: pd.DataFrame) -> go.Figure:
        """
        Create sensor time series plot.
        
        Args:
            sensor_data (pd.DataFrame): Recent sensor data
            
        Returns:
            go.Figure: Plotly figure
        """
        fig = go.Figure()
        
        # Get sensor columns (exclude metadata)
        sensor_cols = [col for col in sensor_data.columns if col not in ['cast_id', 'defect_label']]
        
        for i, sensor in enumerate(sensor_cols[:5]):  # Limit to 5 sensors for readability
            if sensor in sensor_data.columns:
                fig.add_trace(go.Scatter(
                    x=sensor_data.index,
                    y=sensor_data[sensor],
                    mode='lines',
                    name=sensor.replace('_', ' ').title(),
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title="Real-time Sensor Data",
            xaxis_title="Time",
            yaxis_title="Sensor Values",
            height=400,
            template="plotly_white",
            showlegend=True
        )
        
        return fig
    
    def create_sample_sensor_plot(self) -> go.Figure:
        """Create a sample sensor plot for demonstration."""
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Generate sample data
        times = pd.date_range(start=datetime.now() - timedelta(minutes=30), 
                            end=datetime.now(), freq='1min')
        
        data = pd.DataFrame({
            'casting_speed': 1.2 + 0.1 * np.sin(np.arange(len(times)) * 0.1) + 0.05 * np.random.randn(len(times)),
            'mold_temperature': 1520 + 20 * np.sin(np.arange(len(times)) * 0.05) + 5 * np.random.randn(len(times)),
            'mold_level': 150 + 10 * np.sin(np.arange(len(times)) * 0.08) + 2 * np.random.randn(len(times))
        }, index=times)
        
        return self.create_sensor_plot(data)
    
    def create_prediction_gauge(self, prediction_prob: float) -> go.Figure:
        """
        Create prediction probability gauge.
        
        Args:
            prediction_prob (float): Current prediction probability
            
        Returns:
            go.Figure: Gauge chart
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction_prob,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Defect Probability"},
            delta={'reference': 0.5},
            gauge={'axis': {'range': [None, 1]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 0.3], 'color': "lightgreen"},
                       {'range': [0.3, 0.7], 'color': "yellow"},
                       {'range': [0.7, 1], 'color': "red"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 0.8}}))
        
        fig.update_layout(height=400, template="plotly_white")
        return fig
    
    def create_prediction_history_plot(self, history_data: pd.DataFrame) -> go.Figure:
        """
        Create prediction history plot.
        
        Args:
            history_data (pd.DataFrame): Historical predictions
            
        Returns:
            go.Figure: History plot
        """
        fig = go.Figure()
        
        if 'prediction' in history_data.columns:
            fig.add_trace(go.Scatter(
                x=history_data.index,
                y=history_data['prediction'],
                mode='lines+markers',
                name='Prediction Probability',
                line=dict(color='blue', width=2)
            ))
        
        # Add threshold line
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                     annotation_text="Threshold")
        
        fig.update_layout(
            title="Prediction History",
            xaxis_title="Time",
            yaxis_title="Prediction Probability",
            yaxis=dict(range=[0, 1]),
            height=400,
            template="plotly_white"
        )
        
        return fig
    
    def create_sample_prediction_history(self) -> go.Figure:
        """Create sample prediction history for demonstration."""
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta
        
        times = pd.date_range(start=datetime.now() - timedelta(hours=2), 
                            end=datetime.now(), freq='10min')
        
        # Generate sample predictions with some trend
        predictions = 0.3 + 0.2 * np.sin(np.arange(len(times)) * 0.2) + 0.1 * np.random.randn(len(times))
        predictions = np.clip(predictions, 0, 1)
        
        history_data = pd.DataFrame({'prediction': predictions}, index=times)
        return self.create_prediction_history_plot(history_data)
    
    def create_model_comparison_plot(self, comparison_data: Dict) -> go.Figure:
        """
        Create model comparison visualization.
        
        Args:
            comparison_data (Dict): Model comparison data
            
        Returns:
            go.Figure: Comparison plot
        """
        # TODO: Implement model comparison plot
        pass
    
    def run(self, debug: bool = False, host: str = "127.0.0.1") -> None:
        """
        Run dashboard server.
        
        Args:
            debug (bool): Run in debug mode
            host (str): Host address
        """
        port = self.config['inference']['dashboard_port']
        self.app.run_server(debug=debug, host=host, port=port)