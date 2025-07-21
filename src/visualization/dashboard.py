import dash
from dash import dcc, html, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
from typing import Dict, Optional
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DefectMonitoringDashboard:
    """Real-time monitoring dashboard for steel defect prediction with multi-page navigation"""
    
    def __init__(self, config: Dict):
        """
        Initialize dashboard with multi-page framework.
        
        Args:
            config (Dict): Dashboard configuration
        """
        # Initialize app with Bootstrap theme for responsive design
        self.app = dash.Dash(
            __name__, 
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        
        self.config = config
        self.default_theme = "plotly_white"
        self.refresh_interval = self.config.get('refresh_interval', 5000)  # Default 5 seconds
        
        # Set up the main layout and callbacks
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self) -> None:
        """Setup multi-page dashboard layout with navigation."""
        # Main layout with navigation and page content
        self.app.layout = dbc.Container([
            # Location component for URL routing
            dcc.Location(id='url', refresh=False),
            
            # Session storage for user preferences
            dcc.Store(id='session-store', storage_type='session'),
            dcc.Store(id='theme-store', storage_type='session', data=self.default_theme),
            dcc.Store(id='refresh-interval-store', storage_type='session', data=self.refresh_interval),
            
            # Navigation bar
            self.create_navbar(),
            
            # Error alert container
            dbc.Alert(
                id="error-alert",
                is_open=False,
                dismissable=True,
                color="danger",
                style={"margin": "10px 0"}
            ),
            
            # Main content area
            html.Div(id='page-content', style={"marginTop": "20px"}),
            
            # Global interval component for real-time updates
            dcc.Interval(
                id='global-interval',
                interval=self.refresh_interval,
                n_intervals=0
            ),
            
        ], fluid=True)
    
    def create_navbar(self) -> dbc.NavbarSimple:
        """Create responsive navigation bar with theme controls."""
        return dbc.NavbarSimple(
            children=[
                dbc.DropdownMenu(
                    children=[
                        dbc.DropdownMenuItem("Real-time Monitoring", href="/", external_link=False),
                        dbc.DropdownMenuItem("Model Comparison", href="/model-comparison", external_link=False),
                        dbc.DropdownMenuItem("Historical Analysis", href="/historical-analysis", external_link=False),
                        dbc.DropdownMenuItem("System Status", href="/system-status", external_link=False),
                    ],
                    nav=True,
                    in_navbar=True,
                    label="Pages",
                    id="pages-dropdown"
                ),
                dbc.DropdownMenu(
                    children=[
                        dbc.DropdownMenuItem("Light Theme", id="theme-light"),
                        dbc.DropdownMenuItem("Dark Theme", id="theme-dark"),
                        dbc.DropdownMenuItem("Default Theme", id="theme-default"),
                    ],
                    nav=True,
                    in_navbar=True,
                    label="Theme",
                    id="theme-dropdown"
                ),
                dbc.DropdownMenu(
                    children=[
                        dbc.DropdownMenuItem("1 Second", id="refresh-1s"),
                        dbc.DropdownMenuItem("5 Seconds", id="refresh-5s"),
                        dbc.DropdownMenuItem("10 Seconds", id="refresh-10s"),
                        dbc.DropdownMenuItem("30 Seconds", id="refresh-30s"),
                    ],
                    nav=True,
                    in_navbar=True,
                    label="Refresh Rate",
                    id="refresh-dropdown"
                ),
            ],
            brand="Steel Defect Prediction Dashboard",
            brand_href="/",
            color="dark",
            dark=True,
            className="mb-3"
        )
    
    def create_page_layout(self, page_name: str) -> html.Div:
        """Create layout for specific page."""
        layouts = {
            'real-time-monitoring': self.create_realtime_layout(),
            'model-comparison': self.create_model_comparison_layout(),
            'historical-analysis': self.create_historical_layout(),
            'system-status': self.create_system_status_layout()
        }
        return layouts.get(page_name, self.create_realtime_layout())
    
    def create_realtime_layout(self) -> html.Div:
        """Create real-time monitoring page layout."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H2("Real-time Monitoring", className="text-center mb-4"),
                ], width=12)
            ]),
            
            # Control panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Control Panel", className="card-title"),
                            dbc.ButtonGroup([
                                dbc.Button("Start Stream", id="start-button", color="success", size="sm"),
                                dbc.Button("Stop Stream", id="stop-button", color="danger", size="sm"),
                                dbc.Button("Reset Stream", id="reset-button", color="warning", size="sm"),
                            ], size="sm")
                        ])
                    ], className="mb-3")
                ], width=12)
            ]),
            
            # Real-time sensor plots and prediction gauge
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Real-time Sensor Data", className="card-title"),
                            dcc.Graph(id='sensor-timeseries')
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Defect Probability", className="card-title"),
                            dcc.Graph(id='prediction-gauge')
                        ])
                    ])
                ], width=4),
            ], className="mb-3"),
            
            # Prediction history
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Prediction History", className="card-title"),
                            dcc.Graph(id='prediction-history')
                        ])
                    ])
                ], width=12)
            ])
        ])
    
    def create_model_comparison_layout(self) -> html.Div:
        """Create model comparison page layout."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H2("Model Comparison", className="text-center mb-4"),
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Model Performance Metrics", className="card-title"),
                            dcc.Graph(id='model-comparison-chart')
                        ])
                    ])
                ], width=12)
            ])
        ])
    
    def create_historical_layout(self) -> html.Div:
        """Create historical analysis page layout."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H2("Historical Analysis", className="text-center mb-4"),
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Historical Trends", className="card-title"),
                            dcc.Graph(id='historical-trends')
                        ])
                    ])
                ], width=12)
            ])
        ])
    
    def create_system_status_layout(self) -> html.Div:
        """Create system status page layout."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H2("System Status", className="text-center mb-4"),
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("System Health", className="card-title"),
                            html.Div(id='system-status-content')
                        ])
                    ])
                ], width=12)
            ])
        ])
        
    def setup_callbacks(self) -> None:
        """Setup interactive callbacks for multi-page navigation and updates."""
        
        # Page routing callback
        @self.app.callback(
            Output('page-content', 'children'),
            [Input('url', 'pathname')]
        )
        def display_page(pathname):
            """Route pages based on URL pathname."""
            try:
                if pathname == '/model-comparison':
                    return self.create_model_comparison_layout()
                elif pathname == '/historical-analysis':
                    return self.create_historical_layout()
                elif pathname == '/system-status':
                    return self.create_system_status_layout()
                else:  # Default to real-time monitoring
                    return self.create_realtime_layout()
            except Exception as e:
                logger.error(f"Error loading page {pathname}: {str(e)}")
                return html.Div([
                    dbc.Alert(
                        f"Error loading page: {str(e)}",
                        color="danger",
                        className="m-3"
                    )
                ])
        
        # Theme management callback
        @self.app.callback(
            [Output('theme-store', 'data'),
             Output('error-alert', 'children'),
             Output('error-alert', 'is_open')],
            [Input('theme-light', 'n_clicks'),
             Input('theme-dark', 'n_clicks'), 
             Input('theme-default', 'n_clicks')],
            [State('theme-store', 'data')]
        )
        def update_theme(light_clicks, dark_clicks, default_clicks, current_theme):
            """Update theme based on user selection."""
            try:
                ctx = dash.callback_context
                if not ctx.triggered:
                    return current_theme, "", False
                
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                theme_mapping = {
                    'theme-light': 'plotly_white',
                    'theme-dark': 'plotly_dark', 
                    'theme-default': 'plotly_white'
                }
                
                new_theme = theme_mapping.get(button_id, current_theme)
                return new_theme, "", False
            except Exception as e:
                logger.error(f"Error updating theme: {str(e)}")
                return current_theme, f"Error updating theme: {str(e)}", True
        
        # Refresh interval management callback
        @self.app.callback(
            [Output('refresh-interval-store', 'data'),
             Output('global-interval', 'interval')],
            [Input('refresh-1s', 'n_clicks'),
             Input('refresh-5s', 'n_clicks'),
             Input('refresh-10s', 'n_clicks'),
             Input('refresh-30s', 'n_clicks')],
            [State('refresh-interval-store', 'data')]
        )
        def update_refresh_interval(clicks_1s, clicks_5s, clicks_10s, clicks_30s, current_interval):
            """Update refresh interval based on user selection."""
            try:
                ctx = dash.callback_context
                if not ctx.triggered:
                    return current_interval, current_interval
                
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                interval_mapping = {
                    'refresh-1s': 1000,
                    'refresh-5s': 5000,
                    'refresh-10s': 10000,
                    'refresh-30s': 30000
                }
                
                new_interval = interval_mapping.get(button_id, current_interval)
                return new_interval, new_interval
            except Exception as e:
                logger.error(f"Error updating refresh interval: {str(e)}")
                return current_interval, current_interval
        
        # Real-time data update callback for monitoring page
        @self.app.callback(
            [Output('sensor-timeseries', 'figure'),
             Output('prediction-gauge', 'figure'),
             Output('prediction-history', 'figure')],
            [Input('global-interval', 'n_intervals')],
            [State('theme-store', 'data'),
             State('url', 'pathname')]
        )
        def update_realtime_data(n_intervals, theme, pathname):
            """Update real-time monitoring components."""
            try:
                # Only update if on the real-time monitoring page
                if pathname != '/' and pathname != '/real-time-monitoring':
                    return {}, {}, {}
                
                # Create sample sensor plot
                sensor_fig = self.create_sample_sensor_plot(theme)
                
                # Create sample prediction gauge  
                import random
                prediction_prob = random.uniform(0.1, 0.9)
                gauge_fig = self.create_prediction_gauge(prediction_prob, theme)
                
                # Create sample prediction history
                history_fig = self.create_sample_prediction_history(theme)
                
                return sensor_fig, gauge_fig, history_fig
            except Exception as e:
                logger.error(f"Error updating real-time data: {str(e)}")
                # Return empty figures on error to prevent crashes
                empty_fig = go.Figure()
                empty_fig.update_layout(title="Data unavailable", template=theme)
                return empty_fig, empty_fig, empty_fig
        
        # Model comparison update callback
        @self.app.callback(
            Output('model-comparison-chart', 'figure'),
            [Input('global-interval', 'n_intervals')],
            [State('theme-store', 'data'),
             State('url', 'pathname')]
        )
        def update_model_comparison(n_intervals, theme, pathname):
            """Update model comparison chart."""
            try:
                if pathname != '/model-comparison':
                    return {}
                return self.create_model_comparison_chart(theme)
            except Exception as e:
                logger.error(f"Error updating model comparison: {str(e)}")
                empty_fig = go.Figure()
                empty_fig.update_layout(title="Model comparison unavailable", template=theme)
                return empty_fig
        
        # Historical analysis update callback
        @self.app.callback(
            Output('historical-trends', 'figure'),
            [Input('global-interval', 'n_intervals')],
            [State('theme-store', 'data'),
             State('url', 'pathname')]
        )
        def update_historical_analysis(n_intervals, theme, pathname):
            """Update historical analysis chart."""
            try:
                if pathname != '/historical-analysis':
                    return {}
                return self.create_historical_trends_chart(theme)
            except Exception as e:
                logger.error(f"Error updating historical analysis: {str(e)}")
                empty_fig = go.Figure()
                empty_fig.update_layout(title="Historical data unavailable", template=theme)
                return empty_fig
        
        # System status update callback
        @self.app.callback(
            Output('system-status-content', 'children'),
            [Input('global-interval', 'n_intervals')],
            [State('url', 'pathname')]
        )
        def update_system_status(n_intervals, pathname):
            """Update system status information."""
            try:
                if pathname != '/system-status':
                    return ""
                return self.create_system_status_content(n_intervals)
            except Exception as e:
                logger.error(f"Error updating system status: {str(e)}")
                return dbc.Alert(f"System status unavailable: {str(e)}", color="warning")
    
    def create_sensor_plot(self, sensor_data: pd.DataFrame, theme: str = "plotly_white") -> go.Figure:
        """
        Create sensor time series plot.
        
        Args:
            sensor_data (pd.DataFrame): Recent sensor data
            theme (str): Plotly theme to use
            
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
            template=theme,
            showlegend=True
        )
        
        return fig
    
    def create_sample_sensor_plot(self, theme: str = "plotly_white") -> go.Figure:
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
        
        return self.create_sensor_plot(data, theme)
    
    def create_prediction_gauge(self, prediction_prob: float, theme: str = "plotly_white") -> go.Figure:
        """
        Create prediction probability gauge.
        
        Args:
            prediction_prob (float): Current prediction probability
            theme (str): Plotly theme to use
            
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
        
        fig.update_layout(height=300, template=theme)
        return fig
    
    def create_prediction_history_plot(self, history_data: pd.DataFrame, theme: str = "plotly_white") -> go.Figure:
        """
        Create prediction history plot.
        
        Args:
            history_data (pd.DataFrame): Historical predictions
            theme (str): Plotly theme to use
            
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
            template=theme
        )
        
        return fig
    
    def create_sample_prediction_history(self, theme: str = "plotly_white") -> go.Figure:
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
        return self.create_prediction_history_plot(history_data, theme)
    
    def create_model_comparison_chart(self, theme: str = "plotly_white") -> go.Figure:
        """Create model comparison visualization."""
        models = ['Baseline', 'LSTM', 'Ensemble']
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        # Sample performance data
        performance_data = {
            'Baseline': [0.82, 0.79, 0.85, 0.82],
            'LSTM': [0.88, 0.86, 0.90, 0.88],
            'Ensemble': [0.91, 0.89, 0.93, 0.91]
        }
        
        fig = go.Figure()
        
        for model in models:
            fig.add_trace(go.Bar(
                name=model,
                x=metrics,
                y=performance_data[model],
                text=[f'{val:.2f}' for val in performance_data[model]],
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Metrics",
            yaxis_title="Score",
            barmode='group',
            height=400,
            template=theme
        )
        
        return fig
    
    def create_historical_trends_chart(self, theme: str = "plotly_white") -> go.Figure:
        """Create historical trends visualization."""
        from datetime import datetime, timedelta
        
        # Generate sample historical data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                            end=datetime.now(), freq='D')
        
        np.random.seed(42)  # For consistent demo data
        defect_rates = 0.1 + 0.05 * np.sin(np.arange(len(dates)) * 0.2) + 0.02 * np.random.randn(len(dates))
        defect_rates = np.clip(defect_rates, 0, 1)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=defect_rates,
            mode='lines+markers',
            name='Daily Defect Rate',
            line=dict(color='red', width=2)
        ))
        
        fig.add_hline(y=0.1, line_dash="dash", line_color="orange",
                     annotation_text="Target Rate")
        
        fig.update_layout(
            title="Historical Defect Rate Trends",
            xaxis_title="Date",
            yaxis_title="Defect Rate",
            height=400,
            template=theme
        )
        
        return fig
    
    def create_system_status_content(self, n_intervals: int) -> html.Div:
        """Create system status content."""
        import psutil
        from datetime import datetime
        
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            status_cards = [
                dbc.Card([
                    dbc.CardBody([
                        html.H5("System Status", className="card-title"),
                        html.P(f"Last Updated: {current_time}"),
                        html.P(f"Refresh Count: {n_intervals}"),
                        html.P("Status: ✅ Operational", style={"color": "green"})
                    ])
                ], className="mb-3"),
                
                dbc.Card([
                    dbc.CardBody([
                        html.H5("System Resources", className="card-title"),
                        html.P(f"CPU Usage: {cpu_percent}%"),
                        html.P(f"Memory Usage: {memory.percent}%"),
                        html.P(f"Disk Usage: {disk.percent}%")
                    ])
                ], className="mb-3"),
                
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Dashboard Metrics", className="card-title"),
                        html.P(f"Active Sessions: 1"),
                        html.P(f"Total Refreshes: {n_intervals}"),
                        html.P("Data Source: 🟢 Connected")
                    ])
                ])
            ]
            
            return html.Div(status_cards)
            
        except Exception as e:
            return dbc.Alert(f"Error getting system status: {str(e)}", color="warning")
    
    def run(self, debug: bool = False, host: str = "127.0.0.1") -> None:
        """
        Run dashboard server.
        
        Args:
            debug (bool): Run in debug mode
            host (str): Host address
        """
        try:
            port = self.config.get('inference', {}).get('dashboard_port', 8050)
            logger.info(f"Starting dashboard on {host}:{port}")
            self.app.run_server(debug=debug, host=host, port=port)
        except Exception as e:
            logger.error(f"Error starting dashboard: {str(e)}")
            raise