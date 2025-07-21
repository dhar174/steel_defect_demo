#!/usr/bin/env python3
"""
Test script for the SensorMonitoringComponent.

This script creates a simple Dash app to test the sensor monitoring component
with mock data and demonstrates all the implemented features.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import time
import threading

# Import our component
from src.visualization.components.sensor_monitoring import SensorMonitoringComponent

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Create sensor monitoring component
sensor_monitor = SensorMonitoringComponent(
    component_id="test-sensor-monitor",
    buffer_size=500,
    update_interval=2000  # 2 seconds for testing
)

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Steel Defect Demo - Sensor Monitoring Test", 
                   className="text-center mb-4"),
            html.Hr(),
        ], width=12)
    ]),
    
    # Sensor monitoring component
    sensor_monitor.create_layout(),
    
], fluid=True)

# Callback to update plots with mock data
@app.callback(
    [Output('test-sensor-monitor-main-plot', 'figure'),
     Output('test-sensor-monitor-detail-plots', 'figure'),
     Output('test-sensor-monitor-health-indicators', 'children'),
     Output('test-sensor-monitor-data-store', 'data')],
    [Input('test-sensor-monitor-interval', 'n_intervals')],
    [State('test-sensor-monitor-config-store', 'data'),
     State('test-sensor-monitor-data-store', 'data')]
)
def update_sensor_plots(n_intervals, config, stored_data):
    """Update sensor plots with new mock data."""
    try:
        # Generate new mock data point
        mock_data = sensor_monitor.generate_mock_data_point()
        sensor_monitor.add_data_point(mock_data)
        
        # Get current data from buffers
        current_data, timestamps = sensor_monitor.get_current_data()
        
        # Update sensor health
        health_status = sensor_monitor.update_sensor_health(current_data, timestamps)
        
        # Create plots
        main_plot = sensor_monitor.create_multi_sensor_plot(current_data, timestamps, config)
        detail_plots = sensor_monitor.create_detail_plots(current_data, timestamps, config)
        health_indicators = sensor_monitor.create_health_indicators(health_status)
        
        # Store data for potential use by other callbacks
        data_store = {
            'current_data': current_data,
            'timestamps': [ts.isoformat() for ts in timestamps],
            'health_status': health_status,
            'last_update': datetime.now().isoformat()
        }
        
        return main_plot, detail_plots, health_indicators, data_store
        
    except Exception as e:
        print(f"Error in update callback: {str(e)}")
        # Return empty plots on error
        import plotly.graph_objects as go
        empty_fig = go.Figure()
        empty_fig.update_layout(title=f"Error: {str(e)}")
        return empty_fig, empty_fig, html.Div("Error loading health indicators"), {}

# Callback for time range controls
@app.callback(
    Output('test-sensor-monitor-config-store', 'data'),
    [Input('test-sensor-monitor-time-5min', 'n_clicks'),
     Input('test-sensor-monitor-time-15min', 'n_clicks'),
     Input('test-sensor-monitor-time-30min', 'n_clicks'),
     Input('test-sensor-monitor-time-1hr', 'n_clicks'),
     Input('test-sensor-monitor-display-options', 'value'),
     Input('test-sensor-monitor-update-1s', 'n_clicks'),
     Input('test-sensor-monitor-update-5s', 'n_clicks'),
     Input('test-sensor-monitor-update-10s', 'n_clicks')],
    [State('test-sensor-monitor-config-store', 'data')]
)
def update_config(time_5min, time_15min, time_30min, time_1hr, 
                 display_options, update_1s, update_5s, update_10s, current_config):
    """Update configuration based on user inputs."""
    try:
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_config or {
                'time_range_minutes': 30,
                'auto_scale': True,
                'show_thresholds': True,
                'show_anomalies': True
            }
        
        new_config = current_config.copy() if current_config else {}
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Handle time range updates
        if 'time-5min' in button_id:
            new_config['time_range_minutes'] = 5
        elif 'time-15min' in button_id:
            new_config['time_range_minutes'] = 15
        elif 'time-30min' in button_id:
            new_config['time_range_minutes'] = 30
        elif 'time-1hr' in button_id:
            new_config['time_range_minutes'] = 60
        
        # Handle display options
        if display_options is not None:
            new_config['auto_scale'] = 'auto_scale' in display_options
            new_config['show_thresholds'] = 'show_thresholds' in display_options
            new_config['show_anomalies'] = 'show_anomalies' in display_options
        
        return new_config
        
    except Exception as e:
        print(f"Error in config callback: {str(e)}")
        return current_config or {}

if __name__ == '__main__':
    print("Starting Steel Defect Demo - Sensor Monitoring Test...")
    print("Navigate to http://127.0.0.1:8050 to view the dashboard")
    print("Press Ctrl+C to stop the server")
    
    # Pre-populate some initial data
    print("Generating initial mock data...")
    for i in range(50):
        mock_data = sensor_monitor.generate_mock_data_point()
        timestamp = datetime.now() - timedelta(seconds=50-i)
        sensor_monitor.add_data_point(mock_data, timestamp)
    
    # Run the app
    app.run(debug=True, host='127.0.0.1', port=8050)