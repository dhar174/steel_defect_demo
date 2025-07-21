"""
Integration example showing how to use SensorMonitoringComponent 
within the existing dashboard structure.

This demonstrates how the sensor monitoring component can be integrated
into the main DefectMonitoringDashboard from dashboard.py.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from dash import html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
from datetime import datetime

from src.visualization.components.sensor_monitoring import SensorMonitoringComponent

def create_enhanced_realtime_layout(sensor_monitor: SensorMonitoringComponent) -> html.Div:
    """
    Create an enhanced real-time monitoring layout that includes the 
    sensor monitoring component alongside existing features.
    
    Args:
        sensor_monitor: Instance of SensorMonitoringComponent
        
    Returns:
        html.Div: Enhanced real-time monitoring layout
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H2("Enhanced Real-time Monitoring", className="text-center mb-4"),
            ], width=12)
        ]),
        
        # Integrated sensor monitoring component
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("🔬 Live Sensor Monitoring", className="mb-0"),
                        html.Small("Real-time steel casting process sensors", className="text-muted")
                    ]),
                    dbc.CardBody([
                        sensor_monitor.create_layout()
                    ])
                ], className="shadow-sm")
            ], width=12)
        ], className="mb-4"),
        
        # Original prediction components (from existing dashboard.py)
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Defect Probability", className="card-title"),
                        dcc.Graph(id='prediction-gauge')
                    ])
                ], className="shadow-sm")
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Prediction History", className="card-title"),
                        dcc.Graph(id='prediction-history')
                    ])
                ], className="shadow-sm")
            ], width=6),
        ], className="mb-4"),
        
        # Additional analytics row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("🔍 Sensor Analytics", className="mb-0")
                    ]),
                    dbc.CardBody([
                        html.Div(id='sensor-analytics-content')
                    ])
                ], className="shadow-sm")
            ], width=12)
        ])
    ])

def create_sensor_analytics_content(sensor_data: dict, health_status: dict) -> html.Div:
    """
    Create additional analytics content based on sensor data.
    
    Args:
        sensor_data: Current sensor data
        health_status: Sensor health status
        
    Returns:
        html.Div: Analytics content
    """
    if not sensor_data:
        return html.Div("No sensor data available", className="text-muted")
    
    # Calculate some basic statistics
    analytics_cards = []
    
    for sensor, data in sensor_data.items():
        if data:
            current_value = data[-1] if data else 0
            avg_value = sum(data) / len(data) if data else 0
            status = health_status.get(sensor, 'unknown')
            
            status_color = {
                'good': 'success',
                'warning': 'warning',
                'critical': 'danger',
                'offline': 'secondary'
            }.get(status, 'light')
            
            analytics_cards.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.Small(sensor.replace('_', ' ').title(), className="text-muted"),
                            dbc.Badge(status, color=status_color, className="float-end")
                        ]),
                        dbc.CardBody([
                            html.H6(f"{current_value:.2f}", className="mb-1"),
                            html.Small(f"Avg: {avg_value:.2f}", className="text-muted")
                        ])
                    ], size="sm")
                ], md=2, className="mb-2")
            )
    
    return dbc.Row(analytics_cards)

# Usage example for integration with existing dashboard
def integrate_with_existing_dashboard():
    """
    Example of how to modify the existing DefectMonitoringDashboard 
    to include the sensor monitoring component.
    """
    
    # Create sensor monitoring component
    sensor_monitor = SensorMonitoringComponent(
        component_id="dashboard-sensor-monitor",
        buffer_size=1000,
        update_interval=5000  # 5 seconds for production
    )
    
    # The following would be added to the existing dashboard.py file:
    
    # 1. Import the component at the top of dashboard.py:
    # from src.visualization.components.sensor_monitoring import SensorMonitoringComponent
    
    # 2. Add to __init__ method of DefectMonitoringDashboard:
    # self.sensor_monitor = SensorMonitoringComponent(
    #     component_id="dashboard-sensor-monitor",
    #     buffer_size=1000,
    #     update_interval=self.refresh_interval
    # )
    
    # 3. Replace the create_realtime_layout method with:
    # def create_realtime_layout(self) -> html.Div:
    #     return create_enhanced_realtime_layout(self.sensor_monitor)
    
    # 4. Add new callback to setup_callbacks method:
    """
    @self.app.callback(
        [Output('dashboard-sensor-monitor-main-plot', 'figure'),
         Output('dashboard-sensor-monitor-detail-plots', 'figure'),
         Output('dashboard-sensor-monitor-health-indicators', 'children'),
         Output('sensor-analytics-content', 'children')],
        [Input('global-interval', 'n_intervals')],
        [State('dashboard-sensor-monitor-config-store', 'data'),
         State('url', 'pathname')]
    )
    def update_sensor_monitoring(n_intervals, config, pathname):
        try:
            # Only update if on the real-time monitoring page
            if pathname not in ['/', '/real-time-monitoring']:
                return {}, {}, html.Div(), html.Div()
            
            # Generate new sensor data (replace with actual data source)
            mock_data = self.sensor_monitor.generate_mock_data_point()
            self.sensor_monitor.add_data_point(mock_data)
            
            # Get current data and update health
            current_data, timestamps = self.sensor_monitor.get_current_data()
            health_status = self.sensor_monitor.update_sensor_health(current_data, timestamps)
            
            # Create visualizations
            main_plot = self.sensor_monitor.create_multi_sensor_plot(current_data, timestamps, config)
            detail_plots = self.sensor_monitor.create_detail_plots(current_data, timestamps, config)
            health_indicators = self.sensor_monitor.create_health_indicators(health_status)
            analytics_content = create_sensor_analytics_content(current_data, health_status)
            
            return main_plot, detail_plots, health_indicators, analytics_content
            
        except Exception as e:
            logger.error(f"Error updating sensor monitoring: {str(e)}")
            empty_fig = go.Figure()
            empty_fig.update_layout(title="Sensor monitoring unavailable", template=theme)
            return empty_fig, empty_fig, html.Div("Health indicators unavailable"), html.Div("Analytics unavailable")
    """
    
    return sensor_monitor

if __name__ == '__main__':
    print("This is an integration example file.")
    print("Copy the integration patterns shown here into the main dashboard.py file.")
    
    # Create and test the component
    sensor_monitor = integrate_with_existing_dashboard()
    print("✅ Sensor monitoring component created successfully")
    
    # Test layout creation
    layout = create_enhanced_realtime_layout(sensor_monitor)
    print("✅ Enhanced layout created successfully")
    
    # Test data generation
    mock_data = sensor_monitor.generate_mock_data_point()
    sensor_monitor.add_data_point(mock_data)
    data, timestamps = sensor_monitor.get_current_data()
    print(f"✅ Data buffers working: {len(timestamps)} data points")
    
    print("\nIntegration example completed successfully!")
    print("See the function comments above for exact integration steps.")