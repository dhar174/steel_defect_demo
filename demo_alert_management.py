#!/usr/bin/env python3
"""
Demo script for the Alert Management Interface.

This script demonstrates the functionality of the AlertManagementComponent
by creating a simple Dash app with the alert management interface.
"""

import sys
import os
from datetime import datetime, timedelta
import numpy as np

# Import the AlertManagementComponent using relative imports
from .src.visualization.components.alert_management import AlertManagementComponent
import dash
from dash import html, Input, Output, State, callback
import dash_bootstrap_components as dbc


def create_demo_app():
    """Create a demo Dash app with the alert management interface."""
    
    # Initialize the alert management component
    alert_mgmt = AlertManagementComponent(
        component_id="demo-alert-mgmt",
        config_file="configs/inference_config.yaml"
    )
    
    # Create the Dash app
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True
    )
    
    # Set up the layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Steel Defect Alert Management Demo", className="text-center mb-4"),
                html.Hr(),
                alert_mgmt.create_layout()
            ], width=12)
        ])
    ], fluid=True)
    
    # Set up callbacks for the demo
    @app.callback(
        [Output('demo-alert-mgmt-alert-table', 'data'),
         Output('demo-alert-mgmt-total-alerts', 'children'),
         Output('demo-alert-mgmt-active-alerts', 'children'),
         Output('demo-alert-mgmt-acknowledged-alerts', 'children'),
         Output('demo-alert-mgmt-resolved-alerts', 'children')],
        [Input('demo-alert-mgmt-interval', 'n_intervals'),
         Input('demo-alert-mgmt-refresh-btn', 'n_clicks'),
         Input('demo-alert-mgmt-test-alert-btn', 'n_clicks')]
    )
    def update_alert_data(n_intervals, refresh_clicks, test_alert_clicks):
        """Update alert table and metrics."""
        # Add a test alert when button is clicked
        if test_alert_clicks and test_alert_clicks > 0:
            severity = np.random.choice(['Medium', 'High', 'Critical'])
            description = f"Test alert generated at {datetime.now().strftime('%H:%M:%S')}"
            alert_mgmt.add_alert(severity, description, "Demo System")
        
        # Get current alerts data
        alerts_data = alert_mgmt.get_alerts_data()
        metrics = alert_mgmt.performance_metrics
        
        return (
            alerts_data,
            str(metrics['total_alerts']),
            str(metrics['active_alerts']),
            str(metrics['acknowledged_alerts']),
            str(metrics['resolved_alerts'])
        )
    
    @app.callback(
        Output('demo-alert-mgmt-frequency-chart', 'figure'),
        [Input('demo-alert-mgmt-interval', 'n_intervals')]
    )
    def update_frequency_chart(n_intervals):
        """Update the frequency chart."""
        return alert_mgmt.create_frequency_chart()
    
    @app.callback(
        Output('demo-alert-mgmt-severity-distribution', 'figure'),
        [Input('demo-alert-mgmt-interval', 'n_intervals')]
    )
    def update_severity_chart(n_intervals):
        """Update the severity distribution chart."""
        return alert_mgmt.create_severity_distribution()
    
    @app.callback(
        Output('demo-alert-mgmt-mtta-metric', 'children'),
        Output('demo-alert-mgmt-mttr-metric', 'children'),
        [Input('demo-alert-mgmt-interval', 'n_intervals')]
    )
    def update_performance_metrics(n_intervals):
        """Update performance metrics."""
        metrics = alert_mgmt.performance_metrics
        mtta = f"{metrics['mean_time_to_acknowledge']:.1f} min"
        mttr = f"{metrics['mean_time_to_resolve']:.1f} min"
        return mtta, mttr
    
    @app.callback(
        Output('demo-alert-mgmt-config-alert', 'children'),
        Output('demo-alert-mgmt-config-alert', 'is_open'),
        Output('demo-alert-mgmt-config-alert', 'color'),
        [Input('demo-alert-mgmt-save-config-btn', 'n_clicks')],
        [State('demo-alert-mgmt-defect-threshold', 'value'),
         State('demo-alert-mgmt-high-risk-threshold', 'value'),
         State('demo-alert-mgmt-alert-threshold', 'value'),
         State('demo-alert-mgmt-suppression-period', 'value')]
    )
    def save_configuration(n_clicks, defect_thresh, high_risk_thresh, alert_thresh, suppression):
        """Save configuration changes."""
        if not n_clicks:
            return "", False, "success"
        
        try:
            new_config = {
                'inference': {
                    'thresholds': {
                        'defect_probability': defect_thresh,
                        'high_risk_threshold': high_risk_thresh,
                        'alert_threshold': alert_thresh
                    }
                },
                'monitoring': {
                    'alerts': {
                        'alert_suppression_minutes': suppression
                    }
                }
            }
            
            success = alert_mgmt.update_config_file(new_config)
            
            if success:
                return "Configuration saved successfully!", True, "success"
            else:
                return "Failed to save configuration!", True, "danger"
                
        except Exception as e:
            return f"Error saving configuration: {str(e)}", True, "danger"
    
    return app


def main():
    """Main function to run the demo."""
    print("=" * 60)
    print("Steel Defect Alert Management Interface Demo")
    print("=" * 60)
    print()
    print("This demo showcases the Alert Management Interface component")
    print("with the following features:")
    print()
    print("✓ Real-time alert feed with sortable DataTable")
    print("✓ Alert history and trend analysis")
    print("✓ Configurable alert thresholds")
    print("✓ Alert acknowledgment and resolution")
    print("✓ Alert performance analytics")
    print()
    print("Starting the demo server...")
    print("Open your browser to http://localhost:8050 to view the interface")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        app = create_demo_app()
        app.run(debug=True, host='0.0.0.0', port=8050)
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    except Exception as e:
        print(f"Error running demo: {e}")


if __name__ == "__main__":
    main()