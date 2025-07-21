"""
Dashboard Integration Example for Historical Analysis Component

This script demonstrates how to integrate the historical analysis component
into the main steel defect monitoring dashboard.

Note: Run 'pip install -e .' from the repository root to install the package in development mode.
"""

from visualization.components.historical_analysis import HistoricalAnalysisComponents
from visualization.components.sensor_monitoring import SensorMonitoringComponent
from visualization.components.prediction_display import PredictionDisplayComponents
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc


class ExtendedDashboard:
    """Extended dashboard with historical analysis integration."""
    
    def __init__(self, config: dict = None):
        """Initialize extended dashboard with historical analysis."""
        self.config = config or {}
        
        # Initialize app
        self.app = dash.Dash(
            __name__, 
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        
        # Initialize components
        self.historical_analysis = HistoricalAnalysisComponents(component_id="main-historical")
        self.sensor_monitoring = SensorMonitoringComponent(component_id="main-sensor")
        self.prediction_display = PredictionDisplayComponents(config)
        
        # Setup layout and callbacks
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup the main dashboard layout with integrated components."""
        self.app.layout = dbc.Container([
            # URL routing
            dcc.Location(id='url', refresh=False),
            
            # Navigation
            self.create_navbar(),
            
            # Main content
            html.Div(id='page-content'),
            
        ], fluid=True)
    
    def create_navbar(self):
        """Create navigation bar with historical analysis option."""
        return dbc.NavbarSimple(
            brand="Steel Defect Monitoring System",
            brand_href="/",
            color="primary",
            dark=True,
            children=[
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("Real-time Monitoring", href="/", active="exact")),
                    dbc.NavItem(dbc.NavLink("Predictions", href="/predictions", active="exact")),
                    dbc.NavItem(dbc.NavLink("Historical Analysis", href="/historical", active="exact")),
                    dbc.NavItem(dbc.NavLink("Reports", href="/reports", active="exact")),
                ], navbar=True)
            ]
        )
    
    def create_home_page(self):
        """Create the home page with real-time monitoring."""
        return html.Div([
            html.H1("Real-time Steel Casting Monitoring", className="mb-4"),
            html.P("Monitor live sensor data and defect predictions in real-time.", className="text-muted mb-4"),
            
            # Sensor monitoring component
            self.sensor_monitoring.create_layout(),
        ])
    
    def create_predictions_page(self):
        """Create the predictions page."""
        return html.Div([
            html.H1("Defect Prediction Analysis", className="mb-4"),
            html.P("View detailed defect predictions and model performance.", className="text-muted mb-4"),
            
            # Prediction display components would go here
            dbc.Alert("Prediction components would be integrated here.", color="info"),
        ])
    
    def create_historical_page(self):
        """Create the historical analysis page."""
        return html.Div([
            html.H1("Historical Data Analysis", className="mb-4"),
            html.P("Analyze historical casting data to identify patterns and trends.", className="text-muted mb-4"),
            
            # Historical analysis component
            self.historical_analysis.create_layout(),
        ])
    
    def create_reports_page(self):
        """Create the reports page."""
        return html.Div([
            html.H1("Analysis Reports", className="mb-4"),
            html.P("Generate and download comprehensive analysis reports.", className="text-muted mb-4"),
            
            dbc.Alert("Report generation features would be integrated here.", color="info"),
        ])
    
    def setup_callbacks(self):
        """Setup navigation callbacks."""
        
        @self.app.callback(
            Output('page-content', 'children'),
            Input('url', 'pathname')
        )
        def display_page(pathname):
            """Navigate between pages based on URL."""
            if pathname == '/historical':
                return self.create_historical_page()
            elif pathname == '/predictions':
                return self.create_predictions_page()
            elif pathname == '/reports':
                return self.create_reports_page()
            else:  # Default to home page
                return self.create_home_page()
    
    def run_server(self, **kwargs):
        """Run the dashboard server."""
        self.app.run_server(**kwargs)


def create_integration_demo():
    """Create a demo showing the integrated dashboard."""
    
    print("🚀 Creating Integrated Dashboard Demo")
    
    # Configuration for the dashboard
    config = {
        'refresh_interval': 5000,
        'risk_colors': {
            'safe': '#2E8B57',
            'warning': '#FFD700', 
            'high_risk': '#FF6B35',
            'alert': '#DC143C'
        }
    }
    
    # Create extended dashboard
    dashboard = ExtendedDashboard(config)
    
    print("✅ Dashboard components initialized")
    print("✅ Historical analysis component integrated")
    print("✅ Navigation configured")
    
    return dashboard


def main():
    """Main function to run the integration demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dashboard Integration Demo")
    parser.add_argument("--host", default="127.0.0.1", help="Host for demo app")
    parser.add_argument("--port", default=8051, type=int, help="Port for demo app")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    print("🏗️ Setting up Integrated Steel Defect Dashboard")
    print("📊 Features:")
    print("   • Real-time sensor monitoring")
    print("   • Defect prediction display") 
    print("   • Historical data analysis")
    print("   • Multi-page navigation")
    print("   • Export capabilities")
    
    # Create and run dashboard
    dashboard = create_integration_demo()
    
    print(f"\n🌐 Starting integrated dashboard at http://{args.host}:{args.port}")
    print("📱 Navigation:")
    print(f"   • Home (Real-time): http://{args.host}:{args.port}/")
    print(f"   • Predictions: http://{args.host}:{args.port}/predictions")
    print(f"   • Historical Analysis: http://{args.host}:{args.port}/historical")
    print(f"   • Reports: http://{args.host}:{args.port}/reports")
    
    dashboard.run_server(
        host=args.host, 
        port=args.port, 
        debug=args.debug
    )


if __name__ == "__main__":
    main()