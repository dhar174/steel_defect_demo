#!/usr/bin/env python3
"""
Historical Analysis Component Demo

This script demonstrates the key features of the historical analysis component
for steel defect detection. It can be run standalone to test functionality
or integrated into the main dashboard.

Note: Run 'pip install -e .' from the repository root to install the package in development mode.
"""

from visualization.components.historical_analysis import HistoricalAnalysisComponents
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go


def create_demo_app():
    """Create a standalone demo app for the historical analysis component."""
    
    # Initialize the component
    ha_component = HistoricalAnalysisComponents(component_id="demo-historical")
    
    # Create Dash app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Load sample data
    sample_data = ha_component.load_sample_data()
    
    # Create app layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Historical Data Analysis Demo", className="text-center mb-4"),
                html.P("Interactive demonstration of historical steel casting data analysis tools.", 
                       className="text-center text-muted mb-4")
            ])
        ]),
        
        # Main component layout
        ha_component.create_layout(),
        
        # Demo results section
        html.Hr(className="my-4"),
        dbc.Row([
            dbc.Col([
                html.H3("Demo Results", className="mb-3"),
                html.Div(id="demo-results")
            ])
        ])
    ], fluid=True)
    
    # Callback to populate demo results
    @app.callback(
        Output("demo-results", "children"),
        Input("demo-historical-load-sample", "n_clicks"),
        prevent_initial_call=False
    )
    def update_demo_results(n_clicks):
        # Generate some demo analyses
        demo_cards = []
        
        # Data Overview
        overview_cards = ha_component.create_data_overview_cards(sample_data)
        demo_cards.append(
            dbc.Card([
                dbc.CardHeader("Data Overview"),
                dbc.CardBody(overview_cards)
            ], className="mb-3")
        )
        
        # SPC Analysis
        spc_stats = ha_component.create_spc_statistics_summary(sample_data, "temperature_1")
        demo_cards.append(
            dbc.Card([
                dbc.CardHeader("SPC Analysis Example"),
                dbc.CardBody(spc_stats)
            ], className="mb-3")
        )
        
        # Clustering Analysis
        cluster_fig, cluster_stats = ha_component.create_clustering_analysis(
            sample_data, n_clusters=4, n_components=2
        )
        cluster_summary = ha_component.create_clustering_statistics_summary(cluster_stats)
        demo_cards.append(
            dbc.Card([
                dbc.CardHeader("Clustering Analysis Example"),
                dbc.CardBody(cluster_summary)
            ], className="mb-3")
        )
        
        # Correlation Analysis
        corr_stats = ha_component.create_correlation_statistics_summary(sample_data)
        demo_cards.append(
            dbc.Card([
                dbc.CardHeader("Correlation Analysis Example"),
                dbc.CardBody(corr_stats)
            ], className="mb-3")
        )
        
        return demo_cards
    
    return app


def run_feature_tests():
    """Run comprehensive feature tests for the historical analysis component."""
    
    print("🚀 Starting Historical Analysis Component Tests\n")
    
    # Initialize component
    ha = HistoricalAnalysisComponents()
    print("✅ Component initialized successfully")
    
    # Test data loading
    print("\n📊 Testing Data Loading...")
    df = ha.load_sample_data()
    print(f"✅ Loaded {len(df)} records with {len(df.columns)} features")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Defect rate: {(df['defect'].sum() / len(df)) * 100:.1f}%")
    
    # Test filtering capabilities
    print("\n🔍 Testing Data Filtering...")
    
    # Filter by defect status
    defect_data = ha.filter_data(df, defect_filter=1)
    print(f"✅ Defect filtering: {len(defect_data)} defect records")
    
    # Filter by date range
    mid_date = df['timestamp'].quantile(0.5).strftime('%Y-%m-%d')
    end_date = df['timestamp'].max().strftime('%Y-%m-%d')
    date_filtered = ha.filter_data(df, date_range=[mid_date, end_date])
    print(f"✅ Date filtering: {len(date_filtered)} records from {mid_date}")
    
    # Test aggregation
    daily_agg = ha.filter_data(df, aggregation='daily')
    print(f"✅ Daily aggregation: {len(daily_agg)} daily records")
    
    # Test SPC analysis
    print("\n📈 Testing SPC Analysis...")
    for sensor in ['temperature_1', 'pressure_1', 'casting_speed']:
        spc_fig = ha.create_spc_charts(df, sensor, 'individual')
        print(f"✅ SPC chart created for {sensor}")
    
    # Test clustering analysis
    print("\n🔬 Testing Clustering Analysis...")
    for n_clusters in [3, 5, 7]:
        cluster_fig, stats = ha.create_clustering_analysis(df, n_clusters, n_components=2)
        explained_var = stats.get('explained_variance', 0)
        print(f"✅ {n_clusters}-cluster analysis: {explained_var:.1%} variance explained")
    
    # Test correlation analysis
    print("\n🔗 Testing Correlation Analysis...")
    for method in ['pearson', 'spearman']:
        corr_fig = ha.create_correlation_heatmap(df, method=method)
        print(f"✅ {method.title()} correlation matrix created")
    
    # Test batch comparison
    print("\n⚖️ Testing Batch Comparison...")
    unique_casts = df['cast_id'].unique()[:4]
    for comparison_type in ['overview', 'statistics', 'prediction']:
        batch_fig = ha.create_batch_comparison(df, unique_casts.tolist(), comparison_type)
        print(f"✅ Batch {comparison_type} comparison created")
    
    # Test export functionality
    print("\n💾 Testing Export Functionality...")
    
    # CSV export
    csv_export = ha.export_data_to_csv(defect_data)
    print(f"✅ CSV export: {len(csv_export['content'])} characters")
    
    # Chart export
    sample_chart = ha.create_distribution_plot(df, 'temperature_1')
    chart_export = ha.export_chart_to_image(sample_chart)
    print(f"✅ Chart export: {len(chart_export['content'])} characters (base64)")
    
    # Test utility functions
    print("\n🛠️ Testing Utility Functions...")
    
    # Data info summary
    info = ha.get_data_info_summary(df)
    print(f"✅ Data info: {info}")
    
    # Statistics summaries
    spc_stats = ha.create_spc_statistics_summary(df, 'temperature_1')
    corr_stats = ha.create_correlation_statistics_summary(df)
    batch_stats = ha.create_batch_statistics_summary(df, unique_casts[:3].tolist())
    print("✅ All statistics summaries created successfully")
    
    print(f"\n🎉 All tests completed successfully!")
    print(f"📊 Component ready for integration with main dashboard")
    
    return True


def main():
    """Main function to run tests or demo app."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Historical Analysis Component Demo")
    parser.add_argument("--mode", choices=['test', 'demo'], default='test',
                       help="Run mode: 'test' for feature tests, 'demo' for interactive app")
    parser.add_argument("--host", default="127.0.0.1", help="Host for demo app")
    parser.add_argument("--port", default=8050, type=int, help="Port for demo app")
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        # Run comprehensive tests
        success = run_feature_tests()
        sys.exit(0 if success else 1)
    
    elif args.mode == 'demo':
        # Run interactive demo app
        print(f"🚀 Starting Historical Analysis Demo App")
        print(f"🌐 Open your browser to: http://{args.host}:{args.port}")
        
        app = create_demo_app()
        app.run_server(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()