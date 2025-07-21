"""
Example demonstrating the ModelComparison component integration.

This script shows how to use the ModelComparison component with real model results
and integrate it into a dashboard page.
"""

import sys
import os
sys.path.insert(0, '/home/runner/work/steel_defect_demo/steel_defect_demo')

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from src.visualization.components.model_comparison import ModelComparison, create_sample_model_results


def create_model_comparison_page():
    """
    Create a model comparison page that can be integrated into the main dashboard.
    
    Returns:
        html.Div: The layout for the model comparison page
    """
    # Initialize the model comparison component
    comparison = ModelComparison(theme="plotly_white")
    
    # For demonstration, use sample data
    # In practice, this would load real model results from files or database
    model_results = create_sample_model_results()
    
    # Get the complete dashboard layout
    layout = comparison.get_dashboard_layout(model_results)
    
    return layout


def create_standalone_app():
    """
    Create a standalone Dash app with the model comparison interface.
    
    Returns:
        dash.Dash: Configured Dash application
    """
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1("Steel Defect Detection - Model Comparison", 
                           className="text-center mb-4"),
                    html.Hr(),
                    create_model_comparison_page()
                ])
            ])
        ])
    ], fluid=True)
    
    return app


def demo_individual_components():
    """
    Demonstrate individual component usage.
    """
    print("🔬 Demonstrating individual ModelComparison components...")
    
    # Create sample data
    model_results = create_sample_model_results()
    comparison = ModelComparison()
    
    print("📊 Available models:", list(model_results.keys()))
    
    # Demo 1: ROC and PR curves
    print("\n1. Creating ROC and PR curves...")
    roc_pr_fig = comparison.create_roc_pr_comparison(model_results)
    print(f"   Figure created with {len(roc_pr_fig.data)} traces")
    
    # Demo 2: Feature importance
    print("\n2. Creating feature importance chart...")
    feature_fig = comparison.create_feature_importance_chart(model_results)
    print(f"   Feature importance chart created")
    
    # Demo 3: Attention visualization
    print("\n3. Creating attention visualization...")
    attention_fig = comparison.create_attention_visualization(model_results)
    print(f"   Attention visualization created")
    
    # Demo 4: Correlation analysis
    print("\n4. Creating prediction correlation analysis...")
    correlation_fig = comparison.create_prediction_correlation_analysis(model_results)
    print(f"   Correlation analysis created")
    
    # Demo 5: Performance metrics table
    print("\n5. Creating performance metrics table...")
    metrics_table = comparison.create_performance_metrics_table(model_results)
    print(f"   Metrics table created with {len(metrics_table.data)} rows")
    
    # Demo 6: Side-by-side charts
    print("\n6. Creating side-by-side comparison charts...")
    side_by_side_fig = comparison.create_side_by_side_charts(model_results)
    print(f"   Side-by-side charts created")
    
    print("\n✅ All individual components demonstrated successfully!")


def show_integration_example():
    """
    Show how to integrate with existing dashboard.
    """
    print("\n🔗 Integration Example:")
    print("""
    To integrate the ModelComparison component into an existing dashboard:
    
    1. Import the component:
       from src.visualization.components import ModelComparison
    
    2. Add to your dashboard pages:
       comparison = ModelComparison()
       model_results = load_your_model_results()  # Your data loading function
       comparison_layout = comparison.get_dashboard_layout(model_results)
    
    3. Add to your routing (if using multi-page dashboard):
       @app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
       def display_page(pathname):
           if pathname == '/model-comparison':
               return comparison_layout
           # ... other pages
    
    4. Expected model_results format:
       {
           'ModelName': {
               'y_true': array,           # True labels
               'y_pred': array,           # Predicted labels  
               'y_pred_proba': array,     # Prediction probabilities
               'feature_importance': dict, # Feature importance scores (optional)
               'attention_weights': array  # Attention weights (optional)
           }
       }
    """)


if __name__ == "__main__":
    print("🚀 ModelComparison Component Demo")
    print("=" * 50)
    
    # Run individual component demos
    demo_individual_components()
    
    # Show integration example
    show_integration_example()
    
    print("\n📝 Summary:")
    print("The ModelComparison component provides a comprehensive interface for:")
    print("  • Comparing model performance side-by-side")
    print("  • Visualizing ROC and Precision-Recall curves") 
    print("  • Analyzing feature importance for baseline models")
    print("  • Interpreting LSTM attention weights")
    print("  • Correlating predictions between models")
    print("  • Displaying performance metrics with statistical tests")
    
    # Optional: Start the standalone app
    start_app = input("\n🌐 Start standalone demo app? (y/N): ").lower().strip()
    if start_app == 'y':
        app = create_standalone_app()
        print("Starting demo app on http://127.0.0.1:8050")
        app.run_server(debug=True)
    else:
        print("Demo completed! Component is ready for integration.")