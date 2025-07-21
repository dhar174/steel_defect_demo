"""
Model Comparison Interface for Steel Defect Detection Dashboard

This module provides a comprehensive interface for comparing the performance 
and behavior of different prediction models (e.g., baseline XGBoost vs. LSTM).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score, 
    average_precision_score, confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from scipy import stats
import logging

# Setup logging
logger = logging.getLogger(__name__)


class ModelComparison:
    """
    A comprehensive interface for comparing multiple prediction models.
    
    This class provides visualization and analysis tools for comparing
    model performance, interpretability, and behavior across different
    model architectures.
    """
    
    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize the model comparison interface.
        
        Args:
            theme (str): Plotly theme for visualizations
        """
        self.theme = theme
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
    def create_side_by_side_charts(self, 
                                   model_results: Dict[str, Dict],
                                   chart_types: List[str] = None) -> go.Figure:
        """
        Create side-by-side performance charts for model comparison.
        
        Args:
            model_results (Dict): Dictionary containing model results
                Format: {model_name: {metrics: dict, predictions: dict, ...}}
            chart_types (List): Types of charts to include ['roc', 'pr', 'confusion']
            
        Returns:
            go.Figure: Plotly figure with subplot grid
        """
        if chart_types is None:
            chart_types = ['roc', 'pr', 'confusion']
            
        n_models = len(model_results)
        n_charts = len(chart_types)
        
        # Create subplot grid
        fig = make_subplots(
            rows=n_charts, 
            cols=n_models,
            subplot_titles=[f"{model} - {chart.upper()}" for chart in chart_types for model in model_results.keys()],
            specs=[[{"secondary_y": False} for _ in range(n_models)] for _ in range(n_charts)],
            horizontal_spacing=0.1,
            vertical_spacing=0.1
        )
        
        for col, (model_name, results) in enumerate(model_results.items(), 1):
            # Extract results
            y_true = results.get('y_true', [])
            y_pred_proba = results.get('y_pred_proba', [])
            y_pred = results.get('y_pred', [])
            
            if len(y_true) == 0 or len(y_pred_proba) == 0:
                logger.warning(f"Missing data for model {model_name}")
                continue
                
            # ROC Curve
            if 'roc' in chart_types:
                row_idx = chart_types.index('roc') + 1
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                auc_score = roc_auc_score(y_true, y_pred_proba)
                
                fig.add_trace(
                    go.Scatter(
                        x=fpr, y=tpr,
                        name=f"{model_name} (AUC={auc_score:.3f})",
                        line=dict(color=self.color_palette[col-1]),
                        showlegend=True
                    ),
                    row=row_idx, col=col
                )
                
                # Add diagonal reference line
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        line=dict(dash='dash', color='gray'),
                        showlegend=False
                    ),
                    row=row_idx, col=col
                )
                
            # Precision-Recall Curve
            if 'pr' in chart_types:
                row_idx = chart_types.index('pr') + 1
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
                ap_score = average_precision_score(y_true, y_pred_proba)
                
                fig.add_trace(
                    go.Scatter(
                        x=recall, y=precision,
                        name=f"{model_name} (AP={ap_score:.3f})",
                        line=dict(color=self.color_palette[col-1]),
                        showlegend=True
                    ),
                    row=row_idx, col=col
                )
                
            # Confusion Matrix
            if 'confusion' in chart_types:
                row_idx = chart_types.index('confusion') + 1
                cm = confusion_matrix(y_true, y_pred)
                
                fig.add_trace(
                    go.Heatmap(
                        z=cm,
                        x=['Predicted Normal', 'Predicted Defect'],
                        y=['Actual Normal', 'Actual Defect'],
                        colorscale='Blues',
                        showscale=False,
                        text=cm,
                        texttemplate="%{text}",
                        textfont={"size": 16}
                    ),
                    row=row_idx, col=col
                )
        
        # Update layout
        fig.update_layout(
            height=300 * n_charts,
            title_text="Model Performance Comparison",
            template=self.theme,
            showlegend=True
        )
        
        return fig
    
    def create_roc_pr_comparison(self, model_results: Dict[str, Dict]) -> go.Figure:
        """
        Create overlaid ROC and Precision-Recall curves for multiple models.
        
        Args:
            model_results (Dict): Dictionary containing model results
            
        Returns:
            go.Figure: Plotly figure with ROC and PR curves
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['ROC Curves', 'Precision-Recall Curves'],
            horizontal_spacing=0.1
        )
        
        for i, (model_name, results) in enumerate(model_results.items()):
            y_true = results.get('y_true', [])
            y_pred_proba = results.get('y_pred_proba', [])
            
            if len(y_true) == 0 or len(y_pred_proba) == 0:
                continue
                
            color = self.color_palette[i % len(self.color_palette)]
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc_score = roc_auc_score(y_true, y_pred_proba)
            
            fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr,
                    name=f"{model_name} (AUC={auc_score:.3f})",
                    line=dict(color=color),
                    legendgroup=model_name
                ),
                row=1, col=1
            )
            
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            ap_score = average_precision_score(y_true, y_pred_proba)
            
            fig.add_trace(
                go.Scatter(
                    x=recall, y=precision,
                    name=f"{model_name} (AP={ap_score:.3f})",
                    line=dict(color=color),
                    legendgroup=model_name,
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Add reference lines
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                      line=dict(dash='dash', color='gray'),
                      showlegend=False),
            row=1, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=400,
            title_text="ROC and Precision-Recall Curve Comparison",
            template=self.theme
        )
        
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
        fig.update_xaxes(title_text="Recall", row=1, col=2)
        fig.update_yaxes(title_text="Precision", row=1, col=2)
        
        return fig
    
    def create_feature_importance_chart(self, 
                                        model_results: Dict[str, Dict],
                                        top_features: int = 15) -> go.Figure:
        """
        Create feature importance comparison chart for baseline models.
        
        Args:
            model_results (Dict): Dictionary containing model results with feature_importance
            top_features (int): Number of top features to display
            
        Returns:
            go.Figure: Horizontal bar chart of feature importances
        """
        fig = go.Figure()
        
        # Get all unique features across models
        all_features = set()
        for results in model_results.values():
            if 'feature_importance' in results:
                all_features.update(results['feature_importance'].keys())
        
        if not all_features:
            # Create placeholder figure
            fig.add_annotation(
                text="No feature importance data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Create comparison data
        feature_data = []
        for model_name, results in model_results.items():
            if 'feature_importance' not in results:
                continue
                
            importance_dict = results['feature_importance']
            
            # Sort and get top features
            sorted_features = sorted(
                importance_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:top_features]
            
            for feature, importance in sorted_features:
                feature_data.append({
                    'Model': model_name,
                    'Feature': feature,
                    'Importance': importance
                })
        
        if not feature_data:
            fig.add_annotation(
                text="No feature importance data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(feature_data)
        
        # Create grouped bar chart
        models = df['Model'].unique()
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]
            
            fig.add_trace(
                go.Bar(
                    x=model_data['Importance'],
                    y=model_data['Feature'],
                    name=model,
                    orientation='h',
                    marker_color=self.color_palette[i % len(self.color_palette)]
                )
            )
        
        fig.update_layout(
            title="Feature Importance Comparison",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=max(400, len(df['Feature'].unique()) * 20),
            template=self.theme,
            barmode='group'
        )
        
        return fig
    
    def create_attention_visualization(self, 
                                       model_results: Dict[str, Dict],
                                       sample_idx: int = 0) -> go.Figure:
        """
        Create LSTM attention weight visualization.
        
        Args:
            model_results (Dict): Dictionary containing model results with attention_weights
            sample_idx (int): Index of sample to visualize
            
        Returns:
            go.Figure: Heatmap of attention weights
        """
        fig = make_subplots(
            rows=1, cols=len(model_results),
            subplot_titles=list(model_results.keys()),
            horizontal_spacing=0.1
        )
        
        for col, (model_name, results) in enumerate(model_results.items(), 1):
            attention_weights = results.get('attention_weights')
            
            if attention_weights is None:
                # Add placeholder
                fig.add_annotation(
                    text=f"No attention data for {model_name}",
                    xref=f"x{col}", yref=f"y{col}",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=12)
                )
                continue
            
            # Handle different attention weight formats
            if isinstance(attention_weights, (list, np.ndarray)):
                attention_weights = np.array(attention_weights)
                
                # If 3D (batch_size, seq_len, features), take the sample
                if attention_weights.ndim == 3:
                    if sample_idx < attention_weights.shape[0]:
                        weights = attention_weights[sample_idx]
                    else:
                        weights = attention_weights[0]  # Fallback to first sample
                elif attention_weights.ndim == 2:
                    weights = attention_weights
                else:
                    # 1D weights - reshape for visualization
                    weights = attention_weights.reshape(1, -1)
                    
                # Create heatmap
                fig.add_trace(
                    go.Heatmap(
                        z=weights,
                        colorscale='Viridis',
                        showscale=(col == len(model_results))  # Show scale only for last subplot
                    ),
                    row=1, col=col
                )
                
                # Update axes labels
                fig.update_xaxes(title_text="Time Steps", row=1, col=col)
                if col == 1:
                    fig.update_yaxes(title_text="Features", row=1, col=1)
        
        fig.update_layout(
            title="LSTM Attention Weight Visualization",
            height=400,
            template=self.theme
        )
        
        return fig
    
    def create_prediction_correlation_analysis(self, model_results: Dict[str, Dict]) -> go.Figure:
        """
        Create correlation analysis between model predictions.
        
        Args:
            model_results (Dict): Dictionary containing model results
            
        Returns:
            go.Figure: Scatter plot matrix and correlation heatmap
        """
        # Extract predictions
        model_predictions = {}
        for model_name, results in model_results.items():
            y_pred_proba = results.get('y_pred_proba', [])
            if len(y_pred_proba) > 0:
                model_predictions[model_name] = y_pred_proba
        
        if len(model_predictions) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Need at least 2 models for correlation analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Create correlation matrix
        pred_df = pd.DataFrame(model_predictions)
        correlation_matrix = pred_df.corr()
        
        # Create subplot: correlation heatmap + scatter plots
        model_names = list(model_predictions.keys())
        n_models = len(model_names)
        
        if n_models == 2:
            # Simple case: one scatter plot + correlation
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['Prediction Correlation', 'Correlation Matrix'],
                horizontal_spacing=0.15
            )
            
            # Scatter plot
            model1, model2 = model_names
            fig.add_trace(
                go.Scatter(
                    x=model_predictions[model1],
                    y=model_predictions[model2],
                    mode='markers',
                    name=f'{model1} vs {model2}',
                    marker=dict(
                        size=6,
                        color=model_predictions[model1],
                        colorscale='Viridis',
                        opacity=0.7
                    )
                ),
                row=1, col=1
            )
            
            # Add perfect correlation line
            min_val = min(min(model_predictions[model1]), min(model_predictions[model2]))
            max_val = max(max(model_predictions[model1]), max(model_predictions[model2]))
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(dash='dash', color='red'),
                    name='Perfect Correlation',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.update_xaxes(title_text=f"{model1} Predictions", row=1, col=1)
            fig.update_yaxes(title_text=f"{model2} Predictions", row=1, col=1)
            
        else:
            # Multiple models: just show correlation matrix
            fig = make_subplots(
                rows=1, cols=1,
                subplot_titles=['Model Prediction Correlation Matrix']
            )
        
        # Correlation heatmap
        fig.add_trace(
            go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.round(3).values,
                texttemplate="%{text}",
                textfont={"size": 12},
                showscale=True
            ),
            row=1, col=2 if n_models == 2 else 1
        )
        
        fig.update_layout(
            title="Model Prediction Correlation Analysis",
            height=500,
            template=self.theme
        )
        
        return fig
    
    def create_performance_metrics_table(self, model_results: Dict[str, Dict]) -> dash_table.DataTable:
        """
        Create performance metrics comparison table with statistical significance.
        
        Args:
            model_results (Dict): Dictionary containing model results
            
        Returns:
            dash_table.DataTable: Interactive table with performance metrics
        """
        metrics_data = []
        
        for model_name, results in model_results.items():
            y_true = results.get('y_true', [])
            y_pred = results.get('y_pred', [])
            y_pred_proba = results.get('y_pred_proba', [])
            
            if len(y_true) == 0 or len(y_pred) == 0:
                continue
            
            # Calculate metrics
            metrics = {
                'Model': model_name,
                'Accuracy': accuracy_score(y_true, y_pred),
                'Precision': precision_score(y_true, y_pred, zero_division=0),
                'Recall': recall_score(y_true, y_pred, zero_division=0),
                'F1-Score': f1_score(y_true, y_pred, zero_division=0),
            }
            
            # Add AUC if probabilities available
            if len(y_pred_proba) > 0:
                metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
                metrics['AP'] = average_precision_score(y_true, y_pred_proba)
            
            metrics_data.append(metrics)
        
        if not metrics_data:
            return dash_table.DataTable(
                data=[{'Model': 'No data available'}],
                columns=[{'name': 'Model', 'id': 'Model'}]
            )
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(metrics_data)
        
        # Add statistical significance tests (if multiple models)
        if len(df) > 1:
            # Simple pairwise comparison for AUC (if available)
            if 'AUC' in df.columns:
                # Add significance indicator (simplified)
                df['AUC_Rank'] = df['AUC'].rank(ascending=False)
        
        # Round numerical columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].round(4)
        
        # Create table columns
        columns = []
        for col in df.columns:
            column_config = {
                'name': col,
                'id': col,
                'type': 'numeric' if col != 'Model' else 'text',
                'format': {'specifier': '.4f'} if col != 'Model' else None
            }
            columns.append(column_config)
        
        # Create the table
        table = dash_table.DataTable(
            data=df.to_dict('records'),
            columns=columns,
            style_cell={
                'textAlign': 'center',
                'padding': '10px',
                'fontFamily': 'Arial',
                'fontSize': '14px'
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 0},
                    'backgroundColor': 'rgb(248, 248, 248)',
                }
            ],
            sort_action="native",
            filter_action="native"
        )
        
        return table
    
    def get_dashboard_layout(self, model_results: Dict[str, Dict]) -> html.Div:
        """
        Get the complete dashboard layout for model comparison.
        
        Args:
            model_results (Dict): Dictionary containing model results
            
        Returns:
            html.Div: Dash HTML layout
        """
        return html.Div([
            dbc.Container([
                # Header
                dbc.Row([
                    dbc.Col([
                        html.H1("Model Comparison Dashboard", className="text-center mb-4"),
                        html.Hr()
                    ])
                ]),
                
                # Performance Metrics Table
                dbc.Row([
                    dbc.Col([
                        html.H3("Performance Metrics Comparison"),
                        self.create_performance_metrics_table(model_results)
                    ])
                ], className="mb-4"),
                
                # ROC and PR Curves
                dbc.Row([
                    dbc.Col([
                        html.H3("ROC and Precision-Recall Curves"),
                        dcc.Graph(
                            figure=self.create_roc_pr_comparison(model_results),
                            id="roc-pr-comparison"
                        )
                    ])
                ], className="mb-4"),
                
                # Feature Importance (if available)
                dbc.Row([
                    dbc.Col([
                        html.H3("Feature Importance Comparison"),
                        dcc.Graph(
                            figure=self.create_feature_importance_chart(model_results),
                            id="feature-importance-comparison"
                        )
                    ])
                ], className="mb-4"),
                
                # Attention Visualization (if available)
                dbc.Row([
                    dbc.Col([
                        html.H3("LSTM Attention Visualization"),
                        dcc.Graph(
                            figure=self.create_attention_visualization(model_results),
                            id="attention-visualization"
                        )
                    ])
                ], className="mb-4"),
                
                # Prediction Correlation
                dbc.Row([
                    dbc.Col([
                        html.H3("Model Prediction Correlation"),
                        dcc.Graph(
                            figure=self.create_prediction_correlation_analysis(model_results),
                            id="prediction-correlation"
                        )
                    ])
                ], className="mb-4"),
                
                # Side-by-side Comparison
                dbc.Row([
                    dbc.Col([
                        html.H3("Side-by-Side Performance Charts"),
                        dcc.Graph(
                            figure=self.create_side_by_side_charts(model_results),
                            id="side-by-side-comparison"
                        )
                    ])
                ])
            ], fluid=True)
        ])


def create_sample_model_results() -> Dict[str, Dict]:
    """
    Create sample model results for testing and demonstration.
    
    Returns:
        Dict: Sample model results in the expected format
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data
    y_true = np.random.binomial(1, 0.3, n_samples)  # 30% defect rate
    
    # XGBoost model (better performance)
    xgb_proba = np.random.beta(2, 5, n_samples)
    xgb_proba[y_true == 1] += np.random.normal(0.3, 0.1, np.sum(y_true == 1))
    xgb_proba = np.clip(xgb_proba, 0, 1)
    xgb_pred = (xgb_proba > 0.5).astype(int)
    
    # LSTM model (slightly different performance)
    lstm_proba = np.random.beta(2.5, 4.5, n_samples)
    lstm_proba[y_true == 1] += np.random.normal(0.25, 0.12, np.sum(y_true == 1))
    lstm_proba = np.clip(lstm_proba, 0, 1)
    lstm_pred = (lstm_proba > 0.5).astype(int)
    
    # Sample feature importance for XGBoost
    feature_names = [f"sensor_{i}" for i in range(1, 11)] + [f"feature_{i}" for i in range(1, 6)]
    xgb_importance = dict(zip(feature_names, np.random.exponential(0.1, len(feature_names))))
    
    # Sample attention weights for LSTM
    seq_len, n_features = 50, 10
    attention_weights = np.random.exponential(0.5, (1, seq_len, n_features))
    attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
    
    return {
        'XGBoost': {
            'y_true': y_true,
            'y_pred': xgb_pred,
            'y_pred_proba': xgb_proba,
            'feature_importance': xgb_importance
        },
        'LSTM': {
            'y_true': y_true,
            'y_pred': lstm_pred,
            'y_pred_proba': lstm_proba,
            'attention_weights': attention_weights
        }
    }


# Example usage for testing
if __name__ == "__main__":
    # Create sample data and test the component
    sample_results = create_sample_model_results()
    comparison = ModelComparison()
    
    # Test individual methods
    roc_pr_fig = comparison.create_roc_pr_comparison(sample_results)
    feature_fig = comparison.create_feature_importance_chart(sample_results)
    attention_fig = comparison.create_attention_visualization(sample_results)
    correlation_fig = comparison.create_prediction_correlation_analysis(sample_results)
    metrics_table = comparison.create_performance_metrics_table(sample_results)
    
    print("Model comparison component created successfully!")
    print("Available methods:")
    print("- create_roc_pr_comparison")
    print("- create_feature_importance_chart") 
    print("- create_attention_visualization")
    print("- create_prediction_correlation_analysis")
    print("- create_performance_metrics_table")
    print("- get_dashboard_layout")