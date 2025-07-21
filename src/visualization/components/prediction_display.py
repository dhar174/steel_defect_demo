"""
Prediction Visualization Components for Steel Defect Detection Dashboard

This module provides components for visualizing defect prediction outputs including:
- Real-time prediction probability gauge
- Historical prediction timeline
- Model ensemble contribution charts
- Alert status indicators
- Prediction confidence visualization
- Model accuracy metrics display
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

import plotly.graph_objects as go
import plotly.express as px
from dash import html, dcc
import dash_bootstrap_components as dbc


class PredictionDisplayComponents:
    """
    Collection of components for displaying prediction results and model performance.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize prediction display components.
        
        Args:
            config (Dict): Configuration dictionary containing thresholds and settings
        """
        self.config = config or {}
        
        # Extract thresholds from config (defaults match inference_config.yaml)
        inference_config = self.config.get('inference', {})
        thresholds = inference_config.get('thresholds', {})
        
        self.defect_threshold = thresholds.get('defect_probability', 0.5)
        self.high_risk_threshold = thresholds.get('high_risk_threshold', 0.7)
        self.alert_threshold = thresholds.get('alert_threshold', 0.8)
        
        # Color scheme for risk levels
        default_risk_colors = {
            'safe': '#2E8B57',      # Green
            'warning': '#FFD700',   # Yellow  
            'high_risk': '#FF6B35', # Orange
            'alert': '#DC143C'      # Red
        }
        self.risk_colors = self.config.get('risk_colors', default_risk_colors)
    
    def create_prediction_gauge(self, 
                              prediction_prob: float, 
                              title: str = "Defect Probability",
                              theme: str = "plotly_white",
                              height: int = 400) -> go.Figure:
        """
        Create a real-time prediction probability gauge with color-coded risk levels.
        
        Args:
            prediction_prob (float): Current prediction probability (0-1)
            title (str): Gauge title
            theme (str): Plotly theme
            height (int): Figure height
            
        Returns:
            go.Figure: Gauge chart with color-coded risk levels
        """
        # Determine gauge bar color based on risk level
        if prediction_prob < self.defect_threshold:
            bar_color = self.risk_colors['safe']
        elif prediction_prob < self.high_risk_threshold:
            bar_color = self.risk_colors['warning']
        elif prediction_prob < self.alert_threshold:
            bar_color = self.risk_colors['high_risk']
        else:
            bar_color = self.risk_colors['alert']
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction_prob,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 20}},
            number={'font': {'size': 24}},
            delta={
                'reference': self.defect_threshold,
                'increasing': {'color': self.risk_colors['alert']},
                'decreasing': {'color': self.risk_colors['safe']}
            },
            gauge={
                'axis': {
                    'range': [None, 1],
                    'tickwidth': 1,
                    'tickcolor': "darkblue"
                },
                'bar': {'color': bar_color, 'thickness': 0.8},
                'steps': [
                    {'range': [0, self.defect_threshold], 'color': self.risk_colors['safe']},
                    {'range': [self.defect_threshold, self.high_risk_threshold], 'color': self.risk_colors['warning']},
                    {'range': [self.high_risk_threshold, self.alert_threshold], 'color': self.risk_colors['high_risk']},
                    {'range': [self.alert_threshold, 1], 'color': self.risk_colors['alert']}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': self.alert_threshold
                }
            }
        ))
        
        fig.update_layout(
            height=height,
            template=theme,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    def create_historical_timeline(self, 
                                 history_data: pd.DataFrame,
                                 title: str = "Prediction History Timeline",
                                 theme: str = "plotly_white",
                                 height: int = 400) -> go.Figure:
        """
        Create historical prediction timeline with color-coded risk levels.
        
        Args:
            history_data (pd.DataFrame): Historical predictions with timestamp index and 'prediction' column
            title (str): Plot title  
            theme (str): Plotly theme
            height (int): Figure height
            
        Returns:
            go.Figure: Timeline plot with color-coded line segments
        """
        if history_data.empty or 'prediction' not in history_data.columns:
            # Return empty figure if no data
            fig = go.Figure()
            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title="Prediction Probability",
                height=height,
                template=theme,
                annotations=[{
                    'text': 'No prediction data available',
                    'showarrow': False,
                    'x': 0.5,
                    'y': 0.5,
                    'xref': 'paper',
                    'yref': 'paper',
                    'font': {'size': 16}
                }]
            )
            return fig
        
        fig = go.Figure()
        
        # Create segments based on risk levels for color coding
        timestamps = history_data.index
        predictions = history_data['prediction'].values
        
        # Add main prediction line
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=predictions,
            mode='lines+markers',
            name='Prediction Probability',
            line=dict(width=3),
            marker=dict(size=6),
            customdata=self._get_risk_levels(predictions),
            hovertemplate='<b>Time:</b> %{x}<br>' +
                         '<b>Probability:</b> %{y:.3f}<br>' +
                         '<b>Risk Level:</b> %{customdata}<br>' +
                         '<extra></extra>'
        ))
        
        # Add colored segments based on risk levels
        self._add_risk_segments(fig, timestamps, predictions)
        
        # Add threshold lines
        fig.add_hline(
            y=self.defect_threshold, 
            line_dash="dash", 
            line_color="gray",
            annotation_text=f"Defect Threshold ({self.defect_threshold})"
        )
        
        fig.add_hline(
            y=self.high_risk_threshold,
            line_dash="dot",
            line_color=self.risk_colors['high_risk'],
            annotation_text=f"High Risk ({self.high_risk_threshold})"
        )
        
        fig.add_hline(
            y=self.alert_threshold,
            line_dash="dashdot", 
            line_color=self.risk_colors['alert'],
            annotation_text=f"Alert ({self.alert_threshold})"
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Prediction Probability",
            yaxis=dict(range=[0, 1]),
            height=height,
            template=theme,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def create_ensemble_contribution_chart(self, 
                                         baseline_contrib: float,
                                         lstm_contrib: float,
                                         title: str = "Model Ensemble Contributions",
                                         theme: str = "plotly_white",
                                         height: int = 400) -> go.Figure:
        """
        Create model ensemble contribution visualization.
        
        Args:
            baseline_contrib (float): Baseline model contribution (0-1)
            lstm_contrib (float): LSTM model contribution (0-1)
            title (str): Chart title
            theme (str): Plotly theme
            height (int): Figure height
            
        Returns:
            go.Figure: Pie chart or bar chart showing model contributions
        """
        # Normalize contributions to sum to 1
        total = baseline_contrib + lstm_contrib
        if total > 0:
            baseline_norm = baseline_contrib / total
            lstm_norm = lstm_contrib / total
        else:
            baseline_norm = lstm_norm = 0.5
        
        models = ['Baseline', 'LSTM']
        contributions = [baseline_norm, lstm_norm]
        colors = ['#1f77b4', '#ff7f0e']
        
        # Create pie chart
        fig = go.Figure(data=[
            go.Pie(
                labels=models,
                values=contributions,
                marker_colors=colors,
                textinfo='label+percent',
                textposition='auto',
                textfont_size=14,
                hole=0.3  # Donut chart for modern look
            )
        ])
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            height=height,
            template=theme,
            showlegend=True,
            annotations=[{
                'text': f'Final<br>Prediction',
                'x': 0.5, 'y': 0.5,
                'font_size': 12,
                'showarrow': False
            }]
        )
        
        return fig
    
    def create_alert_status_indicator(self, 
                                    prediction_prob: float,
                                    cast_id: Optional[str] = None) -> html.Div:
        """
        Create alert status indicator with dynamic styling.
        
        Args:
            prediction_prob (float): Current prediction probability
            cast_id (str): Current cast ID for context
            
        Returns:
            html.Div: Alert status indicator component
        """
        # Determine alert level and styling
        if prediction_prob < self.defect_threshold:
            status_text = "OK - Normal Operation"
            status_icon = "✅"
            alert_class = "alert-success"
            status_color = self.risk_colors['safe']
        elif prediction_prob < self.high_risk_threshold:
            status_text = "CAUTION - Monitor Closely"
            status_icon = "⚠️"
            alert_class = "alert-warning"
            status_color = self.risk_colors['warning']
        elif prediction_prob < self.alert_threshold:
            status_text = "HIGH RISK - Check Parameters"
            status_icon = "🔶"
            alert_class = "alert-danger"
            status_color = self.risk_colors['high_risk']
        else:
            status_text = "DEFECT ALERT - Immediate Action Required"
            status_icon = "🚨"
            alert_class = "alert-danger"
            status_color = self.risk_colors['alert']
        
        cast_info = f" (Cast: {cast_id})" if cast_id else ""
        
        return html.Div([
            dbc.Alert([
                html.H4([
                    html.Span(status_icon, style={'marginRight': '10px'}),
                    status_text,
                    html.Small(cast_info, className="text-muted")
                ], className="alert-heading"),
                html.P([
                    f"Prediction Probability: ",
                    html.Strong(f"{prediction_prob:.3f}", style={'color': status_color}),
                    html.Br(),
                    f"Last Updated: {datetime.now().strftime('%H:%M:%S')}"
                ], className="mb-0")
            ], color=alert_class.split('-')[1], className=f"{alert_class} mb-3"),
        ])
    
    def create_confidence_visualization(self, 
                                      prediction_prob: float,
                                      confidence_interval: Optional[Tuple[float, float]] = None,
                                      uncertainty: Optional[float] = None,
                                      title: str = "Prediction Confidence",
                                      theme: str = "plotly_white",
                                      height: int = 300) -> go.Figure:
        """
        Create prediction confidence visualization with uncertainty bands.
        
        Args:
            prediction_prob (float): Main prediction probability
            confidence_interval (Tuple[float, float]): Optional confidence interval (low, high)
            uncertainty (float): Optional uncertainty estimate
            title (str): Chart title
            theme (str): Plotly theme
            height (int): Figure height
            
        Returns:
            go.Figure: Confidence visualization
        """
        fig = go.Figure()
        
        # If confidence interval is provided, show it as error bars
        if confidence_interval:
            lower_bound, upper_bound = confidence_interval
            error_y = {
                'array': [upper_bound - prediction_prob],
                'arrayminus': [prediction_prob - lower_bound],
                'visible': True,
                'color': 'rgba(0,100,80,0.5)',
                'thickness': 3,
                'width': 10
            }
        elif uncertainty:
            # Use uncertainty to create symmetric error bars
            error_y = {
                'array': [uncertainty],
                'visible': True,
                'color': 'rgba(0,100,80,0.5)',
                'thickness': 3,
                'width': 10
            }
        else:
            error_y = None
        
        # Main prediction point
        fig.add_trace(go.Scatter(
            x=['Current Prediction'],
            y=[prediction_prob],
            mode='markers',
            marker=dict(
                size=20,
                color=self._get_risk_color(prediction_prob),
                line=dict(width=2, color='DarkSlateGrey')
            ),
            error_y=error_y,
            name='Prediction',
            hovertemplate='<b>Prediction:</b> %{y:.3f}<br>' +
                         ('<b>Confidence Interval:</b> [%.3f, %.3f]<br>' % confidence_interval if confidence_interval else '') +
                         ('<b>Uncertainty:</b> ±%.3f<br>' % uncertainty if uncertainty else '') +
                         '<extra></extra>'
        ))
        
        # Add threshold reference lines
        for threshold, name in [(self.defect_threshold, 'Defect'),
                               (self.high_risk_threshold, 'High Risk'),
                               (self.alert_threshold, 'Alert')]:
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="gray",
                opacity=0.5,
                annotation_text=name
            )
        
        fig.update_layout(
            title=title,
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1]),
            height=height,
            template=theme,
            showlegend=False,
            xaxis=dict(showgrid=False)
        )
        
        return fig
    
    def create_accuracy_metrics_display(self, 
                                      metrics: Dict[str, float],
                                      model_name: str = "Ensemble") -> html.Div:
        """
        Create prediction accuracy metrics display.
        
        Args:
            metrics (Dict[str, float]): Performance metrics (accuracy, precision, recall, f1_score)
            model_name (str): Name of the model
            
        Returns:
            html.Div: Metrics display component
        """
        # Default metrics if none provided
        default_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
        metrics = {**default_metrics, **metrics}
        
        # Create metric cards
        metric_cards = []
        metric_info = {
            'accuracy': {'label': 'Accuracy', 'color': 'primary', 'icon': '🎯'},
            'precision': {'label': 'Precision', 'color': 'info', 'icon': '🔍'},
            'recall': {'label': 'Recall', 'color': 'success', 'icon': '📡'},
            'f1_score': {'label': 'F1-Score', 'color': 'warning', 'icon': '⚖️'}
        }
        
        for metric_key, value in metrics.items():
            if metric_key in metric_info:
                info = metric_info[metric_key]
                
                # Color code based on performance
                if value >= 0.9:
                    text_color = "success"
                elif value >= 0.8:
                    text_color = "warning"
                else:
                    text_color = "danger"
                
                card = dbc.Card([
                    dbc.CardBody([
                        html.H6([
                            html.Span(info['icon'], style={'marginRight': '5px'}),
                            info['label']
                        ], className="card-title text-center"),
                        html.H4(
                            f"{value:.3f}",
                            className=f"text-{text_color} text-center mb-0"
                        )
                    ])
                ], color=info['color'], outline=True, className="mb-2")
                
                metric_cards.append(dbc.Col(card, width=6))
        
        return html.Div([
            html.H5(f"{model_name} Performance Metrics", className="mb-3"),
            dbc.Row(metric_cards)
        ])
    
    def _get_risk_levels(self, predictions: np.ndarray) -> List[str]:
        """Get risk level labels for predictions."""
        risk_levels = []
        for pred in predictions:
            if pred < self.defect_threshold:
                risk_levels.append('Safe')
            elif pred < self.high_risk_threshold:
                risk_levels.append('Warning')
            elif pred < self.alert_threshold:
                risk_levels.append('High Risk')
            else:
                risk_levels.append('Alert')
        return risk_levels
    
    def _get_risk_color(self, prediction: float) -> str:
        """Get color for a single prediction based on risk level."""
        if prediction < self.defect_threshold:
            return self.risk_colors['safe']
        elif prediction < self.high_risk_threshold:
            return self.risk_colors['warning']
        elif prediction < self.alert_threshold:
            return self.risk_colors['high_risk']
        else:
            return self.risk_colors['alert']
    
    def _add_risk_segments(self, fig: go.Figure, timestamps: pd.Index, predictions: np.ndarray):
        """Add colored segments to timeline based on risk levels."""
        # This creates a more visually appealing colored line
        # by adding multiple traces for different risk levels
        current_level = None
        segment_start = 0
        
        for i, pred in enumerate(predictions):
            level = self._get_risk_level_key(pred)
            
            if level != current_level and i > 0:
                # Add segment for previous level
                if current_level is not None:
                    self._add_segment_trace(
                        fig, timestamps[segment_start:i+1], 
                        predictions[segment_start:i+1], 
                        current_level
                    )
                segment_start = i
                current_level = level
            elif current_level is None:
                current_level = level
        
        # Add final segment
        if current_level is not None and segment_start < len(predictions):
            self._add_segment_trace(
                fig, timestamps[segment_start:], 
                predictions[segment_start:], 
                current_level
            )
    
    def _get_risk_level_key(self, prediction: float) -> str:
        """Get risk level key for a prediction."""
        if prediction < self.defect_threshold:
            return 'safe'
        elif prediction < self.high_risk_threshold:
            return 'warning'
        elif prediction < self.alert_threshold:
            return 'high_risk'
        else:
            return 'alert'
    
    def _add_segment_trace(self, fig: go.Figure, timestamps: pd.Index, 
                          predictions: np.ndarray, risk_level: str):
        """Add a colored segment trace to the timeline."""
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=predictions,
            mode='lines',
            line=dict(color=self.risk_colors[risk_level], width=4),
            showlegend=False,
            hoverinfo='skip'
        ))


def create_sample_data_for_demo(seed: Optional[int] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Create sample data for demonstration purposes.
    
    Args:
        seed (Optional[int]): Random seed for reproducibility. If None, no seed is set.
    
    Returns:
        Tuple[pd.DataFrame, Dict]: Sample historical data and metrics
    """
    # Generate sample historical prediction data
    if seed is not None:
        np.random.seed(seed)  # Set seed for reproducible demo data
    
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=4)
    timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
    
    # Create realistic prediction pattern with some trend
    base_prob = 0.3
    trend = np.linspace(0, 0.4, len(timestamps))
    noise = 0.1 * np.random.randn(len(timestamps))
    predictions = np.clip(base_prob + trend + noise, 0, 1)
    
    history_data = pd.DataFrame({
        'prediction': predictions
    }, index=timestamps)
    
    # Sample metrics
    sample_metrics = {
        'accuracy': 0.891,
        'precision': 0.856,
        'recall': 0.923,
        'f1_score': 0.888
    }
    
    return history_data, sample_metrics