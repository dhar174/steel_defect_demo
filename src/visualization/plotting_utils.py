import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

class PlottingUtils:
    """Utility functions for creating various plots and visualizations"""
    
    def __init__(self, style: str = "seaborn-v0_8"):
        """
        Initialize plotting utilities.
        
        Args:
            style (str): Matplotlib style to use
        """
        plt.style.use(style)
        sns.set_palette("husl")
        self.colors = px.colors.qualitative.Set1
    
    def plot_sensor_timeseries(self, data: pd.DataFrame, 
                             sensors: List[str] = None,
                             title: str = "Sensor Time Series") -> go.Figure:
        """
        Plot multiple sensor time series.
        
        Args:
            data (pd.DataFrame): Time series data with datetime index
            sensors (List[str]): List of sensor columns to plot
            title (str): Plot title
            
        Returns:
            go.Figure: Plotly figure
        """
        # TODO: Implement sensor time series plotting
        pass
    
    def plot_defect_distribution(self, labels: np.ndarray, 
                               title: str = "Defect Distribution") -> go.Figure:
        """
        Plot distribution of defect vs normal cases.
        
        Args:
            labels (np.ndarray): Binary labels
            title (str): Plot title
            
        Returns:
            go.Figure: Bar chart
        """
        # TODO: Implement defect distribution plotting
        pass
    
    def plot_correlation_heatmap(self, data: pd.DataFrame,
                               title: str = "Feature Correlation",
                               correlation_matrix: pd.DataFrame = None) -> go.Figure:
        """
        Plot correlation heatmap.
        
        Args:
            data (pd.DataFrame): Data for correlation analysis (if correlation_matrix not provided)
            title (str): Plot title
            correlation_matrix (pd.DataFrame): Pre-computed correlation matrix
            
        Returns:
            go.Figure: Heatmap
        """
        # Compute correlation matrix if not provided
        if correlation_matrix is None:
            corr_matrix = data.corr()
        else:
            corr_matrix = correlation_matrix
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Correlation"),
            text=np.round(corr_matrix.values, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Sensors",
            yaxis_title="Sensors",
            width=600,
            height=500
        )
        
        return fig
    
    def plot_defect_correlation_comparison(self, 
                                         good_corr: pd.DataFrame,
                                         defect_corr: pd.DataFrame,
                                         difference_corr: pd.DataFrame = None) -> go.Figure:
        """
        Plot comparison of correlations between good and defective casts.
        
        Args:
            good_corr (pd.DataFrame): Correlation matrix for good casts
            defect_corr (pd.DataFrame): Correlation matrix for defective casts
            difference_corr (pd.DataFrame): Difference matrix (defect - good)
            
        Returns:
            go.Figure: Subplot figure with comparison heatmaps
        """
        from plotly.subplots import make_subplots
        
        # Create subplots
        n_plots = 3 if difference_corr is not None else 2
        fig = make_subplots(
            rows=1, cols=n_plots,
            subplot_titles=("Good Casts", "Defective Casts", "Difference (Defect - Good)") if n_plots == 3
                          else ("Good Casts", "Defective Casts"),
            horizontal_spacing=0.1
        )
        
        # Good casts heatmap
        fig.add_trace(
            go.Heatmap(
                z=good_corr.values,
                x=good_corr.columns,
                y=good_corr.index,
                colorscale='RdBu',
                zmid=0,
                showscale=False,
                text=np.round(good_corr.values, 3),
                texttemplate="%{text}",
                textfont={"size": 8}
            ),
            row=1, col=1
        )
        
        # Defective casts heatmap
        fig.add_trace(
            go.Heatmap(
                z=defect_corr.values,
                x=defect_corr.columns,
                y=defect_corr.index,
                colorscale='RdBu',
                zmid=0,
                showscale=False,
                text=np.round(defect_corr.values, 3),
                texttemplate="%{text}",
                textfont={"size": 8}
            ),
            row=1, col=2
        )
        
        # Difference heatmap
        if difference_corr is not None:
            fig.add_trace(
                go.Heatmap(
                    z=difference_corr.values,
                    x=difference_corr.columns,
                    y=difference_corr.index,
                    colorscale='RdBu',
                    zmid=0,
                    colorbar=dict(title="Correlation Difference"),
                    text=np.round(difference_corr.values, 3),
                    texttemplate="%{text}",
                    textfont={"size": 8}
                ),
                row=1, col=3
            )
        
        fig.update_layout(
            title="Correlation Comparison: Good vs Defective Casts",
            height=400,
            width=1200 if n_plots == 3 else 800
        )
        
        return fig
    
    def plot_time_lagged_correlations(self, 
                                    lagged_correlations: Dict[str, pd.DataFrame],
                                    sensor_pair: str = None) -> go.Figure:
        """
        Plot time-lagged correlations showing delayed relationships.
        
        Args:
            lagged_correlations (Dict): Output from compute_time_lagged_correlations
            sensor_pair (str): Specific sensor pair to plot, or None for all
            
        Returns:
            go.Figure: Line plot of lagged correlations
        """
        fig = go.Figure()
        
        pairs_to_plot = [sensor_pair] if sensor_pair else list(lagged_correlations.keys())
        
        for pair in pairs_to_plot:
            if pair in lagged_correlations:
                lag_data = lagged_correlations[pair]
                fig.add_trace(go.Scatter(
                    x=lag_data['lag'],
                    y=lag_data['correlation'],
                    mode='lines+markers',
                    name=pair.replace('_', ' → '),
                    line=dict(width=2),
                    marker=dict(size=4)
                ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title="Time-Lagged Correlations Between Sensors",
            xaxis_title="Lag (seconds)",
            yaxis_title="Correlation",
            hovermode='x unified',
            height=500,
            width=800
        )
        
        return fig
    
    def plot_feature_importance_ranking(self, 
                                      importance_df: pd.DataFrame,
                                      title: str = "Sensor Feature Importance for Defect Prediction") -> go.Figure:
        """
        Plot feature importance ranking for predictive sensor combinations.
        
        Args:
            importance_df (pd.DataFrame): Output from identify_predictive_sensor_combinations
            title (str): Plot title
            
        Returns:
            go.Figure: Horizontal bar chart of feature importance
        """
        # Sort by importance (should already be sorted)
        sorted_df = importance_df.sort_values('importance', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=sorted_df['importance'],
            y=sorted_df['feature'],
            orientation='h',
            marker=dict(
                color=sorted_df['importance'],
                colorscale='Viridis',
                colorbar=dict(title="Importance Score")
            ),
            text=np.round(sorted_df['importance'], 4),
            textposition='outside'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Mutual Information Score",
            yaxis_title="Feature",
            height=max(400, len(sorted_df) * 25),
            width=800,
            margin=dict(l=200)  # Extra margin for feature names
        )
        
        return fig
    
    def plot_feature_importance(self, feature_names: List[str],
                              importances: np.ndarray,
                              title: str = "Feature Importance") -> go.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_names (List[str]): Names of features
            importances (np.ndarray): Importance scores
            title (str): Plot title
            
        Returns:
            go.Figure: Bar chart
        """
        # TODO: Implement feature importance plotting
        pass
    
    def plot_prediction_timeline(self, timestamps: List,
                               predictions: np.ndarray,
                               true_labels: np.ndarray = None,
                               title: str = "Prediction Timeline") -> go.Figure:
        """
        Plot prediction probabilities over time.
        
        Args:
            timestamps (List): Time points
            predictions (np.ndarray): Prediction probabilities
            true_labels (np.ndarray): True labels (optional)
            title (str): Plot title
            
        Returns:
            go.Figure: Time series plot
        """
        # TODO: Implement prediction timeline plotting
        pass
    
    def plot_cast_comparison(self, good_cast: pd.DataFrame,
                           defect_cast: pd.DataFrame,
                           sensor_name: str) -> go.Figure:
        """
        Compare sensor patterns between good and defect casts.
        
        Args:
            good_cast (pd.DataFrame): Data from a good cast
            defect_cast (pd.DataFrame): Data from a defect cast
            sensor_name (str): Name of sensor to compare
            
        Returns:
            go.Figure: Comparison plot
        """
        # TODO: Implement cast comparison plotting
        pass
    
    def plot_model_performance_comparison(self, results: Dict[str, Dict]) -> go.Figure:
        """
        Compare performance metrics across models.
        
        Args:
            results (Dict[str, Dict]): Model evaluation results
            
        Returns:
            go.Figure: Performance comparison chart
        """
        # TODO: Implement model performance comparison
        pass
    
    def plot_threshold_analysis(self, thresholds: np.ndarray,
                              precisions: np.ndarray,
                              recalls: np.ndarray,
                              f1_scores: np.ndarray) -> go.Figure:
        """
        Plot threshold analysis for binary classification.
        
        Args:
            thresholds (np.ndarray): Threshold values
            precisions (np.ndarray): Precision scores
            recalls (np.ndarray): Recall scores
            f1_scores (np.ndarray): F1 scores
            
        Returns:
            go.Figure: Threshold analysis plot
        """
        # TODO: Implement threshold analysis plotting
        pass
    
    def save_plot(self, figure: go.Figure, filename: str, 
                 format: str = "png") -> None:
        """
        Save plot to file.
        
        Args:
            figure (go.Figure): Plotly figure to save
            filename (str): Output filename
            format (str): Output format ('png', 'html', 'pdf', 'svg')
        """
        from pathlib import Path
        
        # Create output directory if it doesn't exist
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'html':
            figure.write_html(filename)
        elif format.lower() == 'png':
            figure.write_image(filename, format='png')
        elif format.lower() == 'pdf':
            figure.write_image(filename, format='pdf')
        elif format.lower() == 'svg':
            figure.write_image(filename, format='svg')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Plot saved to {filename}")