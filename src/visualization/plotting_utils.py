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
        if sensors is None:
            sensors = [col for col in data.columns if col not in ['cast_id', 'defect_label']]
        
        fig = go.Figure()
        
        for i, sensor in enumerate(sensors):
            if sensor in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[sensor],
                    mode='lines',
                    name=sensor.replace('_', ' ').title(),
                    line=dict(color=self.colors[i % len(self.colors)])
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Sensor Value",
            showlegend=True,
            height=600,
            template="plotly_white"
        )
        
        return fig
    
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
        unique, counts = np.unique(labels, return_counts=True)
        labels_text = ['Good Cast' if label == 0 else 'Defect Cast' for label in unique]
        colors = ['blue', 'red']
        
        fig = go.Figure(data=[
            go.Bar(
                x=labels_text,
                y=counts,
                marker_color=colors[:len(unique)],
                text=counts,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Cast Type",
            yaxis_title="Count",
            showlegend=False,
            height=400,
            template="plotly_white"
        )
        
        return fig
    
    def plot_correlation_heatmap(self, data: pd.DataFrame,
                               title: str = "Feature Correlation") -> go.Figure:
        """
        Plot correlation heatmap.
        
        Args:
            data (pd.DataFrame): Data for correlation analysis
            title (str): Plot title
            
        Returns:
            go.Figure: Heatmap
        """
        # Calculate correlation matrix
        numeric_data = data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            height=600,
            width=600,
            template="plotly_white"
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
        fig = go.Figure()
        
        # Plot prediction probabilities
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=predictions,
            mode='lines+markers',
            name='Prediction Probability',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # Add threshold line at 0.5
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                     annotation_text="Decision Threshold (0.5)")
        
        # Plot true labels if available
        if true_labels is not None:
            # Create colored background regions for true defects
            defect_regions = []
            for i, label in enumerate(true_labels):
                if label == 1:  # Defect cast
                    defect_regions.append(i)
            
            if defect_regions:
                fig.add_trace(go.Scatter(
                    x=[timestamps[i] for i in defect_regions],
                    y=[1.0] * len(defect_regions),
                    mode='markers',
                    name='True Defects',
                    marker=dict(symbol='x', size=8, color='red')
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Prediction Probability",
            yaxis=dict(range=[0, 1]),
            showlegend=True,
            height=500,
            template="plotly_white"
        )
        
        return fig
    
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
        fig = go.Figure()
        
        # Plot good cast
        if sensor_name in good_cast.columns:
            fig.add_trace(go.Scatter(
                x=good_cast.index,
                y=good_cast[sensor_name],
                mode='lines',
                name='Good Cast',
                line=dict(color='blue', width=2),
                opacity=0.8
            ))
        
        # Plot defect cast
        if sensor_name in defect_cast.columns:
            fig.add_trace(go.Scatter(
                x=defect_cast.index,
                y=defect_cast[sensor_name],
                mode='lines',
                name='Defect Cast',
                line=dict(color='red', width=2),
                opacity=0.8
            ))
        
        fig.update_layout(
            title=f"{sensor_name.replace('_', ' ').title()} - Good vs Defect Cast Comparison",
            xaxis_title="Time",
            yaxis_title=f"{sensor_name.replace('_', ' ').title()}",
            showlegend=True,
            height=500,
            template="plotly_white"
        )
        
        return fig
    
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
            format (str): Output format
        """
        if format.lower() == "html":
            figure.write_html(filename)
        else:
            figure.write_image(filename, format=format)
    
    def create_multi_sensor_dashboard(self, cast_data: pd.DataFrame, 
                                    cast_metadata: Dict = None,
                                    sensors: List[str] = None) -> go.Figure:
        """
        Create a comprehensive multi-sensor dashboard for a single cast.
        
        Args:
            cast_data (pd.DataFrame): Time series data for a single cast
            cast_metadata (Dict): Metadata about the cast (optional)
            sensors (List[str]): List of sensors to display (optional)
            
        Returns:
            go.Figure: Multi-panel dashboard figure
        """
        if sensors is None:
            sensors = [col for col in cast_data.columns if col not in ['cast_id', 'defect_label']]
        
        # Create subplot layout - configurable columns and rows based on number of sensors
        rows = max(self.MIN_ROWS, (len(sensors) + self.DEFAULT_COLUMNS - 1) // self.DEFAULT_COLUMNS)
        
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=rows, cols=self.DEFAULT_COLUMNS,
            subplot_titles=sensors + ['Summary Statistics'],
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        # Plot each sensor in its own subplot
        for i, sensor in enumerate(sensors):
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            if sensor in cast_data.columns:
                # Determine color based on cast type if metadata available
                color = 'red' if (cast_metadata and cast_metadata.get('defect_label', 0) == 1) else 'blue'
                
                fig.add_trace(
                    go.Scatter(
                        x=cast_data.index,
                        y=cast_data[sensor],
                        mode='lines',
                        name=sensor.replace('_', ' ').title(),
                        line=dict(color=color, width=1.5),
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        # Add summary statistics in the last panel if there's space
        if len(sensors) < rows * 3:
            summary_row = ((len(sensors)) // 3) + 1
            summary_col = ((len(sensors)) % 3) + 1
            
            # Create summary statistics
            summary_stats = []
            for sensor in sensors:
                if sensor in cast_data.columns:
                    mean_val = cast_data[sensor].mean()
                    std_val = cast_data[sensor].std()
                    summary_stats.append(f"{sensor}: μ={mean_val:.2f}, σ={std_val:.2f}")
            
            # Add text annotation for summary
            fig.add_annotation(
                text="<br>".join(summary_stats),
                xref="paper", yref="paper",
                x=0.85, y=0.5,
                showarrow=False,
                font=dict(size=10),
                bgcolor="lightgray",
                bordercolor="gray",
                borderwidth=1
            )
        
        # Update layout
        cast_type = "DEFECT" if (cast_metadata and cast_metadata.get('defect_label', 0) == 1) else "GOOD"
        cast_id = cast_metadata.get('cast_id', 'Unknown') if cast_metadata else 'Unknown'
        
        fig.update_layout(
            title=f"Multi-Sensor Dashboard - Cast {cast_id} ({cast_type})",
            height=200 * rows,
            showlegend=False,
            template="plotly_white"
        )
        
        return fig
    
    def create_side_by_side_comparison(self, good_casts: List[pd.DataFrame],
                                     defect_casts: List[pd.DataFrame],
                                     sensors: List[str] = None) -> go.Figure:
        """
        Create side-by-side comparison plots for normal vs defect casts.
        
        Args:
            good_casts (List[pd.DataFrame]): List of good cast data
            defect_casts (List[pd.DataFrame]): List of defect cast data
            sensors (List[str]): List of sensors to compare
            
        Returns:
            go.Figure: Side-by-side comparison figure
        """
        if sensors is None:
            # Get sensors from the first available cast
            all_casts = good_casts + defect_casts
            for cast_data in all_casts:
                if not cast_data.empty:
                    sensors = [col for col in cast_data.columns if col not in ['cast_id', 'defect_label']]
                    break
        
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=len(sensors), cols=2,
            subplot_titles=['Good Casts', 'Defect Casts'] * len(sensors),
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        for i, sensor in enumerate(sensors):
            row = i + 1
            
            # Plot good casts
            for j, good_cast in enumerate(good_casts):
                if sensor in good_cast.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=good_cast.index,
                            y=good_cast[sensor],
                            mode='lines',
                            name=f'Good Cast {j+1}' if i == 0 else '',
                            line=dict(color='blue', width=1.5),
                            opacity=0.7,
                            showlegend=(i == 0)
                        ),
                        row=row, col=1
                    )
            
            # Plot defect casts
            for j, defect_cast in enumerate(defect_casts):
                if sensor in defect_cast.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=defect_cast.index,
                            y=defect_cast[sensor],
                            mode='lines',
                            name=f'Defect Cast {j+1}' if i == 0 else '',
                            line=dict(color='red', width=1.5),
                            opacity=0.7,
                            showlegend=(i == 0)
                        ),
                        row=row, col=2
                    )
            
            # Update y-axis labels
            fig.update_yaxes(title_text=sensor.replace('_', ' ').title(), row=row, col=1)
            fig.update_yaxes(title_text=sensor.replace('_', ' ').title(), row=row, col=2)
        
        fig.update_layout(
            title="Side-by-Side Sensor Comparison: Normal vs Defect Casts",
            height=300 * len(sensors),
            template="plotly_white"
        )
        
        return fig