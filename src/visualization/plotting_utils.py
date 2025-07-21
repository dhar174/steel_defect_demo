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
        # Count good vs defect cases
        unique, counts = np.unique(labels, return_counts=True)
        class_names = ['Good', 'Defect']
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=class_names,
                y=counts,
                text=[f'{count} ({count/len(labels)*100:.1f}%)' for count in counts],
                textposition='auto',
                marker_color=['#2E8B57', '#DC143C']
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Class",
            yaxis_title="Count",
            showlegend=False
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
        # Compute correlation matrix
        corr_matrix = data.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            showscale=True
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Features",
            yaxis_title="Features",
            height=600,
            width=800
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
            format (str): Output format
        """
        if format == "png":
            figure.write_image(filename)
        elif format == "html":
            figure.write_html(filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def plot_sensor_histograms(self, data: pd.DataFrame, sensor_name: str,
                             class_column: str = 'defect_label',
                             title: str = None) -> go.Figure:
        """
        Plot histograms for sensor values, stratified by class.
        
        Args:
            data (pd.DataFrame): Data containing sensor values
            sensor_name (str): Name of sensor to plot
            class_column (str): Column containing class labels
            title (str): Plot title
            
        Returns:
            go.Figure: Histogram plot
        """
        if title is None:
            title = f"{sensor_name} Distribution by Class"
        
        # Get sensor columns
        sensor_cols = [col for col in data.columns if col.startswith(sensor_name)]
        
        if not sensor_cols:
            raise ValueError(f"No columns found for sensor: {sensor_name}")
        
        # For demonstration, use the mean column if available
        sensor_col = None
        for col in sensor_cols:
            if 'mean' in col:
                sensor_col = col
                break
        
        if sensor_col is None:
            sensor_col = sensor_cols[0]  # Use first available column
        
        # Separate by class
        good_data = data[data[class_column] == 0][sensor_col].dropna()
        defect_data = data[data[class_column] == 1][sensor_col].dropna()
        
        fig = go.Figure()
        
        # Add histograms
        fig.add_trace(go.Histogram(
            x=good_data,
            name='Good',
            opacity=0.7,
            marker_color='#2E8B57',
            nbinsx=20
        ))
        
        fig.add_trace(go.Histogram(
            x=defect_data,
            name='Defect',
            opacity=0.7,
            marker_color='#DC143C',
            nbinsx=20
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=sensor_col,
            yaxis_title="Frequency",
            barmode='overlay',
            legend=dict(x=0.8, y=0.9)
        )
        
        return fig
    
    def plot_sensor_boxplots(self, data: pd.DataFrame, sensor_names: List[str],
                           class_column: str = 'defect_label',
                           title: str = "Sensor Box Plots by Class") -> go.Figure:
        """
        Plot box plots for multiple sensors, stratified by class.
        
        Args:
            data (pd.DataFrame): Data containing sensor values
            sensor_names (List[str]): Names of sensors to plot
            class_column (str): Column containing class labels
            title (str): Plot title
            
        Returns:
            go.Figure: Box plot
        """
        fig = go.Figure()
        
        for i, sensor_name in enumerate(sensor_names):
            # Get sensor columns and use mean if available
            sensor_cols = [col for col in data.columns if col.startswith(sensor_name)]
            sensor_col = None
            
            for col in sensor_cols:
                if 'mean' in col:
                    sensor_col = col
                    break
            
            if sensor_col is None and sensor_cols:
                sensor_col = sensor_cols[0]
            
            if sensor_col:
                # Good class
                good_data = data[data[class_column] == 0][sensor_col].dropna()
                fig.add_trace(go.Box(
                    y=good_data,
                    name=f'{sensor_name} - Good',
                    marker_color='#2E8B57',
                    legendgroup='good',
                    showlegend=(i == 0),
                    legendgrouptitle_text='Good'
                ))
                
                # Defect class
                defect_data = data[data[class_column] == 1][sensor_col].dropna()
                fig.add_trace(go.Box(
                    y=defect_data,
                    name=f'{sensor_name} - Defect',
                    marker_color='#DC143C',
                    legendgroup='defect',
                    showlegend=(i == 0),
                    legendgrouptitle_text='Defect'
                ))
        
        fig.update_layout(
            title=title,
            yaxis_title="Sensor Values",
            boxmode='group'
        )
        
        return fig
    
    def plot_outlier_detection(self, data: pd.DataFrame, outlier_results: Dict,
                             sensor_name: str, method: str = 'iqr',
                             title: str = None) -> go.Figure:
        """
        Plot outlier detection results.
        
        Args:
            data (pd.DataFrame): Data containing sensor values
            outlier_results (Dict): Results from outlier detection
            sensor_name (str): Name of sensor to plot
            method (str): Outlier detection method used
            title (str): Plot title
            
        Returns:
            go.Figure: Scatter plot with outliers highlighted
        """
        if title is None:
            title = f"{sensor_name} Outlier Detection ({method.upper()})"
        
        # Get sensor columns
        sensor_cols = [col for col in data.columns if col.startswith(sensor_name)]
        
        if not sensor_cols:
            raise ValueError(f"No columns found for sensor: {sensor_name}")
        
        # Use mean column if available
        sensor_col = None
        for col in sensor_cols:
            if 'mean' in col:
                sensor_col = col
                break
        
        if sensor_col is None:
            sensor_col = sensor_cols[0]
        
        # Get outlier information
        sensor_outliers = outlier_results.get('sensors', {}).get(sensor_name, {})
        outlier_info = sensor_outliers.get(sensor_col, {})
        outlier_indices = outlier_info.get('outlier_indices', [])
        
        # Create scatter plot
        fig = go.Figure()
        
        # Plot all points
        y_values = data[sensor_col]
        x_values = list(range(len(y_values)))
        
        normal_mask = ~data.index.isin(outlier_indices)
        outlier_mask = data.index.isin(outlier_indices)
        
        # Normal points
        fig.add_trace(go.Scatter(
            x=np.array(x_values)[normal_mask],
            y=y_values[normal_mask],
            mode='markers',
            name='Normal',
            marker=dict(color='#2E8B57', size=6),
            hovertemplate='Index: %{x}<br>Value: %{y}<extra></extra>'
        ))
        
        # Outlier points
        if outlier_mask.any():
            fig.add_trace(go.Scatter(
                x=np.array(x_values)[outlier_mask],
                y=y_values[outlier_mask],
                mode='markers',
                name='Outliers',
                marker=dict(color='#DC143C', size=10, symbol='x'),
                hovertemplate='Index: %{x}<br>Value: %{y}<br>OUTLIER<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Cast Index",
            yaxis_title=sensor_col,
            hovermode='closest'
        )
        
        return fig
    
    def plot_ks_test_results(self, ks_results: Dict, 
                           title: str = "Kolmogorov-Smirnov Test Results") -> go.Figure:
        """
        Plot KS test results showing p-values for different sensors.
        
        Args:
            ks_results (Dict): Results from KS tests
            title (str): Plot title
            
        Returns:
            go.Figure: Bar chart of p-values
        """
        # Extract sensor results
        sensors = list(ks_results['sensors'].keys())
        p_values = []
        feature_names = []
        
        for sensor in sensors:
            sensor_data = ks_results['sensors'][sensor]
            for feature, results in sensor_data.items():
                p_values.append(results['p_value'])
                feature_names.append(f"{sensor}_{feature.split('_', 1)[1]}")
        
        # Create bar chart
        colors = ['#DC143C' if p < 0.05 else '#2E8B57' for p in p_values]
        
        fig = go.Figure(data=[
            go.Bar(
                x=feature_names,
                y=p_values,
                marker_color=colors,
                text=[f'{p:.3f}' for p in p_values],
                textposition='auto',
                hovertemplate='Feature: %{x}<br>P-value: %{y:.4f}<extra></extra>'
            )
        ])
        
        # Add significance line
        fig.add_hline(y=0.05, line_dash="dash", line_color="red",
                     annotation_text="α = 0.05")
        
        fig.update_layout(
            title=title,
            xaxis_title="Sensor Features",
            yaxis_title="P-value",
            xaxis_tickangle=-45,
            showlegend=False
        )
        
        return fig