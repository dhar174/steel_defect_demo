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
            
        Raises:
            ValueError: If no numeric columns are found in the data
        """

        # Calculate correlation matrix
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Validate that we have numeric data to compute correlations
        if numeric_data.empty:
            raise ValueError("No numeric columns found in the data. Cannot compute correlation matrix.")
        
        corr_matrix = numeric_data.corr()
        

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            showscale=True,
            texttemplate='%{text}',
            colorbar=dict(title="Correlation"),
            textfont={"size": 10},
            hoverongaps=False

        ))
        
        fig.update_layout(
            title=title,
            template="plotly_white",
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
            showlegend=False)

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
