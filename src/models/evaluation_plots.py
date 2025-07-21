"""
Advanced plotting utilities for model evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import warnings


class EvaluationPlots:
    """Advanced plotting utilities for model evaluation"""
    
    def __init__(self, style: str = 'seaborn', output_dir: str = "results/plots"):
        """
        Initialize plotting utilities
        
        Args:
            style: Plotting style ('seaborn', 'ggplot', 'default')
            output_dir: Directory to save plots
        """
        self.style = style
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup plotting style
        self._setup_style()
        
        # Color palettes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8'
        }
        
    def _setup_style(self):
        """Setup matplotlib style"""
        if self.style == 'seaborn':
            sns.set_style("whitegrid")
            plt.rcParams['figure.facecolor'] = 'white'
            plt.rcParams['axes.facecolor'] = 'white'
        elif self.style == 'ggplot':
            plt.style.use('ggplot')
        
        # Common settings
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 16
    
    def create_dashboard(self, evaluation_results: Dict[str, Any]) -> plt.Figure:
        """
        Create comprehensive evaluation dashboard
        
        Args:
            evaluation_results: Complete evaluation results
            
        Returns:
            Combined dashboard figure
        """
        fig = plt.figure(figsize=(20, 15))
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Extract data
        metrics = evaluation_results['metrics']
        model_name = evaluation_results.get('model_name', 'Model')
        
        # 1. Key Metrics Summary (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_metrics_summary(ax1, metrics, model_name)
        
        # 2. Confusion Matrix (top center-left)
        ax2 = fig.add_subplot(gs[0, 1])
        if 'confusion_matrix' in evaluation_results:
            cm = np.array(evaluation_results['confusion_matrix'])
            self._plot_confusion_matrix_subplot(ax2, cm, model_name)
        
        # 3. ROC Curve (top center-right)
        ax3 = fig.add_subplot(gs[0, 2])
        if 'threshold_analysis' in evaluation_results and evaluation_results['threshold_analysis']:
            # Placeholder for ROC curve - would need actual curve data
            ax3.text(0.5, 0.5, 'ROC Curve\n(Requires curve data)', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('ROC Curve')
        
        # 4. PR Curve (top right)
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.text(0.5, 0.5, 'PR Curve\n(Requires curve data)', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Precision-Recall Curve')
        
        # 5. Threshold Analysis (middle row)
        ax5 = fig.add_subplot(gs[1, :])
        if 'threshold_analysis' in evaluation_results and evaluation_results['threshold_analysis']:
            self._plot_threshold_analysis_subplot(ax5, evaluation_results['threshold_analysis'])
        
        # 6. Class Distribution (bottom left)
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_class_distribution(ax6, evaluation_results)
        
        # 7. Performance Metrics Radar (bottom center-left)
        ax7 = fig.add_subplot(gs[2, 1])
        self._plot_performance_radar(ax7, metrics)
        
        # 8. Calibration Info (bottom center-right)
        ax8 = fig.add_subplot(gs[2, 2])
        if 'calibration_metrics' in evaluation_results and evaluation_results['calibration_metrics']:
            self._plot_calibration_info(ax8, evaluation_results['calibration_metrics'])
        
        # 9. Model Info (bottom right)
        ax9 = fig.add_subplot(gs[2, 3])
        self._plot_model_info(ax9, evaluation_results)
        
        # 10. Feature Importance placeholder (bottom row)
        ax10 = fig.add_subplot(gs[3, :])
        ax10.text(0.5, 0.5, 'Feature Importance\n(Requires feature data)', 
                 ha='center', va='center', transform=ax10.transAxes)
        ax10.set_title('Feature Importance')
        
        plt.suptitle(f'Model Evaluation Dashboard - {model_name}', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        return fig
    
    def _plot_metrics_summary(self, ax, metrics: Dict[str, float], model_name: str):
        """Plot key metrics summary"""
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        values = [metrics.get(metric, 0) for metric in key_metrics]
        labels = [metric.replace('_', ' ').title() for metric in key_metrics]
        
        bars = ax.bar(labels, values, color=[self.colors['primary'], self.colors['success'], 
                                           self.colors['warning'], self.colors['info']])
        ax.set_ylim(0, 1)
        ax.set_title('Key Metrics')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    def _plot_confusion_matrix_subplot(self, ax, cm: np.ndarray, model_name: str):
        """Plot confusion matrix as subplot"""
        labels = ['Normal', 'Defect']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   ax=ax, xticklabels=labels, yticklabels=labels)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    def _plot_threshold_analysis_subplot(self, ax, threshold_analysis: Dict[str, Any]):
        """Plot threshold analysis"""
        thresholds = threshold_analysis['thresholds']
        precision = threshold_analysis['precision']
        recall = threshold_analysis['recall']
        f1_score = threshold_analysis['f1_score']
        
        ax.plot(thresholds, precision, label='Precision', color=self.colors['primary'])
        ax.plot(thresholds, recall, label='Recall', color=self.colors['success'])
        ax.plot(thresholds, f1_score, label='F1 Score', color=self.colors['danger'])
        
        # Mark optimal thresholds
        if 'optimal_thresholds' in threshold_analysis:
            optimal_f1_thresh = threshold_analysis['optimal_thresholds']['f1'][0]
            ax.axvline(x=optimal_f1_thresh, color=self.colors['danger'], 
                      linestyle='--', alpha=0.7, label=f'Optimal F1 ({optimal_f1_thresh:.3f})')
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Metric Value')
        ax.set_title('Threshold Sensitivity Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_class_distribution(self, ax, evaluation_results: Dict[str, Any]):
        """Plot class distribution"""
        pos_ratio = evaluation_results.get('positive_class_ratio', 0.5)
        sizes = [1 - pos_ratio, pos_ratio]
        labels = ['Normal', 'Defect']
        colors = [self.colors['primary'], self.colors['danger']]
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Class Distribution')
    
    def _plot_performance_radar(self, ax, metrics: Dict[str, float]):
        """Plot performance metrics as radar chart"""
        # Simple bar chart since radar is complex in matplotlib
        key_metrics = ['precision', 'recall', 'f1_score', 'accuracy']
        values = [metrics.get(metric, 0) for metric in key_metrics]
        labels = [metric.replace('_', ' ').title() for metric in key_metrics]
        
        bars = ax.barh(labels, values, color=self.colors['info'])
        ax.set_xlim(0, 1)
        ax.set_title('Performance Metrics')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=9)
    
    def _plot_calibration_info(self, ax, calibration_metrics: Dict[str, float]):
        """Plot calibration information"""
        metrics_names = list(calibration_metrics.keys())
        metrics_values = list(calibration_metrics.values())
        
        # Filter out NaN values
        valid_pairs = [(name, value) for name, value in zip(metrics_names, metrics_values) 
                      if not (np.isnan(value) if isinstance(value, (int, float)) else False)]
        
        if valid_pairs:
            names, values = zip(*valid_pairs)
            ax.bar(range(len(names)), values, color=self.colors['warning'])
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels([name.replace('_', ' ').title() for name in names], rotation=45)
            ax.set_title('Calibration Metrics')
        else:
            ax.text(0.5, 0.5, 'No valid calibration metrics', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Calibration Metrics')
    
    def _plot_model_info(self, ax, evaluation_results: Dict[str, Any]):
        """Plot model information"""
        info_text = []
        info_text.append(f"Model: {evaluation_results.get('model_name', 'Unknown')}")
        info_text.append(f"Threshold: {evaluation_results.get('threshold', 0.5):.3f}")
        info_text.append(f"Sample Size: {evaluation_results.get('sample_size', 'Unknown')}")
        
        if 'evaluation_time' in evaluation_results:
            info_text.append(f"Eval Time: {evaluation_results['evaluation_time']:.2f}s")
        
        ax.text(0.1, 0.5, '\n'.join(info_text), transform=ax.transAxes, 
               fontsize=11, verticalalignment='center')
        ax.set_title('Model Information')
        ax.axis('off')
    
    def plot_metric_comparison(self, metrics_dict: Dict[str, Dict[str, float]]) -> plt.Figure:
        """
        Plot comparison of metrics across different models
        
        Args:
            metrics_dict: Dictionary of model names to metrics
            
        Returns:
            Comparison plot figure
        """
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(metrics_dict).T
        
        # Select key metrics for comparison
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        available_metrics = [m for m in key_metrics if m in df.columns]
        
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(4*len(available_metrics), 6))
        if len(available_metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            values = df[metric].values
            models = df.index.tolist()
            
            bars = ax.bar(models, values, color=self.colors['primary'], alpha=0.7)
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylim(0, 1)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Model Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_threshold_analysis(self, threshold_results: Dict[str, Any]) -> plt.Figure:
        """
        Plot threshold sensitivity analysis
        
        Args:
            threshold_results: Results from threshold analysis
            
        Returns:
            Threshold analysis figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        thresholds = threshold_results['thresholds']
        
        # Plot 1: Precision, Recall, F1
        ax1.plot(thresholds, threshold_results['precision'], 
                label='Precision', color=self.colors['primary'], linewidth=2)
        ax1.plot(thresholds, threshold_results['recall'], 
                label='Recall', color=self.colors['success'], linewidth=2)
        ax1.plot(thresholds, threshold_results['f1_score'], 
                label='F1 Score', color=self.colors['danger'], linewidth=2)
        
        # Mark optimal thresholds
        if 'optimal_thresholds' in threshold_results:
            optimal_f1 = threshold_results['optimal_thresholds']['f1']
            ax1.axvline(x=optimal_f1[0], color=self.colors['danger'], 
                       linestyle='--', alpha=0.7, 
                       label=f'Optimal F1 ({optimal_f1[0]:.3f}, {optimal_f1[1]:.3f})')
        
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Metric Value')
        ax1.set_title('Precision, Recall, and F1 Score vs Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Accuracy and Specificity
        ax2.plot(thresholds, threshold_results['accuracy'], 
                label='Accuracy', color=self.colors['info'], linewidth=2)
        ax2.plot(thresholds, threshold_results['specificity'], 
                label='Specificity', color=self.colors['warning'], linewidth=2)
        
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Metric Value')
        ax2.set_title('Accuracy and Specificity vs Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_plots(self, evaluation_results: Dict[str, Any]) -> Dict[str, go.Figure]:
        """
        Create interactive Plotly visualizations
        
        Args:
            evaluation_results: Complete evaluation results
            
        Returns:
            Dictionary of interactive plots
        """
        plots = {}
        
        # Interactive confusion matrix
        if 'confusion_matrix' in evaluation_results:
            plots['confusion_matrix'] = self._create_interactive_confusion_matrix(
                evaluation_results['confusion_matrix'], 
                evaluation_results.get('model_name', 'Model')
            )
        
        # Interactive threshold analysis
        if 'threshold_analysis' in evaluation_results and evaluation_results['threshold_analysis']:
            plots['threshold_analysis'] = self._create_interactive_threshold_plot(
                evaluation_results['threshold_analysis']
            )
        
        # Interactive metrics dashboard
        plots['metrics_dashboard'] = self._create_interactive_metrics_dashboard(
            evaluation_results['metrics'],
            evaluation_results.get('model_name', 'Model')
        )
        
        return plots
    
    def _create_interactive_confusion_matrix(self, cm: List[List], model_name: str) -> go.Figure:
        """Create interactive confusion matrix"""
        cm_array = np.array(cm)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm_array,
            x=['Normal', 'Defect'],
            y=['Normal', 'Defect'],
            colorscale='Blues',
            text=cm_array,
            texttemplate="%{text}",
            textfont={"size": 20},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f'Confusion Matrix - {model_name}',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            width=500,
            height=500
        )
        
        return fig
    
    def _create_interactive_threshold_plot(self, threshold_analysis: Dict[str, Any]) -> go.Figure:
        """Create interactive threshold analysis plot"""
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        
        thresholds = threshold_analysis['thresholds']
        
        fig.add_trace(go.Scatter(
            x=thresholds, y=threshold_analysis['precision'],
            mode='lines', name='Precision', line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=thresholds, y=threshold_analysis['recall'],
            mode='lines', name='Recall', line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=thresholds, y=threshold_analysis['f1_score'],
            mode='lines', name='F1 Score', line=dict(color='red')
        ))
        
        # Add optimal threshold markers
        if 'optimal_thresholds' in threshold_analysis:
            optimal_f1 = threshold_analysis['optimal_thresholds']['f1']
            fig.add_vline(
                x=optimal_f1[0], 
                line_dash="dash", 
                annotation_text=f"Optimal F1: {optimal_f1[0]:.3f}"
            )
        
        fig.update_layout(
            title='Threshold Sensitivity Analysis',
            xaxis_title='Threshold',
            yaxis_title='Metric Value',
            hovermode='x unified'
        )
        
        return fig
    
    def _create_interactive_metrics_dashboard(self, metrics: Dict[str, float], model_name: str) -> go.Figure:
        """Create interactive metrics dashboard"""
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        if 'roc_auc' in metrics:
            key_metrics.append('roc_auc')
        
        metric_values = [metrics.get(metric, 0) for metric in key_metrics]
        metric_labels = [metric.replace('_', ' ').title() for metric in key_metrics]
        
        fig = go.Figure(data=go.Bar(
            x=metric_labels,
            y=metric_values,
            text=[f'{val:.3f}' for val in metric_values],
            textposition='auto',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title=f'Performance Metrics - {model_name}',
            xaxis_title='Metric',
            yaxis_title='Score',
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def save_all_plots(self, plots: Dict[str, plt.Figure], prefix: str = "") -> None:
        """
        Save all plots to output directory
        
        Args:
            plots: Dictionary of plot names to figures
            prefix: Filename prefix
        """
        for plot_name, fig in plots.items():
            filename = f"{prefix}_{plot_name}.png" if prefix else f"{plot_name}.png"
            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)  # Close to free memory
        
        print(f"Saved {len(plots)} plots to {self.output_dir}")