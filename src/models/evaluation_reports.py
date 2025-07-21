"""
Report generation utilities for model evaluation
"""

import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from jinja2 import Template


class EvaluationReports:
    """Comprehensive report generation for model evaluation"""
    
    def __init__(self, output_dir: str = "results/reports"):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_comprehensive_report(self, 
                                    evaluation_results: Dict[str, Any],
                                    model_info: Optional[Dict[str, Any]] = None,
                                    include_plots: bool = True) -> Dict[str, str]:
        """
        Generate comprehensive evaluation report in multiple formats
        
        Args:
            evaluation_results: Complete evaluation results
            model_info: Additional model information
            include_plots: Include plot references in report
            
        Returns:
            Dictionary of report paths
        """
        report_paths = {}
        
        # Generate different format reports
        report_paths['html'] = self.generate_html_report(evaluation_results, model_info, include_plots)
        report_paths['markdown'] = self.generate_markdown_report(evaluation_results, model_info)
        report_paths['json'] = self.generate_json_report(evaluation_results, model_info)
        report_paths['excel'] = self.generate_excel_report(evaluation_results, model_info)
        
        return report_paths
    
    def generate_html_report(self, 
                           evaluation_results: Dict[str, Any],
                           model_info: Optional[Dict[str, Any]] = None,
                           include_plots: bool = True) -> str:
        """
        Generate HTML evaluation report
        
        Args:
            evaluation_results: Complete evaluation results
            model_info: Additional model information
            include_plots: Include plot references
            
        Returns:
            Path to generated HTML report
        """
        model_name = evaluation_results.get('model_name', 'Model')
        timestamp = evaluation_results.get('evaluation_timestamp', datetime.now().isoformat())
        
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Evaluation Report - {{ model_name }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { background-color: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 30px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background-color: #f9f9f9; padding: 15px; border-radius: 5px; text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; color: #333; }
        .metric-label { font-size: 0.9em; color: #666; text-transform: uppercase; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; }
        .section { margin: 30px 0; }
        .confusion-matrix { display: inline-block; margin: 20px; }
        .plot-placeholder { background-color: #e9e9e9; padding: 40px; text-align: center; border-radius: 5px; margin: 20px 0; }
        .alert { padding: 15px; margin: 20px 0; border-radius: 5px; }
        .alert-info { background-color: #d1ecf1; border-left: 5px solid #17a2b8; }
        .alert-warning { background-color: #fff3cd; border-left: 5px solid #ffc107; }
        .alert-success { background-color: #d4edda; border-left: 5px solid #28a745; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Model Evaluation Report</h1>
        <h2>{{ model_name }}</h2>
        <p><strong>Generated:</strong> {{ timestamp }}</p>
        <p><strong>Evaluation Time:</strong> {{ evaluation_time }}s</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <div class="alert alert-info">
            <p><strong>Model Performance Overview:</strong></p>
            <ul>
                <li>Sample Size: {{ sample_size }} samples</li>
                <li>Positive Class Ratio: {{ positive_ratio }}%</li>
                <li>Classification Threshold: {{ threshold }}</li>
            </ul>
        </div>
    </div>

    <div class="section">
        <h2>Key Performance Metrics</h2>
        <div class="metric-grid">
            {% for metric_name, metric_value in key_metrics.items() %}
            <div class="metric-card">
                <div class="metric-value">{{ "%.3f"|format(metric_value) if metric_value is number else metric_value }}</div>
                <div class="metric-label">{{ metric_name.replace('_', ' ').title() }}</div>
            </div>
            {% endfor %}
        </div>
    </div>

    <div class="section">
        <h2>Detailed Metrics</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Interpretation</th>
                </tr>
            </thead>
            <tbody>
                {% for metric_name, metric_value in all_metrics.items() %}
                <tr>
                    <td>{{ metric_name.replace('_', ' ').title() }}</td>
                    <td>{{ "%.4f"|format(metric_value) if metric_value is number else metric_value }}</td>
                    <td>{{ get_metric_interpretation(metric_name, metric_value) }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    {% if confusion_matrix_analysis %}
    <div class="section">
        <h2>Confusion Matrix Analysis</h2>
        <div class="confusion-matrix">
            <table>
                <thead>
                    <tr>
                        <th></th>
                        <th>Predicted Normal</th>
                        <th>Predicted Defect</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <th>Actual Normal</th>
                        <td>{{ confusion_matrix_analysis.true_negatives }}</td>
                        <td>{{ confusion_matrix_analysis.false_positives }}</td>
                    </tr>
                    <tr>
                        <th>Actual Defect</th>
                        <td>{{ confusion_matrix_analysis.false_negatives }}</td>
                        <td>{{ confusion_matrix_analysis.true_positives }}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <div class="alert alert-warning">
            <p><strong>Impact Analysis:</strong></p>
            <ul>
                <li>False Positives (unnecessary inspections): {{ confusion_matrix_analysis.false_positives }}</li>
                <li>False Negatives (missed defects): {{ confusion_matrix_analysis.false_negatives }}</li>
                <li>Detection Rate: {{ "%.1f"|format(detection_rate) if detection_rate is number else detection_rate }}%</li>
            </ul>
        </div>
    </div>
    {% endif %}

    {% if threshold_analysis %}
    <div class="section">
        <h2>Threshold Analysis</h2>
        <p>Optimal thresholds for different objectives:</p>
        <ul>
            {% for objective, threshold_data in threshold_analysis.optimal_thresholds.items() %}
            <li><strong>{{ objective.title() }}:</strong> {{ "%.3f"|format(threshold_data[0]) if threshold_data[0] is number else threshold_data[0] }} (Score: {{ "%.3f"|format(threshold_data[1]) if threshold_data[1] is number else threshold_data[1] }})</li>
            {% endfor %}
        </ul>
        {% if include_plots %}
        <div class="plot-placeholder">
            Threshold Sensitivity Plot<br>
            <small>(Plot file: {{ model_name }}_threshold_analysis.png)</small>
        </div>
        {% endif %}
    </div>
    {% endif %}

    {% if calibration_metrics %}
    <div class="section">
        <h2>Calibration Analysis</h2>
        <table>
            <thead>
                <tr>
                    <th>Calibration Metric</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                {% for metric_name, metric_value in calibration_metrics.items() %}
                <tr>
                    <td>{{ metric_name.replace('_', ' ').title() }}</td>
                    <td>{{ "%.4f"|format(metric_value) if not (metric_value != metric_value) else 'N/A' }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    {% endif %}

    {% if model_info %}
    <div class="section">
        <h2>Model Information</h2>
        <table>
            <thead>
                <tr>
                    <th>Property</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                {% for key, value in model_info.items() %}
                <tr>
                    <td>{{ key.replace('_', ' ').title() }}</td>
                    <td>{{ value }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    {% endif %}

    <div class="section">
        <h2>Recommendations</h2>
        <div class="alert alert-success">
            {{ recommendations }}
        </div>
    </div>

    <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #666;">
        <p>Generated by Steel Defect Detection Model Evaluation Framework</p>
        <p>Report generated on {{ timestamp }}</p>
    </footer>
</body>
</html>
        """
        
        # Prepare template data
        template_data = self._prepare_template_data(evaluation_results, model_info, include_plots)
        
        # Render template
        template = Template(html_template)
        html_content = template.render(**template_data)
        
        # Save report
        report_path = self.output_dir / f"{template_data['model_name']}_evaluation_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def generate_markdown_report(self, 
                               evaluation_results: Dict[str, Any],
                               model_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate Markdown evaluation report
        
        Args:
            evaluation_results: Complete evaluation results
            model_info: Additional model information
            
        Returns:
            Path to generated Markdown report
        """
        model_name = evaluation_results.get('model_name', 'Model')
        timestamp = evaluation_results.get('evaluation_timestamp', datetime.now().isoformat())
        
        md_content = []
        
        # Header
        md_content.append(f"# Model Evaluation Report: {model_name}")
        md_content.append(f"**Generated:** {timestamp}")
        md_content.append(f"**Evaluation Time:** {evaluation_results.get('evaluation_time', 'N/A')}s")
        md_content.append("")
        
        # Executive Summary
        md_content.append("## Executive Summary")
        md_content.append(f"- **Model:** {model_name}")
        md_content.append(f"- **Threshold:** {evaluation_results.get('threshold', 'N/A')}")
        md_content.append(f"- **Sample Size:** {evaluation_results.get('sample_size', 'N/A')}")
        md_content.append(f"- **Positive Class Ratio:** {evaluation_results.get('positive_class_ratio', 0)*100:.1f}%")
        md_content.append("")
        
        # Key Metrics
        metrics = evaluation_results.get('metrics', {})
        md_content.append("## Key Performance Metrics")
        
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'average_precision']
        for metric in key_metrics:
            if metric in metrics:
                md_content.append(f"- **{metric.replace('_', ' ').title()}:** {metrics[metric]:.3f}")
        md_content.append("")
        
        # Detailed Metrics Table
        md_content.append("## Detailed Metrics")
        md_content.append("| Metric | Value |")
        md_content.append("|--------|-------|")
        
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                md_content.append(f"| {metric_name.replace('_', ' ').title()} | {metric_value:.4f} |")
        md_content.append("")
        
        # Confusion Matrix Analysis
        if 'confusion_matrix_analysis' in evaluation_results:
            cm_analysis = evaluation_results['confusion_matrix_analysis']
            md_content.append("## Confusion Matrix Analysis")
            md_content.append(f"- **True Positives:** {cm_analysis.get('true_positives', 'N/A')}")
            md_content.append(f"- **True Negatives:** {cm_analysis.get('true_negatives', 'N/A')}")
            md_content.append(f"- **False Positives:** {cm_analysis.get('false_positives', 'N/A')}")
            md_content.append(f"- **False Negatives:** {cm_analysis.get('false_negatives', 'N/A')}")
            md_content.append("")
        
        # Threshold Analysis
        if 'threshold_analysis' in evaluation_results and evaluation_results['threshold_analysis']:
            threshold_analysis = evaluation_results['threshold_analysis']
            md_content.append("## Threshold Analysis")
            md_content.append("Optimal thresholds for different objectives:")
            
            if 'optimal_thresholds' in threshold_analysis:
                for objective, threshold_data in threshold_analysis['optimal_thresholds'].items():
                    md_content.append(f"- **{objective.title()}:** {threshold_data[0]:.3f} (Score: {threshold_data[1]:.3f})")
            md_content.append("")
        
        # Model Information
        if model_info:
            md_content.append("## Model Information")
            for key, value in model_info.items():
                md_content.append(f"- **{key.replace('_', ' ').title()}:** {value}")
            md_content.append("")
        
        # Recommendations
        recommendations = self._generate_recommendations(evaluation_results)
        md_content.append("## Recommendations")
        for rec in recommendations:
            md_content.append(f"- {rec}")
        md_content.append("")
        
        # Save report
        report_content = '\n'.join(md_content)
        report_path = self.output_dir / f"{model_name}_evaluation_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return str(report_path)
    
    def generate_json_report(self, 
                           evaluation_results: Dict[str, Any],
                           model_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate JSON evaluation report
        
        Args:
            evaluation_results: Complete evaluation results
            model_info: Additional model information
            
        Returns:
            Path to generated JSON report
        """
        model_name = evaluation_results.get('model_name', 'Model')
        
        # Combine all data
        report_data = {
            'evaluation_results': evaluation_results,
            'model_info': model_info or {},
            'recommendations': self._generate_recommendations(evaluation_results),
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'framework_version': '1.0.0',
                'report_type': 'comprehensive_evaluation'
            }
        }
        
        # Convert numpy types for JSON serialization
        report_data = self._convert_numpy_types(report_data)
        
        # Save report
        report_path = self.output_dir / f"{model_name}_evaluation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return str(report_path)
    
    def generate_excel_report(self, 
                            evaluation_results: Dict[str, Any],
                            model_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate Excel evaluation report with multiple sheets
        
        Args:
            evaluation_results: Complete evaluation results
            model_info: Additional model information
            
        Returns:
            Path to generated Excel report
        """
        model_name = evaluation_results.get('model_name', 'Model')
        report_path = self.output_dir / f"{model_name}_evaluation_report.xlsx"
        
        with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
            # Sheet 1: Summary
            summary_data = {
                'Model Name': [model_name],
                'Evaluation Date': [evaluation_results.get('evaluation_timestamp', 'N/A')],
                'Sample Size': [evaluation_results.get('sample_size', 'N/A')],
                'Threshold': [evaluation_results.get('threshold', 'N/A')],
                'Evaluation Time (s)': [evaluation_results.get('evaluation_time', 'N/A')]
            }
            
            # Add key metrics
            metrics = evaluation_results.get('metrics', {})
            key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'average_precision']
            for metric in key_metrics:
                if metric in metrics:
                    summary_data[metric.replace('_', ' ').title()] = [metrics[metric]]
            
            summary_df = pd.DataFrame(summary_data).T
            summary_df.columns = ['Value']
            summary_df.to_excel(writer, sheet_name='Summary')
            
            # Sheet 2: All Metrics
            metrics_data = []
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    metrics_data.append({
                        'Metric': metric_name.replace('_', ' ').title(),
                        'Value': metric_value
                    })
            
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            
            # Sheet 3: Confusion Matrix Analysis
            if 'confusion_matrix_analysis' in evaluation_results:
                cm_analysis = evaluation_results['confusion_matrix_analysis']
                cm_data = [
                    {'Component': 'True Positives', 'Count': cm_analysis.get('true_positives', 0)},
                    {'Component': 'True Negatives', 'Count': cm_analysis.get('true_negatives', 0)},
                    {'Component': 'False Positives', 'Count': cm_analysis.get('false_positives', 0)},
                    {'Component': 'False Negatives', 'Count': cm_analysis.get('false_negatives', 0)}
                ]
                cm_df = pd.DataFrame(cm_data)
                cm_df.to_excel(writer, sheet_name='Confusion Matrix', index=False)
            
            # Sheet 4: Threshold Analysis
            if 'threshold_analysis' in evaluation_results and evaluation_results['threshold_analysis']:
                threshold_data = evaluation_results['threshold_analysis']
                threshold_df = pd.DataFrame({
                    'Threshold': threshold_data.get('thresholds', []),
                    'Precision': threshold_data.get('precision', []),
                    'Recall': threshold_data.get('recall', []),
                    'F1_Score': threshold_data.get('f1_score', []),
                    'Accuracy': threshold_data.get('accuracy', []),
                    'Specificity': threshold_data.get('specificity', [])
                })
                threshold_df.to_excel(writer, sheet_name='Threshold Analysis', index=False)
            
            # Sheet 5: Model Info
            if model_info:
                info_data = [{'Property': k.replace('_', ' ').title(), 'Value': v} 
                           for k, v in model_info.items()]
                info_df = pd.DataFrame(info_data)
                info_df.to_excel(writer, sheet_name='Model Info', index=False)
        
        return str(report_path)
    
    def _prepare_template_data(self, evaluation_results: Dict[str, Any], 
                             model_info: Optional[Dict[str, Any]] = None,
                             include_plots: bool = True) -> Dict[str, Any]:
        """Prepare data for template rendering"""
        metrics = evaluation_results.get('metrics', {})
        
        # Key metrics for highlighting
        key_metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'average_precision']:
            if metric in metrics:
                key_metrics[metric] = metrics[metric]
        
        # Calculate detection rate
        detection_rate = 0
        if 'confusion_matrix_analysis' in evaluation_results:
            cm_analysis = evaluation_results['confusion_matrix_analysis']
            tp = cm_analysis.get('true_positives', 0)
            fn = cm_analysis.get('false_negatives', 0)
            detection_rate = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
        
        return {
            'model_name': evaluation_results.get('model_name', 'Model'),
            'timestamp': evaluation_results.get('evaluation_timestamp', datetime.now().isoformat()),
            'evaluation_time': evaluation_results.get('evaluation_time', 0),
            'sample_size': evaluation_results.get('sample_size', 0),
            'positive_ratio': evaluation_results.get('positive_class_ratio', 0) * 100,
            'threshold': evaluation_results.get('threshold', 0.5),
            'key_metrics': key_metrics,
            'all_metrics': metrics,
            'confusion_matrix_analysis': evaluation_results.get('confusion_matrix_analysis'),
            'threshold_analysis': evaluation_results.get('threshold_analysis'),
            'calibration_metrics': evaluation_results.get('calibration_metrics'),
            'model_info': model_info,
            'include_plots': include_plots,
            'detection_rate': detection_rate,
            'recommendations': '<br>'.join(self._generate_recommendations(evaluation_results)),
            'get_metric_interpretation': self._get_metric_interpretation
        }
    
    def _generate_recommendations(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        metrics = evaluation_results.get('metrics', {})
        
        # Check F1 score
        f1_score = metrics.get('f1_score', 0)
        if f1_score < 0.7:
            recommendations.append("Consider improving model performance - F1 score is below 0.7")
        elif f1_score > 0.9:
            recommendations.append("Excellent F1 score - model shows strong performance")
        
        # Check precision vs recall balance
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        
        if precision > 0 and recall > 0:
            if precision > recall + 0.1:
                recommendations.append("Model has high precision but lower recall - consider adjusting threshold to catch more defects")
            elif recall > precision + 0.1:
                recommendations.append("Model has high recall but lower precision - consider adjusting threshold to reduce false alarms")
        
        # Check calibration
        if 'calibration_metrics' in evaluation_results:
            brier_score = evaluation_results['calibration_metrics'].get('brier_score', None)
            if brier_score is not None and brier_score > 0.25:
                recommendations.append("Model predictions may be poorly calibrated - consider calibration techniques")
        
        # Check confusion matrix for class imbalance issues
        if 'confusion_matrix_analysis' in evaluation_results:
            cm_analysis = evaluation_results['confusion_matrix_analysis']
            fn = cm_analysis.get('false_negatives', 0)
            tp = cm_analysis.get('true_positives', 0)
            
            if fn > tp:
                recommendations.append("High number of missed defects - consider techniques to improve sensitivity")
        
        if not recommendations:
            recommendations.append("Model performance appears satisfactory across key metrics")
        
        return recommendations
    
    def _get_metric_interpretation(self, metric_name: str, metric_value: Any) -> str:
        """Get interpretation for a metric"""
        if not isinstance(metric_value, (int, float)):
            return "N/A"
        
        interpretations = {
            'accuracy': 'Overall correctness of predictions',
            'precision': 'Proportion of positive predictions that were correct',
            'recall': 'Proportion of actual positives that were identified',
            'f1_score': 'Harmonic mean of precision and recall',
            'roc_auc': 'Area under ROC curve - ability to distinguish classes',
            'average_precision': 'Area under precision-recall curve',
            'specificity': 'Proportion of actual negatives correctly identified',
            'matthews_corrcoef': 'Correlation between observed and predicted classifications'
        }
        
        base_interpretation = interpretations.get(metric_name.lower(), 'Performance metric')
        
        # Add performance assessment
        if metric_value >= 0.9:
            return f"{base_interpretation} (Excellent)"
        elif metric_value >= 0.8:
            return f"{base_interpretation} (Good)"
        elif metric_value >= 0.7:
            return f"{base_interpretation} (Fair)"
        else:
            return f"{base_interpretation} (Needs Improvement)"
    
    def _convert_numpy_types(self, obj: Any) -> Any:
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        return obj