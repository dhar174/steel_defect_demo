#!/usr/bin/env python3
"""
Integration example showing how to use template validation in the steel defect monitoring system.

This demonstrates how the template validation utilities would be used in practice
for generating reports and dashboards with proper metric validation.
"""

import sys
import os
import random
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.template_validation import (
    TemplateDataValidator, 
    SafeTemplateRenderer
)


class SteelDefectReportGenerator:
    """
    Example report generator using template validation for steel defect monitoring.
    
    This shows how the template validation would integrate with the existing system.
    """
    
    def __init__(self):
        """Initialize the report generator."""
        self.renderer = SafeTemplateRenderer()
        
    def generate_sensor_report(self, sensor_data: dict, cast_id: str) -> str:
        """
        Generate a sensor report with proper validation.
        
        Args:
            sensor_data: Dictionary of sensor readings
            cast_id: Cast identifier
            
        Returns:
            str: HTML report
        """
        # Template for sensor report
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Steel Cast Sensor Report - {{ cast_id }}</title>
            <style>
                .sensor-valid { color: green; }
                .sensor-invalid { color: red; }
                .metric-card { border: 1px solid #ccc; margin: 10px; padding: 10px; }
            </style>
        </head>
        <body>
            <h1>Steel Cast Sensor Report</h1>
            <h2>Cast ID: {{ cast_id }}</h2>
            <p>Generated: {{ timestamp }}</p>
            
            <div class="sensor-grid">
            {% for sensor in sensor_data %}
                <div class="metric-card">
                    <h3>{{ sensor.name }}</h3>
                    {% if sensor.is_valid %}
                        <div class="sensor-valid">
                            <strong>Value:</strong> {{ "%.2f"|format(sensor.value) }} {{ sensor.unit }}
                        </div>
                    {% else %}
                        <div class="sensor-invalid">
                            <strong>Error:</strong> {{ sensor.error }}
                        </div>
                    {% endif %}
                </div>
            {% endfor %}
            </div>
            
            <div class="summary">
                <h3>Validation Summary</h3>
                <p>Valid sensors: {{ valid_count }} / {{ total_count }}</p>
                {% if valid_count == total_count %}
                    <p style="color: green;">✅ All sensors reporting valid data</p>
                {% else %}
                    <p style="color: orange;">⚠️ Some sensors have validation issues</p>
                {% endif %}
            </div>
        </body>
        </html>
        """
        
        # Validate all sensor data and prepare for template
        sensor_list = []
        sensor_units = {
            'mold_temperature': '°C',
            'casting_speed': 'm/min',
            'mold_level': 'mm',
            'cooling_water_flow': 'L/min',
            'superheat': '°C'
        }
        
        for sensor_name, value in sensor_data.items():
            is_valid = TemplateDataValidator.is_valid_metric_value(value)
            sensor_list.append({
                'name': sensor_name.replace('_', ' ').title(),
                'value': value,
                'is_valid': is_valid,
                'error': None if is_valid else "Invalid or non-numeric value",
                'unit': sensor_units.get(sensor_name, '')
            })
        
        valid_count = sum(1 for sensor in sensor_list if sensor['is_valid'])
        
        template_data = {
            'cast_id': cast_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sensor_data': sensor_list,
            'valid_count': valid_count,
            'total_count': len(sensor_list)
        }
        
        return self.renderer.render_template_string(template, **template_data)
    
    def generate_defect_probability_alert(self, prediction_data: dict) -> str:
        """
        Generate defect probability alert with validation.
        
        Args:
            prediction_data: Dictionary containing prediction information
            
        Returns:
            str: HTML alert
        """
        template = """
        <div class="alert {{ alert_class }}">
            <h2>🚨 Defect Probability Alert</h2>
            {% if is_probability_valid %}
                <p><strong>Defect Probability:</strong> {{ "%.1f"|format(probability_value * 100) }}%</p>
                
                {% if probability_value >= 0.8 %}
                    <p style="color: red;">🔥 <strong>HIGH RISK</strong> - Immediate action required!</p>
                {% elif probability_value >= 0.5 %}
                    <p style="color: orange;">⚠️ <strong>MEDIUM RISK</strong> - Monitor closely</p>
                {% else %}
                    <p style="color: green;">✅ <strong>LOW RISK</strong> - Normal operation</p>
                {% endif %}
                
                {% if is_confidence_valid %}
                    <p><small>Confidence: {{ "%.1f"|format(confidence_value * 100) }}%</small></p>
                {% endif %}
            {% else %}
                <p style="color: red;">❌ <strong>INVALID PREDICTION DATA</strong></p>
                <p>Error: {{ probability_error }}</p>
            {% endif %}
            
            <p><small>Generated: {{ timestamp }}</small></p>
        </div>
        """
        
        # Validate prediction data
        validated_data = TemplateDataValidator.prepare_metric_data(
            prediction_data.get('probability'), 'probability'
        )
        
        confidence_data = TemplateDataValidator.prepare_metric_data(
            prediction_data.get('confidence', 0.0), 'confidence'
        )
        
        # Determine alert class
        prob_value = prediction_data.get('probability', 0)
        if TemplateDataValidator.is_valid_metric_value(prob_value):
            if prob_value >= 0.8:
                alert_class = 'alert-danger'
            elif prob_value >= 0.5:
                alert_class = 'alert-warning'
            else:
                alert_class = 'alert-info'
        else:
            alert_class = 'alert-error'
        
        template_data = {
            **validated_data,
            **confidence_data,
            'alert_class': alert_class,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return self.renderer.render_template_string(template, **template_data)


def demonstrate_integration():
    """Demonstrate the integration with steel defect monitoring."""
    print("=" * 70)
    print("STEEL DEFECT MONITORING - TEMPLATE VALIDATION INTEGRATION")
    print("=" * 70)
    
    generator = SteelDefectReportGenerator()
    
    # Example 1: Generate sensor report with mixed valid/invalid data
    print("\n📊 Example 1: Sensor Report Generation")
    print("-" * 40)
    
    sensor_data = {
        'mold_temperature': 1520.5,     # Valid
        'casting_speed': 1.2,           # Valid
        'mold_level': "invalid_reading", # Invalid - string
        'cooling_water_flow': float('inf'), # Invalid - infinity
        'superheat': 25.3               # Valid
    }
    
    report_html = generator.generate_sensor_report(sensor_data, "CAST_001")
    
    # Save report
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)
    
    report_file = output_dir / "sensor_report_validation_demo.html"
    with open(report_file, 'w') as f:
        f.write(report_html)
    
    print(f"✅ Sensor report generated: {report_file}")
    print("   Contains validation for all sensor types including invalid data")
    
    # Example 2: Generate defect probability alerts
    print("\n🚨 Example 2: Defect Probability Alerts")
    print("-" * 40)
    
    test_cases = [
        {'probability': 0.95, 'confidence': 0.88, 'description': 'High risk with good confidence'},
        {'probability': 0.65, 'confidence': 0.72, 'description': 'Medium risk'},
        {'probability': 0.15, 'confidence': 0.91, 'description': 'Low risk'},
        {'probability': 'invalid', 'confidence': 0.5, 'description': 'Invalid probability'},
        {'probability': float('nan'), 'confidence': 0.8, 'description': 'NaN probability'},
    ]
    
    for i, case in enumerate(test_cases):
        alert_html = generator.generate_defect_probability_alert(case)
        alert_file = output_dir / f"alert_{i+1}_validation_demo.html"
        
        with open(alert_file, 'w') as f:
            f.write(f"<html><body>{alert_html}</body></html>")
        
        print(f"✅ Alert {i+1} generated: {alert_file}")
        print(f"   {case['description']}")
    
    # Example 3: Show validation statistics
    print("\n📈 Example 3: Validation Statistics")
    print("-" * 40)
    
    all_test_values = [
        1520.5,           # Valid float
        42,               # Valid int
        "1234.5",         # Invalid string
        None,             # Invalid None
        True,             # Invalid boolean (edge case)
        float('inf'),     # Invalid infinity
        float('nan'),     # Invalid NaN
        0,                # Valid zero
        -273.15,          # Valid negative
    ]
    
    valid_count = sum(1 for val in all_test_values 
                     if TemplateDataValidator.is_valid_metric_value(val))
    
    print(f"Total test values: {len(all_test_values)}")
    print(f"Valid metric values: {valid_count}")
    print(f"Invalid metric values: {len(all_test_values) - valid_count}")
    print(f"Validation accuracy: {valid_count/len(all_test_values)*100:.1f}%")
    
    print(f"\n🎉 All examples completed! Check {output_dir}/ for generated files.")


if __name__ == "__main__":
    demonstrate_integration()