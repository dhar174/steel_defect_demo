"""
Template validation utilities for steel defect monitoring system.

This module provides utilities to properly validate data before passing to Jinja2 templates,
addressing the issue where 'metric_value is number' tests might be used incorrectly.
"""

import numbers
from typing import Any, Dict, Union, Optional
from jinja2 import Environment, select_autoescape


class TemplateDataValidator:
    """Validates data before passing to Jinja2 templates."""
    
    @staticmethod
    def is_numeric(value: Any) -> bool:
        """
        Check if a value is numeric using Python's isinstance check.
        
        This is more reliable than relying on Jinja2's 'is number' test
        in all contexts, especially when dealing with edge cases.
        
        Args:
            value: The value to check
            
        Returns:
            bool: True if the value is numeric, False otherwise
        """
        return isinstance(value, numbers.Number) and not isinstance(value, bool)
    
    @staticmethod
    def is_valid_metric_value(value: Any) -> bool:
        """
        Validate if a value is a valid metric value.
        
        A valid metric value should be:
        - Numeric (int or float)
        - Not None
        - Not boolean (booleans are technically numbers in Python)
        - Finite (not inf or nan)
        
        Args:
            value: The value to validate
            
        Returns:
            bool: True if valid metric value, False otherwise
        """
        if not TemplateDataValidator.is_numeric(value):
            return False
        
        try:
            # Check for inf or nan values
            import math
            return math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return False
    
    @staticmethod
    def prepare_metric_data(metric_value: Any, metric_name: str = "metric") -> Dict[str, Any]:
        """
        Prepare metric data for template rendering with proper validation.
        
        Args:
            metric_value: The metric value to validate and prepare
            metric_name: Name of the metric (for error messages)
            
        Returns:
            dict: Data ready for template rendering with validation flags
        """
        is_valid = TemplateDataValidator.is_valid_metric_value(metric_value)
        
        return {
            f"{metric_name}_value": metric_value,
            f"is_{metric_name}_valid": is_valid,
            f"{metric_name}_error": None if is_valid else "Invalid or non-numeric value"
        }
    
    @staticmethod
    def prepare_sensor_data(sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare sensor data for template rendering with validation.
        
        Args:
            sensor_data: Dictionary of sensor name -> value pairs
            
        Returns:
            dict: Validated sensor data ready for template rendering
        """
        prepared_data = {}
        
        for sensor_name, value in sensor_data.items():
            validation_result = TemplateDataValidator.prepare_metric_data(value, sensor_name)
            prepared_data.update(validation_result)
        
        return prepared_data


class SafeTemplateRenderer:
    """Safe template renderer with built-in data validation."""
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize the safe template renderer.
        
        Args:
            template_dir: Directory containing templates, if any
        """
        self.env = Environment(
            autoescape=select_autoescape(['html', 'xml']),
            loader=None if template_dir is None else None  # Add FileSystemLoader if needed
        )
        
        # Add custom tests to Jinja2 environment
        self.env.tests['valid_metric'] = TemplateDataValidator.is_valid_metric_value
        self.env.tests['numeric_safe'] = TemplateDataValidator.is_numeric
    
    def render_template_string(self, template_string: str, **kwargs) -> str:
        """
        Render a template string with validated data.
        
        Args:
            template_string: The Jinja2 template string
            **kwargs: Data to pass to the template
            
        Returns:
            str: Rendered template
        """
        template = self.env.from_string(template_string)
        return template.render(**kwargs)
    
    def render_metric_template(self, template_string: str, metric_value: Any, 
                             metric_name: str = "metric") -> str:
        """
        Render a template with a single metric value, properly validated.
        
        Args:
            template_string: The Jinja2 template string
            metric_value: The metric value to render
            metric_name: Name of the metric
            
        Returns:
            str: Rendered template
        """
        validated_data = TemplateDataValidator.prepare_metric_data(metric_value, metric_name)
        
        # For backward compatibility, also add non-prefixed versions
        # if the template expects 'metric_value', 'is_metric_valid', etc.
        if metric_name == "metric":
            # No need to duplicate
            template_data = validated_data
        else:
            # Add both prefixed and non-prefixed versions
            template_data = validated_data.copy()
            template_data.update({
                'metric_value': metric_value,
                'is_metric_valid': TemplateDataValidator.is_valid_metric_value(metric_value),
                'metric_error': None if TemplateDataValidator.is_valid_metric_value(metric_value) 
                              else "Invalid or non-numeric value"
            })
        
        return self.render_template_string(template_string, **template_data)


# Example templates demonstrating proper usage
SAFE_METRIC_TEMPLATE = """
{%- if is_metric_valid %}
<div class="metric-value valid">
    <span class="value">{{ metric_value }}</span>
    <span class="unit">{{ metric_unit|default('') }}</span>
</div>
{%- else %}
<div class="metric-value invalid">
    <span class="error">{{ metric_error }}</span>
</div>
{%- endif %}
"""

SENSOR_DASHBOARD_TEMPLATE = """
<div class="sensor-dashboard">
{%- for sensor in sensors %}
    <div class="sensor-panel">
        <h3>{{ sensor.name.replace('_', ' ').title() }}</h3>
        {%- if sensor.is_valid %}
        <div class="sensor-value valid">{{ sensor.value }}</div>
        {%- else %}
        <div class="sensor-value invalid">{{ sensor.error }}</div>
        {%- endif %}
    </div>
{%- endfor %}
</div>
"""