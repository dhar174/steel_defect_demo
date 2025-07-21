#!/usr/bin/env python3
"""
Example demonstrating the Jinja2 template issue and fix.

This file demonstrates the problem mentioned in the issue:
"The template uses 'metric_value is number' test which is not a valid Jinja2 test."
"""

from jinja2 import Environment, Template
import numbers
from typing import Any


def demonstrate_issue():
    """Demonstrate the problematic template and show how to fix it."""
    
    # The actual issue might be that "is number" test exists but behaves differently
    # than expected, or there might be an edge case issue
    test_template_str = """
Testing 'is number' with various values:
{% for item in test_values %}
  Value: {{ item.value }} | Type: {{ item.type }} | Is Number: {{ item.value is number }}
{% endfor %}
"""
    
    # Test various values to understand the behavior
    test_values = [
        {'value': 42, 'type': 'int'},
        {'value': 42.5, 'type': 'float'},
        {'value': '42', 'type': 'str'},
        {'value': 'not_a_number', 'type': 'str'},
        {'value': None, 'type': 'None'},
        {'value': True, 'type': 'bool'},
        {'value': [], 'type': 'list'},
    ]
    
    print("Demonstrating Jinja2 'is number' test behavior:")
    print("=" * 50)
    
    # Test the 'is number' behavior
    try:
        env = Environment()
        test_template = env.from_string(test_template_str)
        result = test_template.render(test_values=test_values)
        print("✅ 'is number' test results:")
        print(result)
    except Exception as e:
        print(f"❌ Error with 'is number' test: {e}")


def validate_metric_value(value: Any) -> bool:
    """
    Validate if a value is a number using Python's isinstance().
    
    Args:
        value: The value to check
        
    Returns:
        bool: True if the value is a number, False otherwise
    """
    return isinstance(value, numbers.Number)


def prepare_template_data(metric_value: Any) -> dict:
    """
    Prepare data for template rendering with proper validation.
    
    Args:
        metric_value: The metric value to validate
        
    Returns:
        dict: Data ready for template rendering
    """
    return {
        'metric_value': metric_value,
        'is_metric_valid': validate_metric_value(metric_value)
    }


if __name__ == "__main__":
    demonstrate_issue()