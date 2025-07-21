#!/usr/bin/env python3
"""
Test script to reproduce the format filter issue.

This demonstrates the case where invalid sensors might reach the template 
with value=0, masking the actual invalid value.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.template_validation import SafeTemplateRenderer
from jinja2 import TemplateError


def test_format_filter_issue():
    """Test the format filter issue scenario."""
    print("Testing format filter issue scenarios...")
    
    renderer = SafeTemplateRenderer()
    
    # Template with problematic format filter (similar to the original issue)
    problematic_template = """
    {% for sensor in sensors %}
        <div>{{ sensor.name }}: {{ "%.2f"|format(sensor.value) }}</div>
    {% endfor %}
    """
    
    # Test case 1: Non-numeric value that would cause format filter to fail
    print("\n1. Testing with non-numeric sensor value...")
    test_data_1 = {
        'sensors': [
            {'name': 'temperature', 'value': 'invalid_reading'},
            {'name': 'pressure', 'value': 150.5}
        ]
    }
    
    try:
        result = renderer.render_template_string(problematic_template, **test_data_1)
        print("✅ Template rendered successfully (unexpected)")
        print(result)
    except Exception as e:
        print(f"❌ Template failed as expected: {type(e).__name__}: {e}")
    
    # Test case 2: Masked invalid sensor with value=0
    print("\n2. Testing with masked invalid sensor (value=0)...")
    test_data_2 = {
        'sensors': [
            {'name': 'temperature', 'value': 0},  # This could mask an invalid reading
            {'name': 'pressure', 'value': 150.5}
        ]
    }
    
    try:
        result = renderer.render_template_string(problematic_template, **test_data_2)
        print("✅ Template rendered successfully")
        print("📝 Result:")
        print(result.strip())
        print("⚠️  Note: value=0 might be masking an actual invalid sensor reading!")
    except Exception as e:
        print(f"❌ Template failed: {type(e).__name__}: {e}")
    
    # Test case 3: Safe template with conditional formatting
    print("\n3. Testing safe template with conditional formatting...")
    safe_template = """
    {% for sensor in sensors %}
        <div>{{ sensor.name }}: 
        {%- if sensor.value is number %}
            {{ "%.2f"|format(sensor.value) }}
        {%- else %}
            {{ sensor.value }} (invalid)
        {%- endif %}
        </div>
    {% endfor %}
    """
    
    # Test with mixed valid/invalid data
    test_data_3 = {
        'sensors': [
            {'name': 'temperature', 'value': 'invalid_reading'},
            {'name': 'pressure', 'value': 150.5},
            {'name': 'flow', 'value': 0},
            {'name': 'vibration', 'value': None}
        ]
    }
    
    try:
        result = renderer.render_template_string(safe_template, **test_data_3)
        print("✅ Safe template rendered successfully")
        print("📝 Result:")
        print(result.strip())
    except Exception as e:
        print(f"❌ Safe template failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    test_format_filter_issue()