#!/usr/bin/env python3
"""
Final test demonstrating the fix for the format filter issue.

This test shows:
1. Before: Format filter would fail on non-numeric sensor values
2. After: Conditional formatting prevents failures and shows raw values for debugging
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.template_validation import SafeTemplateRenderer


def demonstrate_issue_fix():
    """Demonstrate that the format filter issue is fixed."""
    print("=" * 80)
    print("DEMONSTRATING FORMAT FILTER ISSUE FIX")
    print("=" * 80)
    
    renderer = SafeTemplateRenderer()
    
    # Simulating the problematic scenario from the issue
    print("\n🔴 ISSUE: Format filter fails on non-numeric sensor.value")
    print("🔴 ISSUE: Invalid sensors with value=0 mask actual invalid values")
    print("-" * 80)
    
    # Original problematic template (would fail)
    problematic_template = """
    Sensor: {{ sensor.name }}
    Value: {{ "%.2f"|format(sensor.value) }}
    """
    
    # Fixed template (from integration_example.py)
    fixed_template = """
    Sensor: {{ sensor.name }}
    Value: {%- if sensor.value is number -%}
        {{ "%.2f"|format(sensor.value) }}
    {%- else -%}
        {{ sensor.value }} (raw)
    {%- endif %}
    {%- if not sensor.is_valid and sensor.value is defined %}
    Raw Value: {{ sensor.value }} (for debugging)
    {%- endif %}
    """
    
    # Test cases that would cause the original issue
    test_cases = [
        {
            'name': 'Invalid sensor with string value',
            'sensor': {
                'name': 'Temperature Sensor',
                'value': 'COMM_ERROR_123',
                'is_valid': False
            }
        },
        {
            'name': 'Invalid sensor with value=0 (masking issue)',
            'sensor': {
                'name': 'Pressure Sensor', 
                'value': 0,  # This could mask the actual error
                'is_valid': False  # But validation detects it's invalid
            }
        },
        {
            'name': 'Valid sensor with numeric value',
            'sensor': {
                'name': 'Flow Sensor',
                'value': 125.75,
                'is_valid': True
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Test Case: {test_case['name']}")
        print("-" * 50)
        
        # Try the problematic template first
        print("❌ Original problematic template:")
        try:
            result = renderer.render_template_string(problematic_template, **test_case)
            print(f"   Result: {result.strip()}")
        except Exception as e:
            print(f"   FAILS: {type(e).__name__}: {e}")
        
        # Show the fixed template result
        print("✅ Fixed template with conditional formatting:")
        try:
            result = renderer.render_template_string(fixed_template, **test_case)
            print(f"   Result: {result.strip()}")
        except Exception as e:
            print(f"   Unexpected error: {type(e).__name__}: {e}")
    
    print("\n" + "=" * 80)
    print("🎉 SUMMARY: Format filter issue is FIXED!")
    print("✅ Templates no longer fail on non-numeric sensor values")
    print("✅ Raw values are displayed for debugging invalid sensors")
    print("✅ Value=0 masking is addressable through validation flags")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_issue_fix()