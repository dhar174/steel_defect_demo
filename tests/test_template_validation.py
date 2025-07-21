"""
Tests for template validation utilities.

This test file demonstrates the proper way to handle metric values in templates
and validates the fix for the Jinja2 'is number' issue.
"""

import sys
import unittest
import math
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.template_validation import (
    TemplateDataValidator, 
    SafeTemplateRenderer, 
    SAFE_METRIC_TEMPLATE
)


class TestTemplateDataValidator(unittest.TestCase):
    """Test the template data validator."""
    
    def test_is_numeric(self):
        """Test numeric value detection."""
        # Valid numeric values
        self.assertTrue(TemplateDataValidator.is_numeric(42))
        self.assertTrue(TemplateDataValidator.is_numeric(42.5))
        self.assertTrue(TemplateDataValidator.is_numeric(0))
        self.assertTrue(TemplateDataValidator.is_numeric(-1.5))
        
        # Invalid numeric values
        self.assertFalse(TemplateDataValidator.is_numeric("42"))
        self.assertFalse(TemplateDataValidator.is_numeric(None))
        self.assertFalse(TemplateDataValidator.is_numeric([]))
        self.assertFalse(TemplateDataValidator.is_numeric({}))
        self.assertFalse(TemplateDataValidator.is_numeric(True))  # Booleans are excluded
        self.assertFalse(TemplateDataValidator.is_numeric(False))
    
    def test_is_valid_metric_value(self):
        """Test metric value validation."""
        # Valid metric values
        self.assertTrue(TemplateDataValidator.is_valid_metric_value(42))
        self.assertTrue(TemplateDataValidator.is_valid_metric_value(42.5))
        self.assertTrue(TemplateDataValidator.is_valid_metric_value(0))
        self.assertTrue(TemplateDataValidator.is_valid_metric_value(-1.5))
        
        # Invalid metric values
        self.assertFalse(TemplateDataValidator.is_valid_metric_value(float('inf')))
        self.assertFalse(TemplateDataValidator.is_valid_metric_value(float('nan')))
        self.assertFalse(TemplateDataValidator.is_valid_metric_value("42"))
        self.assertFalse(TemplateDataValidator.is_valid_metric_value(None))
        self.assertFalse(TemplateDataValidator.is_valid_metric_value(True))
    
    def test_prepare_metric_data(self):
        """Test metric data preparation."""
        # Valid metric
        result = TemplateDataValidator.prepare_metric_data(42.5, "temperature")
        self.assertEqual(result["temperature_value"], 42.5)
        self.assertTrue(result["is_temperature_valid"])
        self.assertIsNone(result["temperature_error"])
        
        # Invalid metric
        result = TemplateDataValidator.prepare_metric_data("invalid", "pressure")
        self.assertEqual(result["pressure_value"], "invalid")
        self.assertFalse(result["is_pressure_valid"])
        self.assertIsNotNone(result["pressure_error"])
    
    def test_prepare_sensor_data(self):
        """Test sensor data preparation."""
        sensor_data = {
            "temperature": 1520.5,
            "pressure": "invalid",
            "flow_rate": 200.0
        }
        
        result = TemplateDataValidator.prepare_sensor_data(sensor_data)
        
        # Check temperature (valid)
        self.assertEqual(result["temperature_value"], 1520.5)
        self.assertTrue(result["is_temperature_valid"])
        
        # Check pressure (invalid)
        self.assertEqual(result["pressure_value"], "invalid")
        self.assertFalse(result["is_pressure_valid"])
        
        # Check flow_rate (valid)
        self.assertEqual(result["flow_rate_value"], 200.0)
        self.assertTrue(result["is_flow_rate_valid"])


class TestSafeTemplateRenderer(unittest.TestCase):
    """Test the safe template renderer."""
    
    def setUp(self):
        """Set up test renderer."""
        self.renderer = SafeTemplateRenderer()
    
    def test_custom_tests_available(self):
        """Test that custom Jinja2 tests are available."""
        template_str = "{{ value is valid_metric }}"
        result = self.renderer.render_template_string(template_str, value=42.5)
        self.assertEqual(result, "True")
        
        result = self.renderer.render_template_string(template_str, value="invalid")
        self.assertEqual(result, "False")
    
    def test_numeric_safe_test(self):
        """Test the numeric_safe test."""
        template_str = "{{ value is numeric_safe }}"
        result = self.renderer.render_template_string(template_str, value=42)
        self.assertEqual(result, "True")
        
        result = self.renderer.render_template_string(template_str, value="42")
        self.assertEqual(result, "False")
    
    def test_render_metric_template(self):
        """Test metric template rendering."""
        # Valid metric
        result = self.renderer.render_metric_template(
            SAFE_METRIC_TEMPLATE, 
            42.5, 
            "temperature"
        )
        self.assertIn("42.5", result)
        self.assertIn("valid", result)
        self.assertNotIn("error", result)
        
        # Invalid metric
        result = self.renderer.render_metric_template(
            SAFE_METRIC_TEMPLATE, 
            "invalid", 
            "temperature"
        )
        self.assertIn("Invalid or non-numeric value", result)
        self.assertIn("invalid", result)
        self.assertNotIn("42.5", result)


class TestProblematicTemplateFixed(unittest.TestCase):
    """Test that demonstrates the fix for the original issue."""
    
    def setUp(self):
        """Set up renderer."""
        self.renderer = SafeTemplateRenderer()
    
    def test_problematic_template_original_issue(self):
        """
        Test the original problematic pattern and show the proper fix.
        
        Original issue: "metric_value is number" test usage
        """
        # The problematic approach (works but is not always reliable)
        problematic_template = """
        {%- if metric_value is number %}
        <div>{{ metric_value }}</div>
        {%- else %}
        <div>Error</div>
        {%- endif %}
        """
        
        # The fixed approach using our validator
        fixed_template = """
        {%- if metric_value is valid_metric %}
        <div>{{ metric_value }}</div>
        {%- else %}
        <div>Error</div>
        {%- endif %}
        """
        
        # Test with various edge cases
        test_cases = [
            42,           # int
            42.5,         # float
            float('inf'), # infinity
            float('nan'), # NaN
            "42",         # string
            None,         # None
            True,         # boolean
        ]
        
        for test_value in test_cases:
            with self.subTest(value=test_value):
                # Our fixed approach should handle edge cases better
                try:
                    result = self.renderer.render_template_string(
                        fixed_template, metric_value=test_value
                    )
                    
                    # Check that inf and nan are properly handled as invalid
                    if test_value is None or isinstance(test_value, str) or \
                       (isinstance(test_value, float) and (math.isinf(test_value) or math.isnan(test_value))):
                        self.assertIn("Error", result)
                    elif isinstance(test_value, bool):
                        self.assertIn("Error", result)  # Booleans should be invalid
                    else:
                        self.assertIn(str(test_value), result)
                        
                except Exception as e:
                    self.fail(f"Template rendering failed for value {test_value}: {e}")


def demonstrate_fixes():
    """Demonstrate the proper fixes for the Jinja2 template issue."""
    print("=" * 60)
    print("DEMONSTRATION: Fixing Jinja2 Template Validation Issues")
    print("=" * 60)
    
    renderer = SafeTemplateRenderer()
    
    # Test cases that show the difference
    test_cases = [
        ("Valid integer", 42),
        ("Valid float", 42.5),
        ("String number", "42"),
        ("Invalid string", "not_a_number"),
        ("None value", None),
        ("Boolean true", True),
        ("Infinity", float('inf')),
        ("NaN", float('nan')),
    ]
    
    print("\n1. Testing 'is number' vs 'is valid_metric':")
    print("-" * 50)
    
    for description, test_value in test_cases:
        # Test built-in 'is number'
        builtin_result = renderer.render_template_string(
            "{{ value is number }}", value=test_value
        )
        
        # Test our custom 'is valid_metric'
        custom_result = renderer.render_template_string(
            "{{ value is valid_metric }}", value=test_value
        )
        
        print(f"{description:15} | is number: {builtin_result:5} | is valid_metric: {custom_result:5}")
    
    print("\n2. Recommended template pattern:")
    print("-" * 50)
    
    recommended_template = """
    {%- if is_metric_valid %}
    Valid: {{ metric_value }}
    {%- else %}
    Error: {{ metric_error }}
    {%- endif %}
    """
    
    for description, test_value in test_cases[:4]:  # Test first 4 cases
        validated_data = TemplateDataValidator.prepare_metric_data(test_value, "metric")
        result = renderer.render_template_string(recommended_template, **validated_data)
        print(f"{description}: {result.strip()}")
    
    print("\n3. Summary:")
    print("-" * 50)
    print("✅ Use isinstance() validation in Python before passing to templates")
    print("✅ Use custom Jinja2 tests for reliable validation")
    print("✅ Handle edge cases like inf, nan, and boolean values properly")
    print("❌ Avoid relying solely on 'is number' for critical validation")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_fixes()
    
    print("\n" + "=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60)
    
    # Run unit tests
    unittest.main(verbosity=2)