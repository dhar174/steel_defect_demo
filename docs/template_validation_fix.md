# Template Validation Fix for Jinja2 'is number' Issue

## Problem Description

The original issue reported that templates were using `'metric_value is number'` test, which was claimed to not be a valid Jinja2 test. Upon investigation, we found that:

1. **Jinja2 DOES have an 'is number' test** - it's a valid built-in test
2. **However, the 'is number' test has limitations** for robust metric validation:
   - Returns `True` for boolean values (`True`, `False`)
   - Returns `True` for infinity (`float('inf')`)
   - Returns `True` for NaN (`float('nan')`)
   - May not behave as expected in all edge cases

## Solution

We've implemented a comprehensive template validation system that addresses these issues:

### 1. Python-side validation using `isinstance()`

```python
from src.utils.template_validation import TemplateDataValidator

# Validate before passing to template
is_valid = TemplateDataValidator.is_valid_metric_value(metric_value)
```

### 2. Custom Jinja2 tests for reliable validation

```python
from src.utils.template_validation import SafeTemplateRenderer

renderer = SafeTemplateRenderer()
template = "{{ value is valid_metric }}"  # Custom test
result = renderer.render_template_string(template, value=42.5)
```

### 3. Validated data preparation

```python
# Prepare data with validation flags
validated_data = TemplateDataValidator.prepare_metric_data(metric_value, "temperature")
# Results in: {'temperature_value': ..., 'is_temperature_valid': ..., 'temperature_error': ...}
```

## Usage Examples

### Before (Problematic)
```jinja2
{% if metric_value is number %}
  <div>{{ metric_value }}</div>
{% else %}
  <div>Error</div>
{% endif %}
```

**Issues with this approach:**
- `True` is considered a number (returns `True`)
- `float('inf')` is considered a number (returns `True`)
- `float('nan')` is considered a number (returns `True`)

### After (Fixed)

#### Option 1: Use custom Jinja2 test
```jinja2
{% if metric_value is valid_metric %}
  <div>{{ metric_value }}</div>
{% else %}
  <div>Error</div>
{% endif %}
```

#### Option 2: Validate in Python (Recommended)
```python
# In Python code
validated_data = TemplateDataValidator.prepare_metric_data(metric_value, "temperature")
```

```jinja2
{% if is_temperature_valid %}
  <div>{{ temperature_value }}</div>
{% else %}
  <div>Error: {{ temperature_error }}</div>
{% endif %}
```

## Comparison Table

| Value | `is number` | `is valid_metric` | Notes |
|-------|-------------|-------------------|-------|
| `42` | ✅ True | ✅ True | Valid integer |
| `42.5` | ✅ True | ✅ True | Valid float |
| `"42"` | ❌ False | ❌ False | String, not numeric |
| `True` | ✅ **True** | ❌ **False** | Boolean excluded |
| `float('inf')` | ✅ **True** | ❌ **False** | Infinity excluded |
| `float('nan')` | ✅ **True** | ❌ **False** | NaN excluded |
| `None` | ❌ False | ❌ False | None value |

## Implementation Files

- `src/utils/template_validation.py` - Main validation utilities
- `tests/test_template_validation.py` - Comprehensive tests and examples
- `template_example.py` - Simple demonstration script

## Migration Guide

To update existing templates:

1. **Replace direct 'is number' tests:**
   ```jinja2
   <!-- OLD -->
   {% if metric_value is number %}
   
   <!-- NEW -->
   {% if metric_value is valid_metric %}
   ```

2. **Use Python validation (recommended):**
   ```python
   # In your Python code
   from src.utils.template_validation import TemplateDataValidator
   
   validated_data = TemplateDataValidator.prepare_metric_data(value, "metric_name")
   template.render(**validated_data)
   ```

3. **Update template to use validation flags:**
   ```jinja2
   {% if is_metric_valid %}
     Valid: {{ metric_value }}
   {% else %}
     Error: {{ metric_error }}
   {% endif %}
   ```

## Testing

Run the comprehensive test suite:
```bash
python tests/test_template_validation.py
```

This will show the comparison between different validation approaches and verify that all edge cases are handled correctly.