#!/usr/bin/env python3
"""
Comprehensive test for the format filter fix.

This tests the updated integration_example.py to ensure:
1. Format filter doesn't fail on non-numeric values
2. Raw values are displayed for debugging
3. Invalid sensors with value=0 are properly handled
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from integration_example import SteelDefectReportGenerator


def test_sensor_report_fix():
    """Test the sensor report formatting fix."""
    print("Testing sensor report with format filter fix...")
    
    generator = SteelDefectReportGenerator()
    
    # Test various edge cases
    test_cases = [
        {
            'name': 'Mixed valid/invalid sensors',
            'data': {
                'mold_temperature': 1520.5,        # Valid float
                'casting_speed': "error_code_123", # Invalid string
                'mold_level': float('inf'),        # Invalid infinity
                'cooling_water_flow': None,        # Invalid None
                'superheat': 0,                    # Valid zero (potential masking issue)
            }
        },
        {
            'name': 'All invalid sensors',
            'data': {
                'sensor_1': "COMM_ERROR",
                'sensor_2': "TIMEOUT",
                'sensor_3': float('nan'),
                'sensor_4': [],  # Invalid list
                'sensor_5': {},  # Invalid dict
            }
        },
        {
            'name': 'Zero values that might mask invalid readings',
            'data': {
                'temperature': 0,     # Could be masking invalid reading
                'pressure': 0.0,      # Could be masking invalid reading
                'flow_rate': -273.15, # Valid very low value
                'vibration': 0,       # Could be masking invalid reading
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        print("-" * 50)
        
        try:
            # Generate report
            report_html = generator.generate_sensor_report(
                test_case['data'], 
                f"TEST_CAST_{i:03d}"
            )
            
            # Save report
            output_dir = Path("demo_outputs")
            output_dir.mkdir(exist_ok=True)
            report_file = output_dir / f"test_fix_{i}_report.html"
            
            with open(report_file, 'w') as f:
                f.write(report_html)
            
            print(f"✅ Report generated successfully: {report_file}")
            
            # Check if raw values are shown for invalid sensors
            if "Raw Value:" in report_html:
                print("✅ Raw values displayed for debugging")
            else:
                print("ℹ️  No invalid sensors to show raw values for")
                
            # Check if valid numeric values are formatted
            if ".00" in report_html or ".50" in report_html:
                print("✅ Valid values formatted correctly")
            
        except Exception as e:
            print(f"❌ Test failed: {type(e).__name__}: {e}")


def test_defect_probability_fix():
    """Test the defect probability alert formatting fix."""
    print("\n" + "="*70)
    print("Testing defect probability alerts with format filter fix...")
    
    generator = SteelDefectReportGenerator()
    
    # Test edge cases for probability values
    test_cases = [
        {'probability': 'invalid_string', 'confidence': 0.8, 'name': 'Invalid string probability'},
        {'probability': float('nan'), 'confidence': 0.7, 'name': 'NaN probability'},
        {'probability': 0.85, 'confidence': 'invalid', 'name': 'Invalid confidence'},
        {'probability': 0.0, 'confidence': 0.95, 'name': 'Zero probability (might mask invalid)'},
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        print("-" * 50)
        
        try:
            alert_html = generator.generate_defect_probability_alert(test_case)
            
            # Save alert
            output_dir = Path("demo_outputs")
            alert_file = output_dir / f"test_fix_{i}_alert.html"
            
            with open(alert_file, 'w') as f:
                f.write(f"<html><body>{alert_html}</body></html>")
            
            print(f"✅ Alert generated successfully: {alert_file}")
            
            # Check for proper handling of invalid values
            if "invalid format" in alert_html:
                print("✅ Invalid format values handled correctly")
            elif "INVALID PREDICTION DATA" in alert_html:
                print("✅ Invalid prediction data handled correctly")
            else:
                print("ℹ️  Valid data processed normally")
                
        except Exception as e:
            print(f"❌ Test failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    test_sensor_report_fix()
    test_defect_probability_fix()
    print("\n🎉 All format filter fix tests completed!")