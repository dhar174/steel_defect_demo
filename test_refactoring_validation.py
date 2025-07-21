#!/usr/bin/env python3
"""
Test script to validate HistoricalAnalysisComponents refactoring.

This test ensures that:
1. No monkey-patching is present in the code
2. All expected methods are available as class methods
3. Basic functionality works correctly
4. Demo applications can import and use the component
"""

import sys
import os
import inspect
import pandas as pd

# Add the project root to path
sys.path.insert(0, os.path.abspath('.'))

def test_no_monkey_patching():
    """Test that no monkey-patching assignments exist in the code."""
    print("Testing for monkey-patching...")
    
    with open('src/visualization/components/historical_analysis.py', 'r') as f:
        content = f.read()
    
    # Check for monkey-patching patterns
    monkey_patch_patterns = [
        'HistoricalAnalysisComponents.create_',
        'HistoricalAnalysisComponents.export_',
        'HistoricalAnalysisComponents.filter_',
        '.create_data_overview_cards =',
        '.create_distribution_plot =',
        '.create_timeseries_plot =',
    ]
    
    violations = []
    for pattern in monkey_patch_patterns:
        if pattern in content:
            violations.append(pattern)
    
    if violations:
        print(f"❌ Found monkey-patching violations: {violations}")
        return False
    else:
        print("✅ No monkey-patching found")
        return True

def test_class_methods_available():
    """Test that all expected methods are available as class methods."""
    print("Testing class methods availability...")
    
    try:
        from src.visualization.components.historical_analysis import HistoricalAnalysisComponents
        
        expected_methods = [
            'create_data_overview_cards',
            'create_distribution_plot', 
            'create_timeseries_plot',
            'filter_data',
            'export_data_to_csv',
            'create_spc_statistics_summary',
            'create_clustering_analysis',
            'create_clustering_statistics_summary',
            'create_correlation_statistics_summary',
            'create_spc_charts',
            'create_correlation_heatmap',
            'create_batch_comparison',
            'export_chart_to_image',
            'create_batch_statistics_summary'
        ]
        
        ha = HistoricalAnalysisComponents()
        missing_methods = []
        
        for method in expected_methods:
            if not hasattr(ha, method) or not callable(getattr(ha, method)):
                missing_methods.append(method)
        
        if missing_methods:
            print(f"❌ Missing methods: {missing_methods}")
            return False
        else:
            print(f"✅ All {len(expected_methods)} expected methods available")
            return True
            
    except Exception as e:
        print(f"❌ Import or instantiation error: {e}")
        return False

def test_methods_are_bound():
    """Test that methods are properly bound to the class (not standalone functions)."""
    print("Testing method binding...")
    
    try:
        from src.visualization.components.historical_analysis import HistoricalAnalysisComponents
        
        ha = HistoricalAnalysisComponents()
        test_methods = ['filter_data', 'create_clustering_analysis', 'export_data_to_csv']
        
        for method_name in test_methods:
            method = getattr(ha, method_name)
            
            # Check if it's a bound method
            if not inspect.ismethod(method):
                print(f"❌ {method_name} is not a bound method")
                return False
                
            # Check if the method is bound to the correct instance
            if method.__self__ is not ha:
                print(f"❌ {method_name} is not bound to the correct instance")
                return False
                
            # Check if the method can access instance attributes
            if not hasattr(method.__self__, 'sensor_columns'):
                print(f"❌ {method_name} cannot access instance attributes")
                return False
        
        print(f"✅ All tested methods are properly bound")
        return True
        
    except Exception as e:
        print(f"❌ Method binding test error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of the refactored class."""
    print("Testing basic functionality...")
    
    try:
        from src.visualization.components.historical_analysis import HistoricalAnalysisComponents
        
        ha = HistoricalAnalysisComponents()
        
        # Test data generation
        sample_data = ha.load_sample_data()
        if len(sample_data) == 0:
            print("❌ Sample data generation failed")
            return False
        
        # Test filtering
        filtered_data = ha.filter_data(sample_data, defect_filter=1)
        if len(filtered_data) == 0:
            print("❌ Data filtering failed")
            return False
        
        # Test visualization
        overview_cards = ha.create_data_overview_cards(sample_data)
        if overview_cards is None:
            print("❌ Overview cards creation failed")
            return False
        
        # Test clustering
        cluster_fig, stats = ha.create_clustering_analysis(sample_data, 3)
        if cluster_fig is None or not stats:
            print("❌ Clustering analysis failed")
            return False
        
        print("✅ Basic functionality tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

def test_demo_imports():
    """Test that demo applications can import the component."""
    print("Testing demo application imports...")
    
    try:
        from demo_dashboard_integration import ExtendedDashboard
        from demo_historical_analysis import create_demo_app
        
        print("✅ Demo applications import successfully")
        return True
        
    except Exception as e:
        print(f"❌ Demo import error: {e}")
        return False

def test_file_size_reduction():
    """Test that the file size was reduced (indicating cleanup)."""
    print("Testing file size and cleanliness...")
    
    with open('src/visualization/components/historical_analysis.py', 'r') as f:
        lines = f.readlines()
    
    line_count = len(lines)
    
    # The file should be significantly smaller than the original (~2131 lines)
    if line_count > 1500:
        print(f"❌ File still too large: {line_count} lines (expected < 1500)")
        return False
    
    # Check for duplicate function definitions
    function_names = []
    for line in lines:
        if line.strip().startswith('def ') and '(' in line:
            func_name = line.strip().split('(')[0].replace('def ', '').strip()
            if func_name in function_names:
                print(f"❌ Duplicate function definition found: {func_name}")
                return False
            function_names.append(func_name)
    
    print(f"✅ File size reduced to {line_count} lines, no duplicates found")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("HistoricalAnalysisComponents Refactoring Validation")
    print("=" * 60)
    
    tests = [
        test_no_monkey_patching,
        test_class_methods_available,
        test_methods_are_bound,
        test_basic_functionality,
        test_demo_imports,
        test_file_size_reduction
    ]
    
    results = []
    for test in tests:
        print()
        result = test()
        results.append(result)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test.__name__}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Refactoring successful.")
        return 0
    else:
        print("❌ Some tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())