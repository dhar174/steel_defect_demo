#!/usr/bin/env python3
"""
Test suite for the Issue Manager

This script validates that the issue manager correctly parses and generates
sub-issues from the Phase 6 parent issue.
"""

import json
import sys
import os
from pathlib import Path

# Add the scripts directory to the path so we can import issue_manager
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from issue_manager import IssueManager


def test_sub_issue_generation():
    """Test that all expected sub-issues are generated."""
    manager = IssueManager()
    sub_issues = manager.generate_all_sub_issues()
    
    # Expected sub-issues based on the Phase 6 content
    expected_sections = [
        '6.1', '6.2', '6.3', '6.4', '6.5', '6.6', 
        '6.7', '6.8', '6.9', '6.10', '6.11'
    ]
    
    assert len(sub_issues) == len(expected_sections), f"Expected {len(expected_sections)} sub-issues, got {len(sub_issues)}"
    
    generated_sections = [issue['metadata']['section_number'] for issue in sub_issues]
    
    for expected in expected_sections:
        assert expected in generated_sections, f"Missing expected section {expected}"
    
    print("✓ All expected sub-issues generated")


def test_sub_issue_structure():
    """Test that each sub-issue has the required structure."""
    manager = IssueManager()
    sub_issues = manager.generate_all_sub_issues()
    
    required_fields = ['title', 'body', 'labels', 'metadata', 'parent_issue']
    required_metadata = ['section_number', 'priority', 'estimated_effort', 'prerequisites']
    
    for issue in sub_issues:
        # Test main structure
        for field in required_fields:
            assert field in issue, f"Missing required field '{field}' in issue {issue.get('title', 'Unknown')}"
        
        # Test metadata structure
        for field in required_metadata:
            assert field in issue['metadata'], f"Missing metadata field '{field}' in issue {issue.get('title', 'Unknown')}"
        
        # Test parent issue reference
        assert issue['parent_issue']['number'] == 10
        assert issue['parent_issue']['url'] == "https://github.com/dhar174/steel_defect_demo/issues/10"
        
        # Test that title includes section number
        section_num = issue['metadata']['section_number']
        assert issue['title'].startswith(section_num), f"Title doesn't start with section number for {section_num}"
    
    print("✓ All sub-issues have correct structure")


def test_prerequisites_dependencies():
    """Test that prerequisites are correctly assigned."""
    manager = IssueManager()
    sub_issues = manager.generate_all_sub_issues()
    
    # Create a mapping for easy lookup
    issues_by_section = {issue['metadata']['section_number']: issue for issue in sub_issues}
    
    # Test specific dependency relationships
    assert len(issues_by_section['6.1']['metadata']['prerequisites']) == 0, "6.1 should have no prerequisites"
    assert '6.1' in issues_by_section['6.2']['metadata']['prerequisites'], "6.2 should depend on 6.1"
    assert '6.1' in issues_by_section['6.3']['metadata']['prerequisites'], "6.3 should depend on 6.1"
    
    # Test that advanced features depend on core infrastructure
    advanced_features = ['6.9', '6.10', '6.11']
    for section in advanced_features:
        prerequisites = issues_by_section[section]['metadata']['prerequisites']
        assert len(prerequisites) > 0, f"{section} should have prerequisites"
        assert '6.1' in prerequisites or '6.2' in prerequisites, f"{section} should depend on core infrastructure"
    
    print("✓ Prerequisites correctly assigned")


def test_priority_assignment():
    """Test that priorities are correctly assigned based on dependencies."""
    manager = IssueManager()
    sub_issues = manager.generate_all_sub_issues()
    
    # Count priorities
    priorities = {}
    for issue in sub_issues:
        priority = issue['metadata']['priority']
        priorities[priority] = priorities.get(priority, 0) + 1
    
    # Should have high, medium, and low priorities
    assert 'High' in priorities, "Should have high priority issues"
    assert 'Medium' in priorities, "Should have medium priority issues"
    assert 'Low' in priorities, "Should have low priority issues"
    
    # Core infrastructure should be high priority
    issues_by_section = {issue['metadata']['section_number']: issue for issue in sub_issues}
    core_infrastructure = ['6.1', '6.2', '6.3']
    
    for section in core_infrastructure:
        priority = issues_by_section[section]['metadata']['priority']
        assert priority == 'High', f"Core infrastructure {section} should be high priority, got {priority}"
    
    print("✓ Priorities correctly assigned")


def test_acceptance_criteria():
    """Test that acceptance criteria are comprehensive."""
    manager = IssueManager()
    sub_issues = manager.generate_all_sub_issues()
    
    for issue in sub_issues:
        body = issue['body']
        
        # Should have acceptance criteria section
        assert '## Acceptance Criteria' in body, f"Missing acceptance criteria in {issue['title']}"
        
        # Should have multiple criteria (count checkboxes)
        checkbox_count = body.count('- [ ]')
        assert checkbox_count >= 10, f"Should have at least 10 acceptance criteria, got {checkbox_count} for {issue['title']}"
        
        # Should have testing requirements
        assert '## Testing Requirements' in body, f"Missing testing requirements in {issue['title']}"
        
        # Should have definition of done
        assert '## Definition of Done' in body, f"Missing definition of done in {issue['title']}"
    
    print("✓ Acceptance criteria are comprehensive")


def test_json_export():
    """Test that JSON export works correctly."""
    manager = IssueManager()
    
    # Test export to temporary file
    temp_file = "/tmp/test_export.json"
    filepath = manager.export_to_json(temp_file)
    
    assert os.path.exists(filepath), "JSON export file should exist"
    
    # Validate JSON structure
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    assert 'parent_issue' in data, "JSON should contain parent_issue"
    assert 'sub_issues' in data, "JSON should contain sub_issues"
    assert 'metadata' in data, "JSON should contain metadata"
    
    assert data['parent_issue']['number'] == 10, "Parent issue number should be 10"
    assert len(data['sub_issues']) == 11, "Should have 11 sub-issues"
    
    # Clean up
    os.remove(filepath)
    
    print("✓ JSON export works correctly")


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_sub_issue_generation,
        test_sub_issue_structure,
        test_prerequisites_dependencies,
        test_priority_assignment,
        test_acceptance_criteria,
        test_json_export
    ]
    
    print("Running Issue Manager Tests...")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {str(e)}")
            failed += 1
    
    print("=" * 50)
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)