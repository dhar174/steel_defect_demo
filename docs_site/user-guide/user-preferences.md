# User Preferences

The User Preferences system allows operators and administrators to customize their experience with the Steel Defect
Prediction System according to their roles, responsibilities, and workflow needs.

## Overview

User preferences cover:

- Dashboard layout and widget configuration
- Alert notification settings
- Data visualization preferences
- Report generation templates
- Language and localization settings
- Theme and accessibility options

## Dashboard Customization

### Widget Configuration

```python

# Configure dashboard widgets

dashboard_config = {
    'user_id': 'operator_001',
    'layout': 'grid',
    'widgets': [
        {
            'type': 'defect_probability_gauge',
            'position': {'row': 1, 'col': 1, 'span': 2},
            'settings': {
                'thresholds': {'warning': 0.6, 'critical': 0.8},
                'update_interval': 1  # seconds
            }
        },
        {
            'type': 'process_parameters_chart',
            'position': {'row': 1, 'col': 3, 'span': 4}, 
            'settings': {
                'parameters': ['mold_temperature', 'casting_speed'],
                'time_window': 300,  # 5 minutes
                'chart_type': 'line'
            }
        },
        {
            'type': 'alert_panel',
            'position': {'row': 2, 'col': 1, 'span': 3},
            'settings': {
                'show_levels': ['critical', 'warning'],
                'max_alerts': 10
            }
        }
    ]
}
```text

### Layout Templates

```python

# Predefined layout templates

layout_templates = {
    'operator_view': {
        'description': 'Focused on real-time monitoring',
        'widgets': ['defect_gauge', 'process_chart', 'alert_panel']
    },
    'supervisor_view': {
        'description': 'Overview with analytics',
        'widgets': ['summary_stats', 'trend_analysis', 'performance_metrics']
    },
    'maintenance_view': {
        'description': 'Equipment status focus',
        'widgets': ['equipment_status', 'maintenance_alerts', 'diagnostic_charts']
    }
}

# Apply template

from src.preferences.dashboard_manager import DashboardManager

dashboard_mgr = DashboardManager()
dashboard_mgr.apply_template('operator_view', user_id='operator_001')
```text

## Notification Preferences

### Alert Notification Settings

```python

# Configure personal alert preferences

alert_preferences = {
    'user_id': 'supervisor_002',
    'channels': {
        'critical': ['email', 'sms', 'dashboard'],
        'warning': ['email', 'dashboard'],
        'info': ['dashboard']
    },
    'quiet_hours': {
        'enabled': True,
        'start_time': '22:00',
        'end_time': '06:00',
        'emergency_override': True  # Critical alerts still come through
    },
    'escalation_preferences': {
        'response_timeout': 600,  # 10 minutes
        'escalate_to': 'manager@company.com'
    }
}
```text

### Custom Alert Rules

```python

# User-specific alert conditions

custom_alerts = {
    'user_id': 'quality_engineer_003',
    'custom_rules': [
        {
            'name': 'quality_concern',
            'condition': 'defect_probability > 0.5 AND steel_grade == "premium"',
            'notification': 'immediate',
            'description': 'Quality concern for premium steel grades'
        },
        {
            'name': 'shift_handover_summary', 
            'condition': 'time == "end_of_shift"',
            'notification': 'email',
            'description': 'Daily summary report'
        }
    ]
}
```text

## Data Visualization Preferences

### Chart Preferences

```python

# Visualization preferences

viz_preferences = {
    'user_id': 'analyst_004',
    'chart_settings': {
        'default_time_range': '24h',
        'color_scheme': 'steel_industry',  # Custom color palette
        'show_confidence_intervals': True,
        'animation_enabled': False,  # For better performance
        'grid_lines': True
    },
    'data_preferences': {
        'decimal_places': 2,
        'units': 'metric',  # or 'imperial'
        'aggregation_method': 'mean',  # or 'median', 'max'
        'missing_data_handling': 'interpolate'  # or 'skip', 'zero'
    }
}
```text

### Custom Metrics

```python

# Define user-specific metrics

custom_metrics = {
    'user_id': 'process_engineer_005',
    'metrics': [
        {
            'name': 'efficiency_index',
            'formula': '(production_rate * quality_score) / energy_consumption',
            'display_name': 'Process Efficiency Index',
            'unit': 'kg/kWh',
            'target_value': 15.0
        },
        {
            'name': 'cost_per_ton',
            'formula': '(energy_cost + material_cost + labor_cost) / production_volume',
            'display_name': 'Cost per Ton',
            'unit': '$/ton',
            'target_value': 850.0
        }
    ]
}
```text

## Report Templates

### Custom Report Configuration

```python

# Personal report templates

report_templates = {
    'user_id': 'manager_006',
    'templates': [
        {
            'name': 'daily_summary',
            'schedule': 'daily_8am',
            'sections': [
                'production_summary',
                'quality_metrics', 
                'alert_summary',
                'efficiency_trends'
            ],
            'format': 'pdf',
            'email_to': ['manager_006@company.com']
        },
        {
            'name': 'weekly_analysis',
            'schedule': 'monday_9am', 
            'sections': [
                'weekly_trends',
                'comparative_analysis',
                'improvement_recommendations'
            ],
            'format': 'html',
            'include_charts': True
        }
    ]
}
```text

### Report Filters

```python

# Custom data filters for reports

report_filters = {
    'user_id': 'quality_manager_007',
    'default_filters': {
        'steel_grades': ['304L', '316L'],  # Only premium grades
        'shifts': ['day', 'afternoon'],    # Exclude night shift
        'production_lines': [1, 2],        # Lines 1 and 2 only
        'exclude_maintenance_periods': True
    },
    'advanced_filters': {
        'statistical_outliers': 'remove',
        'data_quality_threshold': 0.95,   # 95% data completeness required
        'seasonal_adjustment': True
    }
}
```text

## Accessibility and Localization

### Accessibility Settings

```python

# Accessibility preferences

accessibility_settings = {
    'user_id': 'operator_008',
    'visual': {
        'high_contrast': True,
        'font_size_multiplier': 1.2,
        'color_blind_friendly': True,
        'color_palette': 'deuteranopia_safe'
    },
    'audio': {
        'alert_sounds': True,
        'sound_volume': 0.8,
        'speech_alerts': False
    },
    'interaction': {
        'keyboard_navigation': True,
        'mouse_sensitivity': 'low',
        'double_click_delay': 500  # milliseconds
    }
}
```text

### Language and Localization

```python

# Language preferences

localization_settings = {
    'user_id': 'operator_009',
    'language': 'es',  # Spanish
    'region': 'MX',    # Mexico
    'timezone': 'America/Mexico_City',
    'date_format': 'DD/MM/YYYY',
    'time_format': '24h',
    'number_format': {
        'decimal_separator': ',',
        'thousands_separator': '.',
        'currency_symbol': '$'
    }
}
```text

## Theme and UI Preferences

### Theme Configuration

```python

# UI theme preferences

theme_preferences = {
    'user_id': 'night_operator_010',
    'theme': {
        'mode': 'dark',  # 'light', 'dark', 'auto'
        'primary_color': '#1976d2',
        'accent_color': '#ff5722',
        'background_type': 'solid',  # 'solid', 'gradient'
        'sidebar_collapsed': False,
        'compact_mode': True
    },
    'animations': {
        'enabled': True,
        'speed': 'normal',  # 'slow', 'normal', 'fast'
        'transitions': True
    }
}
```text

### Custom CSS

```css
/* User-specific CSS overrides */
.user-custom-010 {
    --primary-color: #2196f3;
    --warning-color: #ff9800; 
    --danger-color: #f44336;
    --chart-background: #263238;
    --text-primary: #ffffff;
    --text-secondary: #b0bec5;
}

.user-custom-010 .dashboard-widget {
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

.user-custom-010 .alert-critical {
    animation: pulse 2s infinite;
}
```text

## Preferences Management

### Save and Load Preferences

```python
from src.preferences.preference_manager import PreferenceManager

# Initialize preference manager

pref_manager = PreferenceManager()

# Save user preferences

pref_manager.save_preferences(
    user_id='operator_001',
    preferences={
        'dashboard': dashboard_config,
        'alerts': alert_preferences,
        'visualization': viz_preferences,
        'theme': theme_preferences
    }
)

# Load user preferences

user_prefs = pref_manager.load_preferences('operator_001')

# Apply preferences to session

pref_manager.apply_preferences(user_prefs)
```text

### Preference Inheritance

```python

# Role-based preference inheritance

role_hierarchy = {
    'operator': {
        'inherits_from': 'base_user',
        'default_preferences': 'operator_defaults.json'
    },
    'supervisor': {
        'inherits_from': 'operator',
        'additional_permissions': ['modify_thresholds', 'view_analytics']
    },
    'manager': {
        'inherits_from': 'supervisor', 
        'additional_permissions': ['system_admin', 'user_management']
    }
}

# Apply role-based preferences

pref_manager.apply_role_preferences(user_id='new_operator', role='operator')
```text

## Backup and Sync

### Preference Backup

```python

# Backup user preferences

backup_data = pref_manager.export_preferences(
    user_id='operator_001',
    include_personal_data=False  # Exclude sensitive information
)

# Save backup

with open('user_preferences_backup.json', 'w') as f:
    json.dump(backup_data, f, indent=2)
```text

### Cross-device Sync

```python

# Sync preferences across devices

sync_config = {
    'user_id': 'mobile_operator_011',
    'devices': ['desktop_workstation', 'tablet_001', 'mobile_phone'],
    'sync_settings': {
        'dashboard_layout': True,
        'alert_preferences': True,
        'theme_preferences': True,
        'device_specific_settings': False  # Don't sync device-specific settings
    }
}

# Perform sync

pref_manager.sync_across_devices(sync_config)
```text

## API Endpoints

### REST API for Preferences

```python

# Get user preferences

GET /api/v1/users/{user_id}/preferences

# Update specific preference section

PATCH /api/v1/users/{user_id}/preferences/dashboard

# Reset to defaults

POST /api/v1/users/{user_id}/preferences/reset

# Export preferences

GET /api/v1/users/{user_id}/preferences/export

# Import preferences

POST /api/v1/users/{user_id}/preferences/import
```text

### Example API Usage

```python
import requests

# Update dashboard preferences

response = requests.patch(
    'http://localhost:8000/api/v1/users/operator_001/preferences/dashboard',
    json={
        'layout': 'compact',
        'auto_refresh': True,
        'refresh_interval': 5
    },
    headers={'Authorization': 'Bearer ${ACCESS_TOKEN}'}
)

if response.status_code == 200:
    print("Preferences updated successfully")
```text

This comprehensive preference system ensures that each user can tailor the Steel Defect Prediction System to their specific needs and workflow requirements.
