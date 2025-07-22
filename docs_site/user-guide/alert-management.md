# Alert Management

The Alert Management system provides comprehensive monitoring and notification capabilities for the Steel Defect
Prediction System, ensuring timely response to critical conditions.

## Overview

Alert management features include:

- Real-time defect probability alerts
- Process parameter threshold monitoring
- Multi-channel notification system
- Alert escalation and acknowledgment
- Historical alert tracking and analysis

## Alert Types

### Defect Probability Alerts

**Critical Alert**: Defect probability > 80%

```python
{
    "alert_id": "CRIT-001",
    "type": "defect_probability",
    "level": "critical",
    "probability": 0.85,
    "threshold": 0.80,
    "message": "High defect probability detected in casting line 1"
}
```text

**Warning Alert**: Defect probability 60-80%

```python
{
    "alert_id": "WARN-002", 
    "type": "defect_probability",
    "level": "warning",
    "probability": 0.72,
    "threshold": 0.60,
    "message": "Elevated defect risk in casting line 1"
}
```text

### Process Parameter Alerts

#### Temperature Deviation

```python
{
    "alert_id": "TEMP-003",
    "type": "temperature_deviation", 
    "level": "warning",
    "current_value": 1580.5,
    "target_value": 1525.0,
    "deviation": 55.5,
    "message": "Mold temperature exceeds target by 55°C"
}
```text

#### Flow Rate Alert

```python
{
    "alert_id": "FLOW-004",
    "type": "flow_rate",
    "level": "critical",
    "current_value": 150.2,
    "min_threshold": 180.0,
    "message": "Cooling water flow below minimum threshold"
}
```text

## Alert Configuration

### Setting Alert Rules

```python
from src.alerts.alert_manager import AlertManager

# Initialize alert manager

alert_manager = AlertManager()

# Configure defect probability alerts

alert_manager.add_rule({
    'name': 'high_defect_probability',
    'condition': 'defect_probability > 0.8',
    'level': 'critical',
    'cooldown': 300,  # 5 minutes
    'notifications': ['email', 'sms', 'dashboard']
})

# Configure process parameter alerts

alert_manager.add_rule({
    'name': 'temperature_deviation',
    'condition': 'abs(mold_temperature - target_temperature) > 50',
    'level': 'warning', 
    'cooldown': 600,  # 10 minutes
    'notifications': ['email', 'dashboard']
})
```text

### Threshold Management

```yaml

# alerts.yml configuration

alert_thresholds:
  defect_probability:
    warning: 0.6
    critical: 0.8
  
  mold_temperature:
    min_warning: 1480
    max_warning: 1570
    min_critical: 1450
    max_critical: 1600
  
  casting_speed:
    min_warning: 0.8
    max_warning: 1.4
    min_critical: 0.6
    max_critical: 1.6
  
  cooling_water_flow:
    min_warning: 180
    min_critical: 160
```text

## Notification Channels

### Email Notifications

```python

# Email configuration

email_config = {
    'smtp_server': 'smtp.company.com',
    'smtp_port': 587,
    'username': 'alerts@company.com',
    'password': '${EMAIL_PASSWORD}',
    'recipients': {
        'critical': ['supervisor@company.com', 'manager@company.com'],
        'warning': ['operator@company.com', 'technician@company.com'],
        'info': ['operator@company.com']
    }
}

# Send email alert

from src.alerts.email_notifier import EmailNotifier

notifier = EmailNotifier(email_config)
notifier.send_alert({
    'level': 'critical',
    'subject': 'Critical Defect Alert - Line 1',
    'message': 'High defect probability detected. Immediate attention required.',
    'data': sensor_data
})
```text

### SMS Notifications

```python

# SMS configuration using Twilio

sms_config = {
    'account_sid': '${TWILIO_ACCOUNT_SID}',
    'auth_token': '${TWILIO_AUTH_TOKEN}',
    'from_number': '+1234567890',
    'recipients': {
        'critical': ['+1987654321', '+1456789012'],
        'warning': ['+1987654321']
    }
}

# Send SMS alert

from src.alerts.sms_notifier import SMSNotifier

sms_notifier = SMSNotifier(sms_config)
sms_notifier.send_alert({
    'level': 'critical',
    'message': 'URGENT: Critical defect alert on Line 1. Defect probability: 85%'
})
```text

### Dashboard Notifications

```javascript
// Real-time dashboard alerts
const alertSocket = new WebSocket('ws://localhost:8000/alerts');

alertSocket.onmessage = function(event) {
    const alert = JSON.parse(event.data);
    displayAlert(alert);
};

function displayAlert(alert) {
    const alertElement = document.createElement('div');
    alertElement.className = `alert alert-${alert.level}`;
    alertElement.innerHTML = `
        <strong>${alert.level.toUpperCase()}</strong>
        <p>${alert.message}</p>
        <span class="timestamp">${alert.timestamp}</span>
        <button onclick="acknowledgeAlert('${alert.id}')">Acknowledge</button>
    `;
    
    document.getElementById('alerts-container').appendChild(alertElement);
}
```text

## Alert Escalation

### Escalation Rules

```python

# Define escalation chain

escalation_config = {
    'levels': [
        {
            'name': 'operator',
            'timeout': 300,  # 5 minutes
            'notifications': ['dashboard', 'email']
        },
        {
            'name': 'supervisor', 
            'timeout': 600,  # 10 minutes
            'notifications': ['email', 'sms']
        },
        {
            'name': 'manager',
            'timeout': 1200,  # 20 minutes
            'notifications': ['email', 'sms', 'phone_call']
        }
    ]
}

# Implement escalation

from src.alerts.escalation_manager import EscalationManager

escalation_manager = EscalationManager(escalation_config)
escalation_manager.start_escalation(alert_id='CRIT-001')
```text

### Auto-escalation Logic

```python
class AlertEscalator:
    def __init__(self, escalation_rules):
        self.rules = escalation_rules
        self.active_escalations = {}
    
    def escalate_alert(self, alert):
        escalation_id = self.start_escalation(alert)
        
        # Schedule escalation steps

        for level in self.rules['levels']:
            threading.Timer(
                level['timeout'],
                self.escalate_to_level,
                args=[escalation_id, level]
            ).start()
    
    def escalate_to_level(self, escalation_id, level):
        if not self.is_acknowledged(escalation_id):
            self.send_notifications(level['notifications'])
            logging.info(f"Escalated alert {escalation_id} to {level['name']}")
```text

## Alert Dashboard

### Alert Status Overview

```python

# Get current alert status

from src.alerts.alert_dashboard import AlertDashboard

dashboard = AlertDashboard()
status = dashboard.get_alert_status()

print(f"Active alerts: {status['active_count']}")
print(f"Critical alerts: {status['critical_count']}")
print(f"Acknowledged alerts: {status['acknowledged_count']}")
print(f"Average response time: {status['avg_response_time']} minutes")
```text

### Alert History

```python

# Query alert history

alert_history = dashboard.get_alert_history(
    start_date='2024-01-01',
    end_date='2024-01-31',
    levels=['critical', 'warning'],
    acknowledged=True
)

# Generate alert statistics

stats = dashboard.generate_alert_statistics(alert_history)
print(f"Total alerts: {stats['total_alerts']}")
print(f"Most common alert type: {stats['most_common_type']}")
print(f"Average response time: {stats['avg_response_time']}")
```text

## Alert Acknowledgment

### Manual Acknowledgment

```python

# Acknowledge alert via API

import requests

response = requests.post(
    'http://localhost:8000/alerts/acknowledge',
    json={
        'alert_id': 'CRIT-001',
        'acknowledged_by': 'operator_1',
        'acknowledgment_note': 'Investigated and taking corrective action'
    }
)
```text

### Auto-acknowledgment Rules

```python

# Configure auto-acknowledgment

auto_ack_rules = {
    'defect_probability_resolved': {
        'condition': 'defect_probability < 0.5',
        'delay': 60  # Wait 1 minute before auto-ack
    },
    'temperature_normalized': {
        'condition': 'abs(mold_temperature - target_temperature) < 10',
        'delay': 120  # Wait 2 minutes before auto-ack
    }
}

# Apply auto-acknowledgment

from src.alerts.auto_acknowledger import AutoAcknowledger

auto_ack = AutoAcknowledger(auto_ack_rules)
auto_ack.check_for_resolution(current_sensor_data)
```text

## Alert Analytics

### Alert Frequency Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Analyze alert patterns

alert_df = pd.DataFrame(alert_history)
alert_df['timestamp'] = pd.to_datetime(alert_df['timestamp'])

# Alert frequency by hour

hourly_alerts = alert_df.groupby(alert_df['timestamp'].dt.hour).size()

plt.figure(figsize=(12, 6))
plt.bar(hourly_alerts.index, hourly_alerts.values)
plt.xlabel('Hour of Day')
plt.ylabel('Alert Count')
plt.title('Alert Frequency by Hour')
plt.show()
```text

### Response Time Analysis

```python

# Calculate response times

alert_df['response_time'] = (
    pd.to_datetime(alert_df['acknowledged_at']) - 
    pd.to_datetime(alert_df['created_at'])
).dt.total_seconds() / 60  # Convert to minutes

# Response time by alert level

response_by_level = alert_df.groupby('level')['response_time'].agg([
    'mean', 'median', 'std'
])

print("Response Time Statistics (minutes):")
print(response_by_level)
```text

## Integration Examples

### SCADA Integration

```python

# Send alerts to SCADA system

from src.integrations.scada_alerts import SCADAAlertSender

scada_alerts = SCADAAlertSender(
    host='scada.plant.com',
    port=502
)

# Send alert to SCADA alarm system

scada_alerts.send_alarm({
    'alarm_id': 'ALM_001',
    'description': 'High defect probability',
    'severity': 'HIGH',
    'area': 'Casting_Line_1'
})
```text

### Third-party Systems

```python

# Send alerts to external monitoring systems

webhook_config = {
    'url': 'https://monitoring.company.com/webhooks/alerts',
    'headers': {
        'Authorization': 'Bearer ${API_TOKEN}',
        'Content-Type': 'application/json'
    }
}

# Send webhook notification

import requests

def send_webhook_alert(alert_data):
    response = requests.post(
        webhook_config['url'],
        json=alert_data,
        headers=webhook_config['headers']
    )
    return response.status_code == 200
```text

## Troubleshooting

### Common Issues

1. **Alerts not firing**: Check threshold configuration and sensor data flow
2. **Notification failures**: Verify SMTP/SMS credentials and network connectivity
3. **False alarms**: Adjust thresholds and add filtering conditions
4. **Missing escalations**: Check escalation timer configuration

### Debug Tools

```python

# Enable alert debugging

import logging
logging.getLogger('alerts').setLevel(logging.DEBUG)

# Test alert configuration

from src.alerts.tester import AlertTester

tester = AlertTester()
test_results = tester.test_all_rules(test_data)

for rule, result in test_results.items():
    print(f"{rule}: {'PASS' if result['success'] else 'FAIL'}")
```text

This alert management system ensures prompt response to critical conditions and maintains operational safety in your steel casting operations.
