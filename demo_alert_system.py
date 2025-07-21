#!/usr/bin/env python3
"""
Example demonstrating the AlertSystem functionality.

This script shows how to use the AlertSystem to send alerts with suppression logic.

Note: Run 'pip install -e .' from the repository root to install the package in development mode.
"""

import time

import logging
from monitoring.alert_system import AlertSystem

# Configure logging to see console output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    """Demonstrate AlertSystem functionality."""
    print("=== Alert System Demo ===\n")
    
    # Configuration with all alert channels enabled
    config = {
        'monitoring': {
            'alerts': {
                'console_enabled': True,
                'file_enabled': True,
                'file_path': 'logs/demo_alerts.log',
                'webhook_url': 'https://httpbin.org/post',  # Demo webhook endpoint
                'alert_suppression_minutes': 1  # Short suppression for demo
            }
        }
    }
    
    # Create AlertSystem instance
    alert_system = AlertSystem(config)
    print("AlertSystem initialized with configuration:")
    print(f"  - Console alerts: {config['monitoring']['alerts']['console_enabled']}")
    print(f"  - File alerts: {config['monitoring']['alerts']['file_enabled']}")
    print(f"  - File path: {config['monitoring']['alerts']['file_path']}")
    print(f"  - Webhook URL: {config['monitoring']['alerts']['webhook_url']}")
    print(f"  - Suppression period: {config['monitoring']['alerts']['alert_suppression_minutes']} minute(s)")
    print()
    
    # Demo 1: Send initial alert
    print("Demo 1: Sending initial high-risk alert...")
    alert_system.send_alert(
        level='warning',
        alert_type='high_risk_prediction',
        message='High defect probability detected in production line A',
        details={
            'prediction': 0.75,
            'confidence': 0.92,
            'line_id': 'A',
            'timestamp': '2023-07-21 14:30:00'
        }
    )
    print("Alert sent successfully.\n")
    
    # Demo 2: Try to send same alert type immediately (should be suppressed)
    print("Demo 2: Sending same alert type immediately (should be suppressed)...")
    alert_system.send_alert(
        level='warning',
        alert_type='high_risk_prediction',
        message='Another high defect probability detected in production line A',
        details={
            'prediction': 0.78,
            'confidence': 0.89,
            'line_id': 'A',
            'timestamp': '2023-07-21 14:30:30'
        }
    )
    print("Alert attempt completed (check logs for suppression message).\n")
    
    # Demo 3: Send different alert type (should not be suppressed)
    print("Demo 3: Sending different alert type (should not be suppressed)...")
    alert_system.send_alert(
        level='critical',
        alert_type='system_error',
        message='Model inference pipeline failure',
        details={
            'error_code': 'INFERENCE_001',
            'component': 'LSTM Model',
            'timestamp': '2023-07-21 14:31:00'
        }
    )
    print("Alert sent successfully.\n")
    
    # Demo 4: Wait for suppression period to end and retry
    print("Demo 4: Waiting for suppression period to end...")
    print("Waiting 65 seconds for suppression period to expire...")
    time.sleep(65)  # Wait longer than suppression period
    
    print("Sending high-risk alert again (should not be suppressed now)...")
    alert_system.send_alert(
        level='warning',
        alert_type='high_risk_prediction',
        message='High defect probability detected in production line B',
        details={
            'prediction': 0.82,
            'confidence': 0.94,
            'line_id': 'B',
            'timestamp': '2023-07-21 14:32:00'
        }
    )
    print("Alert sent successfully.\n")
    
    # Demo 5: Show file logging results
    print("Demo 5: Checking file-based alerts...")
    log_file = Path(config['monitoring']['alerts']['file_path'])
    if log_file.exists():
        print(f"Alert log file contents ({log_file}):")
        print("-" * 50)
        with open(log_file, 'r') as f:
            print(f.read())
        print("-" * 50)
    else:
        print(f"Alert log file not found: {log_file}")
    
    print("\n=== Demo Complete ===")
    print("Check the console output above to see alert processing.")
    print("Note: Webhook alerts are placeholder implementations in this demo.")

if __name__ == '__main__':
    main()