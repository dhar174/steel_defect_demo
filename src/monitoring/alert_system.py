import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional
from pathlib import Path

# Placeholder for a potential webhook library
# import requests 

class AlertSystem:
    """Handles alert dispatching to multiple channels with suppression logic."""
    
    def __init__(self, config: Dict):
        """
        Initializes the alert system with configuration.
        
        Args:
            config (Dict): The application configuration.
        """
        self.config = config.get('monitoring', {})
        self.alert_config = self.config.get('alerts', {})
        self.last_alert_times: Dict[str, datetime] = {}
        self.suppression_period = timedelta(
            minutes=self.alert_config.get('alert_suppression_minutes', 5)
        )
        self.logger = logging.getLogger(__name__)

    def send_alert(self, level: str, alert_type: str, message: str, details: Dict):
        """
        Sends an alert if it's not suppressed.
        
        Args:
            level (str): The severity level ('warning', 'critical').
            alert_type (str): A unique identifier for the alert type (e.g., 'high_risk_prediction').
            message (str): The main alert message.
            details (Dict): Additional details to include in the alert.
        """
        # Check if alert should be suppressed
        if self._is_suppressed(alert_type):
            self.logger.debug(f"Alert suppressed for type '{alert_type}' due to recent alert")
            return

        # Update the last alert time for this type
        self.last_alert_times[alert_type] = datetime.now()

        # Format the alert message
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_alert = f"[{timestamp}] {level.upper()}: {message}"
        if details:
            formatted_alert += f" | Details: {details}"

        # Create JSON payload for webhook
        alert_payload = {
            'timestamp': timestamp,
            'level': level,
            'alert_type': alert_type,
            'message': message,
            'details': details
        }

        # Dispatch to configured channels
        self._dispatch(formatted_alert, alert_payload)

    def _is_suppressed(self, alert_type: str) -> bool:
        """
        Checks if an alert of a given type should be suppressed.
        
        Args:
            alert_type (str): The alert type to check.
            
        Returns:
            bool: True if the alert should be suppressed, False otherwise.
        """
        if alert_type not in self.last_alert_times:
            return False
        
        last_alert_time = self.last_alert_times[alert_type]
        time_elapsed = datetime.now() - last_alert_time
        
        return time_elapsed < self.suppression_period

    def _dispatch(self, formatted_alert: str, alert_payload: Dict):
        """
        Dispatches the alert to all configured channels.
        
        Args:
            formatted_alert (str): The formatted alert message.
            alert_payload (Dict): The JSON payload for webhook notifications.
        """
        # Console notifications
        if self.alert_config.get('console_enabled', True):
            self._send_to_console(formatted_alert)

        # File notifications
        if self.alert_config.get('file_enabled', False):
            self._send_to_file(formatted_alert)

        # Webhook notifications
        webhook_url = self.alert_config.get('webhook_url')
        if webhook_url:
            self._send_to_webhook(alert_payload, webhook_url)

    def _send_to_console(self, alert: str):
        """Handler for console notifications."""
        self.logger.info(f"CONSOLE ALERT: {alert}")

    def _send_to_file(self, alert: str):
        """Handler for file-based notifications."""
        try:
            # Get file path from config, default to logs/alerts.log
            file_path = self.alert_config.get('file_path', 'logs/alerts.log')
            
            # Ensure the directory exists
            log_dir = Path(file_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Append alert to file
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(f"{alert}\n")
                
        except Exception as e:
            self.logger.error(f"Failed to write alert to file: {e}")

    def _send_to_webhook(self, alert_payload: Dict, webhook_url: str):
        """Handler for webhook notifications."""
        try:
            # This is a placeholder implementation
            # In a real implementation, you would use requests library:
            # response = requests.post(webhook_url, json=alert_payload, timeout=10)
            # response.raise_for_status()
            
            self.logger.info(f"WEBHOOK ALERT would be sent to {webhook_url}: {json.dumps(alert_payload)}")
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert to {webhook_url}: {e}")