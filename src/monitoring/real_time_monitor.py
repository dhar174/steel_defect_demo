import pandas as pd
import numpy as np
import logging
from collections import deque
from typing import Dict, List
import time


class RealTimeMonitor:
    """Monitors real-time predictions, system performance, and data quality."""
    
    def __init__(self, config: Dict):
        """
        Initializes the real-time monitor.
        
        Args:
            config (Dict): The application configuration, including monitoring thresholds.
        """
        self.config = config['monitoring']
        self.thresholds = config['inference']['thresholds']
        self.prediction_history = deque(maxlen=1000)  # Store last 1000 predictions
        self.latency_history = deque(maxlen=1000)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def track_prediction(self, prediction_result: Dict):
        """
        Logs a new prediction and checks for alerts.
        
        Args:
            prediction_result (Dict): A dictionary containing 'prediction', 'confidence', 'latency', etc.
        """
        # Store prediction and timestamp
        prediction_data = {
            'prediction': prediction_result.get('ensemble_prediction', prediction_result.get('prediction', 0.0)),
            'confidence': prediction_result.get('confidence', 0.0),
            'timestamp': time.time()
        }
        self.prediction_history.append(prediction_data)
        
        # Store latency information
        latency = prediction_result.get('latency', {})
        if isinstance(latency, dict):
            latency_ms = latency.get('total_time', 0.0) * 1000  # Convert to milliseconds
        else:
            latency_ms = float(latency) * 1000 if latency else 0.0
        
        self.latency_history.append({
            'latency_ms': latency_ms,
            'timestamp': time.time()
        })
        
        # Check for alerts
        self._generate_alert(prediction_result)

    def check_data_quality(self, data_buffer: pd.DataFrame) -> List[str]:
        """
        Performs basic checks on the incoming data buffer for anomalies.
        
        Returns:
            A list of detected data quality issues.
        """
        issues = []
        
        if data_buffer is None or data_buffer.empty:
            issues.append("Data buffer is empty or None")
            return issues
        
        # Check for NaN values
        nan_columns = data_buffer.columns[data_buffer.isnull().any()].tolist()
        if nan_columns:
            issues.append(f"NaN values detected in columns: {nan_columns}")
        
        # Check for reasonable sensor value ranges
        for column in data_buffer.columns:
            if column in ['temperature', 'mold_temperature']:
                # Temperature should be positive and reasonable for steel processing
                invalid_temp = data_buffer[(data_buffer[column] < 0) | (data_buffer[column] > 2000)]
                if not invalid_temp.empty:
                    issues.append(f"Temperature values out of range (0-2000) in column {column}: {len(invalid_temp)} rows")
            
            elif column in ['pressure']:
                # Pressure should be positive
                invalid_pressure = data_buffer[data_buffer[column] < 0]
                if not invalid_pressure.empty:
                    issues.append(f"Negative pressure values detected in column {column}: {len(invalid_pressure)} rows")
            
            elif column in ['flow_rate', 'cooling_water_flow']:
                # Flow rates should be positive
                invalid_flow = data_buffer[data_buffer[column] < 0]
                if not invalid_flow.empty:
                    issues.append(f"Negative flow rate values detected in column {column}: {len(invalid_flow)} rows")
        
        return issues

    def get_system_performance_metrics(self) -> Dict:
        """
        Calculates and returns current system performance metrics.
        
        Returns:
            A dictionary with metrics like 'avg_latency_ms' and 'throughput_preds_per_sec'.
        """
        metrics = {
            'avg_latency_ms': 0.0,
            'throughput_preds_per_sec': 0.0,
            'total_predictions': len(self.prediction_history),
            'high_risk_predictions': 0
        }
        
        if not self.latency_history:
            return metrics
        
        # Calculate average latency
        latencies = [entry['latency_ms'] for entry in self.latency_history]
        metrics['avg_latency_ms'] = np.mean(latencies) if latencies else 0.0
        
        # Calculate throughput (predictions per second)
        if len(self.prediction_history) >= 2:
            time_span = self.prediction_history[-1]['timestamp'] - self.prediction_history[0]['timestamp']
            if time_span > 0:
                metrics['throughput_preds_per_sec'] = len(self.prediction_history) / time_span
        
        # Count high risk predictions
        high_risk_threshold = self.thresholds.get('high_risk_threshold', 0.7)
        high_risk_count = sum(1 for pred in self.prediction_history 
                             if pred['prediction'] >= high_risk_threshold)
        metrics['high_risk_predictions'] = high_risk_count
        
        return metrics

    def _generate_alert(self, prediction_result: Dict):
        """
        Checks if a prediction warrants an alert and logs it.
        """
        prediction = prediction_result.get('ensemble_prediction', prediction_result.get('prediction', 0.0))
        confidence = prediction_result.get('confidence', 0.0)
        
        high_risk_threshold = self.thresholds.get('high_risk_threshold', 0.7)
        alert_threshold = self.thresholds.get('alert_threshold', 0.8)
        
        # Handle case where thresholds might be None
        if high_risk_threshold is None:
            high_risk_threshold = 0.7
        if alert_threshold is None:
            alert_threshold = 0.8
        
        if prediction >= alert_threshold:
            self.logger.error(
                f"CRITICAL ALERT: High defect probability detected! "
                f"Prediction: {prediction:.4f}, Confidence: {confidence:.4f}, "
                f"Threshold: {alert_threshold}"
            )
        elif prediction >= high_risk_threshold:
            self.logger.warning(
                f"HIGH RISK: Elevated defect probability detected. "
                f"Prediction: {prediction:.4f}, Confidence: {confidence:.4f}, "
                f"Threshold: {high_risk_threshold}"
            )