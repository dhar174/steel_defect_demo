"""
Example integration of StreamAnalyticsEngine with data pipeline.
Shows how the engine could be integrated into a real-time monitoring system.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from analytics.stream_analytics import StreamAnalyticsEngine


class MockDataStream:
    """Simulates a streaming data source."""
    
    def __init__(self, anomaly_probability=0.05, trend_rate=0.1):
        self.anomaly_probability = anomaly_probability
        self.trend_rate = trend_rate
        self.time_step = 0
        
    def get_next_batch(self, batch_size=20):
        """Generate next batch of sensor data."""
        np.random.seed(self.time_step)
        
        # Base sensor readings
        data = {
            'temperature': np.random.normal(100 + self.time_step * self.trend_rate, 5, batch_size),
            'pressure': np.random.normal(200, 10, batch_size),
            'vibration': np.random.normal(50, 3, batch_size),
            'flow_rate': np.random.normal(75, 8, batch_size),
        }
        
        # Inject anomalies
        for sensor in data:
            for i in range(batch_size):
                if np.random.random() < self.anomaly_probability:
                    # Create outlier
                    data[sensor][i] *= np.random.uniform(2.0, 3.0)
        
        # Add timestamp
        start_time = self.start_date + pd.Timedelta(minutes=self.time_step * 5)
        data['timestamp'] = pd.date_range(start_time, periods=batch_size, freq='5min')
        
        self.time_step += 1
        return pd.DataFrame(data)


class RealTimeMonitor:
    """Real-time monitoring system using StreamAnalyticsEngine."""
    
    def __init__(self):
        self.config = {
            'spc_sigma_threshold': 2.5,  # More sensitive
            'trend_min_points': 15,
            'trend_significance_level': 0.01  # More stringent
        }
        self.analytics_engine = StreamAnalyticsEngine(self.config)
        self.data_buffer = pd.DataFrame()
        self.buffer_size = 100  # Keep rolling window of 100 points
        
    def process_new_data(self, new_data):
        """Process new data batch through analytics engine."""
        # Add to buffer
        self.data_buffer = pd.concat([self.data_buffer, new_data], ignore_index=True)
        
        # Maintain buffer size
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer = self.data_buffer.tail(self.buffer_size)
        
        # Run analytics
        analytics_result = self.analytics_engine.update_with_new_data(self.data_buffer)
        
        return analytics_result
    
    def generate_alerts(self, analytics_result):
        """Generate alerts based on analytics results."""
        alerts = []
        
        # SPC alerts
        for sensor, violation in analytics_result['spc_violations'].items():
            alerts.append({
                'type': 'SPC_VIOLATION',
                'sensor': sensor,
                'description': violation,
                'severity': 'HIGH'
            })
        
        # Trend alerts
        for sensor, trend in analytics_result['trends'].items():
            alerts.append({
                'type': 'TREND_DETECTED',
                'sensor': sensor,
                'description': trend,
                'severity': 'MEDIUM'
            })
        
        # Anomaly alerts
        if analytics_result['anomalies']:
            alerts.append({
                'type': 'ANOMALIES_DETECTED',
                'count': len(analytics_result['anomalies']),
                'description': f"{len(analytics_result['anomalies'])} anomalous data points detected",
                'severity': 'MEDIUM'
            })
        
        return alerts


def main():
    """Demonstrate real-time monitoring integration."""
    print("Real-Time Monitoring Integration Demo")
    print("=" * 50)
    
    # Initialize components
    data_stream = MockDataStream(anomaly_probability=0.1, trend_rate=0.2)
    monitor = RealTimeMonitor()
    
    print("Starting real-time monitoring simulation...")
    print("Processing batches of sensor data...")
    print()
    
    # Simulate real-time processing
    for iteration in range(10):
        print(f"Batch {iteration + 1}:")
        
        # Get new data
        new_data = data_stream.get_next_batch(batch_size=15)
        
        # Process through analytics
        analytics_result = monitor.process_new_data(new_data)
        
        # Generate alerts
        alerts = monitor.generate_alerts(analytics_result)
        
        # Display results
        print(f"  Buffer size: {len(monitor.data_buffer)} points")
        print(f"  Analytics summary: {analytics_result['summary']}")
        
        if alerts:
            print("  🚨 ALERTS:")
            for alert in alerts:
                print(f"    {alert['type']} [{alert['severity']}]: {alert['description']}")
        else:
            print("  ✅ No alerts")
        
        print()
        
        # Simulate processing delay
        time.sleep(0.5)
    
    print("Real-time monitoring simulation completed!")
    print("\nFinal buffer statistics:")
    print(f"Total data points processed: {len(monitor.data_buffer)}")
    
    # Show buffer summary
    numeric_cols = monitor.data_buffer.select_dtypes(include=[np.number]).columns
    print("\nSensor value ranges in buffer:")
    for col in numeric_cols:
        values = monitor.data_buffer[col]
        print(f"  {col}: {values.min():.2f} - {values.max():.2f} (mean: {values.mean():.2f})")


if __name__ == '__main__':
    main()