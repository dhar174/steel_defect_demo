# Real-time Monitoring

The Steel Defect Prediction System provides comprehensive real-time monitoring capabilities for continuous steel casting operations.

## Overview

Real-time monitoring allows operators to:

- Track live sensor data streams
- Monitor prediction confidence levels
- Detect anomalies in real-time
- Receive immediate alerts for potential defects
- Visualize process parameters and trends

## Dashboard Features

### Live Data Stream

The monitoring dashboard displays real-time data from multiple sensors:

```python
# Example sensor data structure
sensor_data = {
    "timestamp": "2024-01-15T10:30:00Z",
    "mold_temperature": 1525.4,
    "casting_speed": 1.12,
    "cooling_water_flow": 195.8,
    "oxygen_content": 0.025,
    "steel_grade": "304L"
}
```

### Real-time Metrics

Key metrics displayed in real-time:

- **Defect Probability**: Current prediction confidence (0-100%)
- **Process Stability**: Variance in key parameters
- **Quality Index**: Overall process quality score
- **Alert Status**: Current system alerts and warnings

### Visual Indicators

#### Status Indicators

- 🟢 **Green**: Normal operation (defect probability < 30%)
- 🟡 **Yellow**: Caution zone (defect probability 30-70%)
- 🔴 **Red**: High risk (defect probability > 70%)

#### Trend Charts

Real-time charts show:

- Temperature trends over time
- Casting speed variations
- Cooling system performance
- Prediction confidence levels

## Setting Up Real-time Monitoring

### 1. Configure Data Sources

```yaml
# config/monitoring.yml
data_sources:
  primary_sensors:
    - mold_temperature
    - casting_speed
    - cooling_water_flow
  secondary_sensors:
    - oxygen_content
    - steel_grade
    - tundish_temperature

update_frequency: 1  # seconds
buffer_size: 1000   # data points
```

### 2. Start Monitoring Service

```bash
# Start the monitoring service
python -m src.monitoring.real_time_monitor

# Or with Docker
docker run -d --name steel-monitor \
  -p 8001:8001 \
  steel-defect-prediction:latest \
  python -m src.monitoring.real_time_monitor
```

### 3. Access Monitoring Dashboard

Navigate to `http://localhost:8001/monitoring` to access the real-time dashboard.

## Monitoring Components

### Data Acquisition

```python
from src.monitoring.data_collector import RealTimeCollector

# Initialize collector
collector = RealTimeCollector(
    sensors=['mold_temp', 'casting_speed', 'cooling_flow'],
    update_interval=1.0  # seconds
)

# Start data collection
collector.start()

# Get latest data
latest_data = collector.get_latest()
```

### Prediction Engine

```python
from src.inference.real_time_predictor import RealTimePredictor

# Initialize predictor
predictor = RealTimePredictor(
    model_path='models/production_model.pth',
    threshold=0.7
)

# Process incoming data
prediction = predictor.predict(sensor_data)
print(f"Defect probability: {prediction['probability']:.3f}")
print(f"Confidence: {prediction['confidence']:.3f}")
```

### Alert System

```python
from src.monitoring.alert_manager import AlertManager

# Configure alerts
alert_manager = AlertManager(
    rules=[
        {
            'name': 'high_defect_risk',
            'condition': 'defect_probability > 0.8',
            'severity': 'critical',
            'notification': ['email', 'sms']
        },
        {
            'name': 'temperature_anomaly',
            'condition': 'abs(mold_temperature - target_temp) > 50',
            'severity': 'warning',
            'notification': ['dashboard']
        }
    ]
)

# Process alerts
alert_manager.evaluate(sensor_data, prediction)
```

## Configuration Options

### Update Frequency

Control how often data is refreshed:

```python
# High frequency for critical operations
MONITOR_UPDATE_INTERVAL = 0.5  # 500ms

# Standard frequency for normal operations
MONITOR_UPDATE_INTERVAL = 1.0  # 1 second

# Low frequency for overview monitoring
MONITOR_UPDATE_INTERVAL = 5.0  # 5 seconds
```

### Data Retention

Configure how long data is kept in memory:

```python
# Monitoring configuration
REALTIME_BUFFER_SIZE = 3600    # 1 hour of data at 1Hz
HISTORY_RETENTION_DAYS = 7     # Keep 7 days of detailed history
ARCHIVE_RETENTION_MONTHS = 12  # Keep 12 months of summary data
```

### Display Preferences

Customize dashboard appearance:

```javascript
// Dashboard configuration
const dashboardConfig = {
    refreshRate: 1000,           // Refresh every second
    chartTimeWindow: 300,        // Show last 5 minutes
    alertTimeout: 30000,         // Alert timeout: 30 seconds
    theme: 'dark',              // 'light' or 'dark'
    showPredictionBands: true,   // Show confidence intervals
    enableSounds: true          // Audio alerts
};
```

## Alert Configuration

### Alert Levels

Configure different alert severity levels:

```yaml
alert_levels:
  info:
    color: blue
    sound: false
    auto_dismiss: true
    timeout: 10
  
  warning:
    color: orange
    sound: true
    auto_dismiss: false
    timeout: 30
  
  critical:
    color: red
    sound: true
    auto_dismiss: false
    requires_acknowledgment: true
```

### Notification Channels

Set up multiple notification channels:

```yaml
notifications:
  email:
    enabled: true
    smtp_server: smtp.company.com
    recipients:
      - operator@company.com
      - supervisor@company.com
  
  sms:
    enabled: true
    provider: twilio
    numbers:
      - "+1234567890"
  
  webhook:
    enabled: true
    url: https://api.company.com/alerts
    headers:
      Authorization: "Bearer ${API_TOKEN}"
```

## Performance Optimization

### Efficient Data Handling

```python
# Use efficient data structures
import collections
import numpy as np

class CircularBuffer:
    def __init__(self, size):
        self.buffer = collections.deque(maxlen=size)
    
    def add(self, item):
        self.buffer.append(item)
    
    def get_array(self):
        return np.array(self.buffer)

# Batch processing for efficiency
def process_sensor_batch(sensor_batch):
    """Process multiple sensor readings at once"""
    predictions = model.predict_batch(sensor_batch)
    return predictions
```

### Memory Management

```python
# Limit memory usage
MAX_MEMORY_MB = 512

# Monitor memory usage
import psutil
import gc

def check_memory():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    if memory_mb > MAX_MEMORY_MB:
        gc.collect()  # Force garbage collection
        
    return memory_mb
```

## Troubleshooting

### Common Issues

1. **Data lag**: Check network connection and sensor communication
2. **High CPU usage**: Reduce update frequency or optimize prediction pipeline
3. **Memory leaks**: Monitor buffer sizes and implement proper cleanup
4. **Missing alerts**: Verify alert configuration and notification settings

### Debugging Tools

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor performance
from src.monitoring.performance_monitor import PerformanceMonitor

perf_monitor = PerformanceMonitor()
perf_monitor.start()

# Check metrics
metrics = perf_monitor.get_metrics()
print(f"Average processing time: {metrics['avg_processing_time']:.3f}s")
print(f"Predictions per second: {metrics['predictions_per_second']:.1f}")
```

### System Status

Check real-time monitoring system status:

```bash
# Check service status
systemctl status steel-defect-monitor

# View service logs
journalctl -u steel-defect-monitor -f

# Check resource usage
htop
```

## Integration Examples

### SCADA Integration

```python
# Connect to SCADA system
from src.integrations.scada_connector import SCADAConnector

scada = SCADAConnector(
    host="scada.plant.com",
    port=502,
    unit_id=1
)

# Read sensor values
sensor_values = scada.read_sensors([
    'mold_temp_tag',
    'casting_speed_tag',
    'cooling_flow_tag'
])
```

### PLC Integration

```python
# Connect to PLC
from src.integrations.plc_connector import PLCConnector

plc = PLCConnector(
    ip_address="192.168.1.100",
    rack=0,
    slot=1
)

# Read process data
process_data = plc.read_data_block(
    db_number=1,
    start_address=0,
    size=100
)
```

This real-time monitoring system provides comprehensive visibility into your steel casting operations, enabling proactive quality management and immediate response to potential defects.