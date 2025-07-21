# Real-Time Sensor Monitoring Components

## Overview

The `SensorMonitoringComponent` provides a comprehensive real-time monitoring solution for steel casting process sensors. This component implements all the requirements specified in issue #94, including multi-sensor visualization, real-time streaming, anomaly detection, and health monitoring.

## Features

### ✅ Multi-Sensor Time Series Visualization
- **Synchronized Cross-Filtering**: All sensor plots share synchronized x-axis for comparative analysis
- **6 Key Sensors Monitored**:
  - Casting Speed (m/min)
  - Mold Temperature (°C)
  - Mold Level (mm)
  - Strand Temperature (°C)
  - Cooling Water Flow (L/min)
  - Tundish Temperature (°C)
- **Individual Subplot Design**: Each sensor gets its own subplot with appropriate scaling

### ✅ Real-Time Streaming Plot
- **Rolling Window Buffer**: Uses `collections.deque` with configurable size (default: 1000 points)
- **Automatic Memory Management**: Efficiently handles continuous data streams
- **Configurable Update Intervals**: 1s, 5s, 10s options via dashboard controls
- **Data Point Alignment**: Maintains temporal alignment across all sensors

### ✅ Configurable Time Range and Auto-Scaling
- **Time Range Controls**: 5min, 15min, 30min, 1hr time windows
- **Auto-Scaling Options**:
  - Smart auto-scaling with 10% padding
  - Threshold-based manual scaling
  - Statistical-based scaling using sensor data distribution
- **User Controls**: Interactive buttons for real-time configuration changes

### ✅ Anomaly and Threshold Highlighting
- **Visual Threshold Lines**:
  - Critical thresholds: Red solid lines
  - Warning thresholds: Orange dashed lines
- **Multi-Method Anomaly Detection**:
  - Threshold-based detection (configurable per sensor)
  - Statistical outlier detection (IQR method)
  - Rate-of-change detection for sudden spikes
- **Visual Highlighting**: Red X markers for detected anomalies
- **Real-time Processing**: Anomalies detected and displayed instantly

### ✅ Sensor Health and Data Quality Indicators
- **Color-Coded Status Badges**:
  - 🟢 Good: Normal operation
  - 🟡 Warning: Minor issues detected
  - 🔴 Critical: Serious problems
  - ⚪ Offline: No data received
  - 🔵 Stale: Data older than 30 seconds
- **Data Quality Metrics**:
  - Variance analysis for stuck sensors
  - Update frequency monitoring
  - Missing data detection
- **Health Status Bar**: Visual summary of all sensor states

### ✅ Interactive Tooltips
- **Rich Hover Information**:
  - Precise timestamp
  - Sensor reading with 3 decimal precision
  - Current health status
  - Last update time
  - Anomaly indicators
- **Context-Aware Content**: Different tooltip content for normal vs. anomaly data points
- **Cross-Plot Consistency**: Unified tooltip format across all visualizations

## API Reference

### SensorMonitoringComponent

```python
class SensorMonitoringComponent:
    def __init__(self, 
                 component_id: str = "sensor-monitoring",
                 buffer_size: int = 1000,
                 sensors: List[str] = None,
                 thresholds: Dict[str, Dict[str, float]] = None,
                 update_interval: int = 1000):
```

#### Parameters
- `component_id`: Unique identifier for the component
- `buffer_size`: Maximum number of data points in rolling window
- `sensors`: List of sensor names to monitor
- `thresholds`: Dictionary of sensor thresholds
- `update_interval`: Update interval in milliseconds

#### Key Methods

##### Data Management
```python
def add_data_point(self, sensor_data: Dict[str, float], timestamp: datetime = None)
def get_current_data(self) -> Tuple[Dict[str, List], List[datetime]]
def generate_mock_data_point(self) -> Dict[str, float]
```

##### Visualization
```python
def create_layout(self) -> html.Div
def create_multi_sensor_plot(self, data: Dict, timestamps: List, config: Dict) -> go.Figure
def create_detail_plots(self, data: Dict, timestamps: List, config: Dict) -> go.Figure
```

##### Health Monitoring
```python
def update_sensor_health(self, data: Dict, timestamps: List) -> Dict[str, str]
def create_health_indicators(self, health_status: Dict) -> html.Div
```

##### Anomaly Detection
```python
def _detect_anomalies(self, sensor_data: List[float], sensor_name: str) -> List[int]
```

## Usage Examples

### Basic Usage

```python
from src.visualization.components.sensor_monitoring import SensorMonitoringComponent

# Create component
sensor_monitor = SensorMonitoringComponent(
    component_id="my-sensor-monitor",
    buffer_size=500,
    update_interval=2000
)

# Generate layout for Dash app
layout = sensor_monitor.create_layout()

# Add real-time data
sensor_data = {
    'casting_speed': 1.2,
    'mold_temperature': 1530,
    'mold_level': 150
}
sensor_monitor.add_data_point(sensor_data)
```

### Integration with Existing Dashboard

```python
# In dashboard.py __init__ method:
self.sensor_monitor = SensorMonitoringComponent(
    component_id="dashboard-sensor-monitor",
    buffer_size=1000,
    update_interval=self.refresh_interval
)

# In callback function:
@self.app.callback(...)
def update_sensor_monitoring(n_intervals, config, pathname):
    # Get new sensor data from your data source
    new_data = get_real_sensor_data()  # Your data source
    self.sensor_monitor.add_data_point(new_data)
    
    # Update visualizations
    data, timestamps = self.sensor_monitor.get_current_data()
    health_status = self.sensor_monitor.update_sensor_health(data, timestamps)
    
    main_plot = self.sensor_monitor.create_multi_sensor_plot(data, timestamps, config)
    detail_plots = self.sensor_monitor.create_detail_plots(data, timestamps, config)
    health_indicators = self.sensor_monitor.create_health_indicators(health_status)
    
    return main_plot, detail_plots, health_indicators
```

### Custom Threshold Configuration

```python
custom_thresholds = {
    'casting_speed': {
        'min': 0.8, 'max': 1.5,
        'warning_min': 0.9, 'warning_max': 1.4
    },
    'mold_temperature': {
        'min': 1480, 'max': 1580,
        'warning_min': 1500, 'warning_max': 1560
    }
}

component = SensorMonitoringComponent(thresholds=custom_thresholds)
```

## File Structure

```
src/visualization/components/
├── __init__.py                 # Package initialization
└── sensor_monitoring.py       # Main component implementation

# Test and example files
test_sensor_monitoring.py       # Standalone test application
test_sensor_monitoring_unit.py  # Unit tests
integration_sensor_monitoring.py # Integration examples
```

## Dependencies

- `plotly>=5.15.0` - Interactive plotting
- `dash>=3.0.0` - Web framework
- `dash-bootstrap-components>=2.0.0` - UI components
- `pandas>=1.5.0` - Data handling
- `numpy>=1.23.0` - Numerical computations

## Testing

### Running the Test Application

```bash
cd /path/to/steel_defect_demo
python test_sensor_monitoring.py
# Navigate to http://127.0.0.1:8050
```

### Running Unit Tests

```bash
python test_sensor_monitoring_unit.py
```

### Integration Testing

```bash
python integration_sensor_monitoring.py
```

## Performance Characteristics

- **Memory Efficient**: Rolling window buffer prevents unlimited memory growth
- **Real-time Capable**: Handles 1Hz update rates smoothly
- **Scalable**: Tested with 1000+ data points without performance degradation
- **Responsive UI**: Interactive controls update visualizations instantly

## Configuration Options

### Display Options (User Configurable)
- ☑️ Auto-scale Y-axis
- ☑️ Show Thresholds
- ☑️ Highlight Anomalies

### Time Range Options
- 5 minutes
- 15 minutes  
- 30 minutes (default)
- 1 hour

### Update Rate Options
- 1 second (default)
- 5 seconds
- 10 seconds

### Stream Control Options
- ⏸️ Pause updates
- 🔄 Reset data buffer

## Screenshot

![Sensor Monitoring Dashboard](https://github.com/user-attachments/assets/e8bf58ce-1d2c-4261-ac79-110e52553c47)

The screenshot shows the complete working dashboard with:
1. **Control Panel** - Time range, display options, update controls
2. **Health Status Bar** - Color-coded sensor health indicators
3. **Main Sensor Plot** - Real-time synchronized multi-sensor visualization
4. **Detail Plots** - Individual sensor analysis with statistical overlays

## Future Enhancements

Potential improvements for future versions:
- WebSocket integration for real-time data streaming
- Export functionality for plots and data
- Advanced anomaly detection with machine learning
- Sensor correlation analysis
- Historical data comparison overlays
- Custom alert configuration per sensor
- Mobile-responsive design optimizations

## Support

For questions or issues with the sensor monitoring component:
1. Check the integration examples in `integration_sensor_monitoring.py`
2. Run the test application for troubleshooting
3. Review the unit tests for expected behavior
4. Refer to existing dashboard.py for integration patterns