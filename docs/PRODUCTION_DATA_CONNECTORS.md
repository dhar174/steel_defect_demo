# Production Data Connectors

This module provides a flexible architecture for ingesting data from various industrial sources, enabling the steel defect detection system to work with real production data from OPC UA servers, MQTT brokers, REST APIs, and databases.

## Overview

The Production Data Connectors module implements a unified interface for connecting to different industrial data sources. It uses lazy imports to avoid hard dependencies on specialized libraries, making it suitable for deployment in various environments.

## Architecture

### Base Class: `BaseDataConnector`

All connectors inherit from `BaseDataConnector`, which provides a standardized interface:

```python
class BaseDataConnector(ABC):
    def connect(self) -> bool
    def read_data(self) -> Optional[pd.DataFrame]
    def disconnect(self) -> bool
    def is_connection_active(self) -> bool
```

### Connector Types

1. **OPCUAConnector** - Connects to OPC UA servers for real-time industrial data
2. **MQTTConnector** - Subscribes to MQTT brokers for IoT sensor data
3. **RESTConnector** - Polls REST APIs for web-based sensor data
4. **DatabaseConnector** - Queries SQL databases for historical data

## Installation

### Core Module (No additional dependencies)
The connectors module works out of the box with placeholder functionality:

```bash
# Already included in requirements.txt
pip install pandas numpy
```

### Production Dependencies (Optional)
For full functionality with real industrial systems:

```bash
# OPC UA support
pip install asyncua

# MQTT support  
pip install paho-mqtt

# Database support
pip install sqlalchemy

# Additional database drivers as needed
pip install psycopg2-binary  # PostgreSQL
pip install pymysql          # MySQL
```

## Usage Examples

### Basic Usage

```python
from src.connectors.data_connectors import OPCUAConnector, MQTTConnector

# OPC UA Configuration
opcua_config = {
    'server_url': 'opc.tcp://industrial-server:4840',
    'nodes': ['ns=2;i=1001', 'ns=2;i=1002'],
    'username': 'operator',
    'password': 'password'
}

opcua = OPCUAConnector(opcua_config)
if opcua.connect():
    data = opcua.read_data()  # Returns pandas DataFrame
    opcua.disconnect()
```

### MQTT Real-time Data

```python
mqtt_config = {
    'broker_host': 'mqtt.factory.com',
    'broker_port': 1883,
    'topic': 'sensors/+',
    'qos': 1
}

mqtt = MQTTConnector(mqtt_config)
if mqtt.connect():
    # Messages accumulate in background
    time.sleep(5)
    data = mqtt.read_data()  # Get accumulated messages
    mqtt.disconnect()
```

### REST API Polling

```python
rest_config = {
    'base_url': 'https://api.sensors.com',
    'endpoints': ['temperature', 'pressure'],
    'headers': {'Authorization': 'Bearer token'},
    'timeout': 10.0
}

rest = RESTConnector(rest_config)
if rest.connect():
    data = rest.read_data()  # Poll all endpoints
    rest.disconnect()
```

### Database Queries

```python
db_config = {
    'connection_string': 'postgresql://user:pass@host:5432/db',
    'query': 'SELECT * FROM sensors WHERE timestamp > NOW() - INTERVAL \'1 hour\''
}

db = DatabaseConnector(db_config)
if db.connect():
    data = db.read_data()  # Execute query
    db.disconnect()
```

## Integration with Inference System

The connectors are designed to integrate seamlessly with the existing `DefectPredictionEngine`:

```python
from src.connectors.data_connectors import OPCUAConnector
from src.inference.inference_engine import DefectPredictionEngine

# Set up data collection
connector = OPCUAConnector(config)
connector.connect()

# Set up inference
engine = DefectPredictionEngine('configs/inference_config.yaml')

# Real-time inference loop
while True:
    # Collect data
    sensor_data = connector.read_data()
    
    if sensor_data is not None:
        # Apply feature engineering
        features = feature_engineer.transform(sensor_data)
        
        # Run prediction
        predictions = engine.predict(features)
        
        # Handle results
        if predictions['defect_probability'] > threshold:
            send_alert(predictions)
    
    time.sleep(polling_interval)
```

## Configuration

### OPC UA Configuration

```yaml
opcua:
  server_url: "opc.tcp://steel-mill:4840"
  nodes:
    - "ns=2;i=1001"  # Temperature
    - "ns=2;i=1002"  # Pressure
    - "ns=2;i=1003"  # Flow rate
  sampling_interval: 1.0
  username: "operator"
  password: "secure_password"
```

### MQTT Configuration

```yaml
mqtt:
  broker_host: "mqtt.factory.com"
  broker_port: 1883
  topic: "production/sensors/+"
  qos: 1
  username: "mqtt_user"
  password: "mqtt_password"
  keep_alive: 60
```

### REST Configuration

```yaml
rest:
  base_url: "https://api.quality-system.com"
  endpoints:
    - "sensors/metallurgy"
    - "sensors/environmental"
  headers:
    Authorization: "Bearer your-token"
    Content-Type: "application/json"
  timeout: 10.0
  poll_interval: 5.0
```

### Database Configuration

```yaml
database:
  connection_string: "postgresql://user:password@host:5432/sensors"
  query: |
    SELECT timestamp, sensor_id, value 
    FROM sensor_readings 
    WHERE timestamp > NOW() - INTERVAL '1 hour'
    ORDER BY timestamp DESC
  poll_interval: 30.0
```

## Data Format

All connectors return data in a standardized pandas DataFrame format:

| Column | Description |
|--------|-------------|
| `timestamp` | Data collection timestamp |
| `sensor`/`node_id` | Sensor identifier |
| `value` | Sensor reading |
| `source` | Connector name (when using multiple sources) |

## Error Handling

The connectors implement comprehensive error handling:

- **Missing Libraries**: Graceful degradation with clear error messages
- **Connection Failures**: Proper error reporting and cleanup
- **Data Format Issues**: Robust parsing with fallbacks
- **Network Issues**: Timeout handling and retry logic

## Testing

Comprehensive unit tests are provided:

```bash
# Run connector tests
python -m pytest tests/test_connectors.py -v

# Run specific connector tests
python -m pytest tests/test_connectors.py::TestOPCUAConnector -v
```

## Demo Scripts

Several demo scripts are provided:

1. **`demo_connectors.py`** - Basic connector functionality demo
2. **`integration_example_connectors.py`** - Integration with inference pipeline

```bash
# Run basic demo
python demo_connectors.py

# Run integration example
python integration_example_connectors.py
```

## Production Deployment Considerations

### Security
- Use secure connection strings with encrypted passwords
- Implement proper authentication for all protocols
- Use TLS/SSL where available (MQTT over TLS, HTTPS APIs)

### Reliability
- Implement connection retry logic
- Add data buffering for offline periods
- Set up monitoring and alerting for connection status

### Performance
- Configure appropriate polling intervals
- Use connection pooling for database connectors
- Implement data filtering at source to reduce network traffic

### Monitoring
- Log connection status and data collection metrics
- Monitor data quality and freshness
- Set up alerts for missing or anomalous data

## Troubleshooting

### Common Issues

1. **Library Not Available**
   ```
   ERROR: OPC UA library not available. Install 'asyncua' for full functionality.
   ```
   Solution: Install the required library: `pip install asyncua`

2. **Connection Timeout**
   ```
   ERROR: Failed to connect to OPC UA server: timeout
   ```
   Solution: Check network connectivity and server availability

3. **Authentication Failed**
   ```
   ERROR: MQTT connection failed, return code: 5
   ```
   Solution: Verify username/password and broker settings

4. **Data Format Issues**
   ```
   WARNING: Could not parse JSON payload
   ```
   Solution: Check data format and implement custom parsing if needed

## Future Enhancements

Planned improvements for future versions:

1. **Additional Protocols**: Support for Modbus, EtherNet/IP
2. **Data Validation**: Schema validation for incoming data
3. **Caching**: Local data caching for offline operation
4. **Compression**: Data compression for network efficiency
5. **Encryption**: End-to-end data encryption
6. **Load Balancing**: Multiple connection support with failover