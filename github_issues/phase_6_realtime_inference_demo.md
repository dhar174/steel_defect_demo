# Phase 6: Real-Time Inference Demo

## Description

Implement a real-time streaming inference demonstration that simulates live continuous casting data processing. This system will replay cast data in pseudo real-time, perform incremental feature computation and model inference, and provide live monitoring capabilities including optional dashboard visualization.

## Context

This phase demonstrates the practical deployment capabilities of the predictive quality monitoring system by creating a realistic streaming inference environment. It validates that the trained models can operate effectively in a real-time context and provides the foundation for production deployment with actual plant data feeds.

## Objectives

- Create realistic streaming data simulation from historical cast records
- Implement incremental feature computation and model inference pipelines
- Develop real-time monitoring and alerting capabilities
- Build optional live dashboard for operator visualization
- Validate end-to-end system performance under streaming conditions

## Acceptance Criteria

### Streaming Data Simulation (`simulate_stream.py`)
- [ ] **Cast replay functionality**: Stream historical casts at configurable real-time factors
- [ ] **Data buffering**: Maintain sliding windows for feature computation
- [ ] **Timing accuracy**: Precise timestamp control for realistic simulation
- [ ] **Multiple cast support**: Concurrent streaming from multiple historical casts
- [ ] **Pause/resume/reset controls**: Interactive simulation management

### Real-Time Inference Engine
- [ ] **Incremental feature computation**: Rolling window feature updates
- [ ] **Model inference pipeline**: Both baseline and sequence model support
- [ ] **Prediction buffering**: Store and track prediction history
- [ ] **Latency monitoring**: Track processing time and system performance
- [ ] **Configurable inference intervals**: Adjustable prediction frequency (e.g., every 30 seconds)

### Monitoring and Alerting System
- [ ] **Risk score tracking**: Continuous defect probability monitoring
- [ ] **Threshold-based alerts**: Configurable warning and critical thresholds
- [ ] **Alert management**: Alert generation, escalation, and acknowledgment
- [ ] **Logging infrastructure**: Structured logging of all inference events
- [ ] **Performance metrics**: System health and processing performance tracking

### Optional Live Dashboard
- [ ] **Real-time visualization**: Live sensor data streams and risk curves
- [ ] **Interactive controls**: Start/stop simulation, adjust parameters
- [ ] **Alert display**: Visual alert notifications and history
- [ ] **Model comparison**: Side-by-side baseline vs. sequence model results
- [ ] **Historical overlay**: Compare current patterns with historical data

### Configuration and Deployment
- [ ] **Configuration management**: YAML-based system configuration
- [ ] **Containerization**: Docker setup for portable deployment
- [ ] **Scalability considerations**: Architecture supporting multiple strands
- [ ] **Error handling**: Robust handling of data and model errors

## Implementation Tasks

### Streaming Data Infrastructure

#### Cast Replay Engine
```python
class CastStreamer:
    def __init__(self, cast_data, realtime_factor=1.0, buffer_size=300):
        """
        Stream historical cast data in real-time
        
        Args:
            cast_data: Historical cast time series
            realtime_factor: Speed multiplier (1.0 = real-time)
            buffer_size: Seconds of data to maintain in buffer
        """
        self.cast_data = cast_data
        self.realtime_factor = realtime_factor
        self.buffer = collections.deque(maxlen=buffer_size)
        self.start_time = None
        self.current_position = 0
        
    def start_streaming(self):
        # Initialize streaming state
        # Start real-time data emission
        
    def get_next_reading(self):
        # Emit next sensor reading with accurate timing
        # Update internal buffer
        # Return timestamped sensor data
        
    def get_buffer_data(self, window_seconds=60):
        # Return last N seconds of buffered data
        # For feature computation and model inference
```

#### Data Buffer Management
```python
class SensorDataBuffer:
    def __init__(self, sensors, max_duration=600):
        """Circular buffer for streaming sensor data"""
        self.sensors = sensors
        self.max_duration = max_duration
        self.data = {sensor: collections.deque() for sensor in sensors}
        self.timestamps = collections.deque()
        
    def add_reading(self, timestamp, sensor_data):
        # Add new reading to buffer
        # Maintain time window limits
        # Clean up old data
        
    def get_window(self, duration_seconds=60):
        # Extract time window for analysis
        # Return DataFrame for feature computation
        
    def get_latest_values(self):
        # Get most recent sensor readings
        # For real-time display
```

### Real-Time Inference Pipeline

#### Incremental Feature Computer
```python
class IncrementalFeatureComputer:
    def __init__(self, feature_config, baseline_scaler):
        self.feature_config = feature_config
        self.baseline_scaler = baseline_scaler
        self.feature_cache = {}
        
    def update_features(self, buffer_data, timestamp):
        """Compute features from current buffer window"""
        # Extract statistical features
        # Compute stability indicators
        # Calculate physics-based metrics
        # Apply normalization
        return feature_vector
        
    def get_sequence_data(self, buffer_data, sequence_length=100):
        """Prepare sequence data for deep model"""
        # Extract and normalize sequence
        # Handle variable length padding
        # Return tensor ready for inference
```

#### Real-Time Model Inference
```python
class RealTimeInferenceEngine:
    def __init__(self, baseline_model, sequence_model, config):
        self.baseline_model = baseline_model
        self.sequence_model = sequence_model
        self.config = config
        self.prediction_history = []
        
    def predict_baseline(self, features):
        # Baseline model inference
        # Return probability and feature importance
        
    def predict_sequence(self, sequence_data):
        # Sequence model inference
        # Return probability and attention weights
        
    def predict_ensemble(self, features, sequence_data):
        # Combined prediction from both models
        # Weighted ensemble or voting
        
    def log_prediction(self, timestamp, predictions, metadata):
        # Store prediction with context
        # Update prediction history buffer
```

### Monitoring and Alerting Framework

#### Alert Management System
```python
class AlertManager:
    def __init__(self, config):
        self.thresholds = config['alert_thresholds']
        self.active_alerts = {}
        self.alert_history = []
        
    def check_alerts(self, timestamp, risk_score, model_type):
        """Evaluate current risk against thresholds"""
        alert_level = self.get_alert_level(risk_score)
        
        if alert_level:
            alert = {
                'timestamp': timestamp,
                'level': alert_level,
                'risk_score': risk_score,
                'model': model_type,
                'message': self.generate_alert_message(alert_level, risk_score)
            }
            self.trigger_alert(alert)
            
    def trigger_alert(self, alert):
        # Generate alert notification
        # Log to alert history
        # Trigger external notifications (email, SMS, etc.)
        
    def acknowledge_alert(self, alert_id):
        # Mark alert as acknowledged
        # Update alert status
```

#### Performance Monitoring
```python
class SystemMonitor:
    def __init__(self):
        self.metrics = {
            'inference_latency': [],
            'processing_rate': [],
            'memory_usage': [],
            'error_count': 0
        }
        
    def log_inference_time(self, start_time, end_time):
        latency = end_time - start_time
        self.metrics['inference_latency'].append(latency)
        
    def log_processing_rate(self, readings_per_second):
        self.metrics['processing_rate'].append(readings_per_second)
        
    def get_system_health(self):
        # Return current system performance metrics
        # Check for performance degradation
```

### Live Dashboard Implementation (Optional)

#### Dash/Plotly Dashboard
```python
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

class LiveDashboard:
    def __init__(self, inference_engine, data_buffer):
        self.app = dash.Dash(__name__)
        self.inference_engine = inference_engine
        self.data_buffer = data_buffer
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Create dashboard layout with live charts"""
        self.app.layout = html.Div([
            # Real-time sensor displays
            dcc.Graph(id='live-sensors'),
            
            # Risk score tracking
            dcc.Graph(id='risk-timeline'),
            
            # Model comparison
            dcc.Graph(id='model-comparison'),
            
            # Alert panel
            html.Div(id='alert-panel'),
            
            # Control panel
            html.Div([
                html.Button('Start', id='start-btn'),
                html.Button('Stop', id='stop-btn'),
                html.Button('Reset', id='reset-btn'),
            ]),
            
            # Auto-refresh interval
            dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
        ])
        
    def setup_callbacks(self):
        """Define interactive callback functions"""
        @self.app.callback(
            Output('live-sensors', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_sensor_display(n):
            # Get latest sensor data
            # Create real-time line plots
            # Return updated figure
            
        @self.app.callback(
            Output('risk-timeline', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_risk_display(n):
            # Get prediction history
            # Plot risk score over time
            # Add alert threshold lines
```

#### WebSocket Real-Time Updates (Alternative)
```python
import asyncio
import websockets
import json

class WebSocketDashboard:
    def __init__(self, inference_engine):
        self.inference_engine = inference_engine
        self.connected_clients = set()
        
    async def register_client(self, websocket, path):
        self.connected_clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.connected_clients.remove(websocket)
            
    async def broadcast_update(self, data):
        """Send real-time updates to all connected clients"""
        if self.connected_clients:
            message = json.dumps(data)
            await asyncio.gather(
                *[client.send(message) for client in self.connected_clients],
                return_exceptions=True
            )
```

### Configuration and Deployment

#### System Configuration
```yaml
# config/realtime_inference.yaml
streaming:
  realtime_factor: 1.0  # 1.0 = real-time, 10.0 = 10x speed
  buffer_duration: 300  # seconds
  inference_interval: 30  # seconds between predictions

models:
  baseline:
    path: "models/baseline/model.pkl"
    enabled: true
  sequence:
    path: "models/sequence/model.onnx"
    enabled: true
  ensemble:
    enabled: true
    weights: [0.4, 0.6]  # baseline, sequence

alerts:
  thresholds:
    warning: 0.7
    critical: 0.85
  notification:
    email: true
    dashboard: true
    log_file: "logs/alerts.log"

dashboard:
  enabled: true
  port: 8050
  update_interval: 1000  # milliseconds

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/inference.log"
```

#### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

EXPOSE 8050

CMD ["python", "src/realtime_demo.py"]
```

### Main Demo Application

#### Orchestration Script
```python
class RealTimeDemo:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.setup_components()
        
    def setup_components(self):
        # Initialize data streamer
        # Load trained models
        # Create inference engine
        # Setup monitoring and alerts
        # Initialize dashboard (if enabled)
        
    def run_demo(self, cast_id=None, duration=None):
        """Run complete real-time demonstration"""
        # Start data streaming
        # Begin inference loop
        # Monitor performance
        # Handle graceful shutdown
        
    async def inference_loop(self):
        """Main real-time processing loop"""
        while self.running:
            # Get new sensor readings
            # Update data buffer
            # Check if inference interval reached
            # Compute features and run inference
            # Check alerts and update displays
            # Log performance metrics
            await asyncio.sleep(0.1)  # 100ms processing cycle
```

## Dependencies

- **Prerequisite**: Phase 4 (Baseline Model) and Phase 5 (Sequence Model) complete
- **Models**: Trained baseline and sequence models available
- **Data**: Historical cast data for streaming simulation
- **Optional**: Web dashboard dependencies (Dash, WebSocket libraries)

## Expected Deliverables

1. **Streaming Infrastructure**: `src/realtime/`
   - `cast_streamer.py`: Data streaming simulation
   - `data_buffer.py`: Real-time data buffering
   - `feature_computer.py`: Incremental feature computation

2. **Inference Engine**: `src/inference/`
   - `realtime_inference.py`: Model inference pipeline
   - `alert_manager.py`: Alert and monitoring system
   - `performance_monitor.py`: System performance tracking

3. **Dashboard (Optional)**: `src/dashboard/`
   - `live_dashboard.py`: Real-time visualization interface
   - `websocket_server.py`: WebSocket-based updates
   - `static/`: HTML/CSS/JS assets

4. **Demo Application**: `src/realtime_demo.py`
   - Main orchestration script
   - Command-line interface
   - Configuration management

5. **Documentation**: `docs/realtime_demo.md`
   - Usage instructions
   - Configuration options
   - Performance considerations

## Technical Considerations

### Performance Requirements
- **Latency**: < 1 second for inference pipeline
- **Throughput**: Handle 1 Hz sensor data with 15-20 sensors
- **Memory**: Efficient buffer management for continuous operation
- **CPU**: Optimized feature computation and model inference

### Reliability and Error Handling
- **Data validation**: Check for missing or invalid sensor readings
- **Model fallback**: Graceful degradation if models fail
- **Connection handling**: Robust streaming and dashboard connections
- **Recovery**: Automatic restart and state recovery capabilities

### Scalability Considerations
- **Multi-strand support**: Architecture for multiple concurrent casters
- **Horizontal scaling**: Microservice-ready component design
- **Resource management**: Configurable resource limits and monitoring

## Success Metrics

- [ ] **Streaming accuracy**: Maintain precise timing within 1% error
- [ ] **Inference latency**: < 500ms for complete inference pipeline
- [ ] **System uptime**: Run continuously for 1+ hours without errors
- [ ] **Dashboard responsiveness**: < 2 second update latency
- [ ] **Alert reliability**: 100% alert generation for threshold breaches
- [ ] **Resource usage**: Stable memory consumption, reasonable CPU usage

## Demo Scenarios

### Standard Demo Flow
1. **Initialization**: Load models and configure system
2. **Normal operation**: Stream normal casting data, show stable predictions
3. **Defect scenario**: Stream cast with developing defect, show escalating risk
4. **Alert demonstration**: Show alert generation and operator notification
5. **Model comparison**: Demonstrate differences between baseline and sequence models

### Interactive Features
- **Speed control**: Adjust real-time factor for demonstration purposes
- **Parameter tuning**: Live adjustment of alert thresholds
- **Model toggling**: Enable/disable different model types
- **Historical comparison**: Overlay current data with historical patterns

## Notes

This phase demonstrates the complete end-to-end capability of the predictive quality monitoring system. Focus on:

1. **Realistic simulation**: Create believable real-time environment
2. **Robust implementation**: Handle edge cases and errors gracefully
3. **Clear visualization**: Make predictions and alerts easily understandable
4. **Performance validation**: Verify system meets real-time requirements
5. **Production readiness**: Architecture that scales to real deployment

The demo should convincingly show how the system would operate in a production environment while highlighting the value of predictive defect detection for continuous casting operations.

## Labels
`enhancement`, `phase-6`, `real-time`, `inference`, `demo`, `visualization`

## Priority
**High** - Critical demonstration of system capabilities and production readiness