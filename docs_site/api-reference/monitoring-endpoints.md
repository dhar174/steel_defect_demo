# Monitoring Endpoints

The Monitoring API provides system health, performance metrics, and operational monitoring capabilities for the Steel Defect Prediction System.

## Base URL

```
http://localhost:8000/api/v1/monitoring
```

## Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/monitoring/health` | System health check |
| GET | `/monitoring/metrics` | Performance metrics |
| GET | `/monitoring/alerts` | Active system alerts |
| GET | `/monitoring/status` | Overall system status |
| POST | `/monitoring/alerts/acknowledge` | Acknowledge alerts |

## Health Check

**GET** `/api/v1/monitoring/health`

Basic health check endpoint for load balancers and monitoring systems.

### Response

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "2.1.0",
  "uptime_seconds": 86400,
  "components": {
    "database": "healthy",
    "prediction_engine": "healthy", 
    "cache": "healthy",
    "message_queue": "healthy"
  }
}
```

### Status Codes

- `200 OK`: System is healthy
- `503 Service Unavailable`: System is unhealthy

## Detailed Health

**GET** `/api/v1/monitoring/health/detailed`

Comprehensive health information including component details.

### Response

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "overall_score": 0.98,
  "components": {
    "database": {
      "status": "healthy",
      "response_time_ms": 15,
      "connection_pool": {
        "active": 5,
        "idle": 15,
        "max": 20
      },
      "last_check": "2024-01-15T10:29:55Z"
    },
    "prediction_engine": {
      "status": "healthy",
      "model_version": "v2.1.0",
      "avg_prediction_time_ms": 45,
      "predictions_per_second": 22.5,
      "gpu_utilization": 0.65,
      "memory_usage_mb": 2048
    },
    "cache": {
      "status": "healthy",
      "hit_rate": 0.89,
      "memory_usage_mb": 512,
      "evictions_per_hour": 150
    },
    "message_queue": {
      "status": "healthy",
      "queue_depth": 23,
      "messages_per_second": 45.2,
      "consumer_lag_ms": 120
    }
  },
  "dependencies": {
    "external_sensors": {
      "LINE_01": "connected",
      "LINE_02": "connected",
      "LINE_03": "disconnected"
    },
    "scada_system": "connected",
    "notification_service": "connected"
  }
}
```

## Performance Metrics

**GET** `/api/v1/monitoring/metrics`

Retrieve system performance metrics in Prometheus format.

### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `format` | string | Response format (prometheus, json) |
| `category` | string | Metric category filter |
| `time_range` | string | Time range for metrics |

### Response (JSON Format)

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "metrics": {
    "system": {
      "cpu_usage_percent": 45.2,
      "memory_usage_percent": 67.8,
      "disk_usage_percent": 34.1,
      "network_io_mbps": 125.6
    },
    "application": {
      "total_requests": 150245,
      "requests_per_second": 42.5,
      "average_response_time_ms": 89,
      "error_rate_percent": 0.12,
      "active_connections": 156
    },
    "predictions": {
      "total_predictions": 25689,
      "predictions_per_hour": 1250,
      "average_confidence": 0.87,
      "high_risk_predictions": 256,
      "model_accuracy": 0.923
    },
    "data_quality": {
      "sensor_uptime_percent": 98.5,
      "data_completeness_percent": 97.2,
      "anomaly_rate_percent": 2.1,
      "validation_errors_per_hour": 5
    }
  }
}
```

### Response (Prometheus Format)

```
# HELP steel_predictions_total Total number of predictions made
# TYPE steel_predictions_total counter
steel_predictions_total 25689

# HELP steel_prediction_accuracy Current model accuracy
# TYPE steel_prediction_accuracy gauge  
steel_prediction_accuracy 0.923

# HELP steel_response_time_seconds Response time in seconds
# TYPE steel_response_time_seconds histogram
steel_response_time_seconds_bucket{le="0.1"} 12450
steel_response_time_seconds_bucket{le="0.5"} 24890
steel_response_time_seconds_bucket{le="1.0"} 25689
```

## System Status

**GET** `/api/v1/monitoring/status`

Overall system operational status with summary information.

### Response

```json
{
  "overall_status": "operational",
  "status_code": 200,
  "last_updated": "2024-01-15T10:30:00Z",
  "summary": {
    "total_lines": 3,
    "active_lines": 2,
    "inactive_lines": 1,
    "total_alerts": 5,
    "critical_alerts": 0,
    "warning_alerts": 3,
    "info_alerts": 2
  },
  "production_lines": {
    "LINE_01": {
      "status": "active",
      "current_cast": "CAST_20240115_001",
      "defect_probability": 0.156,
      "last_prediction": "2024-01-15T10:29:55Z"
    },
    "LINE_02": {
      "status": "active", 
      "current_cast": "CAST_20240115_002",
      "defect_probability": 0.734,
      "last_prediction": "2024-01-15T10:29:58Z"
    },
    "LINE_03": {
      "status": "maintenance",
      "last_active": "2024-01-15T08:00:00Z",
      "estimated_return": "2024-01-15T14:00:00Z"
    }
  },
  "recent_activity": {
    "predictions_last_hour": 1250,
    "alerts_last_hour": 8,
    "data_points_received": 15600
  }
}
```

## Active Alerts

**GET** `/api/v1/monitoring/alerts`

Retrieve current system alerts and notifications.

### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `severity` | string | Filter by severity (critical, warning, info) |
| `category` | string | Filter by category |
| `acknowledged` | boolean | Filter by acknowledgment status |
| `limit` | integer | Maximum number of alerts |

### Response

```json
{
  "alerts": [
    {
      "alert_id": "ALT_001",
      "severity": "warning",
      "category": "data_quality",
      "title": "Sensor Communication Intermittent",
      "description": "LINE_03 temperature sensor experiencing intermittent communication",
      "created_at": "2024-01-15T09:15:00Z",
      "acknowledged": false,
      "line_id": "LINE_03",
      "sensor_id": "TEMP_03",
      "impact": "Medium - May affect prediction accuracy"
    },
    {
      "alert_id": "ALT_002",
      "severity": "info",
      "category": "system",
      "title": "High CPU Usage",
      "description": "CPU usage above 80% for 10 minutes",
      "created_at": "2024-01-15T10:20:00Z",
      "acknowledged": true,
      "acknowledged_by": "admin_001",
      "acknowledged_at": "2024-01-15T10:25:00Z"
    }
  ],
  "summary": {
    "total_alerts": 5,
    "by_severity": {
      "critical": 0,
      "warning": 3,
      "info": 2
    },
    "acknowledged": 2,
    "unacknowledged": 3
  }
}
```

## Acknowledge Alerts

**POST** `/api/v1/monitoring/alerts/acknowledge`

Acknowledge one or more system alerts.

### Request Body

```json
{
  "alert_ids": ["ALT_001", "ALT_002"],
  "acknowledged_by": "operator_001",
  "note": "Investigating sensor communication issue"
}
```

### Response

```json
{
  "acknowledged_count": 2,
  "failed_count": 0,
  "results": [
    {
      "alert_id": "ALT_001",
      "status": "acknowledged",
      "acknowledged_at": "2024-01-15T10:30:00Z"
    },
    {
      "alert_id": "ALT_002", 
      "status": "acknowledged",
      "acknowledged_at": "2024-01-15T10:30:00Z"
    }
  ]
}
```

## Performance History

**GET** `/api/v1/monitoring/performance/history`

Historical performance metrics for trend analysis.

### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `metric` | string | Specific metric name |
| `start_time` | string | Start time (ISO 8601) |
| `end_time` | string | End time (ISO 8601) |
| `resolution` | string | Data resolution (minute, hour, day) |

### Response

```json
{
  "metric": "predictions_per_hour",
  "resolution": "hour",
  "data_points": [
    {
      "timestamp": "2024-01-15T08:00:00Z",
      "value": 1150,
      "quality": "good"
    },
    {
      "timestamp": "2024-01-15T09:00:00Z",
      "value": 1275,
      "quality": "good"
    },
    {
      "timestamp": "2024-01-15T10:00:00Z",
      "value": 1320,
      "quality": "good"
    }
  ],
  "statistics": {
    "average": 1248.3,
    "min": 1150,
    "max": 1320,
    "trend": "increasing"
  }
}
```

## System Logs

**GET** `/api/v1/monitoring/logs`

Access system logs for troubleshooting and analysis.

### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `level` | string | Log level (debug, info, warning, error) |
| `component` | string | Component filter |
| `start_time` | string | Start time filter |
| `end_time` | string | End time filter |
| `limit` | integer | Maximum log entries |

### Response

```json
{
  "logs": [
    {
      "timestamp": "2024-01-15T10:29:55Z",
      "level": "info",
      "component": "prediction_engine",
      "message": "Prediction completed for LINE_01",
      "metadata": {
        "prediction_id": "pred_789abc",
        "processing_time_ms": 45,
        "confidence": 0.892
      }
    },
    {
      "timestamp": "2024-01-15T10:29:50Z",
      "level": "warning",
      "component": "data_collector",
      "message": "Sensor reading outside normal range",
      "metadata": {
        "line_id": "LINE_02",
        "sensor": "mold_temperature",
        "value": 1580.5,
        "normal_range": "1480-1570"
      }
    }
  ],
  "pagination": {
    "total": 5647,
    "limit": 100,
    "offset": 0
  }
}
```

## WebSocket Monitoring

**WebSocket** `/api/v1/monitoring/stream`

Real-time monitoring updates via WebSocket connection.

### Connection Example

```python
import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    if data['type'] == 'metric_update':
        print(f"Metric update: {data['metric']} = {data['value']}")
    elif data['type'] == 'alert':
        print(f"New alert: {data['severity']} - {data['title']}")

def on_open(ws):
    # Subscribe to specific monitoring channels
    ws.send(json.dumps({
        "action": "subscribe",
        "channels": ["system_metrics", "alerts", "performance"]
    }))

ws = websocket.WebSocketApp(
    "ws://localhost:8000/api/v1/monitoring/stream",
    header={"Authorization": "Bearer your_access_token"},
    on_message=on_message,
    on_open=on_open
)

ws.run_forever()
```

## Custom Monitoring

**POST** `/api/v1/monitoring/custom-metrics`

Submit custom application metrics.

### Request Body

```json
{
  "metrics": [
    {
      "name": "custom_efficiency_score",
      "value": 0.87,
      "timestamp": "2024-01-15T10:30:00Z",
      "tags": {
        "line_id": "LINE_01",
        "shift": "day"
      }
    }
  ]
}
```

### Response

```json
{
  "accepted_metrics": 1,
  "rejected_metrics": 0,
  "processing_time_ms": 12
}
```

This monitoring API provides comprehensive observability into the Steel Defect Prediction System, enabling proactive maintenance and optimization of system performance.