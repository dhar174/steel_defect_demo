# Data Endpoints

The Data API provides access to sensor data, historical records, and data management capabilities for the Steel
Defect Prediction System.

## Base URL

```text
http://localhost:8000/api/v1/data
```text

## Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/data/sensors/current` | Get current sensor readings |
| GET | `/data/sensors/history` | Get historical sensor data |
| POST | `/data/sensors` | Submit new sensor data |
| GET | `/data/casts` | Get casting information |
| GET | `/data/quality` | Get quality inspection data |
| POST | `/data/export` | Export data to various formats |

## Current Sensor Data

**GET** `/api/v1/data/sensors/current`

Get the most recent sensor readings from all production lines.

### Response

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "lines": {
    "LINE_01": {
      "mold_temperature": 1525.4,
      "casting_speed": 1.12,
      "cooling_water_flow": 195.8,
      "oxygen_content": 0.025,
      "carbon_content": 0.18,
      "steel_grade": "304L",
      "status": "active",
      "last_updated": "2024-01-15T10:29:55Z"
    },
    "LINE_02": {
      "mold_temperature": 1530.1,
      "casting_speed": 1.08,
      "cooling_water_flow": 198.2,
      "oxygen_content": 0.028,
      "carbon_content": 0.16,
      "steel_grade": "316L",
      "status": "active",
      "last_updated": "2024-01-15T10:29:58Z"
    }
  }
}
```text

### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `line_id` | string | Filter by production line |
| `format` | string | Response format (json, csv) |

## Historical Sensor Data

**GET** `/api/v1/data/sensors/history`

Retrieve historical sensor data with flexible filtering options.

### Query Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `start_date` | string | Start date (ISO 8601) | Yes |
| `end_date` | string | End date (ISO 8601) | Yes |
| `line_id` | string | Production line ID | No |
| `parameters` | array | Sensor parameters to include | No |
| `aggregation` | string | Data aggregation (raw, minute, hour, day) | No |
| `limit` | integer | Maximum records (max 10000) | No |

### Example Request

```python
import requests

response = requests.get(
    'http://localhost:8000/api/v1/data/sensors/history',
    headers={'Authorization': 'Bearer your_access_token'},
    params={
        'start_date': '2024-01-14T00:00:00Z',
        'end_date': '2024-01-15T00:00:00Z',
        'line_id': 'LINE_01',
        'parameters': ['mold_temperature', 'casting_speed'],
        'aggregation': 'minute'
    }
)

data = response.json()
```text

### Response

```json
{
  "data": [
    {
      "timestamp": "2024-01-14T00:01:00Z",
      "line_id": "LINE_01",
      "mold_temperature": 1523.2,
      "casting_speed": 1.10
    },
    {
      "timestamp": "2024-01-14T00:02:00Z", 
      "line_id": "LINE_01",
      "mold_temperature": 1524.1,
      "casting_speed": 1.11
    }
  ],
  "metadata": {
    "total_records": 1440,
    "aggregation": "minute",
    "parameters": ["mold_temperature", "casting_speed"]
  }
}
```text

## Submit Sensor Data

**POST** `/api/v1/data/sensors`

Submit new sensor readings to the system.

### Request Body

```json
{
  "line_id": "LINE_01",
  "timestamp": "2024-01-15T10:30:00Z",
  "sensor_data": {
    "mold_temperature": 1525.4,
    "casting_speed": 1.12,
    "cooling_water_flow": 195.8,
    "oxygen_content": 0.025,
    "carbon_content": 0.18,
    "steel_grade": "304L"
  },
  "quality_flags": {
    "temperature_sensor": "good",
    "speed_sensor": "good", 
    "flow_sensor": "warning"
  }
}
```text

### Response

```json
{
  "status": "success",
  "data_id": "data_abc123",
  "timestamp": "2024-01-15T10:30:00Z",
  "validation_results": {
    "valid_parameters": 6,
    "warnings": ["flow_sensor quality flag is warning"],
    "errors": []
  }
}
```text

## Casting Information

**GET** `/api/v1/data/casts`

Retrieve information about casting operations.

### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `cast_id` | string | Specific cast ID |
| `start_date` | string | Start date filter |
| `end_date` | string | End date filter |
| `steel_grade` | string | Steel grade filter |
| `status` | string | Cast status (active, completed, aborted) |

### Response

```json
{
  "casts": [
    {
      "cast_id": "CAST_20240115_001",
      "line_id": "LINE_01",
      "steel_grade": "304L",
      "start_time": "2024-01-15T08:00:00Z",
      "end_time": "2024-01-15T12:30:00Z",
      "status": "completed",
      "total_weight": 25.4,
      "quality_score": 0.92,
      "defect_count": 2,
      "operator_id": "OP_001"
    }
  ],
  "pagination": {
    "total": 156,
    "limit": 20,
    "offset": 0
  }
}
```text

## Quality Data

**GET** `/api/v1/data/quality`

Access quality inspection data and test results.

### Response

```json
{
  "quality_records": [
    {
      "inspection_id": "QI_20240115_001",
      "cast_id": "CAST_20240115_001",
      "inspection_time": "2024-01-15T13:00:00Z",
      "inspector_id": "QC_002",
      "results": {
        "surface_quality": "excellent",
        "dimensional_accuracy": 0.98,
        "mechanical_properties": {
          "tensile_strength": 520,
          "yield_strength": 210,
          "elongation": 40
        },
        "chemical_composition": {
          "carbon": 0.18,
          "chromium": 18.2,
          "nickel": 8.1
        },
        "defects": [
          {
            "type": "surface_crack",
            "severity": "minor",
            "location": "section_B"
          }
        ]
      },
      "overall_grade": "A"
    }
  ]
}
```text

## Data Export

**POST** `/api/v1/data/export`

Export data in various formats for analysis or reporting.

### Request Body

```json
{
  "export_type": "sensor_data",
  "format": "csv",
  "filters": {
    "start_date": "2024-01-14T00:00:00Z",
    "end_date": "2024-01-15T00:00:00Z",
    "line_id": "LINE_01",
    "parameters": ["mold_temperature", "casting_speed", "defect_probability"]
  },
  "options": {
    "include_headers": true,
    "aggregation": "minute",
    "compression": "gzip"
  }
}
```text

### Response

```json
{
  "export_id": "export_xyz789",
  "status": "processing",
  "estimated_completion": "2024-01-15T10:35:00Z",
  "download_url": null,
  "file_size_estimate": "2.5 MB"
}
```text

### Check Export Status

**GET** `/api/v1/data/export/{export_id}/status`

```json
{
  "export_id": "export_xyz789",
  "status": "completed",
  "download_url": "https://api.example.com/downloads/export_xyz789.csv.gz",
  "file_size": "2.3 MB",
  "expires_at": "2024-01-16T10:35:00Z"
}
```text

## Data Validation

**POST** `/api/v1/data/validate`

Validate sensor data before submission.

### Request Body

```json
{
  "sensor_data": {
    "mold_temperature": 1525.4,
    "casting_speed": 1.12,
    "cooling_water_flow": 195.8
  },
  "validation_rules": ["range_check", "trend_analysis", "anomaly_detection"]
}
```text

### Response

```json
{
  "validation_results": {
    "overall_status": "valid",
    "checks": {
      "range_check": {
        "status": "passed",
        "details": "All parameters within normal ranges"
      },
      "trend_analysis": {
        "status": "warning",
        "details": "Temperature trending upward"
      },
      "anomaly_detection": {
        "status": "passed", 
        "details": "No anomalies detected"
      }
    },
    "recommendations": [
      "Monitor temperature trend closely"
    ]
  }
}
```text

## Data Statistics

**GET** `/api/v1/data/statistics`

Get statistical summaries of sensor data.

### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `parameter` | string | Sensor parameter name |
| `time_period` | string | Time period (hour, day, week, month) |
| `line_id` | string | Production line filter |

### Response

```json
{
  "parameter": "mold_temperature",
  "time_period": "day",
  "statistics": {
    "count": 1440,
    "mean": 1525.7,
    "median": 1525.4,
    "std_dev": 12.3,
    "min": 1498.2,
    "max": 1553.1,
    "percentiles": {
      "25": 1517.8,
      "75": 1533.2,
      "95": 1546.9
    }
  },
  "trend": {
    "direction": "stable",
    "slope": 0.02,
    "r_squared": 0.15
  }
}
```text

## Real-time Data Stream

**GET** `/api/v1/data/stream` (WebSocket)

Subscribe to real-time sensor data updates.

### WebSocket Example

```python
import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    print(f"Line {data['line_id']}: Temp={data['mold_temperature']}")

def on_open(ws):

    # Subscribe to specific parameters

    ws.send(json.dumps({
        "action": "subscribe",
        "line_id": "LINE_01",
        "parameters": ["mold_temperature", "casting_speed"]
    }))

ws = websocket.WebSocketApp(
    "ws://localhost:8000/api/v1/data/stream",
    header={"Authorization": "Bearer your_access_token"},
    on_message=on_message,
    on_open=on_open
)

ws.run_forever()
```text

## Error Handling

### Common Error Codes

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 400 | `invalid_date_range` | Invalid date range specified |
| 400 | `invalid_parameters` | Invalid sensor parameters |
| 413 | `payload_too_large` | Request payload exceeds limit |
| 422 | `validation_failed` | Data validation failed |
| 429 | `rate_limit_exceeded` | Too many requests |

### Example Error Response

```json
{
  "error": {
    "code": "invalid_date_range",
    "message": "End date must be after start date",
    "details": {
      "start_date": "2024-01-15T00:00:00Z",
      "end_date": "2024-01-14T00:00:00Z"
    }
  },
  "request_id": "req_67890"
}
```text

This Data API provides comprehensive access to all sensor data and operational information needed for effective steel defect monitoring and analysis.
