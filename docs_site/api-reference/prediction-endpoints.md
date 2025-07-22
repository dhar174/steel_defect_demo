# Prediction Endpoints

The Prediction API provides real-time defect prediction capabilities for continuous steel casting operations.

## Base URL

```
http://localhost:8000/api/v1/predictions
```

## Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predictions` | Create new prediction |
| GET | `/predictions/{id}` | Get prediction by ID |
| GET | `/predictions` | List predictions |
| POST | `/predictions/batch` | Batch predictions |
| GET | `/predictions/stream` | Real-time prediction stream |

## Create Prediction

**POST** `/api/v1/predictions`

Create a new defect prediction based on sensor data.

### Request Body

```json
{
  "sensor_data": {
    "mold_temperature": 1525.4,
    "casting_speed": 1.12,
    "cooling_water_flow": 195.8,
    "oxygen_content": 0.025,
    "carbon_content": 0.18,
    "steel_grade": "304L",
    "tundish_temperature": 1545.0
  },
  "metadata": {
    "cast_id": "CAST_20240115_001",
    "line_id": "LINE_01",
    "operator_id": "OP_001",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### Response

```json
{
  "prediction_id": "pred_789abc123",
  "defect_probability": 0.156,
  "confidence": 0.892,
  "risk_level": "low",
  "processing_time_ms": 45,
  "model_version": "v2.1.0",
  "prediction_details": {
    "feature_importance": {
      "mold_temperature": 0.342,
      "casting_speed": 0.287,
      "cooling_water_flow": 0.198,
      "oxygen_content": 0.173
    },
    "threshold_analysis": {
      "warning_threshold": 0.6,
      "critical_threshold": 0.8,
      "current_status": "normal"
    }
  },
  "timestamp": "2024-01-15T10:30:00.123Z"
}
```

### Example Usage

```python
import requests

# Prediction request
response = requests.post(
    'http://localhost:8000/api/v1/predictions',
    headers={
        'Authorization': 'Bearer your_access_token',
        'Content-Type': 'application/json'
    },
    json={
        "sensor_data": {
            "mold_temperature": 1525.4,
            "casting_speed": 1.12,
            "cooling_water_flow": 195.8,
            "oxygen_content": 0.025,
            "carbon_content": 0.18,
            "steel_grade": "304L"
        },
        "metadata": {
            "cast_id": "CAST_20240115_001",
            "line_id": "LINE_01"
        }
    }
)

prediction = response.json()
print(f"Defect probability: {prediction['defect_probability']:.3f}")
print(f"Risk level: {prediction['risk_level']}")
```

## Get Prediction by ID

**GET** `/api/v1/predictions/{prediction_id}`

Retrieve a specific prediction by its ID.

### Response

```json
{
  "prediction_id": "pred_789abc123",
  "defect_probability": 0.156,
  "confidence": 0.892,
  "risk_level": "low",
  "sensor_data": {
    "mold_temperature": 1525.4,
    "casting_speed": 1.12,
    "cooling_water_flow": 195.8
  },
  "created_at": "2024-01-15T10:30:00.123Z",
  "model_version": "v2.1.0"
}
```

## List Predictions

**GET** `/api/v1/predictions`

Retrieve a list of predictions with optional filtering.

### Query Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `limit` | integer | Number of results (max 100) | 20 |
| `offset` | integer | Results offset | 0 |
| `start_date` | string | Start date (ISO 8601) | - |
| `end_date` | string | End date (ISO 8601) | - |
| `cast_id` | string | Filter by cast ID | - |
| `line_id` | string | Filter by production line | - |
| `risk_level` | string | Filter by risk level | - |
| `min_probability` | float | Minimum defect probability | - |
| `max_probability` | float | Maximum defect probability | - |

### Example Request

```python
# List recent high-risk predictions
response = requests.get(
    'http://localhost:8000/api/v1/predictions',
    headers={'Authorization': 'Bearer your_access_token'},
    params={
        'risk_level': 'high',
        'limit': 50,
        'start_date': '2024-01-15T00:00:00Z'
    }
)

predictions = response.json()
```

### Response

```json
{
  "predictions": [
    {
      "prediction_id": "pred_789abc123",
      "defect_probability": 0.156,
      "risk_level": "low",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "pagination": {
    "total": 1523,
    "limit": 20,
    "offset": 0,
    "has_next": true
  }
}
```

## Batch Predictions

**POST** `/api/v1/predictions/batch`

Process multiple sensor readings in a single request for improved efficiency.

### Request Body

```json
{
  "predictions": [
    {
      "id": "batch_001",
      "sensor_data": {
        "mold_temperature": 1525.4,
        "casting_speed": 1.12,
        "cooling_water_flow": 195.8
      }
    },
    {
      "id": "batch_002", 
      "sensor_data": {
        "mold_temperature": 1530.1,
        "casting_speed": 1.08,
        "cooling_water_flow": 198.2
      }
    }
  ],
  "options": {
    "return_details": true,
    "async_processing": false
  }
}
```

### Response

```json
{
  "batch_id": "batch_xyz789",
  "total_predictions": 2,
  "processing_time_ms": 89,
  "results": [
    {
      "id": "batch_001",
      "prediction_id": "pred_001abc",
      "defect_probability": 0.156,
      "risk_level": "low",
      "status": "success"
    },
    {
      "id": "batch_002",
      "prediction_id": "pred_002def", 
      "defect_probability": 0.734,
      "risk_level": "high",
      "status": "success"
    }
  ]
}
```

## Real-time Prediction Stream

**GET** `/api/v1/predictions/stream`

Establish a Server-Sent Events (SSE) connection for real-time prediction updates.

### WebSocket Connection

```python
import websocket
import json

def on_message(ws, message):
    prediction = json.loads(message)
    print(f"New prediction: {prediction['defect_probability']:.3f}")

def on_open(ws):
    # Subscribe to specific production line
    ws.send(json.dumps({
        "action": "subscribe",
        "line_id": "LINE_01"
    }))

# Connect to WebSocket
ws = websocket.WebSocketApp(
    "ws://localhost:8000/api/v1/predictions/stream",
    header={"Authorization": "Bearer your_access_token"},
    on_message=on_message,
    on_open=on_open
)

ws.run_forever()
```

### SSE Connection

```python
import requests

# Stream predictions via SSE
response = requests.get(
    'http://localhost:8000/api/v1/predictions/stream',
    headers={
        'Authorization': 'Bearer your_access_token',
        'Accept': 'text/event-stream'
    },
    stream=True
)

for line in response.iter_lines():
    if line.startswith(b'data: '):
        data = json.loads(line[6:])
        print(f"Prediction: {data['defect_probability']:.3f}")
```

## Model Information

**GET** `/api/v1/predictions/model/info`

Get information about the current prediction model.

### Response

```json
{
  "model_name": "LSTM_Defect_Predictor",
  "version": "v2.1.0",
  "training_date": "2024-01-10T08:00:00Z",
  "accuracy": 0.923,
  "precision": 0.891,
  "recall": 0.956,
  "f1_score": 0.922,
  "feature_count": 12,
  "supported_steel_grades": ["304L", "316L", "410"],
  "update_frequency": "weekly"
}
```

## Feature Importance

**GET** `/api/v1/predictions/features/importance`

Get global feature importance for the prediction model.

### Response

```json
{
  "feature_importance": {
    "mold_temperature": 0.298,
    "casting_speed": 0.245,
    "cooling_water_flow": 0.187,
    "oxygen_content": 0.134,
    "carbon_content": 0.089,
    "steel_grade": 0.047
  },
  "model_version": "v2.1.0",
  "calculation_date": "2024-01-15T12:00:00Z"
}
```

## Error Responses

### Common Error Codes

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 400 | `invalid_sensor_data` | Invalid or missing sensor data |
| 401 | `unauthorized` | Authentication required |
| 403 | `insufficient_permissions` | Lack required permissions |
| 422 | `validation_error` | Request validation failed |
| 429 | `rate_limit_exceeded` | Too many requests |
| 500 | `prediction_error` | Internal prediction error |

### Error Response Format

```json
{
  "error": {
    "code": "invalid_sensor_data",
    "message": "Required field 'mold_temperature' is missing",
    "details": {
      "missing_fields": ["mold_temperature"],
      "provided_fields": ["casting_speed", "cooling_water_flow"]
    }
  },
  "request_id": "req_12345",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Rate Limiting

Rate limits apply to prediction endpoints:

- **Standard users**: 1000 predictions/hour
- **Premium users**: 5000 predictions/hour  
- **Batch endpoint**: 100 batches/hour (max 50 predictions per batch)

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 997
X-RateLimit-Reset: 1640995200
```

## SDK Examples

### Python SDK

```python
from steel_defect_client import SteelDefectClient

client = SteelDefectClient(
    base_url='http://localhost:8000',
    api_key='your_api_key'
)

# Simple prediction
prediction = client.predictions.create({
    'mold_temperature': 1525.4,
    'casting_speed': 1.12,
    'cooling_water_flow': 195.8
})

# Batch predictions
batch_result = client.predictions.create_batch([
    {'mold_temperature': 1525.4, 'casting_speed': 1.12},
    {'mold_temperature': 1530.1, 'casting_speed': 1.08}
])

# Stream predictions
for prediction in client.predictions.stream():
    print(f"Real-time prediction: {prediction.defect_probability}")
```

### JavaScript SDK

```javascript
import { SteelDefectAPI } from 'steel-defect-sdk';

const api = new SteelDefectAPI({
    baseURL: 'http://localhost:8000',
    apiKey: 'your_api_key'
});

// Create prediction
const prediction = await api.predictions.create({
    sensor_data: {
        mold_temperature: 1525.4,
        casting_speed: 1.12,
        cooling_water_flow: 195.8
    }
});

// Real-time stream
const stream = api.predictions.stream();
stream.on('prediction', (data) => {
    console.log('New prediction:', data.defect_probability);
});
```

This prediction API provides comprehensive capabilities for integrating defect prediction into your steel casting operations with high performance and reliability.