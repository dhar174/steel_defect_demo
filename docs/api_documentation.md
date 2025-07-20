# API Documentation

## Steel Defect Prediction API

This document provides comprehensive API documentation for the Steel Defect Prediction system.

### Overview

The Steel Defect Prediction API provides endpoints for real-time defect prediction, model management, and system monitoring.

### Base URL

```
http://localhost:8000/api/v1
```

### Authentication

TODO: Define authentication mechanism (API keys, JWT, etc.)

### Endpoints

#### 1. Health Check

**GET** `/health`

Check the health status of the API service.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2023-01-01T12:00:00Z",
  "version": "1.0.0"
}
```

#### 2. Predict Defect

**POST** `/predict`

Predict defect probability for steel casting data.

**Request Body:**
```json
{
  "cast_id": "string",
  "sensor_data": [
    {
      "timestamp": "2023-01-01T12:00:00Z",
      "casting_speed": 1.2,
      "mold_temperature": 1520,
      "mold_level": 150,
      "cooling_water_flow": 200,
      "superheat": 25
    }
  ],
  "metadata": {
    "steel_grade": "string",
    "composition": {}
  }
}
```

**Response:**
```json
{
  "cast_id": "string",
  "predictions": {
    "baseline_model": {
      "probability": 0.15,
      "confidence": 0.85
    },
    "lstm_model": {
      "probability": 0.18,
      "confidence": 0.82
    },
    "ensemble": {
      "probability": 0.16,
      "confidence": 0.83
    }
  },
  "risk_level": "low",
  "timestamp": "2023-01-01T12:00:00Z"
}
```

#### 3. Batch Predict

**POST** `/predict/batch`

Predict defects for multiple casts in batch.

TODO: Define batch prediction endpoint structure.

#### 4. Model Information

**GET** `/models`

Get information about available models.

**Response:**
```json
{
  "models": [
    {
      "name": "baseline",
      "type": "xgboost",
      "version": "1.0.0",
      "status": "active",
      "metrics": {
        "auc_roc": 0.87,
        "precision": 0.82,
        "recall": 0.79
      }
    },
    {
      "name": "lstm",
      "type": "neural_network",
      "version": "1.0.0",
      "status": "active",
      "metrics": {
        "auc_roc": 0.89,
        "precision": 0.84,
        "recall": 0.81
      }
    }
  ]
}
```

#### 5. System Metrics

**GET** `/metrics`

Get system performance metrics.

TODO: Define metrics endpoint structure.

### Error Handling

All API endpoints return standard HTTP status codes:

- `200` - Success
- `400` - Bad Request
- `401` - Unauthorized
- `404` - Not Found
- `422` - Validation Error
- `500` - Internal Server Error

**Error Response Format:**
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid sensor data format",
    "details": {}
  }
}
```

### Rate Limiting

TODO: Define rate limiting policies.

### SDKs and Client Libraries

TODO: Document available SDKs and client libraries.

### Examples

#### Python Client Example

```python
import requests

# Predict defect for sensor data
response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={
        "cast_id": "cast_001",
        "sensor_data": [
            {
                "timestamp": "2023-01-01T12:00:00Z",
                "casting_speed": 1.2,
                "mold_temperature": 1520,
                "mold_level": 150,
                "cooling_water_flow": 200,
                "superheat": 25
            }
        ]
    }
)

result = response.json()
print(f"Defect probability: {result['predictions']['ensemble']['probability']}")
```

#### cURL Example

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "cast_id": "cast_001",
    "sensor_data": [
      {
        "timestamp": "2023-01-01T12:00:00Z",
        "casting_speed": 1.2,
        "mold_temperature": 1520,
        "mold_level": 150,
        "cooling_water_flow": 200,
        "superheat": 25
      }
    ]
  }'
```

### Changelog

#### Version 1.0.0
- Initial API release
- Basic prediction endpoints
- Health monitoring

TODO: Expand documentation as API evolves.