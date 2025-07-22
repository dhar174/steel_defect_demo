# First Prediction Tutorial

This tutorial walks you through making your first defect prediction with the Steel Defect Prediction System. You'll learn to load data, run models, and interpret results.

## Prerequisites

- Completed [Quick Start](quick-start.md) setup
- System running and tested
- Basic understanding of steel casting processes

## Tutorial Overview

We'll cover:
1. [Loading sample data](#loading-sample-data)
2. [Running a prediction](#running-a-prediction)
3. [Understanding the output](#understanding-the-output)
4. [Exploring the dashboard](#exploring-the-dashboard)

## Loading Sample Data

The system includes synthetic sample data that mimics real steel casting sensor readings.

### Using Python

```python
from src.data.data_connectors import DataConnector
import pandas as pd

# Initialize data connector
connector = DataConnector()

# Load sample sensor data
sample_data = connector.get_sample_data()
print(f"Loaded {len(sample_data)} data points")

# View first few rows
print(sample_data.head())
```

Expected output:
```
Loaded 1000 data points
   timestamp  mold_temperature  mold_level  casting_speed  cooling_water_flow  superheat
0 2024-01-01               1520      150.0            1.2               200.0       25.0
1 2024-01-01               1522      149.8            1.2               201.2       24.8
2 2024-01-01               1521      150.2            1.1               199.8       25.2
```

### Understanding the Data Format

The sample data includes these key parameters:

| Parameter | Unit | Range | Description |
|-----------|------|-------|-------------|
| `mold_temperature` | °C | 1500-1550 | Steel temperature in mold |
| `mold_level` | mm | 140-160 | Steel level in mold |
| `casting_speed` | m/min | 0.8-1.5 | Speed of continuous casting |
| `cooling_water_flow` | L/min | 180-220 | Cooling water flow rate |
| `superheat` | °C | 20-35 | Temperature above liquidus |

## Running a Prediction

### Single Prediction

Make a prediction for current sensor readings:

```python
from src.inference.prediction_engine import PredictionEngine

# Initialize prediction engine
engine = PredictionEngine()

# Sample sensor readings
sensor_data = {
    "mold_temperature": 1525.0,
    "mold_level": 152.5,
    "casting_speed": 1.1,
    "cooling_water_flow": 195.0,
    "superheat": 27.0,
    "cast_id": "DEMO_CAST_001",
    "timestamp": "2024-01-15T10:30:00Z"
}

# Make prediction
result = engine.predict(sensor_data)
print("Prediction Result:")
print(result)
```

### Batch Predictions

Process multiple readings at once:

```python
# Load multiple readings
readings = connector.get_latest_readings(limit=10)

# Make batch predictions
batch_results = engine.predict_batch(readings.to_dict('records'))

print(f"Processed {len(batch_results)} predictions")
for i, result in enumerate(batch_results[:3]):  # Show first 3
    print(f"Prediction {i+1}: {result['defect_probability']:.3f}")
```

## Understanding the Output

### Prediction Structure

Each prediction returns a structured result:

```python
{
    "prediction_id": "pred_abc123def456",
    "timestamp": "2024-01-15T10:30:00Z",
    "cast_id": "DEMO_CAST_001",
    "defect_probability": 0.15,           # 0-1 scale (0=good, 1=defect)
    "confidence_score": 0.89,             # Model confidence
    "model_predictions": {                # Individual model results
        "baseline_xgboost": {
            "probability": 0.12,
            "confidence": 0.87,
            "features_used": 25
        },
        "lstm_sequence": {
            "probability": 0.18,
            "confidence": 0.91,
            "sequence_length": 60
        },
        "ensemble": {
            "probability": 0.15,
            "confidence": 0.89,
            "weights": {"baseline": 0.4, "lstm": 0.6}
        }
    },
    "alert_level": "low",                 # low, medium, high
    "risk_factors": [                     # Contributing factors
        {
            "factor": "temperature_variance",
            "impact": 0.08,
            "description": "Temperature fluctuation detected"
        }
    ]
}
```

### Interpreting Defect Probability

| Range | Alert Level | Interpretation | Action |
|-------|-------------|----------------|--------|
| 0.0 - 0.3 | Low | Good quality expected | Continue normal operation |
| 0.3 - 0.7 | Medium | Moderate risk | Monitor closely |
| 0.7 - 1.0 | High | High defect risk | Consider intervention |

### Understanding Confidence Scores

- **High Confidence (>0.8)**: Model is very certain about prediction
- **Medium Confidence (0.6-0.8)**: Reasonable certainty
- **Low Confidence (<0.6)**: Model uncertainty, treat with caution

## Exploring the Dashboard

### Launching the Dashboard

Start the interactive dashboard:

```bash
python scripts/run_dashboard.py
```

Open your browser to `http://localhost:8050`

### Dashboard Navigation

1. **Real-time Monitoring**
   - Live sensor readings
   - Current predictions
   - Alert status

2. **Model Comparison**
   - Performance metrics
   - ROC curves
   - Feature importance

3. **Historical Analysis**
   - Trend analysis
   - Statistical summaries
   - Pattern recognition

### Making Dashboard Predictions

1. Navigate to the **Real-time Monitoring** tab
2. The dashboard automatically shows live predictions
3. Observe the prediction probability gauge
4. Check alert indicators in the top panel

### Interactive Features

Try these dashboard features:

- **Time Range Selection**: Change the analysis period
- **Model Toggle**: Compare different model outputs
- **Export Data**: Download results as CSV
- **Alert Configuration**: Set custom thresholds

## Advanced Usage

### Custom Sensor Data

Create your own sensor readings:

```python
custom_data = {
    "mold_temperature": 1530.0,    # Higher temperature
    "mold_level": 145.0,           # Lower level
    "casting_speed": 1.4,          # Faster speed
    "cooling_water_flow": 210.0,   # Higher flow
    "superheat": 30.0,             # Higher superheat
}

prediction = engine.predict(custom_data)
print(f"Defect probability: {prediction['defect_probability']:.3f}")
print(f"Alert level: {prediction['alert_level']}")
```

### Analyzing Risk Factors

Understand what drives the predictions:

```python
# Get detailed risk analysis
prediction = engine.predict(sensor_data)

print("Risk Factors:")
for factor in prediction['risk_factors']:
    print(f"- {factor['factor']}: {factor['impact']:.3f}")
    print(f"  {factor['description']}")
```

### Model Performance Comparison

Compare how different models perform:

```python
from src.visualization.components import ModelComparison

# Load test data
test_data = connector.get_test_data()

# Get predictions from all models
results = {}
for model_name in ['baseline_xgboost', 'lstm_sequence']:
    model_results = engine.evaluate_model(model_name, test_data)
    results[model_name] = model_results

# Create comparison
comparison = ModelComparison()
comparison_chart = comparison.create_roc_curves(results)
comparison_chart.show()
```

## Troubleshooting

### Common Issues

!!! warning "Low Confidence Predictions"
    If you see consistently low confidence scores:
    - Check data quality
    - Ensure all required sensors are working
    - Verify timestamp format

!!! warning "Unexpected High Defect Probability"
    For surprisingly high defect predictions:
    - Review sensor readings for outliers
    - Check if conditions are outside normal ranges
    - Consider recalibration if readings seem wrong

### Data Quality Checks

Validate your sensor data:

```python
from src.data.data_validation import DataValidator

validator = DataValidator()
quality_report = validator.validate(sensor_data)

if quality_report['is_valid']:
    print("✅ Data quality OK")
else:
    print("❌ Data quality issues:")
    for issue in quality_report['issues']:
        print(f"- {issue}")
```

### Getting Help

If you encounter issues:

1. Check the [Troubleshooting Guide](../installation/troubleshooting.md)
2. Review [System Requirements](system-requirements.md)
3. Examine error logs in the `logs/` directory
4. [Open an issue](https://github.com/dhar174/steel_defect_demo/issues) on GitHub

## Next Steps

Now that you've made your first prediction:

- **Learn More**: Read the [User Guide](../user-guide/dashboard-overview.md)
- **Integrate**: Explore [API Integration](../api-reference/dashboard-integration.md)
- **Customize**: Try [Advanced Tutorials](../tutorials/advanced-features.md)
- **Contribute**: Check the [Contributing Guide](../development/contributing.md)

---

**Congratulations!** 🎉 You've successfully made your first steel defect prediction. The system is now ready for more advanced usage and integration into your casting operations.