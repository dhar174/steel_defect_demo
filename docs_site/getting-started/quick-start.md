# Quick Start Guide

Get up and running with the Steel Defect Prediction System in just 5 minutes!

## Prerequisites

- Python 3.8 or higher
- Git
- At least 4GB RAM
- Linux, macOS, or Windows with WSL2

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/dhar174/steel_defect_demo.git
cd steel_defect_demo
```text

### 2. Install Dependencies

```bash

# Install core dependencies

pip install -r requirements.txt

# Install the package in development mode

pip install -e .
```text

### 3. Verify Installation

```bash

# Test the installation

python demo_model_comparison.py
```text

You should see output showing model comparison results and charts being generated.

## Your First Prediction

### Option 1: Interactive Dashboard

Launch the interactive dashboard to explore the system:

```bash
python scripts/run_dashboard.py
```text

Then open your browser to `http://localhost:8050` to access the dashboard.

### Option 2: Command Line Demo

Run a quick prediction demo:

```bash

# Run model comparison demo

python demo_model_comparison.py

# Run sensor monitoring demo  

python demo_sensor_visualization.py

# Run historical analysis demo

python demo_historical_analysis.py
```text

## Understanding the Output

### Model Predictions

The system provides predictions in this format:

```json
{
    "cast_id": "CAST_20250720_001",
    "defect_probability": 0.15,
    "confidence_score": 0.89,
    "model_predictions": {
        "baseline": 0.12,
        "lstm": 0.18
    },
    "alert_level": "low",
    "timestamp": "2025-07-20T22:18:46Z"
}
```text

### Dashboard Features

The dashboard includes:

- **Real-time Monitoring**: Live sensor data and predictions
- **Model Comparison**: Side-by-side performance metrics
- **Historical Analysis**: Trends and pattern analysis
- **Alert Management**: Configure thresholds and notifications

## Next Steps

Once you have the system running:

1. **Explore the Dashboard**: Navigate through different monitoring views
2. **Review Architecture**: Understand the [system design](../architecture/system-overview.md)
3. **Explore Dashboard**: Use the [dashboard overview](../user-guide/dashboard-overview.md)
4. **Integration**: Learn about [API integration](../api-reference/dashboard-integration.md)

## Sample Data

The system includes synthetic sample data for testing:

- **Sensor readings**: Temperature, pressure, flow rates
- **Cast parameters**: Speed, superheat, mold level
- **Quality outcomes**: Defect classifications and probabilities

## Common Issues

### Installation Problems

!!! warning "Module Import Errors"
    If you encounter import errors, ensure you've installed the package in development mode:

    ```bash
    pip install -e .
    ```

!!! warning "Missing Dependencies"
    Some demos require additional packages. Install documentation dependencies:

    ```bash
    pip install -r requirements-docs.txt
    ```

### Performance Tips

- Use at least 4GB RAM for optimal performance
- LSTM model training requires more memory than baseline models
- Dashboard responsiveness improves with faster internet for plot rendering

## Getting Help

- Check the [Development Setup Guide](../installation/development-setup.md)
- Review [System Requirements](system-requirements.md)
- Browse the [User Guide](../user-guide/dashboard-overview.md)
- [Open an issue](https://github.com/dhar174/steel_defect_demo/issues) for bugs

---

**Congratulations!** 🎉 You now have the Steel Defect Prediction System running.
Continue to the [Dashboard Overview](../user-guide/dashboard-overview.md) to learn about all available features.
