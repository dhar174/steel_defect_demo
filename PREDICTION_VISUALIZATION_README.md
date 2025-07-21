# Prediction Visualization System

## Overview

The prediction visualization system provides modular components for displaying steel defect prediction results in the dashboard. This implementation fulfills the requirements specified in issue #95.

## Components Implemented

### 1. PredictionDisplayComponents Class

Located in `src/visualization/components/prediction_display.py`, this class provides:

#### Real-Time Prediction Probability Gauge
- Color-coded gauge using `plotly.graph_objects.Indicator`
- Risk levels: Safe (green), Warning (yellow), High Risk (orange), Alert (red)
- Configurable thresholds from `inference_config.yaml`

#### Historical Prediction Timeline
- Line chart with dynamic color coding based on risk levels
- Threshold reference lines
- Hover information with risk level details

#### Model Ensemble Contribution Chart
- Pie chart showing baseline vs LSTM model contributions
- Configurable weights from configuration

#### Alert Status Indicators
- Dynamic Dash HTML components with Bootstrap styling
- Status changes based on prediction probability
- Cast ID integration for context

#### Prediction Confidence Visualization
- Error bars for confidence intervals
- Uncertainty bands around predictions
- Optional confidence interval or uncertainty parameter support

#### Prediction Accuracy Metrics Display
- Performance metrics cards (accuracy, precision, recall, F1-score)
- Color-coded based on performance levels
- Modular metric display system

## Configuration Integration

The system uses thresholds from `configs/inference_config.yaml`:
- `defect_probability`: 0.5
- `high_risk_threshold`: 0.7  
- `alert_threshold`: 0.8

## Usage Example

```python
from src.visualization.components import PredictionDisplayComponents
import yaml

# Load configuration
with open('configs/inference_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize components
components = PredictionDisplayComponents(config)

# Create a prediction gauge
gauge_fig = components.create_prediction_gauge(0.75)

# Create alert status indicator
alert_component = components.create_alert_status_indicator(0.75, cast_id="CAST_001")

# Create historical timeline
timeline_fig = components.create_historical_timeline(history_data)

# Create ensemble contribution chart
ensemble_fig = components.create_ensemble_contribution_chart(0.4, 0.6)
```

## Testing

Comprehensive test suite in `tests/test_prediction_display.py` covers:
- Component initialization with various configurations
- All visualization component creation
- Risk level calculations and color coding
- Edge cases and error handling
- Integration with existing dashboard components

## Dashboard Integration

The components are designed to integrate seamlessly with the existing `DefectMonitoringDashboard`:
- Consistent theming support
- Modular design for independent usage
- Compatible with existing callback patterns
- Uses same configuration structure

## Files Created

- `src/visualization/components/__init__.py`: Package initialization
- `src/visualization/components/prediction_display.py`: Main implementation (22KB)
- `tests/test_prediction_display.py`: Comprehensive test suite (10KB)
- `demo_prediction_display.py`: Demonstration script (6KB)

## Acceptance Criteria Fulfilled

✅ Real-time gauge with correct risk level color coding  
✅ Historical timeline with dynamic colors  
✅ Alert status indicators with dynamic styling  
✅ Modular design for dashboard integration  
✅ Configuration-based thresholds  
✅ Model ensemble contribution visualization  
✅ Prediction confidence display  
✅ Accuracy metrics presentation  
✅ Comprehensive test coverage  
✅ Integration with existing dashboard framework  

The implementation is ready for production use and provides a complete prediction visualization system for the steel defect detection dashboard.