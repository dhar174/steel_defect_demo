# Dashboard Integration API

The Steel Defect Prediction System provides a Dash-based dashboard with programmatic access through Python modules and callback interfaces. This page documents the integration patterns and programmatic APIs.

## Architecture Overview

The system uses **Dash** (Plotly) for the web interface rather than traditional REST APIs. Integration is achieved through:

- **Python Module APIs**: Direct Python integration
- **Dashboard Callbacks**: Real-time updates via Dash callbacks
- **Data Interfaces**: Programmatic access to prediction engines
- **Component APIs**: Reusable dashboard components

## Python Module Integration

### Prediction Engine API

The core prediction functionality is available through Python modules:

```python
from src.inference.prediction_engine import PredictionEngine
from src.models.baseline_model import BaselineModel
from src.models.lstm_model import LSTMModel

# Initialize prediction engine
engine = PredictionEngine()

# Make predictions
prediction = engine.predict(sensor_data)
print(f"Defect probability: {prediction['defect_probability']}")
```

#### PredictionEngine Class

```python
class PredictionEngine:
    """Main prediction engine for defect detection"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize prediction engine with models.
        
        Args:
            config: Configuration dictionary
        """
    
    def predict(self, data: Dict) -> Dict:
        """
        Generate defect predictions.
        
        Args:
            data: Sensor data dictionary
            
        Returns:
            Prediction results with confidence scores
        """
    
    def predict_batch(self, data: List[Dict]) -> List[Dict]:
        """
        Generate batch predictions.
        
        Args:
            data: List of sensor data dictionaries
            
        Returns:
            List of prediction results
        """
```

### Data Interface API

Access and manipulate data through standardized interfaces:

```python
from src.data.data_connectors import DataConnector
from src.features.feature_engineering import FeatureEngineer

# Data access
connector = DataConnector()
data = connector.get_latest_data(limit=100)

# Feature engineering
engineer = FeatureEngineer()
features = engineer.transform(data)
```

## Dashboard Component APIs

### Component Integration

Reusable dashboard components for custom applications:

```python
from src.visualization.components import (
    PredictionDisplay,
    ModelComparison,
    SensorMonitoring,
    AlertManagement
)

# Create component instances
prediction_display = PredictionDisplay()
model_comparison = ModelComparison()
sensor_monitoring = SensorMonitoring()
alert_management = AlertManagement()
```

### PredictionDisplay Component

Real-time prediction visualization:

```python
class PredictionDisplay:
    """Real-time prediction display component"""
    
    def get_layout(self, initial_data: Dict = None) -> dbc.Container:
        """
        Get the component layout.
        
        Args:
            initial_data: Initial data for display
            
        Returns:
            Dash Bootstrap Container with layout
        """
    
    def update_predictions(self, prediction_data: Dict) -> Dict:
        """
        Update prediction display with new data.
        
        Args:
            prediction_data: Latest prediction results
            
        Returns:
            Updated component state
        """
```

#### Usage Example

```python
# Initialize component
display = PredictionDisplay()

# Get layout for embedding
layout = display.get_layout()

# In Dash callback
@app.callback(
    Output('prediction-display', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_display(n):
    # Get latest predictions
    predictions = prediction_engine.get_latest_predictions()
    
    # Update display
    return display.update_predictions(predictions)
```

### ModelComparison Component

Compare ML model performance:

```python
class ModelComparison:
    """Model performance comparison component"""
    
    def get_dashboard_layout(self, model_results: Dict) -> dbc.Container:
        """
        Create model comparison dashboard.
        
        Args:
            model_results: Dictionary with model performance data
            
        Returns:
            Complete dashboard layout
        """
    
    def create_roc_curves(self, model_results: Dict) -> go.Figure:
        """Create ROC curve comparison chart"""
    
    def create_feature_importance(self, model_results: Dict) -> go.Figure:
        """Create feature importance comparison"""
```

#### Model Results Format

```python
model_results = {
    'XGBoost': {
        'y_true': np.array([0, 1, 0, 1, ...]),           # True labels
        'y_pred': np.array([0, 1, 0, 0, ...]),           # Predicted labels
        'y_pred_proba': np.array([0.1, 0.9, 0.2, ...]), # Probabilities
        'feature_importance': {                           # Feature importance
            'temperature': 0.35,
            'pressure': 0.28,
            'flow_rate': 0.20,
            # ...
        },
        'metrics': {                                      # Performance metrics
            'accuracy': 0.87,
            'precision': 0.84,
            'recall': 0.82,
            'f1_score': 0.83
        }
    },
    'LSTM': {
        # Similar structure for LSTM model
        'attention_weights': np.array([...])  # Additional for LSTM
    }
}
```

### SensorMonitoring Component

Real-time sensor data visualization:

```python
class SensorMonitoring:
    """Real-time sensor monitoring component"""
    
    def create_sensor_charts(self, sensor_data: pd.DataFrame) -> List[go.Figure]:
        """
        Create sensor monitoring charts.
        
        Args:
            sensor_data: DataFrame with sensor readings
            
        Returns:
            List of Plotly figures
        """
    
    def get_alert_indicators(self, sensor_data: pd.DataFrame) -> Dict:
        """
        Generate alert indicators based on sensor data.
        
        Args:
            sensor_data: Current sensor readings
            
        Returns:
            Alert status dictionary
        """
```

## Data Formats

### Sensor Data Format

```python
sensor_data = {
    "timestamp": "2025-07-20T22:18:46Z",
    "cast_id": "CAST_20250720_001",
    "sensors": {
        "mold_temperature": 1520.5,        # °C
        "mold_level": 150.2,               # mm
        "casting_speed": 1.2,              # m/min
        "cooling_water_flow": 200.8,       # L/min
        "superheat": 25.3,                 # °C
        "mold_powder_consumption": 0.5,    # kg/min
        "secondary_cooling_zones": [       # Multiple zones
            {"zone": 1, "water_flow": 50.2, "temperature": 800.1},
            {"zone": 2, "water_flow": 45.8, "temperature": 650.3},
            # ...
        ],
        "oscillation_frequency": 180.0,    # cycles/min
        "oscillation_amplitude": 4.5       # mm
    },
    "process_parameters": {
        "steel_grade": "C45",
        "slab_width": 1200,                # mm
        "slab_thickness": 220,             # mm
        "tundish_temperature": 1545.2      # °C
    }
}
```

### Prediction Response Format

```python
prediction_response = {
    "prediction_id": "pred_abc123def456",
    "timestamp": "2025-07-20T22:18:46Z",
    "cast_id": "CAST_20250720_001",
    "defect_probability": 0.15,           # 0-1 scale
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
            "weights": {
                "baseline": 0.4,
                "lstm": 0.6
            }
        }
    },
    "alert_level": "low",                 # low, medium, high
    "risk_factors": [                     # Contributing factors
        {
            "factor": "temperature_variance",
            "impact": 0.08,
            "description": "Temperature fluctuation detected"
        }
    ],
    "recommendations": [                   # Actionable recommendations
        "Monitor mold temperature stability",
        "Check cooling water flow consistency"
    ]
}
```

## Dashboard Callbacks

### Real-time Updates

Dashboard components use Dash callbacks for real-time updates:

```python
@app.callback(
    [Output('prediction-chart', 'figure'),
     Output('confidence-indicator', 'children'),
     Output('alert-status', 'color')],
    [Input('interval-component', 'n_intervals')],
    [State('theme-store', 'data')]
)
def update_realtime_display(n_intervals, theme):
    """
    Update real-time prediction display.
    
    Args:
        n_intervals: Interval counter
        theme: Current theme setting
        
    Returns:
        Tuple of updated components
    """
    # Get latest predictions
    predictions = prediction_engine.get_latest_predictions()
    
    # Update chart
    fig = create_prediction_chart(predictions, theme)
    
    # Update confidence indicator
    confidence = predictions.get('confidence_score', 0)
    confidence_text = f"Confidence: {confidence:.1%}"
    
    # Update alert status
    alert_level = predictions.get('alert_level', 'low')
    alert_color = {
        'low': 'success',
        'medium': 'warning', 
        'high': 'danger'
    }.get(alert_level, 'secondary')
    
    return fig, confidence_text, alert_color
```

### Interactive Controls

```python
@app.callback(
    Output('model-comparison-chart', 'figure'),
    [Input('model-selector', 'value'),
     Input('metric-selector', 'value')],
    [State('model-results-store', 'data')]
)
def update_model_comparison(selected_models, selected_metric, model_results):
    """
    Update model comparison chart based on user selection.
    
    Args:
        selected_models: List of selected model names
        selected_metric: Selected performance metric
        model_results: Stored model results data
        
    Returns:
        Updated comparison chart
    """
    return model_comparison.create_comparison_chart(
        selected_models, selected_metric, model_results
    )
```

## Configuration API

### Dashboard Configuration

Configure dashboard behavior programmatically:

```python
dashboard_config = {
    "refresh_interval": 5000,              # Milliseconds
    "theme": "plotly_white",               # Default theme
    "auto_refresh": True,                  # Enable auto-refresh
    "alert_thresholds": {                  # Alert configuration
        "high": 0.8,
        "medium": 0.5,
        "low": 0.2
    },
    "chart_settings": {                    # Chart preferences
        "height": 400,
        "show_legend": True,
        "color_scheme": "viridis"
    },
    "data_retention": {                    # Data management
        "realtime_hours": 24,
        "historical_days": 365
    }
}
```

### Model Configuration

```python
model_config = {
    "baseline_model": {
        "type": "xgboost",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "feature_selection": "auto"
    },
    "lstm_model": {
        "type": "lstm",
        "sequence_length": 60,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "bidirectional": True
    },
    "ensemble": {
        "method": "weighted_average",
        "weights": {
            "baseline": 0.4,
            "lstm": 0.6
        }
    }
}
```

## Integration Examples

### Custom Dashboard Integration

```python
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from src.visualization.components import ModelComparison, PredictionDisplay

# Create custom dashboard
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Initialize components
prediction_display = PredictionDisplay()
model_comparison = ModelComparison()

# Custom layout
app.layout = dbc.Container([
    html.H1("Custom Steel Defect Dashboard"),
    
    # Real-time predictions
    dbc.Row([
        dbc.Col([
            html.H3("Real-time Predictions"),
            prediction_display.get_layout()
        ], width=8),
        
        # Alert panel
        dbc.Col([
            html.H3("Alerts"),
            html.Div(id="alert-panel")
        ], width=4)
    ]),
    
    # Model comparison
    dbc.Row([
        dbc.Col([
            html.H3("Model Performance"),
            model_comparison.get_dashboard_layout({})
        ], width=12)
    ]),
    
    # Auto-refresh
    dcc.Interval(
        id='interval-component',
        interval=5000,  # Update every 5 seconds
        n_intervals=0
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
```

### Programmatic Prediction

```python
from src.inference.prediction_engine import PredictionEngine
import pandas as pd

# Initialize engine
engine = PredictionEngine()

# Load sensor data
sensor_data = pd.read_csv('sensor_readings.csv')

# Generate predictions
predictions = []
for _, row in sensor_data.iterrows():
    prediction = engine.predict(row.to_dict())
    predictions.append(prediction)

# Analyze results
results_df = pd.DataFrame(predictions)
high_risk_casts = results_df[results_df['defect_probability'] > 0.8]

print(f"High risk casts: {len(high_risk_casts)}")
print(f"Average confidence: {results_df['confidence_score'].mean():.2f}")
```

## Error Handling

### Common Exceptions

```python
from src.utils.exceptions import (
    PredictionError,
    DataValidationError,
    ModelNotFoundError
)

try:
    prediction = engine.predict(sensor_data)
except PredictionError as e:
    print(f"Prediction failed: {e}")
except DataValidationError as e:
    print(f"Invalid data format: {e}")
except ModelNotFoundError as e:
    print(f"Model not available: {e}")
```

### Graceful Degradation

```python
def safe_prediction(sensor_data):
    """Make prediction with fallback handling"""
    try:
        return engine.predict(sensor_data)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {
            "defect_probability": 0.0,
            "confidence_score": 0.0,
            "alert_level": "unknown",
            "error": str(e)
        }
```

---

Next: [System Overview →](../architecture/system-overview.md)