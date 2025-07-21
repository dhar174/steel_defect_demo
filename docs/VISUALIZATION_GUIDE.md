# Sensor Time Series Visualization

This document describes the comprehensive sensor time series visualization system implemented for steel defect prediction.

## Features Implemented

### 1. Side-by-Side Time Series Plots
- Compare sensor readings between good and defective casts
- Visualize patterns that distinguish defective casting sequences
- Support for multiple sensor channels simultaneously

### 2. Multi-Sensor Dashboard
- Interactive plots showing all sensor channels in a unified view
- Real-time sensor data visualization
- Summary statistics panel
- Color-coded by cast type (good/defect)

### 3. Pattern Recognition Visualization
- Visual identification of defect signatures
- Temporal analysis of sensor evolution
- Correlation heatmaps between sensors
- Statistical distribution comparisons

### 4. Interactive Features
- Plotly-based interactive charts
- Zoom, pan, and hover capabilities
- Export functionality (HTML, PNG, etc.)
- Real-time dashboard updates

## Usage

### Basic Plotting

```python
from visualization.plotting_utils import PlottingUtils
import pandas as pd

# Initialize plotter
plotter = PlottingUtils()

# Create sensor time series plot
sensor_fig = plotter.plot_sensor_timeseries(data, sensors=['casting_speed', 'mold_temperature'])

# Compare good vs defect casts
comparison_fig = plotter.plot_cast_comparison(good_cast_data, defect_cast_data, 'mold_level')

# Create multi-sensor dashboard
dashboard_fig = plotter.create_multi_sensor_dashboard(cast_data, metadata)
```

### Side-by-Side Comparisons

```python
# Compare multiple casts side by side
comparison_fig = plotter.create_side_by_side_comparison(
    good_casts=[good_cast1, good_cast2], 
    defect_casts=[defect_cast1, defect_cast2]
)
```

### Interactive Dashboard

```python
from visualization.dashboard import DefectMonitoringDashboard

# Initialize dashboard
config = {'inference': {'dashboard_port': 8050}}
dashboard = DefectMonitoringDashboard(config)

# Run dashboard server
dashboard.run(debug=True)
```

## Demonstration

Run the comprehensive demonstration:

```bash
python demo_sensor_visualization.py
```

This generates:
- 11 interactive HTML visualizations
- Side-by-side comparisons of normal vs defect casts
- Multi-sensor dashboards
- Correlation analysis
- Pattern recognition plots
- Prediction timeline simulations

## Sensor Channels Supported

The system supports visualization of all steel casting sensor channels:

- **casting_speed**: Casting speed (m/min)
- **mold_temperature**: Mold temperature (°C)
- **mold_level**: Mold level (mm)
- **cooling_water_flow**: Cooling water flow rate (L/min)
- **superheat**: Superheat temperature (°C)

## Defect Pattern Recognition

The visualization system identifies key defect signatures:

1. **Prolonged Mold Level Deviation**: Extended periods outside normal range
2. **Rapid Temperature Drop**: Sudden temperature decreases that can cause thermal stress
3. **High Speed with Low Superheat**: Risk combinations that can lead to solidification issues

## Output Formats

Visualizations can be exported in multiple formats:
- Interactive HTML (default)
- PNG images
- JPEG images
- SVG vector graphics
- PDF (via additional configuration)

## Integration

The visualization system integrates with:
- Synthetic data generator
- Real-time inference engine
- Model prediction outputs
- Historical data analysis

## Requirements

- plotly >= 6.0
- dash >= 3.0
- pandas >= 1.5
- numpy >= 1.23
- matplotlib >= 3.10
- seaborn >= 0.13

## Future Enhancements

- Real-time streaming visualization
- Advanced pattern recognition algorithms
- 3D sensor space visualization
- AR/VR visualization interfaces
- Integration with SCADA systems