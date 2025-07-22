# Dashboard Overview

The Steel Defect Prediction System dashboard provides a comprehensive real-time monitoring interface for steel casting operations. This guide walks you through all major features and components.

## Dashboard Access

The dashboard is accessible at `http://localhost:8050` when running locally, or at your configured production URL.

```bash
# Start the dashboard
python scripts/run_dashboard.py
```

## Main Interface Layout

The dashboard is organized into several key sections:

### Navigation Bar

The top navigation provides access to all major dashboard sections:

- **Real-time Monitoring**: Live sensor data and predictions
- **Model Comparison**: Performance analysis between different models
- **Historical Analysis**: Trends and pattern analysis over time
- **Alert Management**: Configure and manage alert thresholds
- **System Status**: Monitor system health and performance

### Main Content Area

The central area displays the selected dashboard view with interactive charts, tables, and controls.

### Side Panel (when applicable)

Some views include a side panel with:
- Filter controls
- Configuration options
- Additional metrics
- Quick actions

## Key Features

### 1. Real-time Monitoring

![Real-time Dashboard](../assets/images/screenshots/dashboard-realtime.png)

The real-time monitoring view provides:

#### Live Sensor Readings
- **Temperature**: Mold temperature readings
- **Pressure**: Casting pressure metrics
- **Flow Rates**: Cooling water and steel flow
- **Speed**: Casting speed monitoring

#### Prediction Display
- **Defect Probability**: Real-time defect likelihood (0-1 scale)
- **Confidence Score**: Model confidence in predictions
- **Alert Level**: Visual indicators (Green/Yellow/Red)
- **Trend Indicators**: Directional arrows showing trends

#### Key Metrics Cards
```
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Current Defect  │ │ Avg Confidence  │ │ Active Alerts   │
│ Probability     │ │                 │ │                 │
│                 │ │                 │ │                 │
│     0.15        │ │     0.89        │ │       2         │
│                 │ │                 │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### 2. Model Comparison

![Model Comparison](../assets/images/screenshots/dashboard-comparison.png)

Compare performance between different ML models:

#### Performance Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity measure
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve

#### Visualization Charts
- **ROC Curves**: Model discrimination ability
- **Precision-Recall Curves**: Performance trade-offs
- **Feature Importance**: Variable significance
- **Confusion Matrix**: Classification accuracy breakdown

#### Side-by-side Comparison
```
Model Performance Comparison
                    XGBoost    LSTM     Ensemble
Accuracy            0.87       0.89     0.91
Precision           0.84       0.88     0.90
Recall              0.82       0.86     0.89
F1-Score            0.83       0.87     0.89
Training Time       2.3 min    8.7 min  11.2 min
```

### 3. Historical Analysis

![Historical Analysis](../assets/images/screenshots/dashboard-history.png)

Analyze trends and patterns over time:

#### Time Series Charts
- **Defect Rate Trends**: Historical defect rates
- **Sensor Pattern Analysis**: Long-term sensor behavior
- **Model Performance**: Accuracy trends over time
- **Process Stability**: Variance and consistency metrics

#### Interactive Features
- **Date Range Selection**: Choose analysis period
- **Zoom and Pan**: Detailed chart exploration
- **Data Export**: Download analysis results
- **Statistical Summaries**: Automatic trend analysis

### 4. Alert Management

![Alert Management](../assets/images/screenshots/dashboard-alerts.png)

Configure and manage the alerting system:

#### Alert Configuration
- **Threshold Settings**: Defect probability limits
- **Notification Methods**: Email, SMS, dashboard
- **Escalation Rules**: Multi-level alert hierarchy
- **Time-based Rules**: Different thresholds by shift/time

#### Active Alerts Dashboard
```
Current Alerts
┌─────────────────────────────────────────────────────────┐
│ 🔴 HIGH: Defect probability 0.85 (Threshold: 0.8)      │
│ 🟡 MED:  Temperature variance above normal              │
│ 🟡 MED:  Model confidence below 0.7                    │
└─────────────────────────────────────────────────────────┘
```

## Interactive Controls

### Time Range Selector

Most views include time range controls:

```
[Last Hour] [Last 4 Hours] [Last Day] [Custom Range]
```

### Refresh Controls

```
Auto-refresh: [ON/OFF]  Interval: [5s] [15s] [30s] [1m]
```

### Export Options

```
[Export CSV] [Export PNG] [Generate Report]
```

## Dashboard Configuration

### User Preferences

Access user preferences via the settings menu:

- **Display Options**: Chart types, color schemes
- **Refresh Rates**: Auto-update intervals
- **Alert Preferences**: Notification settings
- **Dashboard Layout**: Customize panel arrangement

### Theme Options

The dashboard supports multiple themes:

=== "Light Theme"
    Clean, professional appearance suitable for well-lit environments.

=== "Dark Theme"
    Reduced eye strain for low-light monitoring environments.

=== "High Contrast"
    Enhanced visibility for accessibility requirements.

## Mobile Responsiveness

The dashboard is optimized for various screen sizes:

### Desktop (>1200px)
- Full feature set
- Multi-panel layout
- Detailed charts and tables

### Tablet (768px - 1200px)
- Responsive layout
- Simplified navigation
- Touch-friendly controls

### Mobile (< 768px)
- Essential features only
- Vertical layout
- Large touch targets

## Performance Optimization

### Data Loading

The dashboard implements several performance optimizations:

- **Lazy Loading**: Charts load only when visible
- **Data Caching**: Recent data cached locally
- **Progressive Loading**: Large datasets load incrementally
- **WebSocket Updates**: Real-time data via WebSockets

### Browser Requirements

For optimal performance:
- **RAM**: 2GB+ available memory
- **Network**: Stable internet connection
- **Browser**: Modern browser with JavaScript enabled

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `R` | Refresh current view |
| `F` | Toggle fullscreen |
| `H` | Show/hide help |
| `Space` | Pause/resume auto-refresh |
| `Esc` | Close modals |

## Troubleshooting

### Common Issues

!!! warning "Dashboard Not Loading"
    - Check that the dashboard service is running
    - Verify port 8050 is not blocked by firewall
    - Clear browser cache and cookies

!!! warning "Slow Performance"
    - Reduce auto-refresh frequency
    - Clear browser cache
    - Check network connectivity
    - Restart dashboard service

!!! warning "Data Not Updating"
    - Check data source connections
    - Verify auto-refresh is enabled
    - Check for JavaScript errors in browser console

### Browser Console

Access browser developer tools (F12) to check for errors:

```javascript
// Check WebSocket connection
console.log(window.WebSocket);

// Check for JavaScript errors
console.log('Dashboard loaded successfully');
```

## Getting Help

- **Tooltips**: Hover over charts and controls for help
- **Help Menu**: Click the `?` icon for context-sensitive help
- **Documentation**: This guide and [API Reference](../api-reference/dashboard-integration.md)
- **Support**: [GitHub Issues](https://github.com/dhar174/steel_defect_demo/issues)

---

Next: [Real-time Monitoring →](real-time-monitoring.md)