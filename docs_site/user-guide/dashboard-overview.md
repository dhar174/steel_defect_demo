# Dashboard Overview

The Steel Defect Prediction System dashboard provides a comprehensive real-time monitoring interface for steel casting operations.
This guide walks you through all major features and components.

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

```text
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Current Defect  │ │ Avg Confidence  │ │ Active Alerts   │
│ Probability     │ │                 │ │                 │
│                 │ │                 │ │                 │
│     0.15        │ │     0.89        │ │       2         │
│                 │ │                 │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### 2. Model Comparison

### 2. Model Comparison

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

```text
Model Performance Comparison
                    XGBoost    LSTM     Ensemble
Accuracy            0.87       0.89     0.91
Precision           0.84       0.88     0.90
Recall              0.82       0.86     0.89
F1-Score            0.83       0.87     0.89
Training Time       2.3 min    8.7 min  11.2 min
```

### 3. Historical Analysis

### 3. Historical Analysis

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

### 4. Alert Management

Configure and manage the alerting system:

#### Alert Configuration

- **Threshold Settings**: Defect probability limits
- **Notification Methods**: Email, SMS, dashboard
- **Escalation Rules**: Multi-level alert hierarchy
- **Time-based Rules**: Different thresholds by shift/time

#### Active Alerts Dashboard

```text
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

```text
[Last Hour] [Last 4 Hours] [Last Day] [Custom Range]
```

### Refresh Controls

```text
Auto-refresh: [ON/OFF]  Interval: [5s] [15s] [30s] [1m]
```

### Export Options

```text
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
- **Help Menu**: Click the `?` icon for con-sensitive help
- **Documentation**: This guide and [API Reference](../api-reference/dashboard-integration.md)
- **Support**: [GitHub Issues](https://github.com/dhar174/steel_defect_demo/issues)

---

Next: [API Reference →](../api-reference/dashboard-integration.md)
=======

## Introduction

The Steel Defect Detection Dashboard provides a comprehensive interface for
monitoring continuous steel casting processes and predicting quality defects
in real-time. This guide covers all aspects of using the dashboard effectively.

## Getting Started

### Accessing the Dashboard

The dashboard is accessible through a web browser at the configured URL.
Default local access is typically available at:

```http
http://localhost:8501
```

### Authentication

Currently, the dashboard operates in demonstration mode without authentication
requirements. In production deployments, appropriate access controls should
be implemented.

## Main Interface Components

### Navigation Menu

The dashboard features a sidebar navigation with the following sections:

- **Home**: Overview and system status
- **Real-time Monitoring**: Live sensor data and predictions
- **Historical Analysis**: Past performance and trends
- **Alert Management**: Active alerts and notification settings
- **Model Performance**: ML model metrics and evaluation
- **Configuration**: System settings and parameters

### Status Indicators

Color-coded status indicators provide quick visual feedback:

- **Green**: Normal operation, no issues detected
- **Yellow**: Warning condition, attention recommended
- **Red**: Alert condition, immediate action required
- **Gray**: No data or system offline

## Real-time Monitoring

### Live Data Display

The monitoring interface shows real-time sensor data including:

- **Temperature readings** from multiple casting zones
- **Pressure measurements** throughout the process
- **Flow rates** for cooling water and molten steel
- **Chemical composition** data when available

### Prediction Results

ML model predictions are displayed with:

- **Defect probability scores** for different defect types
- **Confidence intervals** for prediction reliability
- **Risk assessments** based on current process conditions
- **Trend indicators** showing prediction changes over time

### Interactive Charts

Charts and visualizations provide detailed insights:

- **Time series plots** for sensor data trends
- **Correlation matrices** showing parameter relationships
- **Distribution plots** for quality metrics
- **Prediction timelines** with historical con

## Alert Management

### Alert Types

The system monitors for several types of quality issues:

- **Surface cracks**: Longitudinal and transverse defects
- **Internal defects**: Inclusions and porosity
- **Dimensional variations**: Thickness and width deviations
- **Chemical composition**: Out-of-specification conditions

### Alert Configuration

Users can configure alert parameters:

- **Threshold values** for different defect probabilities
- **Notification methods** (dashboard, email, SMS)
- **Escalation rules** for critical conditions
- **Acknowledgment requirements** for alert resolution

### Alert History

The alert history section provides:

- **Chronological log** of all alert events
- **Resolution tracking** and response times
- **Performance metrics** for alert accuracy
- **Export capabilities** for reporting

## Historical Analysis

### Data Exploration

Historical analysis tools include:

- **Date range selection** for specific time periods
- **Filter options** by shift, product type, or conditions
- **Statistical summaries** of process performance
- **Trend analysis** with regression and forecasting

### Performance Metrics

Key performance indicators (KPIs) tracked:

- **Defect detection rate** and false positive percentages
- **Process efficiency** and throughput metrics
- **Quality scores** and customer satisfaction data
- **Cost impact** analysis for defect prevention

### Reporting Features

Generate comprehensive reports with:

- **Executive summaries** for management review
- **Technical details** for process engineers
- **Trend analysis** with recommendations
- **Export formats** (PDF, Excel, CSV)

## Model Performance

### Model Metrics

Monitor ML model performance through:

- **Accuracy scores** and confusion matrices
- **Precision and recall** for different defect types
- **ROC curves** and AUC measurements
- **Feature importance** rankings

### Model Comparison

Compare different model approaches:

- **LSTM vs baseline** model performance
- **Training progress** and validation curves
- **Hyperparameter optimization** results
- **Ensemble model** performance

## Configuration Settings

### System Parameters

Configurable system settings include:

- **Data ingestion** frequency and sources
- **Model prediction** intervals and thresholds
- **Display preferences** and dashboard layout
- **User access** controls and permissions

### Data Sources

Configure data input connections:

- **PLC interfaces** for real-time sensor data
- **Database connections** for historical information
- **File imports** for batch processing
- **API endpoints** for external systems

## Troubleshooting

### Common Issues

Frequently encountered problems and solutions:

- **Data connectivity**: Check network and sensor connections
- **Slow performance**: Review data volume and processing load
- **Prediction accuracy**: Verify model training and data quality
- **Display issues**: Clear browser cache and check compatibility

### Support Resources

Additional help and support:

- **User manual** with detailed instructions
- **Technical documentation** for advanced configuration
- **Contact information** for system administrators
- **Training materials** and video tutorials

## Best Practices

### Monitoring Guidelines

Recommended practices for effective monitoring:

- **Regular review** of alert thresholds and accuracy
- **Shift handover** procedures using dashboard data
- **Trend analysis** for predictive maintenance
- **Quality improvement** based on historical patterns

### Data Management

Maintain data quality through:

- **Regular calibration** of sensors and instruments
- **Data validation** checks and error correction
- **Backup procedures** for critical information
- **Archive management** for long-term storage

## Conclusion

The Steel Defect Detection Dashboard provides powerful tools for monitoring
and improving continuous casting quality. Regular use of all features will
maximize the benefits of predictive defect detection and help maintain
optimal process performance.

For additional support or questions about specific features, consult the
technical documentation or contact the system development team.
