# Historical Analysis

The Historical Analysis module provides comprehensive tools for analyzing past casting operations, identifying patterns, and improving future predictions based on historical data trends.

## Overview

Historical analysis capabilities include:

- Trend analysis of defect patterns over time
- Process parameter correlation analysis
- Seasonal and cyclical pattern detection
- Root cause analysis of defect occurrences
- Performance benchmarking and reporting

## Accessing Historical Data

### Data Sources

Historical data is collected from multiple sources:

```python
from src.analysis.historical_analyzer import HistoricalAnalyzer

# Initialize analyzer with data sources
analyzer = HistoricalAnalyzer(
    data_sources=[
        'sensor_data',
        'quality_inspections',
        'production_logs',
        'maintenance_records'
    ],
    date_range=('2023-01-01', '2024-01-01')
)
```

### Data Loading

```python
# Load historical dataset
historical_data = analyzer.load_data(
    table='production_history',
    filters={
        'steel_grade': ['304L', '316L', '410'],
        'shift': ['morning', 'afternoon', 'night'],
        'quality_status': ['pass', 'fail', 'rework']
    }
)

print(f"Loaded {len(historical_data)} records")
print(f"Date range: {historical_data['timestamp'].min()} to {historical_data['timestamp'].max()}")
```

## Trend Analysis

### Defect Rate Trends

```python
# Analyze defect rate trends over time
defect_trends = analyzer.analyze_defect_trends(
    groupby='month',
    metrics=['defect_rate', 'severity_level', 'defect_type']
)

# Visualize trends
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.lineplot(data=defect_trends, x='month', y='defect_rate')
plt.title('Monthly Defect Rate Trends')
plt.ylabel('Defect Rate (%)')
plt.show()
```

### Process Parameter Trends

```python
# Analyze key process parameters over time
parameter_trends = analyzer.analyze_parameter_trends(
    parameters=[
        'mold_temperature',
        'casting_speed',
        'cooling_water_flow',
        'steel_composition'
    ],
    aggregation='daily'
)

# Multi-parameter trend visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
parameters = ['mold_temperature', 'casting_speed', 'cooling_water_flow', 'steel_composition']

for i, param in enumerate(parameters):
    ax = axes[i//2, i%2]
    sns.lineplot(data=parameter_trends, x='date', y=param, ax=ax)
    ax.set_title(f'{param.replace("_", " ").title()} Trend')
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

## Correlation Analysis

### Parameter Correlation

```python
# Calculate correlation between process parameters and defect occurrence
correlation_matrix = analyzer.calculate_correlations(
    target='defect_probability',
    features=[
        'mold_temperature',
        'casting_speed',
        'cooling_water_flow',
        'oxygen_content',
        'carbon_content'
    ]
)

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Process Parameter Correlations with Defect Probability')
plt.show()
```

### Cross-correlation Analysis

```python
# Time-lagged correlation analysis
cross_correlations = analyzer.cross_correlation_analysis(
    primary_signal='mold_temperature',
    secondary_signal='defect_rate',
    max_lag=24  # hours
)

# Plot cross-correlation
plt.figure(figsize=(10, 6))
plt.plot(cross_correlations['lag'], cross_correlations['correlation'])
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Lag (hours)')
plt.ylabel('Cross-correlation')
plt.title('Cross-correlation: Mold Temperature vs Defect Rate')
plt.grid(True, alpha=0.3)
plt.show()
```

## Pattern Detection

### Seasonal Patterns

```python
from src.analysis.pattern_detector import SeasonalPatternDetector

# Detect seasonal patterns in defect occurrence
seasonal_detector = SeasonalPatternDetector()
seasonal_patterns = seasonal_detector.detect_patterns(
    data=historical_data,
    target='defect_rate',
    periods=['hourly', 'daily', 'weekly', 'monthly']
)

# Visualize seasonal patterns
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
periods = ['hourly', 'daily', 'weekly', 'monthly']

for i, period in enumerate(periods):
    ax = axes[i//2, i%2]
    pattern_data = seasonal_patterns[period]
    ax.plot(pattern_data['time'], pattern_data['average_defect_rate'])
    ax.set_title(f'{period.title()} Defect Rate Pattern')
    ax.set_ylabel('Average Defect Rate (%)')

plt.tight_layout()
plt.show()
```

### Anomaly Detection

```python
from src.analysis.anomaly_detector import HistoricalAnomalyDetector

# Detect historical anomalies
anomaly_detector = HistoricalAnomalyDetector(
    method='isolation_forest',
    contamination=0.1  # 10% expected anomalies
)

anomalies = anomaly_detector.detect_anomalies(
    data=historical_data,
    features=['mold_temperature', 'casting_speed', 'cooling_water_flow']
)

# Visualize anomalies
plt.figure(figsize=(12, 8))
plt.scatter(historical_data['timestamp'], historical_data['defect_rate'], 
           c='blue', alpha=0.6, label='Normal')
plt.scatter(anomalies['timestamp'], anomalies['defect_rate'], 
           c='red', alpha=0.8, label='Anomalies')
plt.xlabel('Time')
plt.ylabel('Defect Rate (%)')
plt.title('Historical Anomalies in Defect Rate')
plt.legend()
plt.xticks(rotation=45)
plt.show()
```

## Root Cause Analysis

### Statistical Analysis

```python
from src.analysis.root_cause_analyzer import RootCauseAnalyzer

# Perform root cause analysis for high defect periods
rca_analyzer = RootCauseAnalyzer()

# Identify periods with high defect rates
high_defect_periods = historical_data[historical_data['defect_rate'] > 15]

# Analyze contributing factors
root_causes = rca_analyzer.analyze_factors(
    high_defect_data=high_defect_periods,
    normal_data=historical_data[historical_data['defect_rate'] <= 5],
    factors=[
        'mold_temperature',
        'casting_speed',
        'steel_grade',
        'shift',
        'operator_id',
        'maintenance_status'
    ]
)

# Display top contributing factors
print("Top Contributing Factors to High Defect Rates:")
for factor, importance in root_causes.items():
    print(f"  {factor}: {importance:.3f}")
```

### Decision Tree Analysis

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Build decision tree for defect prediction
features = ['mold_temperature', 'casting_speed', 'cooling_water_flow', 
           'oxygen_content', 'carbon_content']

X = historical_data[features]
y = (historical_data['defect_rate'] > 10).astype(int)  # Binary classification

dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X, y)

# Visualize decision tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=features, class_names=['Normal', 'High Defect'],
          filled=True, fontsize=10)
plt.title('Decision Tree for Defect Prediction')
plt.show()
```

## Time Series Analysis

### Trend Decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose time series into trend, seasonal, and residual components
defect_ts = historical_data.set_index('timestamp')['defect_rate'].resample('D').mean()

decomposition = seasonal_decompose(defect_ts, model='additive', period=7)

# Plot decomposition
fig, axes = plt.subplots(4, 1, figsize=(12, 10))
decomposition.observed.plot(ax=axes[0], title='Original')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
plt.show()
```

### Forecast Analysis

```python
from statsmodels.tsa.arima.model import ARIMA

# ARIMA forecasting for defect rates
model = ARIMA(defect_ts, order=(2, 1, 2))
fitted_model = model.fit()

# Generate forecast
forecast = fitted_model.forecast(steps=30)  # 30 days ahead
forecast_index = pd.date_range(start=defect_ts.index[-1] + pd.Timedelta(days=1), 
                              periods=30, freq='D')

# Plot historical data and forecast
plt.figure(figsize=(12, 6))
plt.plot(defect_ts.index[-90:], defect_ts[-90:], label='Historical', color='blue')
plt.plot(forecast_index, forecast, label='Forecast', color='red', linestyle='--')
plt.fill_between(forecast_index, 
                forecast - 1.96*np.sqrt(fitted_model.forecast(steps=30, return_conf_int=True)[1][:, 1]),
                forecast + 1.96*np.sqrt(fitted_model.forecast(steps=30, return_conf_int=True)[1][:, 1]),
                alpha=0.3, color='red')
plt.xlabel('Date')
plt.ylabel('Defect Rate (%)')
plt.title('Defect Rate Forecast')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Performance Analysis

### Production Efficiency

```python
# Analyze production efficiency metrics
efficiency_metrics = analyzer.calculate_efficiency_metrics(
    metrics=[
        'overall_equipment_effectiveness',
        'first_pass_yield',
        'defect_rate',
        'production_rate',
        'downtime_percentage'
    ],
    groupby=['month', 'steel_grade', 'shift']
)

# Efficiency trend analysis
plt.figure(figsize=(12, 8))
for steel_grade in efficiency_metrics['steel_grade'].unique():
    grade_data = efficiency_metrics[efficiency_metrics['steel_grade'] == steel_grade]
    plt.plot(grade_data['month'], grade_data['overall_equipment_effectiveness'], 
            label=steel_grade, marker='o')

plt.xlabel('Month')
plt.ylabel('Overall Equipment Effectiveness (%)')
plt.title('OEE Trends by Steel Grade')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Quality Benchmarking

```python
# Benchmark against industry standards or internal targets
benchmarks = {
    'defect_rate_target': 5.0,  # Target: < 5%
    'first_pass_yield_target': 95.0,  # Target: > 95%
    'oee_target': 85.0  # Target: > 85%
}

performance_vs_benchmark = analyzer.benchmark_performance(
    data=historical_data,
    benchmarks=benchmarks,
    groupby='month'
)

# Visualize performance vs benchmarks
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['defect_rate', 'first_pass_yield', 'oee']

for i, metric in enumerate(metrics):
    ax = axes[i]
    ax.plot(performance_vs_benchmark['month'], 
           performance_vs_benchmark[metric], 
           label='Actual', marker='o')
    ax.axhline(y=benchmarks[f'{metric}_target'], 
              color='red', linestyle='--', label='Target')
    ax.set_title(f'{metric.replace("_", " ").title()}')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Report Generation

### Automated Reports

```python
from src.reporting.historical_reporter import HistoricalReporter

# Generate comprehensive historical analysis report
reporter = HistoricalReporter()
report = reporter.generate_report(
    data=historical_data,
    analysis_period='2023-01-01 to 2023-12-31',
    include_sections=[
        'executive_summary',
        'trend_analysis',
        'correlation_analysis',
        'pattern_detection',
        'root_cause_analysis',
        'recommendations'
    ]
)

# Save report
reporter.save_report(report, 'reports/historical_analysis_2023.html')
```

### Custom Analysis Templates

```python
# Create custom analysis template
template = {
    'title': 'Monthly Quality Review',
    'sections': [
        {
            'name': 'Defect Rate Summary',
            'type': 'metric_summary',
            'metrics': ['defect_rate', 'severity_distribution']
        },
        {
            'name': 'Process Parameter Analysis',
            'type': 'correlation_matrix',
            'parameters': ['mold_temperature', 'casting_speed']
        },
        {
            'name': 'Trend Charts',
            'type': 'time_series_plots',
            'variables': ['defect_rate', 'production_volume']
        }
    ]
}

# Generate report from template
custom_report = reporter.generate_from_template(template, historical_data)
```

This historical analysis framework provides deep insights into your steel casting operations, enabling data-driven improvements and proactive quality management.