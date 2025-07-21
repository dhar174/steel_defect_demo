# Data Quality Assessment

This document describes the comprehensive data quality assessment system implemented for steel casting sensor data.

## Overview

The Data Quality Assessment module provides comprehensive validation of steel casting sensor data through four core assessment components:

1. **Missing Value Analysis** - Detects gaps in sensor data
2. **Data Consistency Checks** - Verifies sensor readings are within expected ranges
3. **Temporal Continuity** - Ensures proper time sequencing in data
4. **Synthetic Data Realism** - Compares generated patterns to expected steel casting behavior

## Components

### 1. Missing Value Analysis

Analyzes missing values and temporal gaps in sensor data:

- **Missing Value Detection**: Identifies missing values per sensor and calculates percentages
- **Temporal Gap Analysis**: Detects gaps in time series data exceeding threshold
- **Quality Scoring**: Calculates quality score based on data completeness

**Key Metrics:**
- Total missing percentage
- Missing values by sensor
- Temporal gaps with duration and location
- Quality score (0-1 scale)

### 2. Data Consistency Checks

Validates sensor readings against expected operational ranges and physics constraints:

- **Range Violations**: Detects values outside hard limits and typical operational ranges
- **Physics Constraints**: Validates rate of change constraints based on steel casting physics
- **Outlier Detection**: Uses Z-score and IQR methods to identify outliers

**Expected Sensor Ranges:**
- Casting Speed: 0.5-2.5 m/min (typical: 0.8-1.8)
- Mold Temperature: 1400-1650°C (typical: 1480-1580)
- Mold Level: 100-200mm (typical: 120-180)
- Cooling Water Flow: 100-300 L/min (typical: 150-250)
- Superheat: 10-50°C (typical: 15-40)

**Physics Constraints:**
- Temperature gradient: ≤100°C/min
- Speed gradient: ≤0.3 m/min/min
- Level gradient: ≤30 mm/min
- Flow gradient: ≤50 L/min/min
- Superheat gradient: ≤15°C/min

### 3. Temporal Continuity

Ensures proper time sequencing and sampling rate consistency:

- **Sampling Rate Analysis**: Validates 1Hz expected sampling rate
- **Time Sequence Validation**: Checks for monotonic timestamps and duplicates
- **Coverage Analysis**: Calculates temporal coverage percentage

**Key Metrics:**
- Mean/std interval times
- Irregular interval percentage
- Monotonic sequence validation
- Duplicate timestamp detection
- Coverage percentage

### 4. Synthetic Data Realism

Assesses whether generated data exhibits realistic steel casting behavior:

- **Distribution Analysis**: Examines statistical properties (mean, std, skewness, kurtosis)
- **Correlation Analysis**: Identifies expected correlations between sensors
- **Process Behavior**: Validates temperature stability, speed consistency, mold level control

**Realism Criteria:**
- Temperature stability: <20°C standard deviation
- Speed consistency: <0.1 m/min standard deviation
- Mold level control: <10mm standard deviation, excursions outside 130-170mm range
- Expected correlations between related sensors

## Usage

### Basic Usage

```python
from analysis.data_quality_assessor import DataQualityAssessor

# Initialize assessor
assessor = DataQualityAssessor(data_path='data')

# Assess single cast
cast_data = pd.read_parquet('data/raw/cast_timeseries_0001.parquet')
results = assessor.comprehensive_quality_assessment(cast_data)

# Assess entire dataset
results = assessor.comprehensive_quality_assessment()

# Generate report
report_path = assessor.generate_quality_report(results)
```

### Individual Assessments

```python
# Individual assessment components
missing_results = assessor.assess_missing_values(cast_data)
consistency_results = assessor.assess_data_consistency(cast_data)
temporal_results = assessor.assess_temporal_continuity(cast_data)
realism_results = assessor.assess_synthetic_data_realism(cast_data)
```

### Configuration

The assessor accepts configuration parameters:

```python
config = {
    'missing_value_threshold': 0.01,        # Max 1% missing values allowed
    'temporal_gap_threshold_seconds': 5,    # Max gap of 5 seconds
    'outlier_threshold_std': 3.0,          # Standard deviations for outlier detection
    'physics_violation_threshold': 0.05,    # Max 5% physics violations allowed
    'realism_score_threshold': 0.7         # Minimum realism score (0-1)
}

assessor = DataQualityAssessor(data_path='data', config=config)
```

## Quality Scoring

### Overall Quality Score

The overall quality score is calculated as the mean of all component scores:

- **Excellent** (≥0.9): Data meets all quality standards
- **Good** (≥0.8): Data is suitable for analysis with minor issues
- **Acceptable** (≥0.7): Data has some quality issues that should be addressed
- **Poor** (≥0.6): Data has significant quality issues
- **Unacceptable** (<0.6): Data quality is too poor for reliable analysis

### Component Scoring

Each component provides a score from 0 to 1:

- **Missing Values**: 1 - (missing_percentage / 100)
- **Consistency**: 1 - (violation_rate)
- **Temporal Continuity**: Based on sampling regularity, sequence validity, and coverage
- **Realism**: Based on distribution properties, correlations, and process behavior

## Output Format

Assessment results are provided in a structured JSON format:

```json
{
  "summary": {
    "overall_quality_score": 0.975,
    "quality_level": "Excellent",
    "component_scores": {
      "missing_values": 1.000,
      "consistency": 1.000,
      "temporal_continuity": 1.000,
      "realism": 0.900
    },
    "assessment_timestamp": "2025-07-21T07:56:39.224061",
    "data_scope": "single_cast"
  },
  "missing_value_analysis": { ... },
  "consistency_analysis": { ... },
  "temporal_continuity": { ... },
  "realism_analysis": { ... }
}
```

## Demo Script

Run the comprehensive demonstration:

```bash
python demo_data_quality_assessment.py
```

This script demonstrates:
- Single cast analysis
- Complete dataset analysis
- Degraded data comparison
- Individual assessment components
- Report generation

## Integration

### Data Pipeline Integration

The assessor can be integrated into data processing pipelines:

```python
def process_cast_data(cast_file):
    # Load data
    cast_data = pd.read_parquet(cast_file)
    
    # Assess quality
    assessor = DataQualityAssessor()
    results = assessor.comprehensive_quality_assessment(cast_data)
    
    # Check quality threshold
    if results['summary']['overall_quality_score'] < 0.7:
        raise ValueError(f"Data quality too low: {results['summary']['quality_level']}")
    
    # Continue processing if quality is acceptable
    return process_data(cast_data)
```

### Continuous Monitoring

For production monitoring:

```python
def monitor_data_quality(data_path):
    assessor = DataQualityAssessor(data_path=data_path)
    
    # Regular assessment
    results = assessor.comprehensive_quality_assessment()
    
    # Alert on quality degradation
    if results['summary']['overall_quality_score'] < 0.8:
        send_alert(f"Data quality degraded: {results['summary']['quality_level']}")
    
    # Log results
    log_quality_metrics(results)
```

## Testing

Comprehensive test suite covers:

- All assessment components
- Edge cases (empty data, single row, extreme values)
- Quality score calculations
- Report generation
- Error handling

Run tests:

```bash
python -m pytest tests/test_data_quality_assessment.py -v
```

## Performance Considerations

- **Single Cast**: ~1 second for 7200 samples
- **Dataset Analysis**: Scales linearly with number of casts
- **Memory**: Processes one cast at a time for large datasets
- **Storage**: JSON reports typically 10-50KB per cast

## Future Enhancements

Potential improvements:

1. **Real-time Streaming**: Continuous assessment of streaming data
2. **Adaptive Thresholds**: Machine learning-based threshold adjustment
3. **Anomaly Detection**: Advanced pattern recognition for unusual behavior
4. **Visualization**: Interactive dashboards for quality monitoring
5. **Integration**: REST API for microservice deployment
6. **Alerting**: Configurable alert rules and notification systems

## References

- Steel casting process physics constraints
- Statistical quality control methods
- Time series data validation techniques
- Industrial sensor data quality standards