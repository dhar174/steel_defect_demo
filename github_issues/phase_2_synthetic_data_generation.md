# Phase 2: Synthetic Data Generation

## Description

Implement a comprehensive synthetic data generator that creates realistic continuous steel casting sensor data with controllable defect scenarios. This generator will serve as the foundation for training and testing the predictive quality monitoring system.

## Context

As outlined in the Technical Design, the PoC uses synthetic data while maintaining architectural compatibility with real plant data feeds. The generator must produce realistic multivariate time-series data that captures the complex relationships between thermal, mechanical, material, environmental, and equipment condition variables that influence defect formation in continuous casting.

## Objectives

- Create parameterizable synthetic data generator for continuous casting scenarios
- Generate realistic baseline signals with controlled stochastic variations
- Implement deterministic rules mapping process excursions to defect labels
- Produce training datasets and real-time streaming capabilities
- Ensure data quality and realistic process dynamics

## Acceptance Criteria

### Core Data Generator (`data_gen.py`)
- [ ] Parameterizable generation (number of casts, duration, sampling rate)
- [ ] Generate all required data source categories:
  - Process parameters (time-series)
  - Material properties (per heat/sequence)
  - Environmental data (slow time-series)
  - Equipment condition signals
  - Quality outputs (labels/targets)
- [ ] Configurable via YAML configuration files
- [ ] Deterministic seeding for reproducible datasets

### Data Source Implementation

#### Process Parameters (1 Hz time-series)
- [ ] **Casting speed** (m/min): Normal variation ±5%, controllable excursions
- [ ] **Tundish temperature** (°C): Gradual cooling trend with fluctuations
- [ ] **Mold steel temperature** (°C): Related to tundish with heat transfer delays
- [ ] **Mold level** (mm): Target ±5mm, occasional stability issues
- [ ] **Mold level fluctuations**: Standard deviation as quality indicator
- [ ] **Mold oscillation frequency** (Hz): Typically 60-200 Hz with variations
- [ ] **Mold oscillation stroke** (mm): Related to casting speed
- [ ] **Cooling water flow rates** (L/min): Multiple zones with control loops
- [ ] **Cooling water temperatures** (°C): Inlet/outlet with heat transfer
- [ ] **Stopper rod position** or slide gate opening (%): Flow control
- [ ] **Electromagnetic stirring** (if present): Current/frequency parameters

#### Material Properties (per cast)
- [ ] **Steel grade**: Categorical (e.g., 'Low_Carbon', 'HSLA', 'Stainless')
- [ ] **Chemical composition**: Key elements (C, Mn, Si, S, P) with grade-appropriate ranges
- [ ] **Inclusion cleanliness index**: Score affecting defect probability
- [ ] **Superheat** (°C): Above liquidus temperature, affects cooling dynamics

#### Environmental Data (slow time-series, ~0.1 Hz)
- [ ] **Ambient air temperature** (°C): Seasonal/daily variations
- [ ] **Humidity** (%): Affects cooling efficiency
- [ ] **Cooling water supply conditions**: Temperature and pressure variations

#### Equipment Condition Signals
- [ ] **Oscillator vibration**: Accelerometer data with health indicators
- [ ] **Roller vibration**: Multiple measurement points
- [ ] **Hydraulic pressures**: System health indicators
- [ ] **Motor currents**: Load variations and anomaly detection
- [ ] **Maintenance flags**: Scheduled/unscheduled maintenance events

### Defect Logic Implementation
- [ ] **Defect triggering rules** based on process excursions:
  - Prolonged mold level deviation (±20mm for >60s) → Surface cracks
  - Rapid temperature drop (>50°C in 30s) + high speed → Internal cracks
  - Oscillation frequency outside normal range → Corner cracks
  - High superheat + cooling imbalance → Inclusion entrapment
  - Equipment vibration spikes → Shape deformities
- [ ] **Realistic defect probability**: ~10-20% of casts with defects
- [ ] **Multiple defect types**: Binary classification initially, extensible to multi-class
- [ ] **Severity indicators**: Configurable defect intensity levels

### Output Format and Structure
- [ ] **Raw time-series data**: 
  - Parquet format: `cast_id`, `timestamp`, `sensor_name`, `value`
  - Or wide CSV format: `timestamp`, `sensor1`, `sensor2`, ..., `cast_id`
- [ ] **Cast metadata**: JSON files with:
  - Cast ID, steel grade, composition, superheat
  - Start/end timestamps, total duration
  - Defect labels and types
  - Process summary statistics
- [ ] **Training dataset**: 1000+ simulated casts with labels
- [ ] **Streaming capability**: Real-time playback simulation

## Implementation Tasks

### Core Generator Architecture
```python
class SteelCastingDataGenerator:
    def __init__(self, config_path):
        # Load configuration
        # Initialize parameter ranges
        # Set up random seed handling
    
    def generate_cast(self, cast_id, duration_minutes=60):
        # Generate single cast time-series
        # Apply defect triggering logic
        # Return structured data
    
    def generate_dataset(self, n_casts=1000):
        # Generate full training dataset
        # Balance defect/normal ratio
        # Save in specified formats
    
    def simulate_streaming(self, cast_data, realtime_factor=1.0):
        # Replay cast data in pseudo real-time
        # Yield timestamped sensor readings
```

### Process Dynamics Modeling
- [ ] **Thermal modeling**: Heat transfer relationships between tundish, mold, cooling
- [ ] **Mechanical coupling**: Speed/oscillation/mold level interactions
- [ ] **Control system simulation**: PID-like responses for level/temperature control
- [ ] **Noise modeling**: Realistic sensor noise characteristics

### Configuration Management
- [ ] **YAML configuration**: `configs/data_generation.yaml`
  - Parameter ranges for each sensor
  - Defect triggering thresholds
  - Output format specifications
  - Statistical distributions
- [ ] **Scenario templates**: Pre-defined parameter sets for different scenarios
- [ ] **Validation rules**: Ensure generated data meets physical constraints

### Data Quality Assurance
- [ ] **Statistical validation**: Verify distributions match expected ranges
- [ ] **Correlation checks**: Ensure realistic relationships between variables
- [ ] **Defect rate validation**: Confirm target defect percentages
- [ ] **Time series properties**: Check for appropriate autocorrelation and trends

## Dependencies

- **Prerequisite**: Phase 1 (Environment Setup) must be complete
- **Required libraries**: pandas, numpy, pyyaml, scipy (for distributions)
- **Configuration**: YAML configuration framework from Phase 1

## Expected Deliverables

1. **`src/data_generation/data_gen.py`**: Main generator implementation
2. **`configs/data_generation.yaml`**: Comprehensive configuration file
3. **`data/synthetic/`**: Generated training datasets
   - `training_dataset.parquet` or CSV files
   - `cast_metadata.json` files
   - Data quality reports
4. **Documentation**: 
   - Generator usage instructions
   - Data schema documentation
   - Defect logic explanation
5. **Validation scripts**: Data quality verification tools

## Technical Considerations

### Realism and Physics
- Respect physical constraints (temperature ranges, flow rates)
- Model realistic process dynamics and control responses
- Include appropriate measurement noise and sensor characteristics
- Ensure temporal correlations match industrial expectations

### Scalability
- Efficient generation for large datasets (10k+ casts)
- Memory-efficient streaming for real-time simulation
- Configurable complexity vs. performance trade-offs

### Extensibility
- Modular design for adding new sensor types
- Flexible defect logic framework
- Easy parameter tuning and scenario creation

### Data Engineering
- Consistent timestamp handling and alignment
- Robust handling of missing data scenarios
- Efficient storage formats for downstream processing

## Success Metrics

- [ ] Generate 1000+ realistic casts in <10 minutes
- [ ] Defect rate within 10-20% range with configurable targeting
- [ ] Generated time series pass statistical validation tests
- [ ] Data format compatible with downstream ML pipeline
- [ ] Real-time streaming simulation maintains target timing accuracy

## Testing and Validation

- [ ] **Unit tests**: Individual component functionality
- [ ] **Integration tests**: Full dataset generation workflow
- [ ] **Statistical tests**: Distribution validation, correlation checks
- [ ] **Visual validation**: Time series plots, correlation heatmaps
- [ ] **Performance tests**: Generation speed and memory usage

## Notes

The synthetic data generator is crucial for the entire project's success. Focus on creating realistic, physically-plausible data that will train effective models. The generator should be sophisticated enough to capture the complexity of real continuous casting while remaining configurable and maintainable.

Consider implementing the generator incrementally:
1. Start with basic time series generation
2. Add physical constraints and relationships
3. Implement defect triggering logic
4. Add advanced dynamics and noise modeling

## Labels
`enhancement`, `phase-2`, `data-generation`, `synthetic-data`, `core-functionality`

## Priority
**High** - Required for all subsequent model development phases