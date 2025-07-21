# SequenceFeatureExtractor Documentation

## Overview

The `SequenceFeatureExtractor` class provides a comprehensive pipeline for converting raw time-series sensor data into fixed-length sequences suitable for LSTM training. It includes sliding window extraction, normalization, padding/truncation, data augmentation, and temporal-aware dataset splitting.

## Features

### Core Functionality
- **Sliding Window Extraction**: Extract overlapping sequences with configurable window size and stride
- **Normalization**: Z-score and MinMax scaling per sensor channel
- **Sequence Processing**: Padding/truncation to fixed lengths
- **Data Augmentation**: Noise injection and time warping techniques
- **Temporal Splitting**: Train/validation/test splits preserving temporal order

### Configuration Options
- Sequence length (default: 300 for 5 minutes at 1Hz)
- Stride for sliding window (default: 1)
- Augmentation parameters:
  - `noise_std`: Standard deviation for Gaussian noise (default: 0.01)
  - `time_warp_max`: Maximum time warping factor (default: 0.1)
  - `augmentation_ratio`: Fraction of data to augment (default: 0.2)
  - `min_max_scaling`: Use MinMax instead of Z-score scaling (default: False)

## Usage

### Basic Usage

```python
from features.feature_extractor import SequenceFeatureExtractor
import pandas as pd
import numpy as np

# Initialize extractor
extractor = SequenceFeatureExtractor(
    sequence_length=300,    # 5 minutes at 1Hz
    stride=10,             # 10-second overlap
    augmentation_config={
        'noise_std': 0.01,
        'time_warp_max': 0.1,
        'augmentation_ratio': 0.2,
        'min_max_scaling': False
    }
)

# Prepare sensor data
data = pd.DataFrame({
    'temperature': temperature_readings,
    'pressure': pressure_readings,
    'flow_rate': flow_readings,
    'vibration': vibration_readings,
    'power_consumption': power_readings,
    'label': defect_labels
})

# Extract and prepare sequences
sequences, labels = extractor.prepare_sequences(data)
```

### Step-by-Step Pipeline

```python
# 1. Extract sequences using sliding window
feature_data = data.drop('label', axis=1)
sequences = extractor.extract_sequences(feature_data)

# 2. Normalize sequences
normalized_sequences = extractor.normalize_sequences(sequences)

# 3. Apply data augmentation (optional)
augmented_sequences = extractor.augment_sequences(normalized_sequences)

# 4. Split into train/val/test sets
X_train, X_val, X_test, y_train, y_val, y_test = extractor.split_sequences(
    augmented_sequences, labels, split_ratios=(0.7, 0.15, 0.15)
)
```

### Advanced Usage

```python
# Custom configuration for different scenarios
config_high_noise = {
    'noise_std': 0.05,
    'time_warp_max': 0.2,
    'augmentation_ratio': 0.5,
    'min_max_scaling': True
}

extractor = SequenceFeatureExtractor(
    sequence_length=600,  # 10 minutes
    stride=30,           # 30-second stride
    augmentation_config=config_high_noise
)

# Process individual casts
cast_sequences = extractor.create_sequences_from_cast(cast_data)

# Extract sliding windows with custom parameters
windows = extractor.extract_sliding_windows(
    time_series_data, 
    window_size=100, 
    stride=5
)

# Pad variable-length sequences
padded_sequences = extractor.pad_sequences(variable_sequences)
```

## API Reference

### Class: SequenceFeatureExtractor

#### Constructor
```python
__init__(sequence_length: int = 300, stride: int = 1, augmentation_config: dict = None)
```

#### Methods

##### extract_sequences(data: pd.DataFrame) -> np.ndarray
Extract overlapping sequences from continuous sensor data.
- **Parameters**: DataFrame with sensor readings
- **Returns**: Array of shape (n_sequences, sequence_length, n_features)

##### normalize_sequences(sequences: np.ndarray) -> np.ndarray
Apply normalization to sequences per feature channel.
- **Parameters**: Array of sequences to normalize
- **Returns**: Normalized sequences with same shape

##### pad_sequences(sequences: List[np.ndarray]) -> np.ndarray
Pad/truncate sequences to fixed length.
- **Parameters**: List of variable-length sequences
- **Returns**: Fixed-length sequences

##### augment_sequences(sequences: np.ndarray) -> np.ndarray
Apply data augmentation techniques.
- **Parameters**: Original sequences to augment
- **Returns**: Augmented sequences (original + augmented)

##### split_sequences(sequences: np.ndarray, labels: np.ndarray, split_ratios: Tuple[float, float, float]) -> Tuple
Temporal-aware train/validation/test split.
- **Parameters**: Sequences, labels, and split ratios (train, val, test)
- **Returns**: (X_train, X_val, X_test, y_train, y_val, y_test)

##### prepare_sequences(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]
End-to-end sequence preparation from raw data.
- **Parameters**: DataFrame with sensor data and labels
- **Returns**: (sequences, labels)

##### fit_normalizer(sequences: np.ndarray) -> None
Fit normalizer on training sequences.
- **Parameters**: Training sequences for fitting scaler

##### extract_sliding_windows(time_series: pd.DataFrame, window_size: int = None, stride: int = 1) -> List[np.ndarray]
Extract sliding windows with custom parameters.
- **Parameters**: Time series data, window size, stride
- **Returns**: List of sliding windows

##### create_sequences_from_cast(cast_data: pd.DataFrame) -> np.ndarray
Create sequences from single cast data.
- **Parameters**: Time series data for a single cast
- **Returns**: Sequence representation

## Data Augmentation Techniques

### Noise Injection
Adds Gaussian noise to sequences to improve model robustness:
```python
noise = np.random.normal(0, noise_std, sequence.shape)
augmented_sequence = original_sequence + noise
```

### Time Warping
Stretches or compresses the time axis randomly:
- Random warping factor: `1 ± time_warp_max`
- Uses linear interpolation to maintain data integrity
- Preserves original sequence length

## Temporal-Aware Splitting

The splitting functionality preserves temporal order to prevent data leakage:
- Training data comes from earlier time periods
- Validation data from intermediate time periods  
- Test data from latest time periods
- No shuffling across temporal boundaries

## Configuration Examples

### Low-Resource Scenario
```python
config = {
    'noise_std': 0.005,
    'time_warp_max': 0.05,
    'augmentation_ratio': 0.1,
    'min_max_scaling': False
}
```

### High-Augmentation Scenario
```python
config = {
    'noise_std': 0.02,
    'time_warp_max': 0.15,
    'augmentation_ratio': 0.4,
    'min_max_scaling': True
}
```

### Production Deployment
```python
config = {
    'noise_std': 0.01,
    'time_warp_max': 0.1,
    'augmentation_ratio': 0.0,  # No augmentation for inference
    'min_max_scaling': False
}
```

## Integration with LSTM Pipeline

The SequenceFeatureExtractor integrates seamlessly with LSTM training pipelines:

```python
# Feature extraction
extractor = SequenceFeatureExtractor(sequence_length=300, stride=10)
X_train, X_val, X_test, y_train, y_val, y_test = extractor.prepare_and_split(data)

# LSTM model training
from models.lstm_model import LSTMDefectClassifier

model = LSTMDefectClassifier(
    input_dim=X_train.shape[2],
    sequence_length=X_train.shape[1],
    hidden_dim=128,
    num_layers=2
)

model.fit(X_train, y_train, validation_data=(X_val, y_val))
```

## Testing

Comprehensive test suite available in `tests/test_feature_extractor.py`:
- Unit tests for each method
- Edge case handling
- Integration tests
- Configuration validation
- Performance benchmarks

Run tests with:
```bash
pytest tests/test_feature_extractor.py -v
```

## Dependencies

- `numpy>=1.23.0`: Array operations and numerical computing
- `pandas>=1.5.0`: DataFrame operations and data handling
- `scikit-learn>=1.1.0`: Scaling and preprocessing utilities
- `scipy>=1.9.0`: Interpolation for time warping

## Performance Considerations

- Memory usage scales with `sequence_length × n_features × n_sequences`
- Time warping is computationally intensive; adjust `augmentation_ratio` accordingly
- For large datasets, consider processing in batches
- Fitting normalizer once on training data is sufficient

## Best Practices

1. **Sequence Length**: Choose based on your temporal patterns (e.g., 300 for 5-minute windows at 1Hz)
2. **Stride**: Smaller stride = more overlap = more training data but higher computational cost
3. **Augmentation**: Start with conservative parameters and increase gradually
4. **Normalization**: Fit on training data only, then transform all sets
5. **Splitting**: Use temporal splits for time-series data to avoid leakage