# Phase 5: Deep Sequence Model Development

## Description

Develop and train deep learning sequence models (LSTM/CNN-based) to capture temporal dependencies in continuous casting data for defect prediction. This phase implements the "Stage 2 Sequence Model" to evaluate the value added by temporal modeling compared to the baseline feature-based approach.

## Context

Per the Technical Design specification, this phase develops PyTorch-based sequence models that can learn temporal patterns and dependencies that may be missed by feature-based approaches. The models will ingest normalized multivariate time series and output defect probability predictions, potentially capturing complex temporal precursors to defect formation.

## Objectives

- Implement PyTorch sequence models for multivariate time series classification
- Develop efficient data loading and preprocessing pipelines for sequence data
- Train and validate sequence models with appropriate regularization
- Compare sequence model performance against baseline feature-based models
- Create model artifacts suitable for real-time inference deployment

## Acceptance Criteria

### Data Pipeline for Sequence Models
- [ ] **PyTorch Dataset implementation**: Efficient loading of multivariate sequences
- [ ] **Sequence preprocessing**: Normalization, padding, missing value handling
- [ ] **Data augmentation**: Optional sequence augmentation for minority class
- [ ] **Batch loading**: Optimized DataLoader with appropriate batch sizes
- [ ] **Memory efficiency**: Streaming data loading for large datasets

### Model Architecture Implementation
- [ ] **LSTM baseline**: Stacked LSTM layers with dropout regularization
- [ ] **CNN-LSTM hybrid**: 1D CNN feature extraction + LSTM temporal modeling
- [ ] **Attention mechanisms**: Optional self-attention layers for long sequences
- [ ] **Output layers**: Fully connected layers with appropriate activation
- [ ] **Modular design**: Configurable architecture components

### Training Infrastructure
- [ ] **Training loop**: Custom training with validation monitoring
- [ ] **Loss functions**: Class-weighted binary cross-entropy for imbalanced data
- [ ] **Optimization**: Adam/AdamW with learning rate scheduling
- [ ] **Early stopping**: Validation-based stopping with patience
- [ ] **Regularization**: Dropout, weight decay, gradient clipping

### Model Evaluation and Comparison
- [ ] **Sequence-specific metrics**: Temporal prediction accuracy analysis
- [ ] **Baseline comparison**: Direct comparison with GBDT baseline
- [ ] **Ablation studies**: Architecture component contribution analysis
- [ ] **Interpretability**: Attention weights and gradient-based explanations
- [ ] **Performance profiling**: Training time, inference speed, memory usage

### Model Artifacts and Deployment
- [ ] **Model serialization**: State dict and full model persistence
- [ ] **ONNX export**: Cross-platform deployment compatibility
- [ ] **Inference utilities**: Fast prediction functions for real-time use
- [ ] **Model metadata**: Architecture specs, training configuration

## Implementation Tasks

### PyTorch Dataset Implementation

#### Sequence Dataset Class
```python
class CastingSequenceDataset(Dataset):
    def __init__(self, data_path, sequence_length=None, 
                 normalize=True, augment=False):
        """
        Dataset for multivariate casting sequences
        
        Args:
            data_path: Path to processed sequence data
            sequence_length: Fixed length or None for variable
            normalize: Apply z-score normalization
            augment: Data augmentation for minority class
        """
        self.sequences = []
        self.labels = []
        self.sensor_names = []
        self.scaler = None
        
    def __getitem__(self, idx):
        # Return (sequence_tensor, label)
        # Shape: (sequence_length, n_sensors)
        
    def __len__(self):
        return len(self.sequences)
        
    def collate_fn(self, batch):
        # Handle variable-length sequences with padding
        # Return batched tensors
```

#### Data Preprocessing Pipeline
```python
class SequencePreprocessor:
    def __init__(self, normalization='zscore', 
                 missing_strategy='forward_fill'):
        self.normalization = normalization
        self.missing_strategy = missing_strategy
        self.scaler = None
        
    def fit(self, sequences):
        # Fit normalization parameters on training data
        # Handle missing values appropriately
        
    def transform(self, sequences):
        # Apply fitted transformations
        # Ensure consistent preprocessing
```

### Model Architecture Development

#### Base LSTM Model
```python
class CastingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, 
                 dropout=0.3, output_size=1):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x, lengths=None):
        # LSTM forward pass
        # Handle variable length sequences
        # Return defect probability
```

#### CNN-LSTM Hybrid Model
```python
class CastingCNNLSTM(nn.Module):
    def __init__(self, input_size, cnn_features=64, 
                 lstm_hidden=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        # 1D CNN for local pattern extraction
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(input_size, cnn_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_features),
            nn.Conv1d(cnn_features, cnn_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_features),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # LSTM for temporal dependency modeling
        self.lstm = nn.LSTM(
            input_size=cnn_features,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Output classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, 1),
            nn.Sigmoid()
        )
```

#### Attention-Enhanced Model (Optional)
```python
class CastingAttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 attention_heads=8, dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           dropout=dropout, batch_first=True)
        
        # Self-attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # LSTM + attention forward pass
        # Return attention weights for interpretability
```

### Training Framework Implementation

#### Training Configuration
```python
@dataclass
class TrainingConfig:
    model_type: str = 'lstm'  # 'lstm', 'cnn_lstm', 'attention_lstm'
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 15
    class_weight: float = 3.0  # Weight for minority class
    gradient_clip: float = 1.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
```

#### Training Loop Implementation
```python
class SequenceModelTrainer:
    def __init__(self, model, config, train_loader, val_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss function with class weighting
        pos_weight = torch.tensor([config.class_weight])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=5, factor=0.5
        )
        
    def train_epoch(self):
        # Single epoch training
        # Return training loss and metrics
        
    def validate(self):
        # Validation evaluation
        # Return validation metrics
        
    def train(self):
        # Full training loop with early stopping
        # Save best model based on validation AUC
```

### Model Evaluation and Analysis

#### Performance Comparison Framework
```python
def compare_models(baseline_results, sequence_results):
    """Compare baseline and sequence model performance"""
    comparison = {
        'roc_auc': {
            'baseline': baseline_results['roc_auc'],
            'sequence': sequence_results['roc_auc'],
            'improvement': sequence_results['roc_auc'] - baseline_results['roc_auc']
        },
        'pr_auc': {
            'baseline': baseline_results['pr_auc'],
            'sequence': sequence_results['pr_auc'],
            'improvement': sequence_results['pr_auc'] - baseline_results['pr_auc']
        },
        # Additional metrics comparison
    }
    return comparison
```

#### Temporal Analysis Tools
```python
def analyze_temporal_patterns(model, test_loader, device):
    """Analyze how predictions evolve over sequence length"""
    model.eval()
    temporal_predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            sequences, labels = batch
            # Get predictions at different time steps
            # Analyze prediction confidence evolution
    
    return temporal_predictions
```

### Model Architecture Configurations

#### Recommended Architectures

**Basic LSTM Configuration**:
- Input size: Number of sensors (15-20)
- Hidden size: 64-128
- Layers: 1-2 LSTM layers
- Dropout: 0.2-0.4
- Output: Single sigmoid unit

**CNN-LSTM Configuration**:
- CNN: 2-3 conv1d layers, kernel size 3-5
- CNN features: 32-64 channels
- LSTM: 1-2 layers, hidden size 64-128
- Pooling: MaxPool1d between CNN and LSTM

**Attention-Enhanced Configuration**:
- LSTM: 2 layers, hidden size 128
- Attention: 4-8 heads
- Sequence length: 100-500 time steps

### Handling Sequence-Specific Challenges

#### Variable Length Sequences
- **Padding strategy**: Post-padding with zeros or mean values
- **Masking**: Attention masking for padded positions
- **PackedSequence**: PyTorch efficient variable-length handling

#### Class Imbalance
- **Weighted loss**: BCEWithLogitsLoss with pos_weight
- **Focal loss**: Alternative for extreme imbalance
- **Threshold tuning**: Optimize decision threshold post-training

#### Temporal Data Augmentation
- **Jittering**: Add small random noise to sequences
- **Time warping**: Slight temporal distortions
- **Mixup**: Linear interpolation between sequences (advanced)

## Dependencies

- **Prerequisite**: Phase 2 (Data Generation), Phase 3 (EDA), Phase 4 (Baseline Model)
- **Hardware**: GPU recommended for training (CUDA-compatible)
- **Software**: PyTorch, scikit-learn, numpy, pandas

## Expected Deliverables

1. **Model Implementation**: `src/modeling/sequence_models.py`
   - LSTM, CNN-LSTM, and attention model classes
   - Configurable architecture components
   - Forward pass implementations

2. **Data Pipeline**: `src/modeling/sequence_dataset.py`
   - PyTorch Dataset and DataLoader implementations
   - Preprocessing and normalization utilities
   - Batch collation functions

3. **Training Framework**: `src/modeling/train_sequence.py`
   - Training loop implementation
   - Validation and early stopping
   - Model checkpointing

4. **Trained Models**: `models/sequence/`
   - Best model state dictionaries
   - ONNX exported models
   - Training configuration files

5. **Evaluation Results**: `models/sequence/evaluation/`
   - Performance comparison with baseline
   - Temporal analysis results
   - Model interpretability outputs

## Technical Considerations

### Memory and Computational Efficiency
- **Gradient checkpointing**: For very long sequences
- **Mixed precision training**: FP16 for faster training
- **Batch size optimization**: Balance memory usage and convergence
- **Efficient data loading**: Multi-process data loading

### Model Interpretability
- **Attention visualization**: Plot attention weights over time
- **Gradient-based attribution**: Input gradients for feature importance
- **Sequence importance**: Which time steps are most predictive
- **Ablation studies**: Impact of different architecture components

### Deployment Considerations
- **ONNX export**: Cross-platform inference compatibility
- **Model quantization**: Reduced precision for faster inference
- **Batch inference**: Efficient processing of multiple sequences
- **Real-time constraints**: Latency requirements for streaming data

## Success Metrics

- [ ] **Model Performance**: ROC-AUC improvement > 0.05 over baseline
- [ ] **Training Stability**: Consistent convergence across multiple runs
- [ ] **Inference Speed**: < 100ms per sequence prediction
- [ ] **Memory Efficiency**: Training on available GPU memory
- [ ] **Interpretability**: Clear temporal pattern identification
- [ ] **Deployment Ready**: ONNX export successful with verified outputs

## Model Selection Criteria

1. **Performance**: Validation AUC on held-out data
2. **Stability**: Low variance across training runs
3. **Efficiency**: Training time and inference speed
4. **Interpretability**: Ability to explain predictions
5. **Deployment**: Resource requirements for production

## Notes

This phase represents the core deep learning component of the system. Focus on:

1. **Data quality**: Ensure robust preprocessing pipeline
2. **Architecture experimentation**: Try multiple model configurations
3. **Proper evaluation**: Fair comparison with baseline using same data splits
4. **Interpretability**: Understand what temporal patterns the model learns
5. **Production readiness**: Models should be deployable in real-time systems

The sequence models should demonstrate clear value over the feature-based baseline by capturing temporal dependencies that traditional feature engineering misses. If sequence models don't significantly outperform the baseline, investigate:
- Sequence length optimization
- Architecture modifications
- Data preprocessing improvements
- Temporal augmentation strategies

## Labels
`enhancement`, `phase-5`, `deep-learning`, `sequence-modeling`, `pytorch`

## Priority
**High** - Core technical innovation demonstrating temporal modeling capabilities