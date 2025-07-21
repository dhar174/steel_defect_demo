"""
Example demonstrating the complete data pipeline integration with LSTM model
"""

import numpy as np
import torch
import yaml
from pathlib import Path

# Set PYTHONPATH for imports
import sys
sys.path.append('/home/runner/work/steel_defect_demo/steel_defect_demo')

from src.data.sequence_dataset import (
    CastingSequenceDataset,
    SequenceAugmentation, 
    create_data_loaders,
    validate_dataset_config,
    prepare_sequences_from_dataframe
)
from src.features.feature_extractor import SequenceFeatureExtractor
import pandas as pd


def create_synthetic_casting_data(n_casts=200, avg_sequence_length=300, n_features=5):
    """Create synthetic steel casting time series data"""
    np.random.seed(42)
    
    data = []
    for cast_id in range(n_casts):
        # Variable sequence length (200 to 400 time steps)
        seq_length = np.random.randint(200, 401)
        
        # Generate time series for this cast
        for t in range(seq_length):
            # Generate sensor readings with some correlation structure
            base_temp = 1500 + np.random.normal(0, 50)
            temp1 = base_temp + np.random.normal(0, 10)
            temp2 = base_temp + np.random.normal(0, 15)
            
            pressure1 = 100 + np.random.normal(0, 5)
            pressure2 = pressure1 + np.random.normal(0, 3)
            
            flow_rate = 50 + np.random.normal(0, 2)
            
            # Add some temporal correlation
            if t > 0:
                prev_row = data[-1]
                temp1 = 0.8 * prev_row['temperature_1'] + 0.2 * temp1
                temp2 = 0.8 * prev_row['temperature_2'] + 0.2 * temp2
                pressure1 = 0.9 * prev_row['pressure_1'] + 0.1 * pressure1
                pressure2 = 0.9 * prev_row['pressure_2'] + 0.1 * pressure2
                flow_rate = 0.85 * prev_row['flow_rate'] + 0.15 * flow_rate
            
            data.append({
                'cast_id': cast_id,
                'timestamp': t,
                'temperature_1': temp1,
                'temperature_2': temp2, 
                'pressure_1': pressure1,
                'pressure_2': pressure2,
                'flow_rate': flow_rate
            })
    
    df = pd.DataFrame(data)
    
    # Generate defect labels (cast-level)
    cast_defect_rates = []
    for cast_id in range(n_casts):
        cast_data = df[df['cast_id'] == cast_id]
        
        # Simple defect logic: high variance in temperature or pressure indicates defect
        temp_var = cast_data['temperature_1'].var()
        pressure_var = cast_data['pressure_1'].var()
        
        # Defect probability based on variance (more realistic thresholds)
        defect_score = 0
        if temp_var > 200:  # Higher threshold
            defect_score += 0.3
        if pressure_var > 40:  # Higher threshold  
            defect_score += 0.3
        if np.random.random() < 0.1:  # 10% random noise
            defect_score += 0.2
            
        is_defect = 1 if defect_score > 0.3 else 0
        cast_defect_rates.append(is_defect)
    
    # Add defect labels to dataframe
    cast_defects = dict(zip(range(n_casts), cast_defect_rates))
    df['defect'] = df['cast_id'].map(cast_defects)
    
    print(f"Created synthetic data:")
    print(f"  Total casts: {n_casts}")
    print(f"  Total time points: {len(df)}")
    print(f"  Defect rate: {df.groupby('cast_id')['defect'].first().mean():.3f}")
    
    return df


def demonstrate_dataset_pipeline():
    """Demonstrate the complete dataset pipeline"""
    
    print("=== Steel Casting Sequence Dataset Demo ===\n")
    
    # 1. Create synthetic data
    print("1. Creating synthetic steel casting data...")
    df = create_synthetic_casting_data(n_casts=100, n_features=5)
    
    # 2. Prepare sequences from DataFrame
    print("\n2. Preparing sequences from DataFrame...")
    feature_columns = ['temperature_1', 'temperature_2', 'pressure_1', 'pressure_2', 'flow_rate']
    sequences, labels = prepare_sequences_from_dataframe(
        df, 
        sequence_length=300,
        feature_columns=feature_columns,
        label_column='defect',
        cast_id_column='cast_id'
    )
    
    print(f"Prepared sequences shape: {sequences.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Defect rate: {labels.mean():.3f}")
    
    # 3. Split into train/validation
    print("\n3. Splitting data into train/validation...")
    n_train = int(0.8 * len(sequences))
    
    train_sequences = sequences[:n_train]
    train_labels = labels[:n_train]
    val_sequences = sequences[n_train:]
    val_labels = labels[n_train:]
    
    print(f"Training set: {train_sequences.shape[0]} sequences")
    print(f"Validation set: {val_sequences.shape[0]} sequences")
    
    # 4. Create datasets
    print("\n4. Creating PyTorch datasets...")
    train_dataset = CastingSequenceDataset(
        train_sequences, 
        train_labels, 
        augment=True,
        sequence_length=300
    )
    
    val_dataset = CastingSequenceDataset(
        val_sequences, 
        val_labels, 
        augment=False,
        sequence_length=300
    )
    
    # 5. Test augmentation
    print("\n5. Testing data augmentation...")
    aug = SequenceAugmentation(
        noise_std=0.01,
        time_warp_probability=0.3,
        magnitude_warp_probability=0.3
    )
    
    sample_sequence = torch.FloatTensor(train_sequences[0])
    augmented = aug.apply_random_augmentation(sample_sequence)
    
    print(f"Original sequence shape: {sample_sequence.shape}")
    print(f"Augmented sequence shape: {augmented.shape}")
    print(f"Sequences are different: {not torch.equal(sample_sequence, augmented)}")
    
    # 6. Load configuration and create data loaders
    print("\n6. Creating data loaders with configuration...")
    
    # Load config from file or use defaults
    config_path = Path('/home/runner/work/steel_defect_demo/steel_defect_demo/configs/model_config.yaml')
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        lstm_config = config.get('lstm_model', {})
        data_config = lstm_config.get('data_processing', {})
        training_config = lstm_config.get('training', {})
        
        # Merge configs
        combined_config = {**data_config, **training_config}
        combined_config['num_workers'] = 0  # For demo
    else:
        combined_config = {
            'batch_size': 32,
            'num_workers': 0,
            'defect_weight_multiplier': 3.0
        }
    
    # Validate configuration
    validated_config = validate_dataset_config(combined_config)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset, 
        val_dataset, 
        validated_config,
        use_weighted_sampling=True
    )
    
    # 7. Test batch loading
    print("\n7. Testing batch loading...")
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    
    train_sequences_batch, train_labels_batch, train_attention_masks = train_batch
    val_sequences_batch, val_labels_batch, val_attention_masks = val_batch
    
    print(f"Training batch shapes:")
    print(f"  Sequences: {train_sequences_batch.shape}")
    print(f"  Labels: {train_labels_batch.shape}")
    print(f"  Attention masks: {train_attention_masks.shape}")
    
    print(f"Validation batch shapes:")
    print(f"  Sequences: {val_sequences_batch.shape}")
    print(f"  Labels: {val_labels_batch.shape}")
    print(f"  Attention masks: {val_attention_masks.shape}")
    
    # 8. Demonstrate class imbalance handling
    print("\n8. Class imbalance statistics...")
    class_weights = train_dataset.get_class_weights()
    print(f"Class weights: {class_weights}")
    
    # Count defects in training batch
    defect_count = train_labels_batch.sum().item()
    print(f"Defects in training batch: {defect_count}/{len(train_labels_batch)}")
    
    # 9. Memory usage estimation
    print("\n9. Memory usage estimation...")
    sequence_memory = train_sequences_batch.nelement() * train_sequences_batch.element_size()
    print(f"Memory per batch: {sequence_memory / 1024 / 1024:.2f} MB")
    
    print("\n=== Demo completed successfully! ===")
    
    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = demonstrate_dataset_pipeline()