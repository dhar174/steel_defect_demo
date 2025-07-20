import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Tuple, List
import json
from datetime import datetime, timedelta


class SteelCastingDataGenerator:
    """Generates synthetic steel casting process data with realistic defect simulation"""
    
    def __init__(self, config_path: str):
        """
        Initialize the data generator with configuration settings.
        
        Parameters:
            config_path (str): Path to the YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data_generation']
        self.sensor_config = self.data_config['sensors']
        self.defect_config = self.data_config['defect_simulation']
        
        # Initialize random state for reproducibility
        self.random_state = np.random.RandomState(self.data_config['random_seed'])
        
        # Create output directories
        self._create_output_directories()
    
    def _create_output_directories(self):
        """Create necessary output directories"""
        base_path = Path('.')
        directories = [
            'data/raw',
            'data/processed', 
            'data/synthetic'
        ]
        for dir_path in directories:
            (base_path / dir_path).mkdir(parents=True, exist_ok=True)
    
    def _generate_sensor_values(self, num_samples: int, sensor_name: str, base_values: np.ndarray = None) -> np.ndarray:
        """
        Generate realistic sensor values with noise and constraints.
        
        Parameters:
            num_samples (int): Number of samples to generate
            sensor_name (str): Name of the sensor
            base_values (np.ndarray): Optional base trend values
            
        Returns:
            np.ndarray: Generated sensor values
        """
        sensor_cfg = self.sensor_config[sensor_name]
        
        if base_values is None:
            # Generate base values with some random walk
            base_values = np.full(num_samples, sensor_cfg['base_value'])
            # Add gradual drift
            drift = self.random_state.normal(0, sensor_cfg['noise_std'] * 0.1, num_samples)
            base_values = base_values + np.cumsum(drift) * 0.01
        
        # Add noise
        noise = self.random_state.normal(0, sensor_cfg['noise_std'], num_samples)
        values = base_values + noise
        
        # Apply constraints
        values = np.clip(values, sensor_cfg['min_value'], sensor_cfg['max_value'])
        
        return values
    
    def _detect_defect_triggers(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Detect defect triggers based on process conditions.
        
        Parameters:
            df (pd.DataFrame): Time series data for the cast
            
        Returns:
            Tuple[bool, List[str]]: (has_defect, list of trigger events)
        """
        triggers = []
        
        # 1. Prolonged mold level deviation (>30 seconds outside normal range)
        mold_level_normal_range = self.sensor_config['mold_level_normal_range']  # Configurable range
        outside_range = (df['mold_level'] < mold_level_normal_range[0]) | (df['mold_level'] > mold_level_normal_range[1])
        
        # Find consecutive periods outside range
        consecutive_periods = []
        in_deviation = False
        start_idx = 0
        
        for i, is_outside in enumerate(outside_range):
            if is_outside and not in_deviation:
                start_idx = i
                in_deviation = True
            elif not is_outside and in_deviation:
                consecutive_periods.append(i - start_idx)
                in_deviation = False
        
        if in_deviation:  # Handle case where deviation continues to end
            consecutive_periods.append(len(outside_range) - start_idx)
        
        if any(period >= self.defect_config['defect_triggers']['prolonged_mold_level_deviation'] for period in consecutive_periods):
            triggers.append('prolonged_mold_level_deviation')
        
        # 2. Rapid temperature drop (>50°C drop in 60 seconds)
        temp_diff = df['mold_temperature'].diff(periods=60)  # 60 seconds at 1Hz
        rapid_drop_threshold = -self.defect_config['defect_triggers']['rapid_temperature_drop']
        if (temp_diff < rapid_drop_threshold).any():
            triggers.append('rapid_temperature_drop')
        
        # 3. High speed with low superheat (speed >1.5 m/min with superheat <20°C)
        if self.defect_config['defect_triggers']['high_speed_with_low_superheat']:
            high_speed_low_superheat = (df['casting_speed'] > 1.5) & (df['superheat'] < 20)
            if high_speed_low_superheat.any():
                triggers.append('high_speed_with_low_superheat')
        
        # Determine if defect occurs based on triggers and base probability
        has_defect = False
        if triggers:
            # Higher probability if triggers are present
            defect_prob = min(0.8, len(triggers) * 0.3)
        else:
            # Base defect probability
            defect_prob = self.defect_config['defect_probability']
        
        has_defect = self.random_state.random() < defect_prob
        
        return has_defect, triggers
    
    def generate_cast_sequence(self, cast_id: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Generate time series data for a single cast.
        
        Parameters:
            cast_id (str): Unique identifier for the cast
            
        Returns:
            Tuple[pd.DataFrame, Dict]: Time series DataFrame and metadata dict
        """
        # Calculate number of samples
        duration_seconds = self.data_config['cast_duration_minutes'] * 60
        sampling_rate = self.data_config['sampling_rate_hz']
        num_samples = duration_seconds * sampling_rate
        
        # Generate timestamps with a fixed start time for reproducibility
        start_time = datetime(2023, 1, 1, 0, 0, 0)
        timestamps = pd.date_range(
            start=start_time,
            periods=num_samples,
            freq=f'{1/sampling_rate}S'  # Use uppercase 'S' for seconds
        )
        
        # Generate sensor data
        casting_speed = self._generate_sensor_values(num_samples, 'casting_speed')
        mold_temperature = self._generate_sensor_values(num_samples, 'mold_temperature')
        mold_level = self._generate_sensor_values(num_samples, 'mold_level')
        cooling_water_flow = self._generate_sensor_values(num_samples, 'cooling_water_flow')
        superheat = self._generate_sensor_values(num_samples, 'superheat')
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'casting_speed': casting_speed,
            'mold_temperature': mold_temperature,
            'mold_level': mold_level,
            'cooling_water_flow': cooling_water_flow,
            'superheat': superheat
        })
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Detect defects
        has_defect, trigger_events = self._detect_defect_triggers(df)
        
        # Generate metadata
        metadata = {
            'cast_id': cast_id,
            'generation_timestamp': datetime.now().isoformat(),
            'steel_grade': f'Grade_{self.random_state.choice(["A", "B", "C", "D"])}',
            'duration_minutes': self.data_config['cast_duration_minutes'],
            'sampling_rate_hz': self.data_config['sampling_rate_hz'],
            'num_samples': num_samples,
            'defect_label': int(has_defect),
            'defect_trigger_events': trigger_events,
            'process_summary': {
                'avg_casting_speed': float(df['casting_speed'].mean()),
                'avg_mold_temperature': float(df['mold_temperature'].mean()),
                'avg_mold_level': float(df['mold_level'].mean()),
                'avg_cooling_water_flow': float(df['cooling_water_flow'].mean()),
                'avg_superheat': float(df['superheat'].mean()),
                'speed_std': float(df['casting_speed'].std()),
                'temp_std': float(df['mold_temperature'].std()),
                'level_std': float(df['mold_level'].std())
            }
        }
        
        return df, metadata
    
    def generate_dataset(self) -> None:
        """
        Generate complete dataset with specified number of casts.
        """
        print(f"Generating {self.data_config['num_casts']} synthetic steel casting sequences...")
        
        num_casts = self.data_config['num_casts']
        train_split = self.data_config['output']['train_test_split']
        num_train = int(num_casts * train_split)
        
        all_metadata = []
        defect_count = 0
        
        # Generate all casts
        for i in range(num_casts):
            cast_id = f"cast_{i+1:04d}"
            
            # Generate cast data
            df, metadata = self.generate_cast_sequence(cast_id)
            
            # Track defects
            if metadata['defect_label']:
                defect_count += 1
            
            # Save time series data
            output_file = Path(f"data/raw/cast_timeseries_{i+1:04d}.parquet")
            df.to_parquet(output_file)
            
            # Store metadata
            all_metadata.append(metadata)
            
            # Progress update
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_casts} casts...")
        
        # Save dataset metadata
        dataset_metadata = {
            'dataset_info': {
                'total_casts': num_casts,
                'train_casts': num_train,
                'test_casts': num_casts - num_train,
                'defect_count': defect_count,
                'defect_rate': defect_count / num_casts,
                'generation_timestamp': datetime.now().isoformat(),
                'configuration': self.config
            },
            'cast_metadata': all_metadata
        }
        
        with open('data/synthetic/dataset_metadata.json', 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        # Save generation summary
        generation_summary = {
            'total_casts_generated': num_casts,
            'defect_rate_achieved': defect_count / num_casts,
            'target_defect_rate': self.defect_config['defect_probability'],
            'train_test_split': train_split,
            'output_format': self.data_config['output']['raw_data_format'],
            'generation_completed_at': datetime.now().isoformat()
        }
        
        with open('data/synthetic/generation_summary.json', 'w') as f:
            json.dump(generation_summary, f, indent=2)
        
        print(f"\nDataset generation completed!")
        print(f"Total casts: {num_casts}")
        print(f"Defect rate: {defect_count / num_casts:.2%} (target: {self.defect_config['defect_probability']:.2%})")
        print(f"Train casts: {num_train}")
        print(f"Test casts: {num_casts - num_train}")
        print(f"Files saved to data/raw/ and data/synthetic/")