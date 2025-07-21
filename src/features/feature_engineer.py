import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats
from scipy.signal import find_peaks
import logging
from typing import Dict, List, Tuple, Optional, Union
from joblib import Parallel, delayed

class CastingFeatureEngineer:
    """
    Comprehensive feature engineering pipeline for steel casting defect prediction.
    Extracts statistical, stability, duration, interaction, and temporal features.
    """
    
    def __init__(self, 
                 sensor_columns: List[str] = None,
                 scaling_method: str = 'standard',
                 percentiles: List[float] = [10, 25, 75, 90],
                 spike_threshold: float = 2.0,
                 extreme_percentiles: Tuple[float, float] = (5, 95)):
        """
        Initialize the feature engineering pipeline
        
        Args:
            sensor_columns: List of sensor column names
            scaling_method: 'standard', 'robust', or 'none'
            percentiles: Percentiles to calculate for statistical features
            spike_threshold: Standard deviation threshold for spike detection
            extreme_percentiles: Lower and upper percentiles for extreme detection
        """
        self.sensor_columns = sensor_columns or ['temperature', 'pressure', 'flow_rate', 'vibration', 'power_consumption']
        self.scaling_method = scaling_method
        self.percentiles = percentiles
        self.spike_threshold = spike_threshold
        self.extreme_percentiles = extreme_percentiles
        
        # Initialize scaler
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = None
            
        self.feature_columns = []
        self._fitted = False
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def extract_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract statistical features for each sensor
        Features per sensor (7 × 5 sensors = 35 features):
        - mean, std, min, max, median
        - percentiles (10th, 25th, 75th, 90th)
        
        Args:
            df: DataFrame with sensor readings
            
        Returns:
            DataFrame with statistical features
        """
        features = {}
        
        for sensor in self.sensor_columns:
            if sensor not in df.columns:
                self.logger.warning(f"Sensor {sensor} not found in data. Filling with NaN.")
                # Create dummy data filled with NaN
                sensor_data = pd.Series([np.nan] * len(df))
            else:
                sensor_data = df[sensor].dropna()
            
            if len(sensor_data) == 0 or sensor_data.isna().all():
                # Handle empty or all-NaN data
                features[f'{sensor}_mean'] = np.nan
                features[f'{sensor}_std'] = np.nan
                features[f'{sensor}_min'] = np.nan
                features[f'{sensor}_max'] = np.nan
                features[f'{sensor}_median'] = np.nan
                for p in self.percentiles:
                    features[f'{sensor}_p{int(p)}'] = np.nan
            else:
                # Basic statistics
                features[f'{sensor}_mean'] = sensor_data.mean()
                features[f'{sensor}_std'] = sensor_data.std()
                features[f'{sensor}_min'] = sensor_data.min()
                features[f'{sensor}_max'] = sensor_data.max()
                features[f'{sensor}_median'] = sensor_data.median()
                
                # Percentiles
                for p in self.percentiles:
                    features[f'{sensor}_p{int(p)}'] = np.percentile(sensor_data, p)
        
        return pd.DataFrame([features])
    
    def extract_stability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract stability and variability features
        Features per sensor (4 × 5 sensors = 20 features):
        - spike_count: Number of values exceeding spike_threshold standard deviations
        - excursion_frequency: Rate of crossing mean ± 1 std
        - coefficient_of_variation: std/mean ratio
        - range_ratio: (max - min) / mean
        
        Args:
            df: DataFrame with sensor readings
            
        Returns:
            DataFrame with stability features
        """
        features = {}
        
        for sensor in self.sensor_columns:
            if sensor not in df.columns:
                self.logger.warning(f"Sensor {sensor} not found in data. Filling with NaN.")
                sensor_data = pd.Series([np.nan] * len(df))
            else:
                sensor_data = df[sensor].dropna()
            
            if len(sensor_data) == 0 or sensor_data.isna().all():
                features[f'{sensor}_spike_count'] = np.nan
                features[f'{sensor}_excursion_freq'] = np.nan
                features[f'{sensor}_cv'] = np.nan
                features[f'{sensor}_range_ratio'] = np.nan
            else:
                mean_val = sensor_data.mean()
                std_val = sensor_data.std()
                
                # Spike count: values exceeding threshold standard deviations
                if std_val > 0:
                    z_scores = np.abs((sensor_data - mean_val) / std_val)
                    spike_count = (z_scores > self.spike_threshold).sum()
                else:
                    spike_count = 0
                features[f'{sensor}_spike_count'] = spike_count
                
                # Excursion frequency: rate of crossing mean ± 1 std
                if std_val > 0:
                    upper_bound = mean_val + std_val
                    lower_bound = mean_val - std_val
                    crossings = 0
                    for i in range(1, len(sensor_data)):
                        prev_val = sensor_data.iloc[i-1]
                        curr_val = sensor_data.iloc[i]
                        # Check if crossing any boundary
                        if ((prev_val < lower_bound and curr_val > lower_bound) or
                            (prev_val > lower_bound and curr_val < lower_bound) or
                            (prev_val < upper_bound and curr_val > upper_bound) or
                            (prev_val > upper_bound and curr_val < upper_bound)):
                            crossings += 1
                    excursion_freq = crossings / len(sensor_data)
                else:
                    excursion_freq = 0
                features[f'{sensor}_excursion_freq'] = excursion_freq
                
                # Coefficient of variation
                if mean_val != 0:
                    cv = std_val / abs(mean_val)
                else:
                    cv = np.inf if std_val > 0 else 0
                features[f'{sensor}_cv'] = cv
                
                # Range ratio
                range_val = sensor_data.max() - sensor_data.min()
                if mean_val != 0:
                    range_ratio = range_val / abs(mean_val)
                else:
                    range_ratio = np.inf if range_val > 0 else 0
                features[f'{sensor}_range_ratio'] = range_ratio
        
        return pd.DataFrame([features])
    
    def extract_duration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract duration-based features
        Features per sensor (3 × 5 sensors = 15 features):
        - time_at_extremes: Percentage of time spent in extreme percentiles
        - threshold_crossings: Number of mean crossings
        - consecutive_extremes: Maximum consecutive points in extreme regions
        
        Args:
            df: DataFrame with sensor readings
            
        Returns:
            DataFrame with duration features
        """
        features = {}
        
        for sensor in self.sensor_columns:
            if sensor not in df.columns:
                self.logger.warning(f"Sensor {sensor} not found in data. Filling with NaN.")
                sensor_data = pd.Series([np.nan] * len(df))
            else:
                sensor_data = df[sensor].dropna()
            
            if len(sensor_data) == 0 or sensor_data.isna().all():
                features[f'{sensor}_time_extremes'] = np.nan
                features[f'{sensor}_threshold_crossings'] = np.nan
                features[f'{sensor}_consec_extremes'] = np.nan
            else:
                # Calculate extreme percentiles
                lower_extreme = np.percentile(sensor_data, self.extreme_percentiles[0])
                upper_extreme = np.percentile(sensor_data, self.extreme_percentiles[1])
                
                # Time at extremes: percentage in extreme regions
                extreme_mask = (sensor_data <= lower_extreme) | (sensor_data >= upper_extreme)
                time_at_extremes = extreme_mask.sum() / len(sensor_data) * 100
                features[f'{sensor}_time_extremes'] = time_at_extremes
                
                # Threshold crossings: number of mean crossings
                mean_val = sensor_data.mean()
                crossings = 0
                for i in range(1, len(sensor_data)):
                    if ((sensor_data.iloc[i-1] < mean_val and sensor_data.iloc[i] > mean_val) or
                        (sensor_data.iloc[i-1] > mean_val and sensor_data.iloc[i] < mean_val)):
                        crossings += 1
                features[f'{sensor}_threshold_crossings'] = crossings
                
                # Consecutive extremes: maximum consecutive points in extreme regions
                max_consecutive = 0
                current_consecutive = 0
                for value in sensor_data:
                    if value <= lower_extreme or value >= upper_extreme:
                        current_consecutive += 1
                        max_consecutive = max(max_consecutive, current_consecutive)
                    else:
                        current_consecutive = 0
                features[f'{sensor}_consec_extremes'] = max_consecutive
        
        return pd.DataFrame([features])
    
    def extract_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract cross-sensor interaction features
        Features (10 total):
        - temp_pressure_corr: Correlation between temperature and pressure
        - temp_flow_ratio: Mean temperature / mean flow_rate
        - pressure_vibration_diff: Mean pressure - mean vibration
        - flow_power_corr: Correlation between flow_rate and power_consumption
        - temp_power_product: Mean temperature × mean power_consumption
        - pressure_flow_corr: Correlation between pressure and flow_rate
        - vibration_power_ratio: Mean vibration / mean power_consumption
        - temp_vibration_corr: Correlation between temperature and vibration
        - overall_sensor_corr: Mean correlation across all sensor pairs
        - sensor_variance_ratio: Max sensor variance / min sensor variance
        
        Args:
            df: DataFrame with sensor readings
            
        Returns:
            DataFrame with interaction features
        """
        features = {}
        
        # Helper function for safe correlation calculation
        def safe_correlation(x, y):
            if len(x) < 2 or len(y) < 2 or x.std() == 0 or y.std() == 0:
                return 0.0
            try:
                return np.corrcoef(x, y)[0, 1]
            except:
                return 0.0
        
        # Get sensor data, handle missing columns
        sensor_data = {}
        for sensor in self.sensor_columns:
            if sensor in df.columns:
                sensor_data[sensor] = df[sensor].dropna()
            else:
                self.logger.warning(f"Sensor {sensor} not found in data.")
                sensor_data[sensor] = pd.Series([np.nan])
        
        # Temperature-pressure correlation
        if 'temperature' in sensor_data and 'pressure' in sensor_data:
            features['temp_pressure_corr'] = safe_correlation(
                sensor_data['temperature'], sensor_data['pressure'])
        else:
            features['temp_pressure_corr'] = np.nan
        
        # Temperature-flow ratio
        if ('temperature' in sensor_data and 'flow_rate' in sensor_data and
            not sensor_data['temperature'].isna().all() and not sensor_data['flow_rate'].isna().all()):
            temp_mean = sensor_data['temperature'].mean()
            flow_mean = sensor_data['flow_rate'].mean()
            if flow_mean != 0:
                features['temp_flow_ratio'] = temp_mean / flow_mean
            else:
                features['temp_flow_ratio'] = np.nan
        else:
            features['temp_flow_ratio'] = np.nan
        
        # Pressure-vibration difference
        if ('pressure' in sensor_data and 'vibration' in sensor_data and
            not sensor_data['pressure'].isna().all() and not sensor_data['vibration'].isna().all()):
            pressure_mean = sensor_data['pressure'].mean()
            vibration_mean = sensor_data['vibration'].mean()
            features['pressure_vibration_diff'] = pressure_mean - vibration_mean
        else:
            features['pressure_vibration_diff'] = np.nan
        
        # Flow-power correlation
        if 'flow_rate' in sensor_data and 'power_consumption' in sensor_data:
            features['flow_power_corr'] = safe_correlation(
                sensor_data['flow_rate'], sensor_data['power_consumption'])
        else:
            features['flow_power_corr'] = np.nan
        
        # Temperature-power product
        if ('temperature' in sensor_data and 'power_consumption' in sensor_data and
            not sensor_data['temperature'].isna().all() and not sensor_data['power_consumption'].isna().all()):
            temp_mean = sensor_data['temperature'].mean()
            power_mean = sensor_data['power_consumption'].mean()
            features['temp_power_product'] = temp_mean * power_mean
        else:
            features['temp_power_product'] = np.nan
        
        # Pressure-flow correlation
        if 'pressure' in sensor_data and 'flow_rate' in sensor_data:
            features['pressure_flow_corr'] = safe_correlation(
                sensor_data['pressure'], sensor_data['flow_rate'])
        else:
            features['pressure_flow_corr'] = np.nan
        
        # Vibration-power ratio
        if ('vibration' in sensor_data and 'power_consumption' in sensor_data and
            not sensor_data['vibration'].isna().all() and not sensor_data['power_consumption'].isna().all()):
            vibration_mean = sensor_data['vibration'].mean()
            power_mean = sensor_data['power_consumption'].mean()
            if power_mean != 0:
                features['vibration_power_ratio'] = vibration_mean / power_mean
            else:
                features['vibration_power_ratio'] = np.inf if vibration_mean > 0 else 0
        else:
            features['vibration_power_ratio'] = np.nan
        
        # Temperature-vibration correlation
        if 'temperature' in sensor_data and 'vibration' in sensor_data:
            features['temp_vibration_corr'] = safe_correlation(
                sensor_data['temperature'], sensor_data['vibration'])
        else:
            features['temp_vibration_corr'] = np.nan
        
        # Overall sensor correlation (mean of all pairwise correlations)
        correlations = []
        available_sensors = [s for s in self.sensor_columns 
                           if s in sensor_data and not sensor_data[s].isna().all()]
        
        for i, sensor1 in enumerate(available_sensors):
            for sensor2 in available_sensors[i+1:]:
                corr = safe_correlation(sensor_data[sensor1], sensor_data[sensor2])
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        if correlations:
            features['overall_sensor_corr'] = np.mean(correlations)
        else:
            features['overall_sensor_corr'] = np.nan
        
        # Sensor variance ratio
        variances = []
        for sensor in available_sensors:
            if not sensor_data[sensor].isna().all():
                var = sensor_data[sensor].var()
                if not np.isnan(var) and var >= 0:
                    variances.append(var)
        
        if len(variances) >= 2:
            max_var = max(variances)
            min_var = min(variances)
            if min_var > 0:
                features['sensor_variance_ratio'] = max_var / min_var
            else:
                features['sensor_variance_ratio'] = np.inf
        else:
            features['sensor_variance_ratio'] = np.nan
        
        return pd.DataFrame([features])
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal pattern features
        Features per sensor (2 × 5 sensors = 10 features):
        - linear_trend: Slope of linear regression line
        - mean_gradient: Average absolute difference between consecutive points
        
        Args:
            df: DataFrame with sensor readings
            
        Returns:
            DataFrame with temporal features
        """
        features = {}
        
        for sensor in self.sensor_columns:
            if sensor not in df.columns:
                self.logger.warning(f"Sensor {sensor} not found in data. Filling with NaN.")
                sensor_data = pd.Series([np.nan] * len(df))
            else:
                sensor_data = df[sensor].dropna()
            
            if len(sensor_data) < 2 or sensor_data.isna().all():
                features[f'{sensor}_trend'] = np.nan
                features[f'{sensor}_gradient'] = np.nan
            else:
                # Linear trend: slope of linear regression
                x = np.arange(len(sensor_data))
                if len(x) > 1 and sensor_data.std() > 0:
                    try:
                        slope, _, _, _, _ = stats.linregress(x, sensor_data)
                        features[f'{sensor}_trend'] = slope
                    except:
                        features[f'{sensor}_trend'] = 0.0
                else:
                    features[f'{sensor}_trend'] = 0.0
                
                # Mean gradient: average absolute difference between consecutive points
                if len(sensor_data) > 1:
                    gradients = np.abs(np.diff(sensor_data))
                    features[f'{sensor}_gradient'] = np.mean(gradients)
                else:
                    features[f'{sensor}_gradient'] = 0.0
        
        return pd.DataFrame([features])
    
    def engineer_features(self, 
                         data: Union[pd.DataFrame, Dict],
                         cast_id: Optional[str] = None) -> pd.DataFrame:
        """
        Main method to extract all features from a casting sequence
        
        Args:
            data: DataFrame with sensor readings or dict of DataFrames
            cast_id: Optional casting ID for tracking
            
        Returns:
            DataFrame with all engineered features (90+ columns)
        """
        try:
            if isinstance(data, dict):
                if cast_id and cast_id in data:
                    df = data[cast_id]
                else:
                    # Take the first DataFrame if cast_id not specified
                    df = next(iter(data.values()))
            else:
                df = data
            
            if df.empty:
                self.logger.warning(f"Empty DataFrame for cast {cast_id}")
                return pd.DataFrame()
            
            # Extract all feature categories
            stat_features = self.extract_statistical_features(df)
            stability_features = self.extract_stability_features(df)
            duration_features = self.extract_duration_features(df)
            interaction_features = self.extract_interaction_features(df)
            temporal_features = self.extract_temporal_features(df)
            
            # Combine all features
            all_features = pd.concat([
                stat_features,
                stability_features,
                duration_features,
                interaction_features,
                temporal_features
            ], axis=1)
            
            # Add cast_id if provided
            if cast_id:
                all_features['cast_id'] = cast_id
            
            # Store feature columns for later use
            if not self.feature_columns:
                self.feature_columns = [col for col in all_features.columns if col != 'cast_id']
            
            self.logger.info(f"Extracted {len(self.feature_columns)} features for cast {cast_id}")
            return all_features
            
        except Exception as e:
            self.logger.error(f"Error extracting features for cast {cast_id}: {str(e)}")
            return pd.DataFrame()
    
    def engineer_features_batch(self, 
                               data_dict: Dict[str, pd.DataFrame],
                               n_jobs: int = -1) -> pd.DataFrame:
        """
        Process multiple casting sequences in parallel
        
        Args:
            data_dict: Dictionary mapping cast_ids to DataFrames
            n_jobs: Number of parallel jobs (-1 for all cores)
            
        Returns:
            DataFrame with features for all casts
        """
        self.logger.info(f"Processing {len(data_dict)} casting sequences with {n_jobs} jobs")
        
        try:
            # Process in parallel
            results = Parallel(n_jobs=n_jobs)(
                delayed(self.engineer_features)(df, cast_id)
                for cast_id, df in data_dict.items()
            )
            
            # Filter out empty results and combine
            valid_results = [result for result in results if not result.empty]
            
            if not valid_results:
                self.logger.warning("No valid features extracted from any cast")
                return pd.DataFrame()
            
            combined_features = pd.concat(valid_results, ignore_index=True)
            self.logger.info(f"Successfully processed {len(valid_results)} casts with {len(combined_features.columns)} features")
            
            return combined_features
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {str(e)}")
            return pd.DataFrame()

    def transform_cast(self, time_series: pd.DataFrame) -> Dict:
        """
        Transform single cast to feature vector.
        
        Args:
            time_series: Time series data for a single cast
            
        Returns:
            Dict: Complete feature vector for the cast
        """
        # Use the main engineer_features method and convert to dict
        features_df = self.engineer_features(time_series)
        if features_df.empty:
            return {}
        return features_df.iloc[0].to_dict()
    
    def fit_scaler(self, feature_matrix: pd.DataFrame) -> None:
        """
        Fit feature scaler on training data.
        
        Args:
            feature_matrix: Training features
        """
        if self.scaler is None:
            self.logger.info("No scaling method specified, skipping scaler fitting")
            return
        
        # Remove non-numeric columns
        numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns
        if 'cast_id' in numeric_cols:
            numeric_cols = numeric_cols.drop('cast_id')
        
        if len(numeric_cols) == 0:
            self.logger.warning("No numeric features found for scaling")
            return
        
        # Handle missing values by replacing with column means
        feature_matrix_clean = feature_matrix[numeric_cols].fillna(feature_matrix[numeric_cols].mean())
        
        self.scaler.fit(feature_matrix_clean)
        self._fitted = True
        self.logger.info(f"Fitted {self.scaling_method} scaler on {len(numeric_cols)} features")
    
    def scale_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Scale features using fitted scaler.
        
        Args:
            features: Features to scale
            
        Returns:
            Scaled features
        """
        if self.scaler is None:
            self.logger.info("No scaling method specified, returning original features")
            return features
        
        if not self._fitted:
            self.logger.warning("Scaler not fitted yet. Please call fit_scaler first.")
            return features
        
        # Identify numeric columns (excluding cast_id)
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        if 'cast_id' in numeric_cols:
            numeric_cols = numeric_cols.drop('cast_id')
        
        if len(numeric_cols) == 0:
            self.logger.warning("No numeric features found for scaling")
            return features
        
        # Create a copy to avoid modifying the original
        scaled_features = features.copy()
        
        # Handle missing values
        features_clean = features[numeric_cols].fillna(features[numeric_cols].mean())
        
        # Scale the numeric features
        scaled_values = self.scaler.transform(features_clean)
        scaled_features[numeric_cols] = scaled_values
        
        return scaled_features