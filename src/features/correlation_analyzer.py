"""
Correlation analysis functionality for steel casting sensor data.
Provides cross-sensor correlations, defect-specific analysis, and time-lagged correlations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import json
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression


class SensorCorrelationAnalyzer:
    """Analyzes correlations between sensors in steel casting data"""
    
    def __init__(self, sensor_columns: List[str] = None):
        """
        Initialize correlation analyzer.
        
        Args:
            sensor_columns: List of sensor column names. If None, uses default sensors.
        """
        self.sensor_columns = sensor_columns or [
            'casting_speed', 'mold_temperature', 'mold_level', 
            'cooling_water_flow', 'superheat'
        ]
        self.correlation_cache = {}
        
    def compute_cross_sensor_correlations(self, 
                                        data: pd.DataFrame,
                                        method: str = 'pearson') -> pd.DataFrame:
        """
        Compute correlation matrix between all sensor pairs.
        
        Args:
            data: Time series DataFrame with sensor data
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            DataFrame: Correlation matrix
        """
        # Select only sensor columns
        sensor_data = data[self.sensor_columns]
        
        # Compute correlation matrix
        if method == 'pearson':
            corr_matrix = sensor_data.corr(method='pearson')
        elif method == 'spearman':
            corr_matrix = sensor_data.corr(method='spearman')
        elif method == 'kendall':
            corr_matrix = sensor_data.corr(method='kendall')
        else:
            raise ValueError(f"Unknown correlation method: {method}")
            
        return corr_matrix
    
    def compute_time_lagged_correlations(self, 
                                       data: pd.DataFrame,
                                       max_lag: int = 60,
                                       target_sensor: str = None) -> Dict[str, pd.DataFrame]:
        """
        Compute time-lagged correlations to identify delayed relationships.
        
        Args:
            data: Time series DataFrame with sensor data
            max_lag: Maximum lag in time steps to analyze
            target_sensor: If specified, compute lags against this sensor only
            
        Returns:
            Dict: Maps sensor pairs to lagged correlation DataFrames
        """
        sensor_data = data[self.sensor_columns]
        lagged_correlations = {}
        
        if target_sensor:
            # Analyze lags for one target sensor
            target_data = sensor_data[target_sensor]
            for sensor in self.sensor_columns:
                if sensor != target_sensor:
                    lag_results = []
                    for lag in range(-max_lag, max_lag + 1):
                        if lag == 0:
                            corr, _ = pearsonr(target_data, sensor_data[sensor])
                        elif lag > 0:
                            # Positive lag: sensor leads target
                            if len(sensor_data) > lag:
                                corr, _ = pearsonr(
                                    target_data[lag:], 
                                    sensor_data[sensor][:-lag]
                                )
                            else:
                                corr = np.nan
                        else:
                            # Negative lag: target leads sensor
                            lag_abs = abs(lag)
                            if len(sensor_data) > lag_abs:
                                corr, _ = pearsonr(
                                    target_data[:-lag_abs], 
                                    sensor_data[sensor][lag_abs:]
                                )
                            else:
                                corr = np.nan
                                
                        lag_results.append({'lag': lag, 'correlation': corr})
                    
                    lagged_correlations[f"{sensor}_{target_sensor}"] = pd.DataFrame(lag_results)
        else:
            # Analyze all sensor pairs
            for i, sensor1 in enumerate(self.sensor_columns):
                for j, sensor2 in enumerate(self.sensor_columns):
                    if i < j:  # Avoid duplicate pairs
                        lag_results = []
                        for lag in range(-max_lag, max_lag + 1):
                            if lag == 0:
                                corr, _ = pearsonr(sensor_data[sensor1], sensor_data[sensor2])
                            elif lag > 0:
                                # Positive lag: sensor2 leads sensor1
                                if len(sensor_data) > lag:
                                    corr, _ = pearsonr(
                                        sensor_data[sensor1][lag:], 
                                        sensor_data[sensor2][:-lag]
                                    )
                                else:
                                    corr = np.nan
                            else:
                                # Negative lag: sensor1 leads sensor2
                                lag_abs = abs(lag)
                                if len(sensor_data) > lag_abs:
                                    corr, _ = pearsonr(
                                        sensor_data[sensor1][:-lag_abs], 
                                        sensor_data[sensor2][lag_abs:]
                                    )
                                else:
                                    corr = np.nan
                                    
                            lag_results.append({'lag': lag, 'correlation': corr})
                        
                        lagged_correlations[f"{sensor1}_{sensor2}"] = pd.DataFrame(lag_results)
        
        return lagged_correlations
    
    def compute_defect_specific_correlations(self, 
                                          cast_data_list: List[Tuple[pd.DataFrame, Dict]]) -> Dict[str, pd.DataFrame]:
        """
        Compare correlation patterns between good and defective casts.
        
        Args:
            cast_data_list: List of (time_series_df, metadata_dict) tuples
            
        Returns:
            Dict: Contains 'good_casts', 'defect_casts', and 'difference' correlation matrices
        """
        good_casts = []
        defect_casts = []
        
        # Separate good and defective casts
        for time_series, metadata in cast_data_list:
            if metadata['defect_label'] == 0:
                good_casts.append(time_series)
            else:
                defect_casts.append(time_series)
        
        results = {}
        
        if good_casts:
            # Compute average correlations for good casts
            good_corr_matrices = [
                self.compute_cross_sensor_correlations(cast_data) 
                for cast_data in good_casts
            ]
            results['good_casts'] = pd.concat(good_corr_matrices).groupby(level=0).mean()
        
        if defect_casts:
            # Compute average correlations for defective casts
            defect_corr_matrices = [
                self.compute_cross_sensor_correlations(cast_data) 
                for cast_data in defect_casts
            ]
            results['defect_casts'] = pd.concat(defect_corr_matrices).groupby(level=0).mean()
        
        # Compute difference if both types exist
        if 'good_casts' in results and 'defect_casts' in results:
            results['difference'] = results['defect_casts'] - results['good_casts']
        
        return results
    
    def identify_predictive_sensor_combinations(self, 
                                             cast_data_list: List[Tuple[pd.DataFrame, Dict]],
                                             top_k: int = 10) -> pd.DataFrame:
        """
        Identify sensor combinations most predictive of defects.
        
        Args:
            cast_data_list: List of (time_series_df, metadata_dict) tuples
            top_k: Number of top combinations to return
            
        Returns:
            DataFrame: Top predictive sensor combinations with importance scores
        """
        from itertools import combinations
        
        # Check if we have enough samples
        if len(cast_data_list) < 5:
            print(f"Warning: Only {len(cast_data_list)} samples available. Need at least 5 for reliable analysis.")
        
        # Extract features and labels
        feature_rows = []
        labels = []
        
        for time_series, metadata in cast_data_list:
            # Compute summary statistics for each sensor
            sensor_features = {}
            for sensor in self.sensor_columns:
                sensor_data = time_series[sensor]
                sensor_features.update({
                    f"{sensor}_mean": sensor_data.mean(),
                    f"{sensor}_std": sensor_data.std(),
                    f"{sensor}_min": sensor_data.min(),
                    f"{sensor}_max": sensor_data.max(),
                })
            
            # Compute pairwise correlation features
            corr_matrix = self.compute_cross_sensor_correlations(time_series)
            for i, sensor1 in enumerate(self.sensor_columns):
                for j, sensor2 in enumerate(self.sensor_columns):
                    if i < j:
                        corr_value = corr_matrix.loc[sensor1, sensor2]
                        sensor_features[f"corr_{sensor1}_{sensor2}"] = corr_value
            
            feature_rows.append(sensor_features)
            labels.append(metadata['defect_label'])
        
        # Convert to DataFrame
        features_df = pd.DataFrame(feature_rows)
        labels = np.array(labels)
        
        # Compute feature importance
        feature_importance = {}
        
        # Check if we have enough samples and variation for mutual information
        if len(cast_data_list) >= 5 and len(np.unique(labels)) > 1:
            # Use mutual information for feature importance
            for column in features_df.columns:
                # Handle NaN values
                feature_values = features_df[column].fillna(features_df[column].median())
                try:
                    mi_score = mutual_info_regression(
                        feature_values.values.reshape(-1, 1), 
                        labels, 
                        random_state=42
                    )[0]
                except ValueError as e:
                    # Fallback to correlation-based importance if mutual info fails
                    mi_score = abs(np.corrcoef(feature_values, labels)[0, 1])
                    if np.isnan(mi_score):
                        mi_score = 0.0
                
                feature_importance[column] = mi_score
        else:
            # Fallback to simple correlation-based importance
            for column in features_df.columns:
                feature_values = features_df[column].fillna(features_df[column].median())
                corr = abs(np.corrcoef(feature_values, labels)[0, 1])
                feature_importance[column] = 0.0 if np.isnan(corr) else corr
        
        # Sort by importance and return top combinations
        importance_df = pd.DataFrame([
            {'feature': feature, 'importance': importance}
            for feature, importance in feature_importance.items()
        ]).sort_values('importance', ascending=False)
        
        return importance_df.head(top_k)
    
    def compute_rolling_correlations(self, 
                                   data: pd.DataFrame,
                                   window_size: int = 300,
                                   sensor_pair: Tuple[str, str] = None) -> pd.DataFrame:
        """
        Compute rolling correlations to analyze temporal changes in relationships.
        
        Args:
            data: Time series DataFrame with sensor data
            window_size: Size of rolling window in time steps
            sensor_pair: Tuple of sensor names to analyze, or None for all pairs
            
        Returns:
            DataFrame: Rolling correlations over time
        """
        sensor_data = data[self.sensor_columns]
        
        if sensor_pair:
            # Analyze specific sensor pair
            sensor1, sensor2 = sensor_pair
            rolling_corr = sensor_data[sensor1].rolling(window=window_size).corr(
                sensor_data[sensor2]
            )
            return pd.DataFrame({
                'timestamp': data.index,
                f"rolling_corr_{sensor1}_{sensor2}": rolling_corr
            }).set_index('timestamp')
        else:
            # Analyze all pairs
            rolling_correlations = {}
            for i, sensor1 in enumerate(self.sensor_columns):
                for j, sensor2 in enumerate(self.sensor_columns):
                    if i < j:
                        rolling_corr = sensor_data[sensor1].rolling(window=window_size).corr(
                            sensor_data[sensor2]
                        )
                        rolling_correlations[f"rolling_corr_{sensor1}_{sensor2}"] = rolling_corr
            
            result_df = pd.DataFrame(rolling_correlations, index=data.index)
            return result_df
    
    def export_correlation_analysis(self, 
                                  cast_data_list: List[Tuple[pd.DataFrame, Dict]],
                                  output_path: str) -> None:
        """
        Export comprehensive correlation analysis to JSON file.
        
        Args:
            cast_data_list: List of (time_series_df, metadata_dict) tuples
            output_path: Path to save analysis results
        """
        # Compute all analyses
        sample_data = cast_data_list[0][0]  # Use first cast for basic correlations
        
        results = {
            'analysis_metadata': {
                'sensor_columns': self.sensor_columns,
                'num_casts_analyzed': len(cast_data_list),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            },
            'cross_sensor_correlations': self.compute_cross_sensor_correlations(sample_data).to_dict(),
            'defect_specific_correlations': {},
            'predictive_combinations': {},
            'time_lagged_analysis': {}
        }
        
        # Defect-specific analysis
        defect_analysis = self.compute_defect_specific_correlations(cast_data_list)
        for key, matrix in defect_analysis.items():
            results['defect_specific_correlations'][key] = matrix.to_dict()
        
        # Predictive combinations
        predictive_features = self.identify_predictive_sensor_combinations(cast_data_list)
        results['predictive_combinations'] = predictive_features.to_dict('records')
        
        # Save results
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Correlation analysis exported to {output_path}")