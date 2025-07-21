"""
Data validation helpers for feature engineering pipeline.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional


class FeatureValidator:
    """Validate input data and engineered features"""
    
    def __init__(self, 
                 sensor_columns: List[str] = None,
                 min_data_points: int = 50,
                 max_missing_ratio: float = 0.1,
                 correlation_threshold: float = 0.95):
        """
        Initialize feature validator
        
        Args:
            sensor_columns: Expected sensor column names
            min_data_points: Minimum required data points per sequence
            max_missing_ratio: Maximum allowed missing data ratio
            correlation_threshold: Threshold for highly correlated features
        """
        self.sensor_columns = sensor_columns or ['temperature', 'pressure', 'flow_rate', 'vibration', 'power_consumption']
        self.min_data_points = min_data_points
        self.max_missing_ratio = max_missing_ratio
        self.correlation_threshold = correlation_threshold
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def validate_input_data(self, df: pd.DataFrame, cast_id: Optional[str] = None) -> Dict[str, bool]:
        """
        Validate input sensor data format and content
        
        Args:
            df: Input DataFrame to validate
            cast_id: Optional cast ID for logging
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'has_data': False,
            'sufficient_length': False,
            'low_missing_data': False,
            'all_sensors_present': False,
            'sensors_not_constant': False,
            'errors': []
        }
        
        try:
            # Check if DataFrame is empty
            if df.empty:
                validation['errors'].append("DataFrame is empty")
                validation['is_valid'] = False
                return validation
            
            validation['has_data'] = True
            
            # Check sufficient data length
            if len(df) < self.min_data_points:
                validation['errors'].append(f"Insufficient data points: {len(df)} < {self.min_data_points}")
                validation['is_valid'] = False
            else:
                validation['sufficient_length'] = True
            
            # Check missing data ratio
            total_missing = df.isnull().sum().sum()
            total_values = df.size
            missing_ratio = total_missing / total_values if total_values > 0 else 1.0
            
            if missing_ratio > self.max_missing_ratio:
                validation['errors'].append(f"High missing data ratio: {missing_ratio:.3f} > {self.max_missing_ratio}")
                validation['is_valid'] = False
            else:
                validation['low_missing_data'] = True
            
            # Check sensor columns presence
            missing_sensors = [col for col in self.sensor_columns if col not in df.columns]
            if missing_sensors:
                validation['errors'].append(f"Missing sensor columns: {missing_sensors}")
                validation['is_valid'] = False
            else:
                validation['all_sensors_present'] = True
            
            # Check for constant sensor readings (potential malfunctions)
            constant_sensors = []
            for sensor in self.sensor_columns:
                if sensor in df.columns:
                    sensor_data = df[sensor].dropna()
                    if len(sensor_data) > 1 and sensor_data.std() < 1e-10:
                        constant_sensors.append(sensor)
            
            if constant_sensors:
                validation['errors'].append(f"Constant/flatline sensors detected: {constant_sensors}")
                # Don't mark as invalid, just warn
                self.logger.warning(f"Cast {cast_id}: Constant sensors detected: {constant_sensors}")
            else:
                validation['sensors_not_constant'] = True
            
            # Check for extreme values (basic outlier detection)
            extreme_sensors = []
            for sensor in self.sensor_columns:
                if sensor in df.columns:
                    sensor_data = df[sensor].dropna()
                    if len(sensor_data) > 0:
                        q1 = sensor_data.quantile(0.25)
                        q3 = sensor_data.quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 3 * iqr
                        upper_bound = q3 + 3 * iqr
                        
                        outliers = ((sensor_data < lower_bound) | (sensor_data > upper_bound)).sum()
                        outlier_ratio = outliers / len(sensor_data)
                        
                        if outlier_ratio > 0.1:  # More than 10% outliers
                            extreme_sensors.append(f"{sensor}({outlier_ratio:.2%})")
            
            if extreme_sensors:
                validation['errors'].append(f"High outlier ratios in sensors: {extreme_sensors}")
                self.logger.warning(f"Cast {cast_id}: High outlier ratios: {extreme_sensors}")
            
        except Exception as e:
            validation['errors'].append(f"Validation error: {str(e)}")
            validation['is_valid'] = False
            self.logger.error(f"Error validating cast {cast_id}: {str(e)}")
        
        return validation
    
    def validate_features(self, features_df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate engineered features for completeness and quality
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'has_features': False,
            'sufficient_features': False,
            'low_nan_ratio': False,
            'no_infinite_values': False,
            'reasonable_ranges': False,
            'errors': []
        }
        
        try:
            # Check if features exist
            if features_df.empty:
                validation['errors'].append("No features extracted")
                validation['is_valid'] = False
                return validation
            
            validation['has_features'] = True
            
            # Check feature count (expecting 90+ features)
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            if 'cast_id' in numeric_cols:
                numeric_cols = numeric_cols.drop('cast_id')
            
            if len(numeric_cols) < 80:  # Allow some tolerance
                validation['errors'].append(f"Insufficient features: {len(numeric_cols)} < 80")
                validation['is_valid'] = False
            else:
                validation['sufficient_features'] = True
            
            # Check NaN ratio
            if len(numeric_cols) > 0:
                nan_counts = features_df[numeric_cols].isnull().sum()
                total_features = len(numeric_cols) * len(features_df)
                nan_ratio = nan_counts.sum() / total_features if total_features > 0 else 1.0
                
                if nan_ratio > 0.2:  # More than 20% NaN
                    validation['errors'].append(f"High NaN ratio in features: {nan_ratio:.3f}")
                    validation['is_valid'] = False
                else:
                    validation['low_nan_ratio'] = True
            
            # Check for infinite values
            inf_counts = np.isinf(features_df[numeric_cols]).sum().sum()
            if inf_counts > 0:
                validation['errors'].append(f"Infinite values detected: {inf_counts}")
                validation['is_valid'] = False
            else:
                validation['no_infinite_values'] = True
            
            # Check for reasonable feature ranges
            unreasonable_features = []
            for col in numeric_cols:
                values = features_df[col].dropna()
                if len(values) > 0:
                    # Check for extremely large values
                    if values.abs().max() > 1e10:
                        unreasonable_features.append(f"{col}(too_large)")
                    
                    # Check for extremely small non-zero values
                    non_zero_values = values[values != 0]
                    if len(non_zero_values) > 0 and non_zero_values.abs().min() < 1e-10:
                        unreasonable_features.append(f"{col}(too_small)")
            
            if unreasonable_features:
                validation['errors'].append(f"Unreasonable feature ranges: {unreasonable_features[:5]}")
                # Don't mark as invalid, just warn
                self.logger.warning(f"Unreasonable feature ranges detected: {len(unreasonable_features)} features")
            else:
                validation['reasonable_ranges'] = True
                
        except Exception as e:
            validation['errors'].append(f"Feature validation error: {str(e)}")
            validation['is_valid'] = False
            self.logger.error(f"Error validating features: {str(e)}")
        
        return validation
    
    def check_feature_correlations(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Check for highly correlated features that may need removal
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            DataFrame with highly correlated feature pairs
        """
        try:
            # Get numeric columns only
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            if 'cast_id' in numeric_cols:
                numeric_cols = numeric_cols.drop('cast_id')
            
            if len(numeric_cols) < 2:
                self.logger.warning("Insufficient numeric features for correlation analysis")
                return pd.DataFrame()
            
            # Calculate correlation matrix
            corr_matrix = features_df[numeric_cols].corr()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > self.correlation_threshold:
                        high_corr_pairs.append({
                            'feature1': numeric_cols[i],
                            'feature2': numeric_cols[j],
                            'correlation': corr_val
                        })
            
            if high_corr_pairs:
                self.logger.info(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
                return pd.DataFrame(high_corr_pairs)
            else:
                self.logger.info("No highly correlated features found")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error checking feature correlations: {str(e)}")
            return pd.DataFrame()
    
    def generate_validation_report(self, 
                                 input_validation: Dict[str, bool],
                                 feature_validation: Dict[str, bool],
                                 correlation_df: pd.DataFrame,
                                 cast_id: Optional[str] = None) -> Dict:
        """
        Generate comprehensive validation report
        
        Args:
            input_validation: Results from input data validation
            feature_validation: Results from feature validation
            correlation_df: High correlation analysis results
            cast_id: Optional cast ID
            
        Returns:
            Comprehensive validation report
        """
        report = {
            'cast_id': cast_id,
            'overall_valid': input_validation.get('is_valid', False) and feature_validation.get('is_valid', False),
            'input_validation': input_validation,
            'feature_validation': feature_validation,
            'correlation_analysis': {
                'high_corr_count': len(correlation_df),
                'max_correlation': correlation_df['correlation'].abs().max() if not correlation_df.empty else 0,
                'recommended_removals': []
            },
            'warnings': [],
            'recommendations': []
        }
        
        # Add recommendations based on validation results
        if not input_validation.get('sufficient_length', False):
            report['recommendations'].append("Increase data collection duration")
        
        if not input_validation.get('low_missing_data', False):
            report['recommendations'].append("Improve sensor data quality and reduce missing values")
        
        if not feature_validation.get('sufficient_features', False):
            report['recommendations'].append("Check feature extraction pipeline")
        
        if not feature_validation.get('low_nan_ratio', False):
            report['recommendations'].append("Investigate causes of NaN features")
        
        if not correlation_df.empty:
            # Suggest removing highly correlated features
            corr_features = correlation_df.sort_values('correlation', key=abs, ascending=False)
            if len(corr_features) > 0:
                report['correlation_analysis']['recommended_removals'] = corr_features['feature2'].head(5).tolist()
                report['recommendations'].append("Consider removing highly correlated features")
        
        return report