import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE
from typing import Dict, List, Tuple, Union, Any, Optional
import logging


class DataPreprocessor:
    """Handle data preprocessing operations"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor with configuration
        
        Args:
            config: Preprocessing configuration dictionary
        """
        self.config = config
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.scalers = {}
        self.feature_selector = None
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def detect_feature_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Automatically detect numeric and categorical features
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with 'numeric' and 'categorical' feature lists
        """
        numeric_features = []
        categorical_features = []
        
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                # Check if it's actually categorical (low cardinality)
                unique_values = df[column].nunique()
                if unique_values <= 10 and df[column].dtype == 'int64':
                    categorical_features.append(column)
                else:
                    numeric_features.append(column)
            else:
                categorical_features.append(column)
        
        self.logger.info(f"Detected {len(numeric_features)} numeric and {len(categorical_features)} categorical features")
        
        return {
            'numeric': numeric_features,
            'categorical': categorical_features
        }
    
    def handle_missing_values(self, 
                             X: pd.DataFrame, 
                             strategy: str = 'median',
                             categorical_strategy: str = 'most_frequent') -> pd.DataFrame:
        """
        Handle missing values using various strategies
        
        Args:
            X: Input dataframe
            strategy: Strategy for numeric features ('mean', 'median', 'constant')
            categorical_strategy: Strategy for categorical features ('most_frequent', 'constant')
            
        Returns:
            DataFrame with missing values handled
        """
        X_processed = X.copy()
        
        # Detect feature types
        feature_types = self.detect_feature_types(X)
        
        # Handle numeric features
        if feature_types['numeric']:
            if self.numeric_imputer is None:
                self.numeric_imputer = SimpleImputer(strategy=strategy)
                X_processed[feature_types['numeric']] = self.numeric_imputer.fit_transform(
                    X_processed[feature_types['numeric']]
                )
            else:
                X_processed[feature_types['numeric']] = self.numeric_imputer.transform(
                    X_processed[feature_types['numeric']]
                )
        
        # Handle categorical features
        if feature_types['categorical']:
            if self.categorical_imputer is None:
                self.categorical_imputer = SimpleImputer(strategy=categorical_strategy)
                X_processed[feature_types['categorical']] = self.categorical_imputer.fit_transform(
                    X_processed[feature_types['categorical']]
                )
            else:
                X_processed[feature_types['categorical']] = self.categorical_imputer.transform(
                    X_processed[feature_types['categorical']]
                )
        
        missing_before = X.isnull().sum().sum()
        missing_after = X_processed.isnull().sum().sum()
        
        self.logger.info(f"Missing values handled: {missing_before} -> {missing_after}")
        
        return X_processed
    
    def handle_outliers(self, 
                       X: pd.DataFrame, 
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Handle outliers using various methods
        
        Args:
            X: Input dataframe
            method: 'iqr', 'zscore', 'isolation_forest', or 'none'
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        X_processed = X.copy()
        
        if method == 'none':
            return X_processed
        
        # Detect numeric features
        feature_types = self.detect_feature_types(X)
        numeric_features = feature_types['numeric']
        
        if not numeric_features:
            return X_processed
        
        if method == 'iqr':
            for feature in numeric_features:
                Q1 = X_processed[feature].quantile(0.25)
                Q3 = X_processed[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Cap outliers
                X_processed[feature] = X_processed[feature].clip(lower_bound, upper_bound)
        
        elif method == 'zscore':
            for feature in numeric_features:
                mean = X_processed[feature].mean()
                std = X_processed[feature].std()
                z_scores = np.abs((X_processed[feature] - mean) / std)
                
                # Cap outliers beyond threshold standard deviations
                outlier_mask = z_scores > threshold
                if outlier_mask.any():
                    upper_limit = mean + threshold * std
                    lower_limit = mean - threshold * std
                    X_processed[feature] = X_processed[feature].clip(lower_limit, upper_limit)
        
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_mask = iso_forest.fit_predict(X_processed[numeric_features]) == -1
            
            self.logger.info(f"Isolation Forest detected {outlier_mask.sum()} outliers")
            
            # For simplicity, cap outliers using IQR method
            for feature in numeric_features:
                Q1 = X_processed[feature].quantile(0.25)
                Q3 = X_processed[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_rows = outlier_mask & ((X_processed[feature] < lower_bound) | 
                                              (X_processed[feature] > upper_bound))
                X_processed.loc[outlier_rows, feature] = X_processed[feature].clip(lower_bound, upper_bound)[outlier_rows]
        
        else:
            raise ValueError(f"Unknown outlier handling method: {method}")
        
        self.logger.info(f"Outliers handled using {method} method")
        return X_processed
    
    def feature_scaling(self, 
                       X: np.ndarray, 
                       method: str = 'standard',
                       feature_range: Tuple[int, int] = (0, 1)) -> np.ndarray:
        """
        Apply feature scaling
        
        Args:
            X: Input array
            method: 'standard', 'robust', 'minmax', or 'none'
            feature_range: Range for MinMax scaling
            
        Returns:
            Scaled array
        """
        if method == 'none':
            return X
        
        if method not in self.scalers:
            if method == 'standard':
                self.scalers[method] = StandardScaler()
            elif method == 'robust':
                self.scalers[method] = RobustScaler()
            elif method == 'minmax':
                self.scalers[method] = MinMaxScaler(feature_range=feature_range)
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            X_scaled = self.scalers[method].fit_transform(X)
        else:
            X_scaled = self.scalers[method].transform(X)
        
        self.logger.info(f"Features scaled using {method} method")
        return X_scaled
    
    def encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using one-hot encoding
        
        Args:
            X: Input dataframe
            
        Returns:
            DataFrame with encoded categorical features
        """
        feature_types = self.detect_feature_types(X)
        categorical_features = feature_types['categorical']
        
        if not categorical_features:
            return X
        
        X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
        
        self.logger.info(f"Categorical features encoded: {len(categorical_features)} -> {len(X_encoded.columns) - len(feature_types['numeric'])} features")
        
        return X_encoded
    
    def select_features(self,
                       X: Union[pd.DataFrame, np.ndarray],
                       y: Union[pd.Series, np.ndarray],
                       method: str = 'variance',
                       k_best: int = 50,
                       threshold: float = 0.01) -> Union[pd.DataFrame, np.ndarray]:
        """
        Perform feature selection
        
        Args:
            X: Input features
            y: Target variable
            method: 'variance', 'univariate', 'recursive'
            k_best: Number of best features to select
            threshold: Variance threshold
            
        Returns:
            Selected features
        """
        if method == 'variance':
            if self.feature_selector is None:
                self.feature_selector = VarianceThreshold(threshold=threshold)
                X_selected = self.feature_selector.fit_transform(X)
            else:
                X_selected = self.feature_selector.transform(X)
        
        elif method == 'univariate':
            if self.feature_selector is None:
                self.feature_selector = SelectKBest(score_func=f_classif, k=k_best)
                X_selected = self.feature_selector.fit_transform(X, y)
            else:
                X_selected = self.feature_selector.transform(X)
        
        elif method == 'recursive':
            from sklearn.ensemble import RandomForestClassifier
            
            if self.feature_selector is None:
                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
                self.feature_selector = RFE(estimator, n_features_to_select=k_best)
                X_selected = self.feature_selector.fit_transform(X, y)
            else:
                X_selected = self.feature_selector.transform(X)
        
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        n_features_before = X.shape[1] if hasattr(X, 'shape') else len(X.columns)
        n_features_after = X_selected.shape[1]
        
        self.logger.info(f"Feature selection: {n_features_before} -> {n_features_after} features using {method}")
        
        return X_selected
    
    def create_feature_engineering_pipeline(self,
                                           X: pd.DataFrame,
                                           y: pd.Series,
                                           include_polynomial: bool = False,
                                           polynomial_degree: int = 2,
                                           include_interactions: bool = False) -> pd.DataFrame:
        """
        Create advanced feature engineering pipeline
        
        Args:
            X: Input features
            y: Target variable
            include_polynomial: Include polynomial features
            polynomial_degree: Degree for polynomial features
            include_interactions: Include interaction features
            
        Returns:
            Engineered features DataFrame
        """
        X_engineered = X.copy()
        
        # Add polynomial features
        if include_polynomial:
            from sklearn.preprocessing import PolynomialFeatures
            
            feature_types = self.detect_feature_types(X)
            numeric_features = feature_types['numeric']
            
            if numeric_features:
                poly = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
                poly_features = poly.fit_transform(X[numeric_features])
                
                # Create feature names
                poly_names = [f"poly_{i}" for i in range(poly_features.shape[1])]
                poly_df = pd.DataFrame(poly_features, columns=poly_names, index=X.index)
                
                X_engineered = pd.concat([X_engineered, poly_df], axis=1)
                
                self.logger.info(f"Added {poly_features.shape[1]} polynomial features")
        
        # Add interaction features (simple pairwise products)
        if include_interactions:
            feature_types = self.detect_feature_types(X)
            numeric_features = feature_types['numeric']
            
            if len(numeric_features) >= 2:
                for i, feat1 in enumerate(numeric_features):
                    for feat2 in numeric_features[i+1:]:
                        interaction_name = f"{feat1}_x_{feat2}"
                        X_engineered[interaction_name] = X[feat1] * X[feat2]
                
                n_interactions = len(numeric_features) * (len(numeric_features) - 1) // 2
                self.logger.info(f"Added {n_interactions} interaction features")
        
        return X_engineered
    
    def preprocess_pipeline(self,
                           X: pd.DataFrame,
                           y: Optional[pd.Series] = None,
                           fit: bool = True) -> pd.DataFrame:
        """
        Complete preprocessing pipeline
        
        Args:
            X: Input features
            y: Target variable (for feature selection)
            fit: Whether to fit preprocessors or just transform
            
        Returns:
            Preprocessed DataFrame
        """
        X_processed = X.copy()
        
        # Handle missing values
        if self.config.get('handle_missing', True):
            strategy = self.config.get('missing_strategy', 'median')
            cat_strategy = self.config.get('categorical_missing_strategy', 'most_frequent')
            X_processed = self.handle_missing_values(X_processed, strategy, cat_strategy)
        
        # Handle outliers
        outlier_config = self.config.get('outliers', {})
        if outlier_config.get('method', 'none') != 'none':
            method = outlier_config.get('method', 'iqr')
            threshold = outlier_config.get('threshold', 1.5)
            X_processed = self.handle_outliers(X_processed, method, threshold)
        
        # Encode categorical features
        X_processed = self.encode_categorical_features(X_processed)
        
        # Feature selection
        feature_selection_config = self.config.get('feature_selection', {})
        if feature_selection_config.get('enabled', False) and y is not None:
            method = feature_selection_config.get('method', 'variance')
            k_best = feature_selection_config.get('k_best', 50)
            X_processed = self.select_features(X_processed, y, method, k_best)
        
        self.logger.info(f"Preprocessing pipeline completed: {X.shape} -> {X_processed.shape}")
        
        return X_processed