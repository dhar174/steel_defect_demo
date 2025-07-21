import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, 
    RandomizedSearchCV, cross_val_score
)
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, 
    LabelEncoder, OneHotEncoder
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import yaml
import json
import joblib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import warnings

from .baseline_model import BaselineXGBoostModel
from .preprocessing import DataPreprocessor
from .hyperparameter_search import HyperparameterSearcher
from .training_config import TrainingConfig, TrainingPipelineConfig
from .training_utils import TrainingUtils


class ModelTrainer:
    """
    Comprehensive automated training pipeline that orchestrates the entire 
    machine learning workflow from data preprocessing to model evaluation.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 model_type: str = 'xgboost',
                 experiment_name: Optional[str] = None,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Initialize the model training pipeline
        
        Args:
            config_path: Path to training configuration file
            model_type: Type of model to train ('xgboost', 'random_forest', etc.)
            experiment_name: Name for experiment tracking
            random_state: Random seed for reproducibility
            verbose: Enable verbose logging
        """
        self.config_path = config_path
        self.model_type = model_type
        self.experiment_name = experiment_name or f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.random_state = random_state
        self.verbose = verbose
        
        # Initialize components
        self.config = None
        self.preprocessor = None
        self.hyperparameter_searcher = None
        self.model = None
        self.training_history = {}
        self.preprocessing_pipeline = None
        self.early_stopping_config = {}
        
        # Setup logging
        self._setup_logging()
        
        # Load configuration
        if config_path:
            self.config = self.load_config(config_path)
            self.validate_config(self.config)
        
        # Setup experiment tracking
        self.setup_experiment(self.experiment_name)
        
        # Initialize model based on type
        self._initialize_model()
        
        # Initialize supporting components
        self._initialize_components()
    
    def _setup_logging(self) -> None:
        """Setup comprehensive logging configuration"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO if self.verbose else logging.WARNING,
            format=log_format
        )
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.experiment_name}")
        
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Add file handler
        log_file = log_dir / f"{self.experiment_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"ModelTrainer initialized for experiment: {self.experiment_name}")
    
    def _initialize_model(self) -> None:
        """Initialize the appropriate model based on model_type"""
        if self.model_type == 'xgboost':
            model_params = None
            if self.config and 'model' in self.config:
                model_params = self.config['model'].get('parameters', {})
            
            self.model = BaselineXGBoostModel(
                model_params=model_params,
                random_state=self.random_state,
                verbose=self.verbose
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _initialize_components(self) -> None:
        """Initialize supporting components"""
        config_dict = self.config or {}
        
        self.preprocessor = DataPreprocessor(config_dict.get('preprocessing', {}))
        self.hyperparameter_searcher = HyperparameterSearcher(
            config_dict.get('hyperparameter_search', {}).get('method', 'grid')
        )
        self.training_utils = TrainingUtils(self.random_state)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load training configuration from YAML file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found")
            raise
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            raise
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate training configuration parameters
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if configuration is valid
        """
        try:
            training_config = TrainingPipelineConfig.from_dict(config)
            self.logger.info("Configuration validation passed")
            return True
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise ValueError(f"Invalid configuration: {e}")
    
    def setup_experiment(self, experiment_name: str) -> None:
        """
        Set up experiment tracking and logging
        
        Args:
            experiment_name: Name for the experiment
        """
        self.experiment_name = experiment_name
        
        # Create experiment directories
        base_dir = Path("experiments") / experiment_name
        self.experiment_dir = base_dir
        self.model_dir = base_dir / "models"
        self.results_dir = base_dir / "results"
        self.plots_dir = base_dir / "plots"
        
        for dir_path in [self.experiment_dir, self.model_dir, self.results_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Experiment setup completed: {experiment_name}")
    
    def create_preprocessing_pipeline(self, 
                                     numeric_features: List[str],
                                     categorical_features: Optional[List[str]] = None,
                                     scaling_method: str = 'standard') -> ColumnTransformer:
        """
        Create data preprocessing pipeline
        
        Args:
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names
            scaling_method: 'standard', 'robust', 'minmax', or 'none'
            
        Returns:
            Sklearn ColumnTransformer pipeline
        """
        transformers = []
        
        # Numeric preprocessing
        if numeric_features:
            if scaling_method == 'standard':
                numeric_transformer = StandardScaler()
            elif scaling_method == 'robust':
                numeric_transformer = RobustScaler()
            elif scaling_method == 'minmax':
                numeric_transformer = MinMaxScaler()
            elif scaling_method == 'none':
                numeric_transformer = 'passthrough'
            else:
                raise ValueError(f"Unknown scaling method: {scaling_method}")
                
            transformers.append(('num', numeric_transformer, numeric_features))
        
        # Categorical preprocessing
        if categorical_features:
            categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        pipeline = ColumnTransformer(transformers=transformers, remainder='drop')
        self.preprocessing_pipeline = pipeline
        
        self.logger.info(f"Preprocessing pipeline created with {scaling_method} scaling")
        return pipeline
    
    def preprocess_data(self,
                       X: Union[pd.DataFrame, np.ndarray],
                       y: Union[pd.Series, np.ndarray],
                       fit_preprocessor: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply preprocessing pipeline to data
        
        Args:
            X: Input features
            y: Target labels
            fit_preprocessor: Whether to fit the preprocessor
            
        Returns:
            Preprocessed features and labels
        """
        if self.preprocessing_pipeline is None:
            self.logger.warning("No preprocessing pipeline found, returning original data")
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values
            return X, y
        
        if fit_preprocessor:
            X_processed = self.preprocessing_pipeline.fit_transform(X)
        else:
            X_processed = self.preprocessing_pipeline.transform(X)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        self.logger.info(f"Data preprocessed: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
        return X_processed, y
    
    def handle_missing_values(self,
                             X: pd.DataFrame,
                             strategy: str = 'median',
                             categorical_strategy: str = 'most_frequent') -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            X: Input dataframe
            strategy: Strategy for numeric features
            categorical_strategy: Strategy for categorical features
            
        Returns:
            DataFrame with missing values handled
        """
        return self.preprocessor.handle_missing_values(X, strategy, categorical_strategy)
    
    def create_train_validation_split(self,
                                     X: Union[pd.DataFrame, np.ndarray],
                                     y: Union[pd.Series, np.ndarray],
                                     test_size: float = 0.2,
                                     validation_size: float = 0.2,
                                     stratify: bool = True) -> Tuple[np.ndarray, ...]:
        """
        Create train/validation/test splits
        
        Args:
            X: Input features
            y: Target labels
            test_size: Fraction for test set
            validation_size: Fraction for validation set
            stratify: Use stratified sampling
            
        Returns:
            Train, validation, test splits (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        stratify_arg = y if stratify else None
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=stratify_arg
        )
        
        # Second split: train vs val
        val_size_adjusted = validation_size / (1 - test_size)
        stratify_temp = y_temp if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=self.random_state, stratify=stratify_temp
        )
        
        self.logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def setup_cross_validation(self,
                              cv_folds: int = 5,
                              stratify: bool = True) -> StratifiedKFold:
        """
        Set up cross-validation strategy
        
        Args:
            cv_folds: Number of CV folds
            stratify: Use stratified sampling
            
        Returns:
            Cross-validation object
        """
        if stratify:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        self.logger.info(f"Cross-validation setup: {cv_folds} folds, stratified={stratify}")
        return cv
    
    def train_model(self,
                   X_train: Union[pd.DataFrame, np.ndarray],
                   y_train: Union[pd.Series, np.ndarray],
                   X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                   y_val: Optional[Union[pd.Series, np.ndarray]] = None,
                   model_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train a single model with given parameters
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_params: Model hyperparameters
            
        Returns:
            Training results and metrics
        """
        if model_params:
            self.model.set_hyperparameters(model_params)
        
        start_time = datetime.now()
        
        # Train the model
        training_results = self.model.fit(X_train, y_train, X_val, y_val)
        
        training_time = (datetime.now() - start_time).total_seconds()
        training_results['total_training_time'] = training_time
        
        # Store training history
        self.training_history.update(training_results)
        
        self.logger.info(f"Model training completed in {training_time:.2f}s")
        return training_results
    
    def train_with_cross_validation(self,
                                   X: Union[pd.DataFrame, np.ndarray],
                                   y: Union[pd.Series, np.ndarray],
                                   model_params: Optional[Dict[str, Any]] = None,
                                   cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train model with cross-validation
        
        Args:
            X: Full feature set
            y: Full label set
            model_params: Model hyperparameters
            cv_folds: Number of CV folds
            
        Returns:
            Cross-validation results
        """
        if model_params:
            self.model.set_hyperparameters(model_params)
        
        cv_results = self.model.cross_validate(X, y, cv_folds=cv_folds)
        
        self.training_history.update(cv_results)
        self.logger.info(f"Cross-validation completed: {cv_results.get('roc_auc_mean', 'N/A'):.4f} ± {cv_results.get('roc_auc_std', 'N/A'):.4f}")
        
        return cv_results
    
    def train_pipeline(self,
                      data_path: str,
                      target_column: str,
                      feature_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute complete training pipeline from data loading to evaluation
        
        Args:
            data_path: Path to training data
            target_column: Name of target column
            feature_columns: List of feature columns to use
            
        Returns:
            Complete pipeline results
        """
        self.logger.info(f"Starting training pipeline for {data_path}")
        
        # Load data
        data = pd.read_csv(data_path)
        
        # Extract features and target
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        
        X = data[feature_columns]
        y = data[target_column]
        
        # Handle missing values if configured
        if self.config and self.config.get('preprocessing', {}).get('handle_missing', False):
            X = self.handle_missing_values(X)
        
        # Detect feature types and create preprocessing pipeline
        feature_types = self.preprocessor.detect_feature_types(X)
        scaling_method = 'standard'
        if self.config:
            scaling_method = self.config.get('preprocessing', {}).get('scaling', {}).get('method', 'standard')
        
        self.create_preprocessing_pipeline(
            numeric_features=feature_types['numeric'],
            categorical_features=feature_types['categorical'],
            scaling_method=scaling_method
        )
        
        # Create train/validation/test splits
        test_size = 0.2
        validation_size = 0.2
        if self.config:
            data_config = self.config.get('data', {})
            test_size = data_config.get('test_size', 0.2)
            validation_size = data_config.get('validation_size', 0.2)
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.create_train_validation_split(
            X, y, test_size=test_size, validation_size=validation_size
        )
        
        # Preprocess data
        X_train_processed, y_train = self.preprocess_data(X_train, y_train, fit_preprocessor=True)
        X_val_processed, y_val = self.preprocess_data(X_val, y_val, fit_preprocessor=False)
        X_test_processed, y_test = self.preprocess_data(X_test, y_test, fit_preprocessor=False)
        
        pipeline_results = {
            'data_info': {
                'total_samples': len(data),
                'n_features': len(feature_columns),
                'train_samples': len(X_train_processed),
                'val_samples': len(X_val_processed),
                'test_samples': len(X_test_processed)
            }
        }
        
        # Hyperparameter optimization if enabled
        if self.config and self.config.get('hyperparameter_search', {}).get('enabled', False):
            search_results = self.hyperparameter_search(
                X_train_processed, y_train,
                param_grid=self._get_param_grid_from_config(),
                search_method=self.config['hyperparameter_search'].get('method', 'grid')
            )
            pipeline_results['hyperparameter_search'] = search_results
        
        # Train final model
        training_results = self.train_model(X_train_processed, y_train, X_val_processed, y_val)
        pipeline_results['training'] = training_results
        
        # Evaluate on test set
        test_results = self.model.evaluate(X_test_processed, y_test, plot_curves=True)
        pipeline_results['test_evaluation'] = test_results
        
        # Cross-validation
        if self.config and self.config.get('training', {}).get('cross_validation', {}).get('enabled', False):
            cv_folds = self.config['training']['cross_validation'].get('cv_folds', 5)
            cv_results = self.train_with_cross_validation(X_train_processed, y_train, cv_folds=cv_folds)
            pipeline_results['cross_validation'] = cv_results
        
        # Save artifacts
        self._save_pipeline_artifacts(pipeline_results)
        
        self.logger.info("Training pipeline completed successfully")
        return pipeline_results
    
    def _get_param_grid_from_config(self) -> Dict[str, List]:
        """Get parameter grid from configuration"""
        if not self.config:
            return self.model.get_hyperparameter_grid('coarse')
        
        search_config = self.config.get('hyperparameter_search', {})
        param_grids = search_config.get('param_grids', {})
        
        # Use coarse grid by default
        return param_grids.get('coarse', self.model.get_hyperparameter_grid('coarse'))
    
    def hyperparameter_search(self,
                             X: Union[pd.DataFrame, np.ndarray],
                             y: Union[pd.Series, np.ndarray],
                             param_grid: Dict[str, List],
                             search_method: str = 'grid',
                             cv_folds: int = 5,
                             scoring: str = 'roc_auc',
                             n_iter: int = 100,
                             n_jobs: int = -1) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization
        
        Args:
            X: Training features
            y: Training labels
            param_grid: Parameter grid or distributions
            search_method: 'grid', 'random', or 'bayesian'
            cv_folds: Cross-validation folds
            scoring: Optimization metric
            n_iter: Number of iterations for random search
            n_jobs: Parallel jobs
            
        Returns:
            Best parameters and search results
        """
        if search_method == 'bayesian':
            return self.bayesian_optimization(X, y, param_grid, n_calls=n_iter, cv_folds=cv_folds)
        else:
            return self.model.hyperparameter_search(
                X, y, param_grid, search_method, cv_folds, scoring, n_jobs
            )
    
    def get_default_param_grids(self, model_type: str) -> Dict[str, Dict[str, List]]:
        """
        Get default parameter grids for different model types
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary of parameter grids
        """
        if model_type == 'xgboost':
            return {
                'coarse': self.model.get_hyperparameter_grid('coarse'),
                'fine': self.model.get_hyperparameter_grid('fine'),
                'extensive': self.model.get_hyperparameter_grid('extensive')
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def bayesian_optimization(self,
                             X: Union[pd.DataFrame, np.ndarray],
                             y: Union[pd.Series, np.ndarray],
                             param_space: Dict[str, Any],
                             n_calls: int = 50,
                             cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform Bayesian hyperparameter optimization
        
        Args:
            X: Training features
            y: Training labels
            param_space: Parameter space for optimization
            n_calls: Number of optimization calls
            cv_folds: Cross-validation folds
            
        Returns:
            Optimization results
        """
        return self.hyperparameter_searcher.bayesian_search(
            self.model.model, param_space, X, y, n_calls=n_calls, cv_folds=cv_folds
        )
    
    def setup_early_stopping(self,
                             monitor_metric: str = 'val_auc',
                             patience: int = 10,
                             min_delta: float = 0.001,
                             restore_best_weights: bool = True) -> Dict[str, Any]:
        """
        Configure early stopping parameters
        
        Args:
            monitor_metric: Metric to monitor
            patience: Number of epochs to wait
            min_delta: Minimum change threshold
            restore_best_weights: Whether to restore best weights
            
        Returns:
            Early stopping configuration
        """
        self.early_stopping_config = {
            'monitor_metric': monitor_metric,
            'patience': patience,
            'min_delta': min_delta,
            'restore_best_weights': restore_best_weights
        }
        return self.early_stopping_config
    
    def detect_overfitting(self,
                          train_scores: List[float],
                          val_scores: List[float],
                          patience: int = 5) -> bool:
        """
        Detect overfitting in training curves
        
        Args:
            train_scores: Training scores over time
            val_scores: Validation scores over time
            patience: Number of epochs to check
            
        Returns:
            True if overfitting detected
        """
        return self.training_utils.detect_overfitting(train_scores, val_scores, patience)
    
    def apply_regularization(self,
                            model_params: Dict[str, Any],
                            regularization_strength: float = 0.1) -> Dict[str, Any]:
        """
        Apply regularization techniques to model parameters
        
        Args:
            model_params: Base model parameters
            regularization_strength: Strength of regularization
            
        Returns:
            Regularized parameters
        """
        return self.training_utils.apply_regularization(model_params, regularization_strength)
    
    def track_training_progress(self,
                               epoch: int,
                               train_metrics: Dict[str, float],
                               val_metrics: Dict[str, float]) -> None:
        """
        Track training progress and metrics
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        if 'epoch_history' not in self.training_history:
            self.training_history['epoch_history'] = []
        
        self.training_history['epoch_history'].append({
            'epoch': epoch,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'timestamp': datetime.now().isoformat()
        })
    
    def plot_training_history(self,
                             history: Dict[str, List[float]],
                             metrics: List[str] = ['loss', 'auc'],
                             figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
        """
        Plot training history curves
        
        Args:
            history: Training history dictionary
            metrics: Metrics to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        return self.training_utils.plot_training_history(history, metrics, figsize)
    
    def generate_training_report(self,
                               results: Dict[str, Any],
                               output_path: str) -> None:
        """
        Generate comprehensive training report
        
        Args:
            results: Training results
            output_path: Path to save report
        """
        self.training_utils.generate_training_report(results, output_path)
    
    def save_trained_model(self,
                          model: Any,
                          model_path: str,
                          metadata: Dict[str, Any]) -> None:
        """
        Save trained model with metadata
        
        Args:
            model: Trained model object
            model_path: Path to save model
            metadata: Model metadata
        """
        # Save model
        model.save_model(model_path, include_metadata=True)
        
        # Save additional metadata
        metadata_path = str(Path(model_path).with_suffix('.json'))
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Model and metadata saved to {model_path}")
    
    def save_preprocessing_pipeline(self,
                                  pipeline: Any,
                                  pipeline_path: str) -> None:
        """
        Save preprocessing pipeline
        
        Args:
            pipeline: Preprocessing pipeline
            pipeline_path: Path to save pipeline
        """
        joblib.dump(pipeline, pipeline_path)
        self.logger.info(f"Preprocessing pipeline saved to {pipeline_path}")
    
    def create_model_version(self,
                            model_name: str,
                            performance_metrics: Dict[str, float]) -> str:
        """
        Create versioned model identifier
        
        Args:
            model_name: Base model name
            performance_metrics: Model performance metrics
            
        Returns:
            Versioned model identifier
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Use AUC as primary performance metric
        auc_score = performance_metrics.get('roc_auc', performance_metrics.get('test_auc', 0))
        performance_str = f"auc{auc_score:.3f}".replace('.', '')
        
        version = f"{model_name}_{timestamp}_{performance_str}"
        return version
    
    def _save_pipeline_artifacts(self, results: Dict[str, Any]) -> None:
        """Save all pipeline artifacts"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        if self.model and self.model.is_trained:
            model_path = self.model_dir / f"model_{timestamp}.joblib"
            metadata = {
                'experiment_name': self.experiment_name,
                'model_type': self.model_type,
                'timestamp': timestamp,
                'results': results
            }
            self.save_trained_model(self.model, str(model_path), metadata)
        
        # Save preprocessing pipeline
        if self.preprocessing_pipeline:
            pipeline_path = self.model_dir / f"preprocessing_pipeline_{timestamp}.joblib"
            self.save_preprocessing_pipeline(self.preprocessing_pipeline, str(pipeline_path))
        
        # Save results
        results_path = self.results_dir / f"training_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save training history
        history_path = self.results_dir / f"training_history_{timestamp}.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        
        self.logger.info(f"All artifacts saved to {self.experiment_dir}")


# Legacy support - keep LSTMTrainer for backward compatibility  
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    
    class LSTMTrainer:
        """Training pipeline for LSTM model - Legacy support"""
        
        def __init__(self, model: nn.Module, config: Dict):
            """Initialize LSTM trainer (legacy)"""
            self.model = model
            self.config = config
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )
            
            # Weighted loss for class imbalance
            pos_weight = torch.tensor([config['loss_function']['pos_weight']])
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            self.train_losses = []
            self.val_losses = []
            self.best_val_loss = float('inf')
            self.patience_counter = 0
        
        def get_training_history(self) -> Dict:
            """Get training history."""
            return {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses
            }
            
except ImportError:
    # PyTorch not available, skip LSTM trainer
    pass