import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_score, train_test_split
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, 
    precision_score, recall_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import json
import yaml
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class BaselineXGBoostModel:
    """XGBoost baseline model for defect prediction"""
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 model_params: Optional[Dict] = None,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Initialize the baseline XGBoost model
        
        Args:
            config_path: Path to YAML configuration file
            model_params: Dictionary of XGBoost parameters
            random_state: Random seed for reproducibility
            verbose: Enable verbose logging
        """
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        self.feature_importance_ = None
        self.is_trained = False
        self.training_history = {}
        self.feature_names_ = None
        
        # Setup logging
        self._setup_logging()
        
        # Load configuration
        if config_path:
            self.config = self.load_config(config_path)
        elif model_params:
            self.config = {'model_params': model_params}
        else:
            self.config = {'model_params': self.get_default_params()}
            
        # Initialize model with parameters
        self.set_hyperparameters(self.config.get('model_params', self.get_default_params()))
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO if self.verbose else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load model configuration from YAML file
        
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
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            return {'model_params': self.get_default_params()}
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {'model_params': self.get_default_params()}
    
    def set_hyperparameters(self, params: Dict[str, Any]) -> None:
        """
        Set XGBoost hyperparameters with validation
        
        Args:
            params: Dictionary of hyperparameters
        """
        # Add random state if not present
        if 'random_state' not in params:
            params['random_state'] = self.random_state
            
        self.model_params = params
        self.model = xgb.XGBClassifier(**params)
        self.logger.info(f"Model initialized with parameters: {params}")
        
    def get_default_params(self) -> Dict[str, Any]:
        """
        Get default XGBoost parameters optimized for defect prediction
        
        Returns:
            Default parameter dictionary
        """
        return {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': 1 if self.verbose else 0
        }

    def fit(self, 
            X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray],
            X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y_val: Optional[Union[pd.Series, np.ndarray]] = None,
            sample_weight: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the XGBoost model with optional validation set
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            sample_weight: Sample weights for training
            
        Returns:
            Training history and metrics
        """
        start_time = datetime.now()
        
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Prepare evaluation set if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            eval_set = [(X, y), (X_val, y_val)]
            
        try:
            # Train the model
            self.model.fit(
                X, y,
                sample_weight=sample_weight,
                eval_set=eval_set,
                verbose=False
            )
            
            self.is_trained = True
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate training metrics
            y_pred_proba = self.model.predict_proba(X)[:, 1]
            train_auc = roc_auc_score(y, y_pred_proba)
            
            # Store training history
            self.training_history = {
                'training_time': training_time,
                'train_auc': train_auc,
                'n_samples': len(X),
                'n_features': X.shape[1],
                'timestamp': datetime.now().isoformat()
            }
            
            if eval_set is not None:
                val_pred_proba = self.model.predict_proba(X_val)[:, 1]
                val_auc = roc_auc_score(y_val, val_pred_proba)
                self.training_history['val_auc'] = val_auc
                
            self.logger.info(f"Model training completed in {training_time:.2f}s")
            self.logger.info(f"Training AUC: {train_auc:.4f}")
            
            return self.training_history
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise

    # Keep the old train method for backward compatibility
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the baseline model (backward compatibility method).
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Training labels
        """
        self.fit(X, y)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict defect probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict defect classes.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted classes
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.model.predict(X)
        
    def predict_batch(self, 
                     X_batches: List[Union[pd.DataFrame, np.ndarray]],
                     batch_size: int = 1000) -> List[np.ndarray]:
        """
        Process predictions in batches for memory efficiency
        
        Args:
            X_batches: List of feature batches
            batch_size: Size of each batch
            
        Returns:
            List of prediction arrays
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        predictions = []
        for batch in X_batches:
            if isinstance(batch, pd.DataFrame):
                batch = batch.values
            pred = self.model.predict_proba(batch)[:, 1]
            predictions.append(pred)
            
        return predictions
    
    def cross_validate(self,
                      X: Union[pd.DataFrame, np.ndarray],
                      y: Union[pd.Series, np.ndarray],
                      cv_folds: int = 5,
                      scoring: Union[str, List[str]] = ['roc_auc', 'average_precision'],
                      stratify: bool = True) -> Dict[str, Any]:
        """
        Perform stratified k-fold cross-validation
        
        Args:
            X: Features for cross-validation
            y: Labels for cross-validation
            cv_folds: Number of CV folds
            scoring: Scoring metrics to evaluate
            stratify: Use stratified sampling
            
        Returns:
            Cross-validation results and statistics
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Setup cross-validation
        if stratify:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
        # Ensure scoring is a list
        if isinstance(scoring, str):
            scoring = [scoring]
            
        results = {}
        
        # Perform cross-validation for each metric
        for metric in scoring:
            try:
                scores = cross_val_score(self.model, X, y, cv=cv, scoring=metric, n_jobs=-1)
                results[f'{metric}_scores'] = scores.tolist()
                results[f'{metric}_mean'] = scores.mean()
                results[f'{metric}_std'] = scores.std()
                
                self.logger.info(f"CV {metric}: {scores.mean():.4f} ± {scores.std():.4f}")
            except Exception as e:
                self.logger.warning(f"Could not calculate {metric}: {e}")
                
        results['cv_folds'] = cv_folds
        results['stratified'] = stratify
        
        return results
        
    def nested_cross_validate(self,
                             X: Union[pd.DataFrame, np.ndarray],
                             y: Union[pd.Series, np.ndarray],
                             param_grid: Dict[str, List],
                             outer_cv: int = 5,
                             inner_cv: int = 3) -> Dict[str, Any]:
        """
        Perform nested cross-validation for unbiased performance estimation
        
        Args:
            X: Features
            y: Labels
            param_grid: Hyperparameter grid for optimization
            outer_cv: Outer CV folds
            inner_cv: Inner CV folds for hyperparameter tuning
            
        Returns:
            Nested CV results and best parameters
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        outer_cv_obj = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=self.random_state)
        inner_cv_obj = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=self.random_state)
        
        outer_scores = []
        best_params_list = []
        
        for train_idx, test_idx in outer_cv_obj.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Inner CV for hyperparameter tuning
            grid_search = GridSearchCV(
                xgb.XGBClassifier(random_state=self.random_state),
                param_grid,
                cv=inner_cv_obj,
                scoring='roc_auc',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            best_params_list.append(grid_search.best_params_)
            
            # Evaluate on outer test set
            y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
            outer_score = roc_auc_score(y_test, y_pred_proba)
            outer_scores.append(outer_score)
            
        return {
            'outer_scores': outer_scores,
            'outer_score_mean': np.mean(outer_scores),
            'outer_score_std': np.std(outer_scores),
            'best_params_list': best_params_list,
            'outer_cv_folds': outer_cv,
            'inner_cv_folds': inner_cv
        }
    
    def hyperparameter_search(self,
                             X: Union[pd.DataFrame, np.ndarray],
                             y: Union[pd.Series, np.ndarray],
                             param_grid: Dict[str, List],
                             search_method: str = 'grid',
                             cv_folds: int = 5,
                             scoring: str = 'roc_auc',
                             n_jobs: int = -1) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization
        
        Args:
            X: Training features
            y: Training labels
            param_grid: Parameter grid to search
            search_method: 'grid', 'random', or 'bayesian'
            cv_folds: Cross-validation folds
            scoring: Optimization metric
            n_jobs: Parallel jobs
            
        Returns:
            Best parameters and search results
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        if search_method == 'grid':
            search = GridSearchCV(
                xgb.XGBClassifier(random_state=self.random_state),
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1 if self.verbose else 0
            )
        elif search_method == 'random':
            from sklearn.model_selection import RandomizedSearchCV
            search = RandomizedSearchCV(
                xgb.XGBClassifier(random_state=self.random_state),
                param_grid,
                n_iter=50,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                random_state=self.random_state,
                verbose=1 if self.verbose else 0
            )
        else:
            raise ValueError(f"Unsupported search method: {search_method}")
            
        # Perform search
        start_time = datetime.now()
        search.fit(X, y)
        search_time = (datetime.now() - start_time).total_seconds()
        
        # Update model with best parameters
        self.set_hyperparameters(search.best_params_)
        
        self.logger.info(f"Hyperparameter search completed in {search_time:.2f}s")
        self.logger.info(f"Best score: {search.best_score_:.4f}")
        self.logger.info(f"Best params: {search.best_params_}")
        
        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_,
            'search_time': search_time,
            'search_method': search_method
        }
        
    def get_hyperparameter_grid(self, search_type: str = 'coarse') -> Dict[str, List]:
        """
        Get predefined hyperparameter grids
        
        Args:
            search_type: 'coarse', 'fine', or 'extensive'
            
        Returns:
            Hyperparameter grid dictionary
        """
        if search_type == 'coarse':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        elif search_type == 'fine':
            return {
                'n_estimators': [80, 100, 120],
                'max_depth': [5, 6, 7],
                'learning_rate': [0.08, 0.1, 0.12],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2]
            }
        elif search_type == 'extensive':
            return {
                'n_estimators': [50, 100, 150, 200, 300],
                'max_depth': [3, 4, 5, 6, 7, 8, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'min_child_weight': [1, 3, 5, 7],
                'gamma': [0, 0.1, 0.2, 0.3],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [0.5, 1.0, 1.5, 2.0]
            }
        else:
            raise ValueError(f"Unknown search_type: {search_type}")
    
    def get_feature_importance(self, 
                              importance_type: str = 'gain',
                              max_features: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance rankings
        
        Args:
            importance_type: 'weight', 'gain', 'cover', or 'total_gain'
            max_features: Maximum number of features to return
            
        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before extracting feature importance")
            
        importance_dict = self.model.get_booster().get_score(importance_type=importance_type)
        
        # Create feature names if not available
        if self.feature_names_ is None:
            feature_names = [f'feature_{i}' for i in range(len(importance_dict))]
        else:
            feature_names = self.feature_names_
            
        # Convert to DataFrame
        importance_df = pd.DataFrame([
            {'feature': feat, 'importance': importance_dict.get(f'f{i}', 0)}
            for i, feat in enumerate(feature_names)
        ])
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        if max_features:
            importance_df = importance_df.head(max_features)
            
        return importance_df
    
    def plot_feature_importance(self,
                               importance_type: str = 'gain',
                               max_features: int = 20,
                               figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot feature importance
        
        Args:
            importance_type: Type of importance to plot
            max_features: Number of top features to show
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        importance_df = self.get_feature_importance(importance_type, max_features)
        
        plt.figure(figsize=figsize)
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {max_features} Feature Importance ({importance_type})')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        return plt.gcf()
    
    def select_features(self,
                       X: Union[pd.DataFrame, np.ndarray],
                       y: Union[pd.Series, np.ndarray],
                       selection_method: str = 'importance',
                       n_features: Optional[int] = None,
                       threshold: Optional[float] = None) -> List[int]:
        """
        Perform automated feature selection
        
        Args:
            X: Training features
            y: Training labels
            selection_method: 'importance', 'permutation', or 'recursive'
            n_features: Number of features to select
            threshold: Importance threshold for selection
            
        Returns:
            List of selected feature indices
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        if selection_method == 'importance':
            # Train model first if not trained
            if not self.is_trained:
                self.fit(X, y)
                
            importance_df = self.get_feature_importance()
            
            if threshold:
                selected = importance_df[importance_df['importance'] >= threshold].index.tolist()
            elif n_features:
                selected = importance_df.head(n_features).index.tolist()
            else:
                selected = list(range(X.shape[1]))
                
        elif selection_method == 'recursive':
            from sklearn.feature_selection import RFE
            selector = RFE(self.model, n_features_to_select=n_features)
            selector.fit(X, y)
            selected = np.where(selector.support_)[0].tolist()
            
        else:
            raise ValueError(f"Unsupported selection method: {selection_method}")
            
        return selected
    
    def evaluate(self,
                X: Union[pd.DataFrame, np.ndarray],
                y: Union[pd.Series, np.ndarray],
                threshold: float = 0.5,
                plot_curves: bool = True) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            X: Test features
            y: True labels
            threshold: Classification threshold
            plot_curves: Generate ROC and PR curves
            
        Returns:
            Dictionary with all evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Get predictions
        y_proba = self.predict_proba(X)
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y, y_pred, y_proba)
        
        # Plot curves if requested
        if plot_curves:
            metrics['plots'] = self._plot_evaluation_curves(y, y_proba, y_pred)
            
        return metrics
    
    def calculate_metrics(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {
            'accuracy': (y_true == y_pred).mean(),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            metrics['average_precision'] = average_precision_score(y_true, y_proba)
            
        return metrics
    
    def _plot_evaluation_curves(self, y_true: np.ndarray, y_proba: np.ndarray, 
                               y_pred: np.ndarray) -> Dict[str, plt.Figure]:
        """Plot ROC and PR curves"""
        plots = {}
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        ax_roc.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_true, y_proba):.3f})')
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Random')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve')
        ax_roc.legend()
        plots['roc_curve'] = fig_roc
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
        ax_pr.plot(recall, precision, label=f'PR Curve (AP = {average_precision_score(y_true, y_proba):.3f})')
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title('Precision-Recall Curve')
        ax_pr.legend()
        plots['pr_curve'] = fig_pr
        
        return plots
    
    def plot_confusion_matrix(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             normalize: bool = True,
                             figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Normalize confusion matrix
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='.3f' if normalize else 'd', 
                   cmap='Blues', square=True)
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        return plt.gcf()
    
    def save_model(self, 
                   filepath: str,
                   include_metadata: bool = True,
                   compress: bool = True) -> None:
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save model
            include_metadata: Save training metadata
            compress: Compress saved model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
            
        # Prepare model data
        model_data = {
            'model': self.model,
            'model_params': self.model_params,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names_,
            'random_state': self.random_state
        }
        
        if include_metadata:
            model_data['training_history'] = self.training_history
            model_data['timestamp'] = datetime.now().isoformat()
            
        # Save with compression
        if compress:
            joblib.dump(model_data, filepath, compress=3)
        else:
            joblib.dump(model_data, filepath)
            
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model from disk
        
        Args:
            filepath: Path to saved model
        """
        try:
            model_data = joblib.load(filepath)
            
            # Load model components
            self.model = model_data['model']
            self.model_params = model_data.get('model_params', {})
            self.is_trained = model_data.get('is_trained', False)
            self.feature_names_ = model_data.get('feature_names')
            self.random_state = model_data.get('random_state', 42)
            
            # Load metadata if available
            if 'training_history' in model_data:
                self.training_history = model_data['training_history']
                
            self.logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def save_training_history(self, 
                             history: Dict[str, Any],
                             filepath: str) -> None:
        """
        Save training history and metrics
        
        Args:
            history: Training history dictionary
            filepath: Path to save history
        """
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2, default=str)
            
        self.logger.info(f"Training history saved to {filepath}")
        
    def fit_with_early_stopping(self,
                               X: Union[pd.DataFrame, np.ndarray],
                               y: Union[pd.Series, np.ndarray],
                               validation_split: float = 0.2,
                               early_stopping_rounds: int = 10) -> Dict[str, Any]:
        """
        Train model with early stopping mechanism
        
        Args:
            X: Training features
            y: Training labels
            validation_split: Fraction for validation split
            early_stopping_rounds: Rounds to wait before stopping
            
        Returns:
            Training results with early stopping info
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, 
            random_state=self.random_state, stratify=y
        )
        
        # Update model to include early stopping
        model_with_early_stopping = xgb.XGBClassifier(
            **self.model_params,
            early_stopping_rounds=early_stopping_rounds
        )
        
        # Train with early stopping
        start_time = datetime.now()
        
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
            X_val = X_val.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
            y_val = y_val.values
            
        model_with_early_stopping.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        # Update main model
        self.model = model_with_early_stopping
        self.is_trained = True
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate metrics
        train_pred_proba = self.model.predict_proba(X_train)[:, 1]
        val_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        train_auc = roc_auc_score(y_train, train_pred_proba)
        val_auc = roc_auc_score(y_val, val_pred_proba)
        
        # Store results
        self.training_history = {
            'training_time': training_time,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'best_iteration': getattr(model_with_early_stopping, 'best_iteration', None),
            'early_stopping_rounds': early_stopping_rounds,
            'validation_split': validation_split,
            'n_samples': len(X),
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Training completed with early stopping in {training_time:.2f}s")
        self.logger.info(f"Best iteration: {self.training_history.get('best_iteration', 'N/A')}")
        self.logger.info(f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
        
        return self.training_history