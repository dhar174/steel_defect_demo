import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os

from src.models.model_trainer import ModelTrainer
from src.models.preprocessing import DataPreprocessor
from src.models.hyperparameter_search import HyperparameterSearcher
from src.models.training_config import TrainingPipelineConfig, ConfigurationManager
from src.models.training_utils import TrainingUtils


class TestModelTrainer:
    """Test cases for the ModelTrainer class"""
    
    def setup_method(self):
        """Setup test data and configurations"""
        np.random.seed(42)
        
        # Create synthetic dataset
        n_samples = 1000
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple linear relationship
        
        self.df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        self.df['defect'] = y
        
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = os.path.join(self.temp_dir, 'test_data.csv')
        self.df.to_csv(self.data_path, index=False)
        
        # Create test configuration
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        
    def teardown_method(self):
        """Clean up test files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_model_trainer_initialization(self):
        """Test ModelTrainer initialization"""
        trainer = ModelTrainer(
            model_type='xgboost',
            experiment_name='test_experiment',
            random_state=42,
            verbose=False
        )
        
        assert trainer.model_type == 'xgboost'
        assert trainer.experiment_name == 'test_experiment'
        assert trainer.random_state == 42
        assert trainer.model is not None
        assert trainer.preprocessor is not None
        assert trainer.hyperparameter_searcher is not None
    
    def test_configuration_loading_and_validation(self):
        """Test configuration loading and validation"""
        # Create configuration manager
        config_manager = ConfigurationManager()
        
        # Create default config
        config = config_manager.create_default_config(self.config_path)
        
        # Validate configuration
        errors = config.validate()
        assert len(errors) == 0
        
        # Test loading
        loaded_config = config_manager.load_config(self.config_path)
        assert loaded_config.model.type == 'xgboost'
        assert loaded_config.data.test_size == 0.2
    
    def test_data_preprocessing_pipeline(self):
        """Test data preprocessing functionality"""
        preprocessor = DataPreprocessor({})
        
        # Test feature type detection
        feature_types = preprocessor.detect_feature_types(self.df.drop('defect', axis=1))
        assert 'numeric' in feature_types
        assert 'categorical' in feature_types
        
        # Test missing value handling
        df_with_missing = self.df.copy()
        df_with_missing.iloc[0:10, 0] = np.nan
        
        df_handled = preprocessor.handle_missing_values(df_with_missing)
        assert df_handled.isnull().sum().sum() == 0
    
    def test_train_test_split(self):
        """Test data splitting with stratification"""
        trainer = ModelTrainer(model_type='xgboost', verbose=False)
        
        X = self.df.drop('defect', axis=1)
        y = self.df['defect']
        
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.create_train_validation_split(
            X, y, test_size=0.2, validation_size=0.2, stratify=True
        )
        
        # Check splits are correct size
        total_samples = len(X)
        assert len(X_train) == int(total_samples * 0.6)  # 60% for train
        assert len(X_val) == int(total_samples * 0.2)    # 20% for val
        assert len(X_test) == int(total_samples * 0.2)   # 20% for test
        
        # Check stratification worked (roughly balanced)
        train_balance = y_train.mean()
        val_balance = y_val.mean()
        test_balance = y_test.mean()
        overall_balance = y.mean()
        
        assert abs(train_balance - overall_balance) < 0.1
        assert abs(val_balance - overall_balance) < 0.1
        assert abs(test_balance - overall_balance) < 0.1
    
    def test_model_training(self):
        """Test basic model training functionality"""
        trainer = ModelTrainer(model_type='xgboost', verbose=False)
        
        X = self.df.drop('defect', axis=1).values
        y = self.df['defect'].values
        
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.create_train_validation_split(X, y)
        
        # Train model
        results = trainer.train_model(X_train, y_train, X_val, y_val)
        
        assert 'train_auc' in results
        assert 'val_auc' in results
        assert results['train_auc'] > 0.5  # Should be better than random
        assert results['val_auc'] > 0.5
        assert trainer.model.is_trained
    
    def test_cross_validation(self):
        """Test cross-validation training"""
        trainer = ModelTrainer(model_type='xgboost', verbose=False)
        
        X = self.df.drop('defect', axis=1).values
        y = self.df['defect'].values
        
        cv_results = trainer.train_with_cross_validation(X, y, cv_folds=3)
        
        assert 'roc_auc_mean' in cv_results
        assert 'roc_auc_std' in cv_results
        assert cv_results['roc_auc_mean'] > 0.5
        assert cv_results['cv_folds'] == 3
    
    def test_hyperparameter_search(self):
        """Test hyperparameter optimization methods"""
        trainer = ModelTrainer(model_type='xgboost', verbose=False)
        
        X = self.df.drop('defect', axis=1).values
        y = self.df['defect'].values
        
        # Small parameter grid for testing
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.2]
        }
        
        # Test grid search
        search_results = trainer.hyperparameter_search(
            X, y, param_grid, search_method='grid', cv_folds=3
        )
        
        assert 'best_params' in search_results
        assert 'best_score' in search_results
        assert search_results['best_score'] > 0.5
        
        # Test random search
        search_results_random = trainer.hyperparameter_search(
            X, y, param_grid, search_method='random', n_iter=5, cv_folds=3
        )
        
        assert 'best_params' in search_results_random
        assert search_results_random['best_score'] > 0.5
    
    def test_early_stopping_configuration(self):
        """Test early stopping mechanism"""
        trainer = ModelTrainer(model_type='xgboost', verbose=False)
        
        early_stop_config = trainer.setup_early_stopping(
            monitor_metric='val_auc',
            patience=10,
            min_delta=0.001
        )
        
        assert early_stop_config['monitor_metric'] == 'val_auc'
        assert early_stop_config['patience'] == 10
        assert early_stop_config['min_delta'] == 0.001
    
    def test_training_history_tracking(self):
        """Test training progress tracking"""
        trainer = ModelTrainer(model_type='xgboost', verbose=False)
        
        # Track some fake training progress
        trainer.track_training_progress(
            epoch=1,
            train_metrics={'auc': 0.8, 'loss': 0.5},
            val_metrics={'auc': 0.75, 'loss': 0.6}
        )
        
        trainer.track_training_progress(
            epoch=2,
            train_metrics={'auc': 0.85, 'loss': 0.4},
            val_metrics={'auc': 0.78, 'loss': 0.55}
        )
        
        assert 'epoch_history' in trainer.training_history
        assert len(trainer.training_history['epoch_history']) == 2
        assert trainer.training_history['epoch_history'][0]['epoch'] == 1
        assert trainer.training_history['epoch_history'][1]['epoch'] == 2
    
    def test_model_persistence(self):
        """Test model and artifact saving"""
        trainer = ModelTrainer(model_type='xgboost', verbose=False)
        
        X = self.df.drop('defect', axis=1).values
        y = self.df['defect'].values
        
        # Train model
        trainer.train_model(X, y)
        
        # Test model saving
        model_path = os.path.join(self.temp_dir, 'test_model.joblib')
        metadata = {'test': 'metadata'}
        
        trainer.save_trained_model(trainer.model, model_path, metadata)
        
        assert os.path.exists(model_path)
        assert os.path.exists(model_path.replace('.joblib', '.json'))
    
    def test_full_pipeline_execution(self):
        """Test complete training pipeline"""
        # Create configuration
        config_manager = ConfigurationManager()
        config = config_manager.create_default_config(self.config_path)
        
        # Modify config for faster testing
        config.hyperparameter_search.enabled = False  # Skip for speed
        config.training.cross_validation.cv_folds = 3
        
        trainer = ModelTrainer(
            config_path=self.config_path,
            model_type='xgboost',
            verbose=False
        )
        
        # Run full pipeline
        results = trainer.train_pipeline(
            data_path=self.data_path,
            target_column='defect'
        )
        
        assert 'data_info' in results
        assert 'training' in results
        assert 'test_evaluation' in results
        assert results['data_info']['total_samples'] == len(self.df)
        assert results['test_evaluation']['roc_auc'] > 0.5
    
    def test_error_handling(self):
        """Test error handling for various edge cases"""
        # Test invalid model type
        with pytest.raises(ValueError):
            ModelTrainer(model_type='invalid_model')
        
        # Test invalid configuration
        invalid_config = {
            'data': {'test_size': 1.5}  # Invalid test size
        }
        config = TrainingPipelineConfig.from_dict(invalid_config)
        errors = config.validate()
        assert len(errors) > 0
    
    def test_preprocessing_components(self):
        """Test individual preprocessing components"""
        preprocessor = DataPreprocessor({})
        
        # Test outlier handling
        data_with_outliers = self.df.copy()
        data_with_outliers.iloc[0, 0] = 1000  # Add extreme outlier
        
        cleaned_data = preprocessor.handle_outliers(data_with_outliers, method='iqr')
        assert cleaned_data.iloc[0, 0] < 1000  # Outlier should be capped
        
        # Test feature encoding
        categorical_data = pd.DataFrame({
            'cat_feature': ['A', 'B', 'A', 'C', 'B'],
            'num_feature': [1, 2, 3, 4, 5]
        })
        
        encoded_data = preprocessor.encode_categorical_features(categorical_data)
        assert encoded_data.shape[1] > categorical_data.shape[1]  # Should add dummy columns
    
    def test_hyperparameter_searcher(self):
        """Test hyperparameter searcher functionality"""
        from sklearn.ensemble import RandomForestClassifier
        
        searcher = HyperparameterSearcher('random')
        
        X = self.df.drop('defect', axis=1).values
        y = self.df['defect'].values
        
        model = RandomForestClassifier(random_state=42)
        param_dist = {'n_estimators': [10, 50], 'max_depth': [3, 5]}
        
        results = searcher.random_search(model, param_dist, X, y, n_iter=2, cv=3)
        
        assert 'best_params' in results
        assert 'best_score' in results
        assert results['method'] == 'random_search'
    
    def test_training_utils(self):
        """Test training utility functions"""
        utils = TrainingUtils(random_state=42)
        
        # Test overfitting detection with clear overfitting pattern
        train_scores = [0.8, 0.85, 0.9, 0.95, 0.98]  # Continuously improving significantly
        val_scores = [0.75, 0.76, 0.76, 0.76, 0.76]  # Clearly plateaued validation
        
        overfitting = utils.detect_overfitting(train_scores, val_scores, patience=3, min_delta=0.01)
        assert overfitting  # Should detect overfitting
        
        # Test class weights calculation
        y_imbalanced = np.array([0] * 90 + [1] * 10)  # Imbalanced dataset
        class_weights = utils.calculate_class_weights(y_imbalanced)
        
        assert len(class_weights) == 2
        assert class_weights[1] > class_weights[0]  # Minority class should have higher weight
        
        # Test threshold optimization
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_proba = np.array([0.1, 0.3, 0.6, 0.8, 0.2, 0.9, 0.7, 0.4])
        
        best_threshold, best_score = utils.optimize_threshold(y_true, y_proba, metric='f1')
        
        assert 0 < best_threshold < 1
        assert best_score > 0


if __name__ == '__main__':
    pytest.main([__file__])