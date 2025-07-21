import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os

import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.baseline_model import BaselineXGBoostModel

class TestModels:
    """Test suite for model components"""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create sample data for testing
        np.random.seed(42)
        self.sample_features = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.normal(0, 1, 100),
            'feature_3': np.random.normal(0, 1, 100),
            'feature_4': np.random.normal(0, 1, 100),
            'feature_5': np.random.normal(0, 1, 100)
        })
        self.sample_labels = np.random.binomial(1, 0.15, 100)
        
        # Sample sequence data
        self.sample_sequences = np.random.normal(0, 1, (100, 50, 5))
        
        # Model configurations
        self.baseline_config = {
            'parameters': {
                'n_estimators': 10,
                'max_depth': 3,
                'learning_rate': 0.1,
                'random_state': 42
            }
        }
        
        self.lstm_config = {
            'architecture': {
                'input_size': 5,
                'hidden_size': 32,
                'num_layers': 1,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 16,
                'learning_rate': 0.001,
                'weight_decay': 1e-4
            },
            'loss_function': {
                'pos_weight': 3.0
            }
        }
    
    def test_baseline_model_initialization(self):
        """Test baseline model initialization."""
        # Test with default parameters
        model = BaselineXGBoostModel()
        assert model.model is not None
        assert model.random_state == 42
        assert not model.is_trained
        
        # Test with custom parameters
        custom_params = {'n_estimators': 50, 'max_depth': 4}
        model = BaselineXGBoostModel(model_params=custom_params)
        assert model.model_params['n_estimators'] == 50
        assert model.model_params['max_depth'] == 4
    
    def test_baseline_model_training(self):
        """Test baseline model training."""
        model = BaselineXGBoostModel(model_params={'n_estimators': 10, 'random_state': 42})
        
        # Test training
        history = model.fit(self.sample_features, self.sample_labels)
        
        assert model.is_trained
        assert 'train_auc' in history
        assert 'training_time' in history
        assert history['train_auc'] >= 0.0
        assert history['training_time'] > 0
    
    def test_baseline_model_prediction(self):
        """Test baseline model prediction."""
        model = BaselineXGBoostModel(model_params={'n_estimators': 10, 'random_state': 42})
        
        # Train first
        model.fit(self.sample_features, self.sample_labels)
        
        # Test predictions
        y_pred = model.predict(self.sample_features[:5])
        y_proba = model.predict_proba(self.sample_features[:5])
        
        assert len(y_pred) == 5
        assert len(y_proba) == 5
        assert all(pred in [0, 1] for pred in y_pred)
        assert all(0 <= prob <= 1 for prob in y_proba)
    
    def test_cross_validation(self):
        """Test cross-validation functionality."""
        model = BaselineXGBoostModel(model_params={'n_estimators': 10, 'random_state': 42})
        
        # Test cross-validation
        cv_results = model.cross_validate(
            self.sample_features, 
            self.sample_labels, 
            cv_folds=3,
            scoring=['roc_auc']
        )
        
        assert 'roc_auc_mean' in cv_results
        assert 'roc_auc_std' in cv_results
        assert cv_results['cv_folds'] == 3
        assert 0 <= cv_results['roc_auc_mean'] <= 1
    
    def test_feature_importance_extraction(self):
        """Test feature importance extraction."""
        model = BaselineXGBoostModel(model_params={'n_estimators': 10, 'random_state': 42})
        
        # Train first
        model.fit(self.sample_features, self.sample_labels)
        
        # Test feature importance
        importance_df = model.get_feature_importance()
        
        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) == len(self.sample_features.columns)
    
    def test_model_serialization(self):
        """Test model save/load functionality."""
        model = BaselineXGBoostModel(model_params={'n_estimators': 10, 'random_state': 42})
        
        # Train model
        model.fit(self.sample_features, self.sample_labels)
        
        # Test save/load
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, 'test_model.pkl')
            
            # Save model
            model.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Load model
            new_model = BaselineXGBoostModel()
            new_model.load_model(model_path)
            
            assert new_model.is_trained
            
            # Test predictions match
            pred_original = model.predict_proba(self.sample_features[:5])
            pred_loaded = new_model.predict_proba(self.sample_features[:5])
            
            np.testing.assert_array_almost_equal(pred_original, pred_loaded, decimal=6)
    
    def test_model_evaluation_metrics(self):
        """Test model evaluation metrics calculation."""
        model = BaselineXGBoostModel(model_params={'n_estimators': 10, 'random_state': 42})
        
        # Train model
        model.fit(self.sample_features, self.sample_labels)
        
        # Test evaluation
        eval_results = model.evaluate(
            self.sample_features, 
            self.sample_labels, 
            plot_curves=False
        )
        
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'average_precision']
        for metric in expected_metrics:
            assert metric in eval_results
            assert 0 <= eval_results[metric] <= 1
    
    def test_hyperparameter_optimization(self):
        """Test hyperparameter search functionality."""
        model = BaselineXGBoostModel(model_params={'random_state': 42})
        
        # Simple parameter grid for testing
        param_grid = {
            'n_estimators': [5, 10],
            'max_depth': [2, 3]
        }
        
        # Test hyperparameter search
        search_results = model.hyperparameter_search(
            self.sample_features,
            self.sample_labels,
            param_grid,
            cv_folds=2  # Small for speed
        )
        
        assert 'best_params' in search_results
        assert 'best_score' in search_results
        assert search_results['best_params']['n_estimators'] in [5, 10]
        assert search_results['best_params']['max_depth'] in [2, 3]
    
    def test_early_stopping(self):
        """Test early stopping mechanism."""
        model = BaselineXGBoostModel(model_params={'n_estimators': 100, 'random_state': 42})
        
        # Test early stopping
        history = model.fit_with_early_stopping(
            self.sample_features,
            self.sample_labels,
            validation_split=0.3,
            early_stopping_rounds=5
        )
        
        assert model.is_trained
        assert 'train_auc' in history
        assert 'val_auc' in history
        assert 'early_stopping_rounds' in history
    
    def test_prediction_probability_ranges(self):
        """Test that prediction probabilities are in valid range [0,1]."""
        model = BaselineXGBoostModel(model_params={'n_estimators': 10, 'random_state': 42})
        
        # Train model
        model.fit(self.sample_features, self.sample_labels)
        
        # Test probabilities
        probabilities = model.predict_proba(self.sample_features)
        
        assert all(0 <= prob <= 1 for prob in probabilities), "Probabilities should be between 0 and 1"
        assert len(probabilities) == len(self.sample_features)
    
    def test_model_reproducibility(self):
        """Test that models produce reproducible results with same seed."""
        # Train two models with same random seed
        model1 = BaselineXGBoostModel(model_params={'n_estimators': 10, 'random_state': 42})
        model2 = BaselineXGBoostModel(model_params={'n_estimators': 10, 'random_state': 42})
        
        model1.fit(self.sample_features, self.sample_labels)
        model2.fit(self.sample_features, self.sample_labels)
        
        # Test predictions are identical
        pred1 = model1.predict_proba(self.sample_features[:10])
        pred2 = model2.predict_proba(self.sample_features[:10])
        
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=6)
    
    def test_batch_prediction(self):
        """Test batch prediction functionality."""
        model = BaselineXGBoostModel(model_params={'n_estimators': 10, 'random_state': 42})
        
        # Train model
        model.fit(self.sample_features, self.sample_labels)
        
        # Create batches
        batch1 = self.sample_features[:30]
        batch2 = self.sample_features[30:60]
        batches = [batch1, batch2]
        
        # Test batch prediction
        batch_predictions = model.predict_batch(batches)
        
        assert len(batch_predictions) == 2
        assert len(batch_predictions[0]) == 30
        assert len(batch_predictions[1]) == 30
        
        # Verify predictions match individual predictions
        individual_pred = model.predict_proba(batch1)
        np.testing.assert_array_almost_equal(batch_predictions[0], individual_pred, decimal=6)
    
    def test_configuration_loading(self):
        """Test YAML configuration loading and validation."""
        # Test with configuration file
        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_model.yaml'
        
        if config_path.exists():
            model = BaselineXGBoostModel(config_path=str(config_path))
            assert model.config is not None
            assert 'baseline_model' in model.config
    
    # Placeholder tests for other components (to be implemented when PyTorch is available)
    def test_lstm_model_initialization(self):
        """Test LSTM model initialization."""
        # TODO: Implement when PyTorch is available
        pass
    
    def test_lstm_model_forward_pass(self):
        """Test LSTM forward pass."""
        # TODO: Implement when PyTorch is available
        pass
    
    def test_lstm_dataset_creation(self):
        """Test LSTM dataset creation."""
        # TODO: Implement when PyTorch is available
        pass
    
    def test_lstm_trainer_initialization(self):
        """Test LSTM trainer initialization."""
        # TODO: Implement when PyTorch is available
        pass
    
    def test_model_comparison(self):
        """Test model comparison functionality."""
        # TODO: Implement advanced model comparison
        pass