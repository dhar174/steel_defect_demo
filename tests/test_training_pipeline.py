"""Unit tests for the training script and supporting modules"""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from scripts.config_manager import ConfigManager, DataConfig, ModelConfig, TrainingConfig
from scripts.progress_tracker import ProgressTracker
from scripts.artifact_manager import ArtifactManager
from scripts.training_utils import TrainingUtils


class TestConfigManager:
    """Test configuration management functionality"""
    
    def test_config_manager_initialization(self):
        """Test ConfigManager initialization"""
        config_manager = ConfigManager()
        assert config_manager is not None
        assert config_manager.schema is not None
    
    def test_create_default_config(self):
        """Test default configuration creation"""
        config_manager = ConfigManager()
        default_config = config_manager.create_default_config()
        
        assert 'data' in default_config
        assert 'model' in default_config
        assert 'training' in default_config
        assert 'output' in default_config
        assert 'execution' in default_config
        
        # Validate data config
        assert default_config['data']['target_column'] == 'defect'
        assert default_config['data']['test_size'] == 0.2
        assert default_config['data']['random_state'] == 42
    
    def test_config_validation(self):
        """Test configuration validation"""
        config_manager = ConfigManager()
        
        # Valid configuration
        valid_config = {
            'data': {'target_column': 'defect', 'test_size': 0.2},
            'model': {'type': 'xgboost'},
            'training': {'cv_folds': 5}
        }
        
        assert config_manager.validate_config(valid_config) == True
        
        # Invalid configuration
        invalid_config = {
            'data': {'test_size': 1.5},  # Invalid test size
            'model': {'type': 'invalid_model'},  # Invalid model type
            'training': {'cv_folds': 1}  # Invalid CV folds
        }
        
        with pytest.raises(ValueError):
            config_manager.validate_config(invalid_config)
    
    def test_config_merging(self):
        """Test configuration merging"""
        config_manager = ConfigManager()
        
        base_config = {
            'data': {'target_column': 'defect', 'test_size': 0.2},
            'model': {'type': 'xgboost', 'parameters': {'n_estimators': 100}}
        }
        
        override_config = {
            'data': {'test_size': 0.3},
            'model': {'parameters': {'n_estimators': 200, 'max_depth': 6}}
        }
        
        merged = config_manager.merge_configs(base_config, override_config)
        
        assert merged['data']['target_column'] == 'defect'  # From base
        assert merged['data']['test_size'] == 0.3  # Overridden
        assert merged['model']['type'] == 'xgboost'  # From base
        assert merged['model']['parameters']['n_estimators'] == 200  # Overridden
        assert merged['model']['parameters']['max_depth'] == 6  # New
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration files"""
        config_manager = ConfigManager()
        
        test_config = {
            'data': {'target_column': 'defect', 'test_size': 0.2},
            'model': {'type': 'xgboost'}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test YAML format
            yaml_path = Path(temp_dir) / "test_config.yaml"
            config_manager.save_config(test_config, str(yaml_path), format='yaml')
            
            loaded_yaml = config_manager.load_config(str(yaml_path))
            assert loaded_yaml == test_config
            
            # Test JSON format
            json_path = Path(temp_dir) / "test_config.json"
            config_manager.save_config(test_config, str(json_path), format='json')
            
            loaded_json = config_manager.load_config(str(json_path))
            assert loaded_json == test_config


class TestProgressTracker:
    """Test progress tracking functionality"""
    
    def test_progress_tracker_initialization(self):
        """Test ProgressTracker initialization"""
        tracker = ProgressTracker(total_steps=5, verbose=False, use_progress_bar=False)
        assert tracker.total_steps == 5
        assert tracker.current_step_index == 0
        assert len(tracker.steps) == 0
    
    def test_step_management(self):
        """Test step start/complete cycle"""
        tracker = ProgressTracker(total_steps=3, verbose=False, use_progress_bar=False)
        
        # Start first step
        tracker.start_step("Test Step 1")
        assert tracker.current_step is not None
        assert tracker.current_step.name == "Test Step 1"
        assert tracker.current_step.status == "running"
        
        # Complete first step
        tracker.complete_step("Step 1 completed")
        assert tracker.current_step_index == 1
        assert tracker.steps[0].status == "completed"
        assert tracker.steps[0].message == "Step 1 completed"
        
        # Start and fail second step
        tracker.start_step("Test Step 2")
        tracker.fail_step("Step 2 failed")
        assert tracker.steps[1].status == "failed"
        assert tracker.steps[1].message == "Step 2 failed"
    
    def test_progress_updates(self):
        """Test progress updates within steps"""
        tracker = ProgressTracker(total_steps=1, verbose=False, use_progress_bar=False)
        
        tracker.start_step("Test Step")
        
        # Update progress
        tracker.update_progress(25.0, "Quarter done")
        assert tracker.current_step.progress == 25.0
        assert tracker.current_step.message == "Quarter done"
        
        tracker.update_progress(50.0, "Half done")
        assert tracker.current_step.progress == 50.0
        
        tracker.complete_step("Completed")
        assert tracker.current_step is None
    
    def test_context_manager(self):
        """Test progress tracker context manager"""
        tracker = ProgressTracker(total_steps=1, verbose=False, use_progress_bar=False)
        
        with tracker.context_manager("Context Step") as t:
            assert t.current_step.name == "Context Step"
            assert t.current_step.status == "running"
        
        assert tracker.steps[0].status == "completed"
        assert tracker.steps[0].message == "Step completed successfully"
    
    def test_summary_generation(self):
        """Test progress summary generation"""
        tracker = ProgressTracker(total_steps=2, verbose=False, use_progress_bar=False)
        
        tracker.start_step("Step 1")
        tracker.complete_step("Done")
        tracker.start_step("Step 2") 
        tracker.complete_step("Done")
        tracker.finish("All done")
        
        summary = tracker.get_summary()
        
        assert summary['total_steps'] == 2
        assert summary['completed_steps'] == 2
        assert summary['failed_steps'] == 0
        assert len(summary['steps']) == 2


class TestArtifactManager:
    """Test artifact management functionality"""
    
    def test_artifact_manager_initialization(self):
        """Test ArtifactManager initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ArtifactManager(temp_dir)
            assert manager.output_dir == Path(temp_dir)
            assert manager.experiment_name == "default"
            
            # Check directory structure
            assert (Path(temp_dir) / "models").exists()
            assert (Path(temp_dir) / "results").exists()
            assert (Path(temp_dir) / "configs").exists()
    
    def test_save_and_load_results(self):
        """Test saving and loading results"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ArtifactManager(temp_dir)
            
            test_results = {
                'metrics': {'accuracy': 0.95, 'auc': 0.98},
                'training_time': 120.5,
                'model_type': 'xgboost'
            }
            
            # Save results
            results_path = manager.save_results(test_results, "test_results")
            assert results_path.exists()
            
            # Load results
            loaded_results = manager.load_results(str(results_path))
            assert loaded_results == test_results
    
    def test_save_config(self):
        """Test configuration saving"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ArtifactManager(temp_dir)
            
            test_config = {
                'model': {'type': 'xgboost', 'n_estimators': 100},
                'data': {'test_size': 0.2}
            }
            
            # Save as YAML
            config_path = manager.save_config(test_config, "test_config", format='yaml')
            assert config_path.exists()
            assert config_path.suffix == '.yaml'
            
            # Save as JSON
            json_path = manager.save_config(test_config, "test_config_json", format='json')
            assert json_path.exists()
            assert json_path.suffix == '.json'
    
    def test_artifact_indexing(self):
        """Test artifact indexing and metadata"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ArtifactManager(temp_dir, experiment_name="test_experiment")
            
            # Save multiple artifacts
            test_results = {'metric': 0.95}
            test_config = {'param': 'value'}
            
            manager.save_results(test_results, "results")
            manager.save_config(test_config, "config")
            
            # Check artifact index
            artifacts = manager.list_artifacts()
            assert len(artifacts) == 2
            
            # Check specific artifact types
            result_artifacts = manager.list_artifacts('results')
            config_artifacts = manager.list_artifacts('config')
            
            assert len(result_artifacts) == 1
            assert len(config_artifacts) == 1
            assert result_artifacts[0].experiment_name == "test_experiment"
    
    def test_storage_summary(self):
        """Test storage usage summary"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ArtifactManager(temp_dir)
            
            # Save some artifacts
            manager.save_results({'data': 'test'}, "results1")
            manager.save_results({'data': 'test2'}, "results2")
            manager.save_config({'config': 'test'}, "config1")
            
            summary = manager.get_storage_summary()
            
            assert summary['total_artifacts'] == 3
            assert summary['total_size_bytes'] > 0
            assert 'by_type' in summary
            assert 'results' in summary['by_type']
            assert 'config' in summary['by_type']


class TestTrainingUtils:
    """Test training utility functions"""
    
    def test_training_utils_initialization(self):
        """Test TrainingUtils initialization"""
        utils = TrainingUtils(random_state=42)
        assert utils.random_state == 42
    
    def test_overfitting_detection(self):
        """Test overfitting detection"""
        utils = TrainingUtils()
        
        # No overfitting case - validation scores keep improving
        train_scores = [0.8, 0.85, 0.87, 0.88, 0.89]
        val_scores = [0.75, 0.80, 0.82, 0.83, 0.84]
        
        assert not utils.detect_overfitting(train_scores, val_scores, patience=3)
        
        # Overfitting case - training improves but validation deteriorates significantly
        train_scores = [0.8, 0.85, 0.90, 0.95, 0.99, 1.0, 1.0, 1.0]
        val_scores = [0.75, 0.78, 0.76, 0.73, 0.70, 0.68, 0.65, 0.60]
        
        assert utils.detect_overfitting(train_scores, val_scores, patience=3, threshold=0.1)
    
    def test_regularization_application(self):
        """Test regularization parameter application"""
        utils = TrainingUtils()
        
        base_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6
        }
        
        regularized = utils.apply_regularization(base_params, regularization_strength=0.2)
        
        assert 'reg_alpha' in regularized
        assert 'reg_lambda' in regularized
        assert regularized['reg_alpha'] == 0.2
        assert regularized['reg_lambda'] == 0.2
        assert regularized['learning_rate'] < base_params['learning_rate']  # Should be reduced
    
    def test_data_quality_validation(self):
        """Test data quality validation"""
        utils = TrainingUtils()
        
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(5, 2, n_samples),
            'feature3': np.ones(n_samples),  # Constant feature
            'feature4': np.random.choice(['A', 'B', 'C'], n_samples),  # Categorical
        })
        
        # Add some missing values
        X.loc[0:5, 'feature1'] = np.nan
        
        y = pd.Series(np.random.choice([0, 1], n_samples))
        
        quality_report = utils.validate_data_quality(X, y)
        
        assert 'n_samples' in quality_report
        assert 'n_features' in quality_report
        assert 'missing_features' in quality_report
        assert 'constant_features' in quality_report
        
        assert quality_report['n_samples'] == n_samples
        assert quality_report['n_features'] == 4
        assert 'feature1' in quality_report['missing_features']
        assert 'feature3' in quality_report['constant_features']
    
    def test_memory_estimation(self):
        """Test memory usage estimation"""
        utils = TrainingUtils()
        
        data_shape = (10000, 100)  # 10k samples, 100 features
        
        memory_estimate = utils.estimate_memory_usage(data_shape, model_type='xgboost')
        
        assert 'data_memory_mb' in memory_estimate
        assert 'total_estimated_mb' in memory_estimate
        assert 'total_estimated_gb' in memory_estimate
        
        assert memory_estimate['data_memory_mb'] > 0
        assert memory_estimate['total_estimated_mb'] > memory_estimate['data_memory_mb']


# Integration test for the complete pipeline
class TestTrainingPipeline:
    """Integration tests for the complete training pipeline"""
    
    def test_pipeline_components_integration(self):
        """Test that all components work together"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize components
            config_manager = ConfigManager()
            progress_tracker = ProgressTracker(total_steps=3, verbose=False, use_progress_bar=False)
            artifact_manager = ArtifactManager(temp_dir, experiment_name="integration_test")
            
            # Create test configuration
            config = config_manager.create_default_config()
            config_path = artifact_manager.save_config(config, "test_config")
            
            # Load and validate config
            loaded_config = config_manager.load_config(str(config_path))
            assert config_manager.validate_config(loaded_config)
            
            # Test progress tracking
            progress_tracker.start_step("Integration Test")
            progress_tracker.update_progress(50.0, "Halfway")
            progress_tracker.complete_step("Integration completed")
            
            # Save results
            test_results = {'integration_test': True, 'status': 'success'}
            results_path = artifact_manager.save_results(test_results, "integration_results")
            
            # Verify everything was saved
            assert config_path.exists()
            assert results_path.exists()
            
            artifacts = artifact_manager.list_artifacts()
            assert len(artifacts) == 2
            
            summary = progress_tracker.get_summary()
            assert summary['completed_steps'] == 1
            assert summary['failed_steps'] == 0


if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v"])