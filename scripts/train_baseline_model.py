#!/usr/bin/env python3
"""
Comprehensive Command-Line Training Script for Steel Casting Defect Prediction

This script orchestrates the entire machine learning pipeline from data loading 
to model deployment with full configuration management, progress tracking, and 
artifact management.
"""

import argparse
import logging
import sys
import os
import json
import yaml
import pickle
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Local imports
from src.data.data_loader import DataLoader
from src.features.feature_engineer import CastingFeatureEngineer
from src.models.baseline_model import BaselineXGBoostModel  
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from scripts.config_manager import ConfigManager, CompleteTrainingConfig
from scripts.progress_tracker import ProgressTracker
from scripts.artifact_manager import ArtifactManager
from scripts.training_utils import TrainingUtils

import warnings
warnings.filterwarnings('ignore')


@dataclass
class TrainingConfig:
    """Training configuration dataclass"""
    # Data Configuration
    data_path: str
    target_column: str = "defect"
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    
    # Model Configuration
    model_type: str = "xgboost"
    model_params: Dict[str, Any] = None
    
    # Training Configuration
    hyperparameter_search: bool = True
    cross_validation: bool = True
    cv_folds: int = 5
    
    # Output Configuration
    output_dir: str = "results"
    experiment_name: str = "baseline_training"
    save_artifacts: bool = True
    
    # Execution Configuration
    verbose: bool = True
    n_jobs: int = -1


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Set up comprehensive command-line argument parser
    
    Returns:
        Configured ArgumentParser object
    """
    parser = argparse.ArgumentParser(
        description="Train Steel Casting Defect Prediction Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default configuration
  python scripts/train_baseline_model.py --data data/processed/cleaned_data.csv

  # Training with custom configuration
  python scripts/train_baseline_model.py --config config/training_config.yaml

  # Hyperparameter search with custom parameters
  python scripts/train_baseline_model.py --data data/processed/cleaned_data.csv --hyperparameter-search --cv-folds 10

  # Production training with specific output directory
  python scripts/train_baseline_model.py --config config/production.yaml --output-dir models/production --experiment-name production_v1

  # Quick development training without hyperparameter search
  python scripts/train_baseline_model.py --data data/processed/sample_data.csv --no-hyperparameter-search --quick
        """
    )
    
    # Required Arguments
    required_group = parser.add_argument_group('required arguments')
    required_group.add_argument(
        '--data', 
        type=str,
        help='Path to training data CSV file'
    )
    required_group.add_argument(
        '--config', 
        type=str,
        help='Path to training configuration file (YAML/JSON)'
    )
    
    # Data Arguments
    data_args = parser.add_argument_group('data arguments')
    data_args.add_argument(
        '--target-column',
        type=str,
        default='defect',
        help='Name of target column (default: defect)'
    )
    data_args.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size as fraction (default: 0.2)'
    )
    data_args.add_argument(
        '--validation-size',
        type=float,
        default=0.2,
        help='Validation set size as fraction (default: 0.2)'
    )
    
    # Model Arguments
    model_args = parser.add_argument_group('model arguments')
    model_args.add_argument(
        '--model-type',
        type=str,
        default='xgboost',
        choices=['xgboost', 'random_forest', 'logistic_regression'],
        help='Type of model to train (default: xgboost)'
    )
    model_args.add_argument(
        '--model-params',
        type=str,
        help='JSON string of model parameters'
    )
    
    # Training Arguments
    training_args = parser.add_argument_group('training arguments')
    training_args.add_argument(
        '--hyperparameter-search',
        action='store_true',
        default=True,
        help='Enable hyperparameter search (default: True)'
    )
    training_args.add_argument(
        '--no-hyperparameter-search',
        dest='hyperparameter_search',
        action='store_false',
        help='Disable hyperparameter search'
    )
    training_args.add_argument(
        '--cross-validation',
        action='store_true',
        default=True,
        help='Enable cross-validation (default: True)'
    )
    training_args.add_argument(
        '--no-cross-validation',
        dest='cross_validation',
        action='store_false',
        help='Disable cross-validation'
    )
    training_args.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    training_args.add_argument(
        '--search-method',
        type=str,
        default='grid',
        choices=['grid', 'random', 'bayesian'],
        help='Hyperparameter search method (default: grid)'
    )
    
    # Output Arguments
    output_args = parser.add_argument_group('output arguments')
    output_args.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    output_args.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Experiment name for tracking (default: auto-generated)'
    )
    output_args.add_argument(
        '--no-save-artifacts',
        dest='save_artifacts',
        action='store_false',
        default=True,
        help='Disable saving of model artifacts'
    )
    
    # Execution Arguments
    execution_args = parser.add_argument_group('execution arguments')
    execution_args.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    execution_args.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs (default: -1, use all cores)'
    )
    execution_args.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Enable verbose output (default: True)'
    )
    execution_args.add_argument(
        '--quiet',
        dest='verbose',
        action='store_false',
        help='Disable verbose output'
    )
    execution_args.add_argument(
        '--quick',
        action='store_true',
        help='Quick training mode (reduced hyperparameter search, fewer CV folds)'
    )
    execution_args.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with detailed logging'
    )
    
    # Advanced Arguments
    advanced_args = parser.add_argument_group('advanced arguments')
    advanced_args.add_argument(
        '--resume-from',
        type=str,
        help='Resume training from checkpoint file'
    )
    advanced_args.add_argument(
        '--profile',
        action='store_true',
        help='Enable performance profiling'
    )
    advanced_args.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform dry run without actual training'
    )
    
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command-line arguments
    
    Args:
        args: Parsed command-line arguments
        
    Raises:
        ValueError: If arguments are invalid
    """
    # Require either data or config
    if not args.data and not args.config:
        raise ValueError("Either --data or --config must be specified")
    
    # Validate data path
    if args.data and not Path(args.data).exists():
        raise ValueError(f"Data file not found: {args.data}")
    
    # Validate config path
    if args.config and not Path(args.config).exists():
        raise ValueError(f"Config file not found: {args.config}")
    
    # Validate size parameters
    if not 0 < args.test_size < 1:
        raise ValueError(f"Test size must be between 0 and 1, got: {args.test_size}")
    
    if not 0 < args.validation_size < 1:
        raise ValueError(f"Validation size must be between 0 and 1, got: {args.validation_size}")
    
    if args.test_size + args.validation_size >= 1:
        raise ValueError("Sum of test_size and validation_size must be < 1")
    
    # Validate CV folds
    if args.cv_folds < 2:
        raise ValueError(f"CV folds must be >= 2, got: {args.cv_folds}")
    
    # Validate model parameters
    if args.model_params:
        try:
            json.loads(args.model_params)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in model-params: {e}")


def setup_logging(verbose: bool = True, debug: bool = False, log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration
    
    Args:
        verbose: Enable verbose logging
        debug: Enable debug logging
        log_file: Optional log file path
    """
    # Determine log level
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    
    # Set up formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter if not debug else detailed_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)


def load_and_validate_data(data_path: str, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and validate training data
    
    Args:
        data_path: Path to data file
        target_column: Name of target column
        
    Returns:
        Features and target data
        
    Raises:
        ValueError: If data validation fails
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading data from: {data_path}")
    
    # Load data
    try:
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            data = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
    except Exception as e:
        raise ValueError(f"Failed to load data: {e}")
    
    # Validate target column
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Basic data validation
    if data.empty:
        raise ValueError("Data is empty")
    
    if data[target_column].isna().all():
        raise ValueError("Target column contains only missing values")
    
    logger.info(f"Data loaded successfully: {data.shape}")
    logger.info(f"Target distribution:\n{data[target_column].value_counts()}")
    
    # Split features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    return X, y


def create_experiment_name(base_name: Optional[str] = None) -> str:
    """
    Create unique experiment name
    
    Args:
        base_name: Base name for experiment
        
    Returns:
        Unique experiment name
    """
    if base_name is None:
        base_name = "baseline_training"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}"


def run_training_pipeline(config: TrainingConfig, progress_tracker: ProgressTracker) -> Dict[str, Any]:
    """
    Execute the complete training pipeline
    
    Args:
        config: Training configuration
        progress_tracker: Progress tracking object
        
    Returns:
        Training results dictionary
    """
    logger = logging.getLogger(__name__)
    results = {}
    
    try:
        # Step 1: Load and validate data
        progress_tracker.start_step("Loading Data")
        X, y = load_and_validate_data(config.data_path, config.target_column)
        progress_tracker.complete_step("Data loaded successfully")
        
        # Step 2: Feature engineering
        progress_tracker.start_step("Feature Engineering")
        feature_engineer = CastingFeatureEngineer()
        
        # For now, we'll work directly with the data assuming it's already processed
        # In a real scenario, you might need to engineer features from raw sensor data
        X_engineered = X
        progress_tracker.complete_step(f"Features prepared: {X_engineered.shape[1]} features")
        
        # Step 3: Initialize model
        progress_tracker.start_step("Model Initialization")
        model_params = config.model_params or {}
        model = BaselineXGBoostModel(
            model_params=model_params,
            random_state=config.random_state,
            verbose=config.verbose
        )
        progress_tracker.complete_step("Model initialized")
        
        # Step 4: Initialize trainer
        progress_tracker.start_step("Training Setup")
        trainer = ModelTrainer(
            model_type=config.model_type,
            experiment_name=config.experiment_name,
            random_state=config.random_state,
            verbose=config.verbose
        )
        progress_tracker.complete_step("Trainer initialized")
        
        # Step 5: Model training
        progress_tracker.start_step("Model Training")
        
        if config.hyperparameter_search:
            # Get parameter grid
            param_grid = model.get_hyperparameter_grid('coarse')
            
            # Hyperparameter search
            search_results = model.hyperparameter_search(
                X_engineered, y,
                param_grid=param_grid,
                cv_folds=config.cv_folds,
                n_jobs=config.n_jobs
            )
            best_model = model  # Model is updated with best parameters
            results['hyperparameter_search'] = search_results
        else:
            # Direct training
            training_results = model.fit(X_engineered, y)
            best_model = model
            results['training'] = training_results
        
        progress_tracker.complete_step("Model training completed")
        
        # Step 6: Model evaluation
        progress_tracker.start_step("Model Evaluation")
        evaluator = ModelEvaluator(
            model=best_model,
            model_name=config.experiment_name
        )
        
        # Split data for evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=y
        )
        
        # Evaluate model
        evaluation_results = evaluator.evaluate_model(X_test, y_test)
        results['evaluation'] = evaluation_results
        
        progress_tracker.complete_step("Model evaluation completed")
        
        # Step 7: Cross-validation (if enabled)
        if config.cross_validation:
            progress_tracker.start_step("Cross-Validation")
            cv_results = model.cross_validate(X_engineered, y, cv_folds=config.cv_folds)
            results['cross_validation'] = cv_results
            progress_tracker.complete_step("Cross-validation completed")
        
        # Step 8: Save artifacts (if enabled)
        if config.save_artifacts:
            progress_tracker.start_step("Saving Artifacts")
            artifact_manager = ArtifactManager(config.output_dir, config.experiment_name)
            
            # Save model
            model_path = artifact_manager.save_model(
                best_model, 
                f"{config.experiment_name}_model",
                metrics=evaluation_results['metrics'] if 'evaluation' in results else None
            )
            
            # Save feature engineer
            fe_path = artifact_manager.save_feature_engineer(
                feature_engineer,
                f"{config.experiment_name}_feature_engineer"
            )
            
            # Save results
            results_path = artifact_manager.save_results(
                results,
                f"{config.experiment_name}_results"
            )
            
            # Save configuration
            config_path = artifact_manager.save_config(
                asdict(config),
                f"{config.experiment_name}_config"
            )
            
            results['artifacts'] = {
                'model_path': str(model_path),
                'feature_engineer_path': str(fe_path),
                'results_path': str(results_path),
                'config_path': str(config_path)
            }
            
            progress_tracker.complete_step("Artifacts saved successfully")
        
        return results
        
    except Exception as e:
        progress_tracker.fail_step(f"Training failed: {str(e)}")
        raise


def main():
    """Main training script entry point"""
    
    # Set up argument parser
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    try:
        # Validate arguments
        validate_arguments(args)
        
        # Create experiment name
        experiment_name = args.experiment_name or create_experiment_name()
        
        # Set up output directory
        output_dir = Path(args.output_dir) / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        log_file = output_dir / "training.log" if not args.dry_run else None
        setup_logging(args.verbose, args.debug, log_file)
        
        logger = logging.getLogger(__name__)
        logger.info("="*60)
        logger.info("Steel Casting Defect Prediction Model Training")
        logger.info("="*60)
        logger.info(f"Experiment: {experiment_name}")
        logger.info(f"Output directory: {output_dir}")
        
        # Handle quick mode
        if args.quick:
            logger.info("Quick mode enabled - reducing search space and CV folds")
            args.cv_folds = min(args.cv_folds, 3)
        
        # Create training configuration
        if args.config:
            # Load from config file
            config_manager = ConfigManager()
            config_dict = config_manager.load_config(args.config)
            # Override with command line arguments
            cli_config = config_manager.get_config_from_args(args)
            config_dict = config_manager.merge_configs(config_dict, cli_config)
            
            # Apply quick mode adjustments
            if args.quick:
                config_dict = config_manager.get_quick_config(config_dict)
            
            # Extract data path - prioritize command line argument over config
            data_path = args.data or config_dict.get('data', {}).get('data_path')
            if not data_path:
                raise ValueError("Data path must be specified via --data argument or in config file")
            
            config = TrainingConfig(
                data_path=data_path,
                target_column=config_dict.get('data', {}).get('target_column', 'defect'),
                test_size=config_dict.get('data', {}).get('test_size', 0.2),
                validation_size=config_dict.get('data', {}).get('validation_size', 0.2),
                random_state=config_dict.get('data', {}).get('random_state', 42),
                model_type=config_dict.get('model', {}).get('type', 'xgboost'),
                model_params=config_dict.get('model', {}).get('parameters', {}),
                hyperparameter_search=config_dict.get('training', {}).get('hyperparameter_search', True),
                cross_validation=config_dict.get('training', {}).get('cross_validation', True),
                cv_folds=config_dict.get('training', {}).get('cv_folds', 5),
                output_dir=str(output_dir),
                experiment_name=experiment_name,
                save_artifacts=config_dict.get('output', {}).get('save_artifacts', True),
                verbose=config_dict.get('execution', {}).get('verbose', True),
                n_jobs=config_dict.get('execution', {}).get('n_jobs', -1)
            )
        else:
            # Create from command line arguments
            if not args.data:
                raise ValueError("Data path must be specified via --data argument")
                
            model_params = None
            if args.model_params:
                model_params = json.loads(args.model_params)
            
            config = TrainingConfig(
                data_path=args.data,
                target_column=args.target_column,
                test_size=args.test_size,
                validation_size=args.validation_size,
                random_state=args.random_state,
                model_type=args.model_type,
                model_params=model_params,
                hyperparameter_search=args.hyperparameter_search,
                cross_validation=args.cross_validation,
                cv_folds=args.cv_folds,
                output_dir=str(output_dir),
                experiment_name=experiment_name,
                save_artifacts=args.save_artifacts,
                verbose=args.verbose,
                n_jobs=args.n_jobs
            )
        
        # Log configuration
        logger.info("Training Configuration:")
        for key, value in asdict(config).items():
            logger.info(f"  {key}: {value}")
        
        # Dry run check
        if args.dry_run:
            logger.info("Dry run mode - configuration validated successfully")
            logger.info("Training would proceed with the above configuration")
            return
        
        # Initialize progress tracker
        total_steps = 7 if config.save_artifacts else 6
        if config.cross_validation:
            total_steps += 1
            
        progress_tracker = ProgressTracker(
            total_steps=total_steps,
            verbose=args.verbose,
            experiment_name=experiment_name
        )
        
        # Performance profiling
        if args.profile:
            import cProfile
            import pstats
            
            profiler = cProfile.Profile()
            profiler.enable()
        
        # Run training pipeline
        logger.info("Starting training pipeline...")
        start_time = datetime.now()
        
        results = run_training_pipeline(config, progress_tracker)
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        # Stop profiling
        if args.profile:
            profiler.disable()
            stats_file = output_dir / "profiling_stats.txt"
            with open(stats_file, 'w') as f:
                stats = pstats.Stats(profiler, stream=f)
                stats.sort_stats('cumulative')
                stats.print_stats()
            logger.info(f"Profiling results saved to: {stats_file}")
        
        # Finish progress tracking
        progress_tracker.finish("Training completed successfully!")
        
        # Log results summary
        logger.info("="*60)
        logger.info("Training Completed Successfully!")
        logger.info("="*60)
        logger.info(f"Total training time: {training_duration}")
        
        if 'evaluation' in results:
            metrics = results['evaluation']['metrics']
            logger.info("Model Performance:")
            logger.info(f"  AUC-ROC: {metrics.get('roc_auc', 'N/A'):.4f}")
            logger.info(f"  AUC-PR: {metrics.get('average_precision', 'N/A'):.4f}")
            logger.info(f"  F1 Score: {metrics.get('f1_score', 'N/A'):.4f}")
            logger.info(f"  Precision: {metrics.get('precision', 'N/A'):.4f}")
            logger.info(f"  Recall: {metrics.get('recall', 'N/A'):.4f}")
        
        if 'artifacts' in results:
            logger.info("Saved Artifacts:")
            for artifact_type, path in results['artifacts'].items():
                logger.info(f"  {artifact_type}: {path}")
        
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.error("Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        if args.debug:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()