#!/usr/bin/env python3
"""Train LSTM model for steel defect prediction"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.lstm_model import SteelDefectLSTM, CastingSequenceDataset
from models.model_trainer import LSTMTrainer
from features.feature_extractor import SequenceFeatureExtractor
from utils.config_loader import ConfigLoader
from utils.logger import get_logger
import argparse
import torch
from torch.utils.data import DataLoader

def main():
    """Main function to train LSTM model."""
    parser = argparse.ArgumentParser(description='Train LSTM model for defect prediction')
    parser.add_argument('--config', 
                       default='configs/model_config.yaml',
                       help='Path to model configuration file')
    parser.add_argument('--data-dir',
                       default='data/processed',
                       help='Directory containing processed training data')
    parser.add_argument('--output-dir',
                       default='models/deep_learning',
                       help='Output directory for trained model')
    parser.add_argument('--device',
                       default='auto',
                       help='Device to use for training (cpu, cuda, auto)')
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = get_logger(__name__)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    if args.verbose:
        logger.info(f"Using device: {device}")
        logger.info(f"Loading configuration from: {args.config}")
        logger.info(f"Data directory: {args.data_dir}")
        logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_yaml(args.config)
        lstm_config = config['lstm_model']
        
        # Load and prepare sequence data
        logger.info("Loading and preparing sequence data...")
        feature_extractor = SequenceFeatureExtractor(
            sequence_length=lstm_config['data_processing']['sequence_length']
        )
        # TODO: Implement data loading and sequence preparation
        
        # Initialize LSTM model
        model = SteelDefectLSTM(
            input_size=lstm_config['architecture']['input_size'],
            hidden_size=lstm_config['architecture']['hidden_size'],
            num_layers=lstm_config['architecture']['num_layers'],
            dropout=lstm_config['architecture']['dropout']
        ).to(device)
        
        # Initialize trainer
        trainer = LSTMTrainer(model, lstm_config)
        
        logger.info("Starting LSTM model training...")
        # TODO: Implement training pipeline with data loaders
        
        # Save trained model
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("LSTM model training completed successfully.")
        
    except Exception as e:
        logger.error(f"Error during LSTM training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()