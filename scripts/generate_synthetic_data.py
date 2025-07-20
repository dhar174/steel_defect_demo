#!/usr/bin/env python3
"""Script to generate synthetic steel casting data"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.data_generator import SteelCastingDataGenerator
import argparse

def main():
    """Main function to generate synthetic data."""
    parser = argparse.ArgumentParser(description='Generate synthetic steel casting data')
    parser.add_argument('--config', 
                       default='configs/data_generation.yaml',
                       help='Path to data generation configuration file')
    parser.add_argument('--output-dir',
                       default='data/synthetic',
                       help='Output directory for generated data')
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Loading configuration from: {args.config}")
        print(f"Output directory: {args.output_dir}")
    
    try:
        # Initialize data generator
        generator = SteelCastingDataGenerator(args.config)
        
        # Generate synthetic dataset
        if args.verbose:
            print("Starting synthetic data generation...")
        
        generator.generate_dataset()
        
        print("Synthetic data generation completed successfully.")
        
    except Exception as e:
        print(f"Error during data generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()