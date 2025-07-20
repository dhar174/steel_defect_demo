#!/usr/bin/env python3
"""Script to generate synthetic steel casting data"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.data_generator import SteelCastingDataGenerator


def main():
    """Main function to generate synthetic dataset"""
    print("Steel Casting Synthetic Data Generator")
    print("=" * 40)
    
    # Initialize generator with configuration
    config_path = Path(__file__).parent.parent / 'configs' / 'data_generation.yaml'
    
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    
    try:
        generator = SteelCastingDataGenerator(str(config_path))
        generator.generate_dataset()
        print("\nSynthetic data generation completed successfully!")
        
    except Exception as e:
        print(f"Error during data generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()