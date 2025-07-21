#!/usr/bin/env python3
"""Script to generate synthetic steel casting data"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.data_generator import SteelCastingDataGenerator


from utils.config_loader import ConfigLoader
from configs.schemas.data_generation_schema import DataGenerationConfig

def main():
    """Main function to generate synthetic dataset"""
    print("Steel Casting Synthetic Data Generator")
    print("=" * 40)
    
    config_loader = ConfigLoader()
    
    try:
        config = config_loader.load_config("data_generation", DataGenerationConfig)
        generator = SteelCastingDataGenerator(config)
        generator.generate_dataset()
        print("\nSynthetic data generation completed successfully!")
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()