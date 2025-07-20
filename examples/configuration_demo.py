#!/usr/bin/env python3
"""
Configuration System Usage Example

This script demonstrates how to use the comprehensive configuration management system
for the steel defect prediction project.
"""

import sys
import os
from pathlib import Path

# Import configuration loader from src package
from src.utils.config_loader import ConfigLoader, ConfigValidationError
def main():
    """Demonstrate configuration system usage"""
    
    print("=" * 60)
    print("STEEL DEFECT PREDICTION - CONFIGURATION SYSTEM DEMO")
    print("=" * 60)
    
    # Initialize configuration loader
    print("\n1. Initializing Configuration Loader...")
    loader = ConfigLoader()
    print(f"   Config directory: {loader.config_dir}")
    print(f"   Schema directory: {loader.schema_dir}")
    
    # List available configurations and schemas
    print("\n2. Available Configurations and Schemas:")
    configs = loader.list_configs()
    schemas = loader.list_schemas()
    print(f"   Configurations: {configs}")
    print(f"   Schemas: {schemas}")
    
    # Load and validate each configuration
    print("\n3. Loading and Validating Configurations...")
    
    for config_name in configs:
        try:
            print(f"\n   Loading {config_name}...")
            config = loader.load_yaml(f"{config_name}.yaml")
            
            # Validate against schema
            schema_name = f"{config_name}_schema"
            if schema_name in schemas:
                loader.validate_config(config, schema_name)
                print(f"   ✓ {config_name}: Loaded and validated successfully")
            else:
                print(f"   ⚠ {config_name}: Loaded but no schema found")
                
        except Exception as e:
            print(f"   ✗ {config_name}: Failed - {e}")
    
    # Demonstrate environment variable overrides
    print("\n4. Environment Variable Override Demo...")
    
    # Show original value
    data_config = loader.get_config('data_generation')
    original_casts = data_config['data_generation']['num_casts']
    print(f"   Original num_casts: {original_casts}")
    
    # Set environment variable and reload
    os.environ['DATA_GENERATION_DATA_GENERATION_NUM_CASTS'] = '500'
    os.environ['DATA_GENERATION_DATA_GENERATION_RANDOM_SEED'] = '123'
    
    # Reload configuration with overrides
    updated_config = loader.load_yaml('data_generation.yaml')
    new_casts = updated_config['data_generation']['num_casts']
    new_seed = updated_config['data_generation']['random_seed']
    
    print(f"   After env override num_casts: {new_casts}")
    print(f"   After env override random_seed: {new_seed}")
    
    # Clean up environment variables
    del os.environ['DATA_GENERATION_DATA_GENERATION_NUM_CASTS']
    del os.environ['DATA_GENERATION_DATA_GENERATION_RANDOM_SEED']
    
    # Demonstrate configuration merging
    print("\n5. Configuration Merging Demo...")
    
    merged_config = loader.merge_configs('data_generation', 'model_config', 'inference_config')
    print(f"   Merged configuration sections: {list(merged_config.keys())}")
    
    # Show some merged values
    print(f"   Data generation casts: {merged_config['data_generation']['num_casts']}")
    print(f"   Model algorithm: {merged_config['baseline_model']['algorithm']}")
    print(f"   Dashboard port: {merged_config['inference']['output']['dashboard_port']}")
    
    # Demonstrate configuration updates
    print("\n6. Configuration Update Demo...")
    
    updates = {
        'data_generation': {
            'num_casts': 2000,
            'cast_duration_minutes': 150
        }
    }
    
    print(f"   Original duration: {data_config['data_generation']['cast_duration_minutes']} minutes")
    
    # Create a backup and update
    backup_file = loader.config_dir / 'data_generation_backup.yaml'
    loader.save_config(data_config, 'data_generation_backup.yaml')
    
    # Apply updates (this saves the file)
    loader.update_config('data_generation', updates)
    
    # Verify updates
    updated = loader.get_config('data_generation')
    print(f"   Updated casts: {updated['data_generation']['num_casts']}")
    print(f"   Updated duration: {updated['data_generation']['cast_duration_minutes']} minutes")
    
    # Restore original configuration
    original_config = loader.load_yaml('data_generation_backup.yaml')
    loader.save_config(original_config, 'data_generation.yaml')
    
    # Clean up backup
    backup_file.unlink()
    print("   Configuration restored to original values")
    
    # Demonstrate integration with components
    print("\n7. Integration with Components Demo...")
    
    try:
        from data.data_generator import SteelCastingDataGenerator
        
        # Create data generator using configuration
        config_path = loader.config_dir / 'data_generation.yaml'
        generator = SteelCastingDataGenerator(str(config_path))
        print("   ✓ Data generator initialized with configuration")
        
        # Show some configuration values being used
        print(f"   Configured sensors: {len(generator.sensor_config) - 1}")  # -1 for mold_level_normal_range
        print(f"   Defect probability: {generator.defect_config['defect_probability']}")
        print(f"   Random seed: {generator.data_config['random_seed']}")
        
    except ImportError:
        print("   ⚠ Data generator not available for integration demo")
    
    # Show best practices
    print("\n8. Best Practices Examples...")
    
    print("   Environment-specific configuration:")
    print("   export ENVIRONMENT=production")
    print("   export DATA_GENERATION_DATA_GENERATION_NUM_CASTS=10000")
    print("   export INFERENCE_INFERENCE_OUTPUT_DASHBOARD_ENABLED=true")
    
    print("\n   Configuration validation in production:")
    print("   try:")
    print("       loader.validate_config(config, 'schema_name')")
    print("   except ConfigValidationError as e:")
    print("       logger.error(f'Invalid configuration: {e}')")
    print("       sys.exit(1)")
    
    print("\n   Secure configuration management:")
    print("   # Don't put secrets in config files")
    print("   # Use environment variables for sensitive data")
    print("   database_password = os.getenv('DATABASE_PASSWORD')")
    
    # Performance tips
    print("\n9. Performance and Debugging...")
    
    print(f"   Cached configurations: {len(loader.configs)}")
    print(f"   Loaded schemas: {len(loader.schemas)}")
    print("   Tip: Configurations are cached after first load")
    print("   Tip: Use environment variables for deployment-specific settings")
    print("   Tip: Validate configurations in CI/CD pipelines")
    
    print("\n" + "=" * 60)
    print("CONFIGURATION SYSTEM DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nFor more information, see docs/configuration_guide.md")


if __name__ == '__main__':
    main()