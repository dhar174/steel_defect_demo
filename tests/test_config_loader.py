import unittest
import tempfile
import shutil
import os
import yaml
import json
from pathlib import Path
from unittest.mock import patch

# Import ConfigLoader and ConfigValidationError from utils
from src.utils.config_loader import ConfigLoader, ConfigValidationError
class TestConfigLoader(unittest.TestCase):
    """Test cases for ConfigLoader"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.test_dir) / "configs"
        self.schema_dir = self.config_dir / "schemas"
        
        # Create directories
        self.config_dir.mkdir(parents=True)
        self.schema_dir.mkdir(parents=True)
        
        # Create test configuration
        self.test_config = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'username': 'testuser',
                'debug': True,
                'timeout': 30.5,
                'features': ['auth', 'logging']
            },
            'model': {
                'learning_rate': 0.001,
                'batch_size': 32
            }
        }
        
        # Create test schema
        self.test_schema = {
            'type': 'object',
            'required': ['database', 'model'],
            'properties': {
                'database': {
                    'type': 'object',
                    'required': ['host', 'port'],
                    'properties': {
                        'host': {'type': 'string'},
                        'port': {'type': 'integer'},
                        'username': {'type': 'string'},
                        'debug': {'type': 'boolean'},
                        'timeout': {'type': 'number'},
                        'features': {
                            'type': 'array',
                            'items': {'type': 'string'}
                        }
                    }
                },
                'model': {
                    'type': 'object',
                    'properties': {
                        'learning_rate': {'type': 'number'},
                        'batch_size': {'type': 'integer'}
                    }
                }
            }
        }
        
        # Save test files
        self.config_file = self.config_dir / "test_config.yaml"
        self.schema_file = self.schema_dir / "test_config_schema.yaml"
        
        with open(self.config_file, 'w') as f:
            yaml.dump(self.test_config, f)
        
        with open(self.schema_file, 'w') as f:
            yaml.dump(self.test_schema, f)
        
        # Initialize config loader
        self.loader = ConfigLoader(str(self.config_dir))
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_yaml_loading(self):
        """Test YAML configuration loading"""
        config = self.loader.load_yaml("test_config.yaml")
        
        self.assertEqual(config['database']['host'], 'localhost')
        self.assertEqual(config['database']['port'], 5432)
        self.assertEqual(config['database']['debug'], True)
        self.assertEqual(config['model']['learning_rate'], 0.001)
    
    def test_json_loading(self):
        """Test JSON configuration loading"""
        # Create JSON config file
        json_file = self.config_dir / "test_config.json"
        with open(json_file, 'w') as f:
            json.dump(self.test_config, f)
        
        config = self.loader.load_json("test_config.json")
        self.assertEqual(config['database']['host'], 'localhost')
        self.assertEqual(config['database']['port'], 5432)
    
    def test_config_validation_success(self):
        """Test successful configuration validation"""
        config = self.loader.load_yaml("test_config.yaml")
        
        # Should not raise an exception
        is_valid = self.loader.validate_config(config, "test_config_schema")
        self.assertTrue(is_valid)
    
    def test_config_validation_failure(self):
        """Test configuration validation failure"""
        invalid_config = {
            'database': {
                'host': 'localhost',
                'port': 'invalid_port'  # Should be integer
            }
        }
        
        with self.assertRaises(ConfigValidationError):
            self.loader.validate_config(invalid_config, "test_config_schema")
    
    def test_environment_variable_overrides(self):
        """Test environment variable overrides"""
        with patch.dict(os.environ, {
            'TEST_CONFIG_DATABASE_HOST': 'production-server',
            'TEST_CONFIG_DATABASE_PORT': '3306',
            'TEST_CONFIG_DATABASE_DEBUG': 'false',
            'TEST_CONFIG_MODEL_BATCH_SIZE': '64'
        }):
            config = self.loader.load_yaml("test_config.yaml", env_prefix="TEST_CONFIG")
            
            self.assertEqual(config['database']['host'], 'production-server')
            self.assertEqual(config['database']['port'], 3306)
            self.assertEqual(config['database']['debug'], False)
            self.assertEqual(config['model']['batch_size'], 64)
    
    def test_environment_variable_type_conversion(self):
        """Test environment variable type conversion"""
        with patch.dict(os.environ, {
            'TEST_CONFIG_DATABASE_PORT': '8080',
            'TEST_CONFIG_DATABASE_DEBUG': 'true',
            'TEST_CONFIG_DATABASE_TIMEOUT': '45.5',
            'TEST_CONFIG_DATABASE_FEATURES': 'auth,logging,monitoring'
        }):
            config = self.loader.load_yaml("test_config.yaml", env_prefix="TEST_CONFIG")
            
            self.assertEqual(config['database']['port'], 8080)
            self.assertIsInstance(config['database']['port'], int)
            
            self.assertEqual(config['database']['debug'], True)
            self.assertIsInstance(config['database']['debug'], bool)
            
            self.assertEqual(config['database']['timeout'], 45.5)
            self.assertIsInstance(config['database']['timeout'], float)
            
            self.assertEqual(config['database']['features'], ['auth', 'logging', 'monitoring'])
            self.assertIsInstance(config['database']['features'], list)
    
    def test_config_merging(self):
        """Test configuration merging"""
        # Create second config file
        config2 = {
            'database': {
                'timeout': 60,
                'ssl': True
            },
            'new_section': {
                'value': 'test'
            }
        }
        
        config2_file = self.config_dir / "test_config2.yaml"
        with open(config2_file, 'w') as f:
            yaml.dump(config2, f)
        
        merged = self.loader.merge_configs("test_config", "test_config2")
        
        # Check that original values are preserved
        self.assertEqual(merged['database']['host'], 'localhost')
        self.assertEqual(merged['database']['port'], 5432)
        
        # Check that new values are added
        self.assertEqual(merged['database']['ssl'], True)
        self.assertEqual(merged['new_section']['value'], 'test')
        
        # Check that overlapping values are overridden
        self.assertEqual(merged['database']['timeout'], 60)
    
    def test_config_caching(self):
        """Test that configurations are cached after loading"""
        config1 = self.loader.load_yaml("test_config.yaml")
        config2 = self.loader.get_config("test_config")
        
        # Should be the same object reference (cached)
        self.assertIs(config1, config2)
    
    def test_get_nonexistent_config(self):
        """Test getting non-existent configuration"""
        config = self.loader.get_config("nonexistent")
        self.assertIsNone(config)
    
    def test_save_config_yaml(self):
        """Test saving configuration as YAML"""
        test_config = {'test': {'value': 123}}
        self.loader.save_config(test_config, "saved_config.yaml", "yaml")
        
        # Verify file was created and can be loaded
        loaded = self.loader.load_yaml("saved_config.yaml")
        self.assertEqual(loaded['test']['value'], 123)
    
    def test_save_config_json(self):
        """Test saving configuration as JSON"""
        test_config = {'test': {'value': 123}}
        self.loader.save_config(test_config, "saved_config.json", "json")
        
        # Verify file was created and can be loaded
        loaded = self.loader.load_json("saved_config.json")
        self.assertEqual(loaded['test']['value'], 123)
    
    def test_update_config(self):
        """Test updating existing configuration"""
        # Load initial config
        original = self.loader.load_yaml("test_config.yaml")
        original_host = original['database']['host']
        
        # Update config
        updates = {
            'database': {
                'host': 'updated-server',
                'new_field': 'new_value'
            }
        }
        
        self.loader.update_config("test_config", updates)
        
        # Get updated config
        updated = self.loader.get_config("test_config")
        
        self.assertEqual(updated['database']['host'], 'updated-server')
        self.assertEqual(updated['database']['new_field'], 'new_value')
        # Other values should be preserved
        self.assertEqual(updated['database']['port'], 5432)
    
    def test_load_all_configs(self):
        """Test loading all configurations in directory"""
        # Create additional config files
        config2 = {'section': {'value': 1}}
        config3 = {'section': {'value': 2}}
        
        with open(self.config_dir / "config2.yaml", 'w') as f:
            yaml.dump(config2, f)
        
        with open(self.config_dir / "config3.json", 'w') as f:
            json.dump(config3, f)
        
        all_configs = self.loader.load_all_configs()
        
        self.assertIn("test_config", all_configs)
        self.assertIn("config2", all_configs)
        self.assertIn("config3", all_configs)
        
        self.assertEqual(all_configs["config2"]["section"]["value"], 1)
        self.assertEqual(all_configs["config3"]["section"]["value"], 2)
    
    def test_list_configs(self):
        """Test listing available configurations"""
        # Create additional config files
        config2 = {'test': 'value'}
        
        with open(self.config_dir / "config2.yaml", 'w') as f:
            yaml.dump(config2, f)
        
        configs = self.loader.list_configs()
        self.assertIn("test_config", configs)
        self.assertIn("config2", configs)
    
    def test_list_schemas(self):
        """Test listing available schemas"""
        schemas = self.loader.list_schemas()
        self.assertIn("test_config_schema", schemas)
    
    def test_file_not_found_error(self):
        """Test error handling for non-existent files"""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_yaml("nonexistent.yaml")
    
    def test_invalid_yaml_error(self):
        """Test error handling for invalid YAML"""
        invalid_yaml_file = self.config_dir / "invalid.yaml"
        with open(invalid_yaml_file, 'w') as f:
            f.write("invalid: yaml: content: [\n")
        
        with self.assertRaises(ConfigValidationError):
            self.loader.load_yaml("invalid.yaml")
    
    def test_invalid_json_error(self):
        """Test error handling for invalid JSON"""
        invalid_json_file = self.config_dir / "invalid.json"
        with open(invalid_json_file, 'w') as f:
            f.write('{"invalid": json}')
        
        with self.assertRaises(ConfigValidationError):
            self.loader.load_json("invalid.json")


if __name__ == '__main__':
    unittest.main()