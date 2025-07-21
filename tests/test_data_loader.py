import unittest
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import patch
from src.data.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class exception handling"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_loader = DataLoader(self.temp_dir)
        
        # Create examples directory and sample file
        examples_dir = Path(self.temp_dir) / "examples"
        examples_dir.mkdir(parents=True, exist_ok=True)
        self.sample_path = examples_dir / "steel_defect_sample.csv"
        
        # Create sample CSV data
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='h'),
            'temperature': [1500, 1520, 1480, 1510, 1490],
            'pressure': [100, 105, 98, 102, 99]
        })
        sample_data.to_csv(self.sample_path, index=False)

    def tearDown(self):
        """Clean up after each test method."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_load_cleaned_data_file_not_found_fallback(self):
        """Test that load_cleaned_data falls back to sample data when file doesn't exist"""
        non_existent_file = "/path/that/does/not/exist.csv"
        
        # Should fall back to sample data
        result = self.data_loader.load_cleaned_data(non_existent_file)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)  # Sample data has 5 rows
        self.assertIn('temperature', result.columns)

    def test_load_cleaned_data_permission_error_fallback(self):
        """Test that load_cleaned_data falls back to sample data when permission is denied"""
        # Create a file with restricted permissions
        restricted_file = Path(self.temp_dir) / "restricted.csv"
        restricted_file.write_text("col1,col2\n1,2\n")
        
        # Create expected fallback data
        fallback_data = pd.DataFrame({
            'temperature': [1500, 1520, 1480, 1510, 1490],
            'pressure': [100, 105, 98, 102, 99]
        })
        
        # Mock PermissionError when trying to read the file
        with patch('pandas.read_csv') as mock_read_csv:
            # First call (for the restricted file) raises PermissionError
            # Second call (for sample file) returns actual data
            mock_read_csv.side_effect = [
                PermissionError("Permission denied"),
                fallback_data
            ]
            
            result = self.data_loader.load_cleaned_data(str(restricted_file))
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 5)  # Sample data has 5 rows
            self.assertEqual(mock_read_csv.call_count, 2)

    def test_load_cleaned_data_corrupted_file_fallback(self):
        """Test that load_cleaned_data falls back to sample data when file is corrupted"""
        # Create a corrupted CSV file
        corrupted_file = Path(self.temp_dir) / "corrupted.csv"
        corrupted_file.write_text("invalid,csv,data\nwith\ninvalid\nformat")
        
        # Create expected fallback data
        fallback_data = pd.DataFrame({
            'temperature': [1500, 1520, 1480, 1510, 1490],
            'pressure': [100, 105, 98, 102, 99]
        })
        
        # Mock ParserError when trying to read the corrupted file
        with patch('pandas.read_csv') as mock_read_csv:
            # First call (for corrupted file) raises ParserError
            # Second call (for sample file) returns actual data
            mock_read_csv.side_effect = [
                pd.errors.ParserError("Could not parse CSV"),
                fallback_data
            ]
            
            result = self.data_loader.load_cleaned_data(str(corrupted_file))
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 5)  # Sample data has 5 rows
            self.assertEqual(mock_read_csv.call_count, 2)

    def test_load_cleaned_data_empty_file_fallback(self):
        """Test that load_cleaned_data falls back to sample data when file is empty"""
        # Create an empty file
        empty_file = Path(self.temp_dir) / "empty.csv"
        empty_file.touch()
        
        # Create expected fallback data
        fallback_data = pd.DataFrame({
            'temperature': [1500, 1520, 1480, 1510, 1490],
            'pressure': [100, 105, 98, 102, 99]
        })
        
        # Mock EmptyDataError when trying to read the empty file
        with patch('pandas.read_csv') as mock_read_csv:
            # First call (for empty file) raises EmptyDataError
            # Second call (for sample file) returns actual data
            mock_read_csv.side_effect = [
                pd.errors.EmptyDataError("No data"),
                fallback_data
            ]
            
            result = self.data_loader.load_cleaned_data(str(empty_file))
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 5)  # Sample data has 5 rows
            self.assertEqual(mock_read_csv.call_count, 2)

    def test_load_cleaned_data_successful_load(self):
        """Test that load_cleaned_data successfully loads valid file"""
        # Create a valid CSV file
        valid_file = Path(self.temp_dir) / "valid.csv"
        test_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        test_data.to_csv(valid_file, index=False)
        
        result = self.data_loader.load_cleaned_data(str(valid_file))
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertEqual(list(result.columns), ['col1', 'col2'])

    def test_load_cleaned_data_no_fallback_available(self):
        """Test that FileNotFoundError is raised when both primary and fallback files don't exist"""
        # Remove the sample file
        os.remove(self.sample_path)
        
        non_existent_file = "/path/that/does/not/exist.csv"
        
        with self.assertRaises(FileNotFoundError) as context:
            self.data_loader.load_cleaned_data(non_existent_file)
        
        self.assertIn("Neither", str(context.exception))
        self.assertIn("Original error:", str(context.exception))


if __name__ == '__main__':
    unittest.main()