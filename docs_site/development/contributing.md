# Contributing Guide

Thank you for your interest in contributing to the Steel Defect Prediction System! This guide provides everything
you need to know to contribute effectively.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)  
- [Pull Request Process](#pull-request-process)
- [Code of Conduct](#code-of-conduct)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)
- Basic understanding of machine learning concepts
- Familiarity with Dash/Plotly for UI contributions

### Development Setup

1. **Fork and Clone**

   ```bash

   # Fork the repository on GitHub, then clone your fork

   git clone https://github.com/YOUR_USERNAME/steel_defect_demo.git
   cd steel_defect_demo
   
   # Add upstream remote

   git remote add upstream https://github.com/dhar174/steel_defect_demo.git
   ```

2. **Create Virtual Environment**

   ```bash

   # Using venv

   python -m venv steel_defect_env
   source steel_defect_env/bin/activate  # Linux/macOS

   # steel_defect_env\Scripts\activate   # Windows
   
   # Using conda

   conda create -n steel_defect_env python=3.9
   conda activate steel_defect_env
   ```

3. **Install Dependencies**

   ```bash

   # Install in development mode

   pip install -e .
   
   # Install development dependencies

   pip install -r requirements-docs.txt
   
   # Install pre-commit hooks (optional but recommended)

   pip install pre-commit
   pre-commit install
   ```

4. **Verify Setup**

   ```bash

   # Run tests

   python -m pytest tests/ -v
   
   # Run demo to verify installation

   python demo_model_comparison.py
   ```

## Development Workflow

### Branch Strategy

We use a **feature branch workflow**:

```text
main branch (stable)
├── feature/new-model-type
├── bugfix/dashboard-loading-issue
├── docs/api-documentation
└── enhancement/alert-system-improvements
```text

### Creating a Feature Branch

```bash

# Update main branch

git checkout main
git pull upstream main

# Create and switch to feature branch

git checkout -b feature/your-feature-name

# Push feature branch to your fork

git push -u origin feature/your-feature-name
```text

### Branch Naming Convention

- **Features**: `feature/description-of-feature`
- **Bug fixes**: `bugfix/description-of-bug`
- **Documentation**: `docs/description-of-docs`
- **Enhancements**: `enhancement/description-of-enhancement`
- **Experimental**: `experiment/description-of-experiment`

## Code Standards

### Python Code Style

We follow **PEP 8** with some modifications:

```python

# Good: Clear, descriptive names

def calculate_defect_probability(sensor_data: Dict[str, float]) -> float:
    """
    Calculate defect probability based on sensor readings.
    
    Args:
        sensor_data: Dictionary containing sensor measurements
        
    Returns:
        Defect probability between 0 and 1
        
    Raises:
        ValueError: If sensor data is invalid
    """
    if not sensor_data:
        raise ValueError("Sensor data cannot be empty")
    
    # Implementation here

    return probability

# Bad: Unclear names and missing documentation

def calc(data):
    if not data:
        return 0

    # What does this do?

    return result
```text

### Code Formatting

We use **Black** for code formatting:

```bash

# Format all Python files

black .

# Check formatting without changes

black --check .

# Format specific file

black src/models/baseline_model.py
```text

### Import Organization

```python

# Standard library imports

import os
import sys
from typing import Dict, List, Optional
from dataclasses import dataclass

# Third-party imports

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier

# Local imports

from src.models.base_model import BaseModel
from src.utils.config import Config
from src.data.preprocessor import DataPreprocessor
```text

### Type Hints

Use type hints for better code clarity:

```python
from typing import Dict, List, Optional, Union, Tuple

def process_sensor_data(
    data: Dict[str, float],
    window_size: int = 60,
    normalize: bool = True
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Process sensor data with proper type hints."""

    # Implementation

    return processed_data, metadata
```text

### Docstring Style

Use **Google-style** docstrings:

```python
def train_model(
    training_data: pd.DataFrame,
    model_params: Dict[str, Any],
    validation_split: float = 0.2
) -> Tuple[BaseModel, Dict[str, float]]:
    """
    Train a machine learning model for defect prediction.
    
    This function trains a model using the provided training data and
    parameters, with automatic validation splitting and performance evaluation.
    
    Args:
        training_data: DataFrame containing training samples with features
            and target labels. Must include 'defect_label' column.
        model_params: Dictionary of model hyperparameters. Required keys
            depend on model type (e.g., 'n_estimators' for XGBoost).
        validation_split: Fraction of data to use for validation (0.0-1.0).
            Defaults to 0.2 (20% validation).
    
    Returns:
        A tuple containing:

            - Trained model instance
            - Dictionary of validation metrics including accuracy, precision,

              recall, and F1-score
    
    Raises:
        ValueError: If training_data is empty or missing required columns.
        TypeError: If model_params is not a dictionary.
        
    Example:
        >>> training_df = pd.read_csv('training_data.csv')
        >>> params = {'n_estimators': 100, 'max_depth': 6}
        >>> model, metrics = train_model(training_df, params)
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
        Accuracy: 0.876
    """

    # Implementation here

    pass
```text

## Testing Guidelines

### Test Structure

```text
tests/
├── unit/                   # Unit tests
│   ├── test_models/
│   ├── test_data/
│   └── test_utils/
├── integration/            # Integration tests
│   ├── test_pipeline/
│   └── test_dashboard/
├── fixtures/              # Test data and fixtures
│   ├── sample_data.csv
│   └── mock_models/
└── conftest.py            # Shared test configuration
```text

### Writing Unit Tests

```python
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.models.baseline_model import BaselineModel
from src.utils.exceptions import ModelNotTrainedError

class TestBaselineModel:
    """Test suite for BaselineModel class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        return pd.DataFrame({
            'temperature': [1520, 1525, 1518, 1522],
            'pressure': [150, 155, 148, 152],
            'defect_label': [0, 1, 0, 1]
        })
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Create a trained model instance."""
        model = BaselineModel()
        model.train(sample_data)
        return model
    
    def test_model_initialization(self):
        """Test model initialization with default parameters."""
        model = BaselineModel()
        assert model.is_trained is False
        assert model.model_type == 'xgboost'
    
    def test_model_training_success(self, sample_data):
        """Test successful model training."""
        model = BaselineModel()
        metrics = model.train(sample_data)
        
        assert model.is_trained is True
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_model_training_invalid_data(self):
        """Test model training with invalid data."""
        model = BaselineModel()
        
        with pytest.raises(ValueError, match="Training data cannot be empty"):
            model.train(pd.DataFrame())
    
    def test_prediction_without_training(self):
        """Test prediction on untrained model raises error."""
        model = BaselineModel()
        sensor_data = {'temperature': 1520, 'pressure': 150}
        
        with pytest.raises(ModelNotTrainedError):
            model.predict(sensor_data)
    
    def test_prediction_success(self, trained_model):
        """Test successful prediction."""
        sensor_data = {'temperature': 1520, 'pressure': 150}
        prediction = trained_model.predict(sensor_data)
        
        assert 'defect_probability' in prediction
        assert 0 <= prediction['defect_probability'] <= 1
        assert 'confidence_score' in prediction
    
    @patch('src.models.baseline_model.joblib.load')
    def test_model_loading(self, mock_load):
        """Test model loading from file."""
        mock_load.return_value = Mock()
        
        model = BaselineModel()
        model.load('path/to/model.pkl')
        
        assert model.is_trained is True
        mock_load.assert_called_once_with('path/to/model.pkl')
```text

### Integration Tests

```python
import pytest
from src.inference.prediction_engine import PredictionEngine
from src.data.data_connectors import DataConnector

class TestPredictionPipeline:
    """Integration tests for the complete prediction pipeline."""
    
    def test_end_to_end_prediction(self):
        """Test complete prediction pipeline from data to result."""

        # Setup

        engine = PredictionEngine()
        connector = DataConnector()
        
        # Get test data

        test_data = connector.get_test_data()
        
        # Make prediction

        result = engine.predict(test_data)
        
        # Verify result structure

        assert 'defect_probability' in result
        assert 'confidence_score' in result
        assert 'model_predictions' in result
        assert result['defect_probability'] >= 0
        assert result['defect_probability'] <= 1
```text

### Test Data Management

```python

# conftest.py - Shared test fixtures

import pytest
import pandas as pd
import numpy as np

@pytest.fixture(scope="session")
def sample_sensor_data():
    """Create sample sensor data for testing."""
    np.random.seed(42)  # Reproducible results
    
    n_samples = 1000
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
        'temperature': np.random.normal(1520, 10, n_samples),
        'pressure': np.random.normal(150, 5, n_samples),
        'flow_rate': np.random.normal(200, 20, n_samples),
        'defect_label': np.random.binomial(1, 0.1, n_samples)  # 10% defect rate
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    from unittest.mock import Mock
    
    model = Mock()
    model.predict.return_value = {
        'defect_probability': 0.15,
        'confidence_score': 0.89
    }
    model.is_trained = True
    
    return model
```text

### Running Tests

```bash

# Run all tests

pytest

# Run with coverage

pytest --cov=src --cov-report=html

# Run specific test file

pytest tests/unit/test_models/test_baseline_model.py

# Run with verbose output

pytest -v

# Run tests matching pattern

pytest -k "test_prediction"

# Run only failed tests from last run

pytest --lf
```text

## Documentation

### Code Documentation

- **Every public function/class must have docstrings**
- **Use type hints for all function parameters and returns**
- **Include examples in docstrings for complex functions**
- **Document exceptions that can be raised**

## Pull Request Process

### Creating a Pull Request

1. **Push your feature branch** to your fork
2. **Open a pull request** against the main repository
3. **Fill out the PR template** with detailed information
4. **Link related issues** using keywords (fixes #123)
5. **Request review** from appropriate team members

### PR Requirements

- All tests must pass
- Code coverage must not decrease
- Documentation must be updated
- Code must follow style guidelines
- Commit messages must be descriptive

### Review Process

1. **Automated checks** run on all PRs
2. **Code review** by at least one team member
3. **Testing** in staging environment
4. **Approval** required before merging
5. **Squash and merge** to maintain clean history

### Adding New Documentation

1. **Create markdown files** in the appropriate `docs_site/` subdirectory
2. **Update `mkdocs.yml`** navigation if adding new pages
3. **Use consistent formatting** with existing documentation
4. **Include code examples** and practical usage
5. **Add diagrams** using Mermaid when helpful

### Documentation Style

```markdown

# Page Title

Brief introduction paragraph explaining the purpose and scope.

## Major Section

Detailed explanation with examples.

### Subsection

More specific information.

```python

# Code example with comments

def example_function():
    """Example with proper documentation."""
    return "result"

```text
!!! note "Important Note"
    Use admonitions for important information.

!!! warning "Warning"
    Use warnings for potential issues.

!!! tip "Pro Tip"
    Use tips for helpful suggestions.
```text

## Pull Request Template

### Before Submitting

1. **Ensure all tests pass**

   ```bash
   pytest
   ```

1. **Code formatting**

   ```bash
   black .
   flake8 .
   ```

1. **Update documentation** if needed

1. **Update changelog** in `CHANGELOG.md`

### Pull Request Template

When creating a pull request, include:

```markdown

## Description

Brief description of changes made.

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that causes existing functionality to change)
- [ ] Documentation update

## Testing

- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have tested the changes in the dashboard interface

## Checklist

- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings

## Screenshots (if applicable)

Add screenshots to help explain your changes.
```text

### Review Process

1. **Automated checks** must pass (tests, linting, etc.)
2. **Code review** by at least one maintainer
3. **Manual testing** of significant changes
4. **Documentation review** if docs are updated
5. **Merge** after approval

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of experience level, gender, gender identity and expression, sexual orientation, disability, personal appearance, body size, race, ethnicity, age, religion, or nationality.

### Expected Behavior

- **Be respectful** and inclusive
- **Be collaborative** and helpful
- **Be patient** with newcomers
- **Give constructive feedback**
- **Focus on what is best** for the community

### Unacceptable Behavior

- Harassment, discrimination, or exclusionary behavior
- Personal attacks or insults
- Trolling or inflammatory comments
- Publishing private information without permission
- Any other conduct that could reasonably be considered inappropriate

### Reporting Issues

If you experience or witness unacceptable behavior, please contact the project maintainers at [contact information].

## Getting Help

### Resources

- **Documentation**: This comprehensive guide
- **Issues**: [GitHub Issues](https://github.com/dhar174/steel_defect_demo/issues) for bugs and feature requests
- **Discussions**: [GitHub Discussions](https://github.com/dhar174/steel_defect_demo/discussions) for questions and ideas
- **Wiki**: Additional technical notes and examples

### Communication Channels

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions, ideas, and general discussion
- **Pull Request Reviews**: For code-specific feedback

### Getting Started Issues

Look for issues labeled `good first issue` or `help wanted` if you're new to the project. These are typically:

- Documentation improvements
- Small bug fixes
- Test coverage improvements
- Code style cleanup

## Recognition

Contributors will be recognized in:

- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions
- **Documentation credits**

Thank you for contributing to the Steel Defect Prediction System! 🚀

---

Next: [Changelog →](../releases/changelog.md)
