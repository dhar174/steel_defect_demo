# Development Setup Guide

This guide walks you through setting up a complete development environment for the Steel Defect Prediction System.

## Quick Setup

For experienced developers who want to get started immediately:

```bash

# Clone and setup

git clone https://github.com/dhar174/steel_defect_demo.git
cd steel_defect_demo
python -m venv venv && source venv/bin/activate
pip install -e . && pip install -r requirements-docs.txt
python demo_model_comparison.py  # Verify installation
```text

## Detailed Setup Instructions

### 1. System Prerequisites

Ensure you have these installed:

- **Python 3.8+** (3.9-3.11 recommended)
- **Git** for version control
- **Virtual environment tool** (venv, conda, virtualenv)

#### Platform-Specific Requirements

=== "Linux (Ubuntu/Debian)"

    ```bash

    # Update system packages

    sudo apt update && sudo apt upgrade -y
    
    # Install development tools

    sudo apt install -y python3 python3-pip python3-venv git build-essential
    
    # Install optional dependencies

    sudo apt install -y python3-dev libffi-dev libssl-dev
    ```

=== "macOS"

    ```bash

    # Install Homebrew (if not installed)

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Install Python and Git

    brew install python git
    
    # Install development tools

    xcode-select --install
    ```

=== "Windows"

    ```powershell

    # Using Chocolatey package manager

    # Install Chocolatey first from https://chocolatey.org/install
    
    choco install python git
    
    # Or download installers from:

    # Python: https://python.org/downloads/

    # Git: https://git-scm.com/download/win

    ```

### 2. Clone Repository

```bash

# Clone the repository

git clone https://github.com/dhar174/steel_defect_demo.git
cd steel_defect_demo

# Verify repository structure

ls -la
```text

Expected structure:

```text
steel_defect_demo/
├── src/                    # Source code
├── tests/                  # Test suite
├── docs_site/             # Documentation
├── data/                  # Sample data
├── configs/               # Configuration files
├── requirements.txt       # Dependencies
├── setup.py              # Package setup
└── README.md             # Project overview
```text

### 3. Virtual Environment Setup

Choose your preferred virtual environment tool:

=== "venv (Recommended)"

    ```bash

    # Create virtual environment

    python -m venv steel_defect_env
    
    # Activate virtual environment

    # Linux/macOS:

    source steel_defect_env/bin/activate
    
    # Windows:

    # steel_defect_env\Scripts\activate
    
    # Verify activation

    which python  # Should point to venv
    ```

=== "conda"

    ```bash

    # Create conda environment

    conda create -n steel_defect_env python=3.9
    
    # Activate environment

    conda activate steel_defect_env
    
    # Verify activation

    conda info --envs
    ```

=== "virtualenv"

    ```bash

    # Install virtualenv if needed

    pip install virtualenv
    
    # Create virtual environment

    virtualenv steel_defect_env
    
    # Activate environment

    source steel_defect_env/bin/activate  # Linux/macOS

    # steel_defect_env\Scripts\activate   # Windows

    ```

### 4. Install Dependencies

```bash

# Upgrade pip

pip install --upgrade pip

# Install the package in development mode

pip install -e .

# Install documentation dependencies

pip install -r requirements-docs.txt

# Install development tools (optional but recommended)

pip install black flake8 pytest mypy pre-commit
```text

#### Optional: GPU Support

For LSTM model training with GPU acceleration:

```bash

# CUDA 11.x

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.x

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify GPU support

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```text

### 5. Verify Installation

Run these verification steps:

```bash

# Test basic imports

python -c "import src; print('✅ Package imports successfully')"

# Run a simple demo

python demo_model_comparison.py

# Run basic tests

python -m pytest tests/ -v --tb=short

# Build documentation

mkdocs build
```text

Expected output indicates successful setup:

```text
✅ Package imports successfully
🚀 ModelComparison Component Demo
==================================================
✅ All individual components demonstrated successfully!
```text

### 6. IDE Configuration

#### VS Code Setup

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./steel_defect_env/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "88"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "steel_defect_env/": true,
        "site/": true
    }
}
```text

Create `.vscode/launch.json` for debugging:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Run Dashboard",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/scripts/run_dashboard.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        },
        {
            "name": "Run Demo",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/demo_model_comparison.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        }
    ]
}
```text

#### PyCharm Setup

1. **Open Project**: File → Open → Select `steel_defect_demo` folder

2.
**Configure Interpreter**: Settings → Python Interpreter → Add → Existing Environment → Select `steel_defect_env/bin/python`

3. **Enable Tools**: Settings → Tools → Enable pytest, Black formatter
4. **Mark Source Root**: Right-click `src` folder → Mark Directory As → Sources Root

### 7. Development Tools Setup

#### Pre-commit Hooks (Recommended)

```bash

# Install pre-commit

pip install pre-commit

# Install git hooks

pre-commit install

# Test hooks

pre-commit run --all-files
```text

Create `.pre-commit-config.yaml`:

```yaml
repos:

  - repo: https://github.com/psf/black

    rev: '22.10.0'
    hooks:

      - id: black

        language_version: python3

  - repo: https://github.com/pycqa/flake8

    rev: '5.0.4'
    hooks:

      - id: flake8

        args: ['--max-line-length=88', '--extend-ignore=E203,W503']

  - repo: https://github.com/pre-commit/mirrors-mypy

    rev: 'v0.991'
    hooks:

      - id: mypy

        additional_dependencies: [types-all]
```text

#### Code Quality Tools

```bash

# Format code

black .

# Check code style

flake8 .

# Type checking

mypy src/

# Security analysis

pip install bandit
bandit -r src/
```text

### 8. Database Setup (Optional)

For advanced features requiring database:

```bash

# Install database dependencies

pip install psycopg2-binary  # PostgreSQL
pip install redis            # Redis cache

# Setup PostgreSQL (Ubuntu/Debian)

sudo apt install postgresql postgresql-contrib
sudo -u postgres createdb steel_defect_db

# Setup Redis

sudo apt install redis-server
redis-cli ping  # Should return "PONG"
```text

## Development Workflow

### Daily Development

```bash

# Activate environment

source steel_defect_env/bin/activate

# Pull latest changes

git pull origin main

# Install new dependencies (if any)

pip install -e .

# Run tests before starting work

pytest tests/

# Start development server

python scripts/run_dashboard.py
```text

### Code Development Cycle

1. **Create Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

1. **Write Code and Tests**

   ```bash

   # Edit code in src/

   # Add tests in tests/

   ```

1. **Test Changes**

   ```bash

   # Run tests

   pytest tests/

   # Check code quality

   black .
   flake8 .
   ```

1. **Commit Changes**

   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

1. **Push and Create PR**

   ```bash
   git push origin feature/your-feature-name

   # Create pull request on GitHub

   ```

### Documentation Development

```bash

# Start documentation server

mkdocs serve

# Edit documentation files in docs_site/

# Changes automatically reload in browser

# Build for production

mkdocs build
```text

### Testing

```bash

# Run all tests

pytest

# Run specific test file

pytest tests/unit/test_models/test_baseline_model.py

# Run with coverage

pytest --cov=src --cov-report=html

# Run integration tests

pytest tests/integration/
```text

## Troubleshooting

### Common Issues

!!! warning "Import Errors"

    ```bash

    # If you get import errors, ensure package is installed in development mode

    pip install -e .
    
    # Check Python path

    python -c "import sys; print('\n'.join(sys.path))"
    ```

!!! warning "Permission Errors"

    ```bash

    # Linux/macOS permission issues

    sudo chown -R $USER:$USER steel_defect_demo/
    
    # Windows permission issues

    # Run command prompt as administrator

    ```

!!! warning "Memory Issues"

    ```bash

    # For large datasets or LSTM training

    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    
    # Or reduce batch size in training configs

    ```

### Environment Issues

```bash

# Reset virtual environment

deactivate
rm -rf steel_defect_env/
python -m venv steel_defect_env
source steel_defect_env/bin/activate
pip install -e .
```text

### Package Conflicts

```bash

# Check installed packages

pip list

# Check for conflicts

pip check

# Create requirements.txt from current environment

pip freeze > requirements-current.txt
```text

## Performance Optimization

### Development Performance

- **Use SSD storage** for faster file operations
- **Allocate sufficient RAM** (8GB+ recommended)
- **Enable GPU** for LSTM training if available
- **Use pytest-xdist** for parallel test execution:

  ```bash
  pip install pytest-xdist
  pytest -n auto  # Run tests in parallel
  ```

### IDE Performance

- **Exclude build directories** from indexing
- **Disable unnecessary plugins**
- **Use type hints** for better IntelliSense
- **Configure code completion** for faster responses

## Next Steps

Once your development environment is ready:

1. **Explore the Architecture**: Review [System Overview](../architecture/system-overview.md)
2. **Understand the Dashboard**: Read [Dashboard Overview](../user-guide/dashboard-overview.md)
3. **Learn the API**: Check [Dashboard Integration](../api-reference/dashboard-integration.md)
4. **Start Contributing**: See [Contributing Guide](../development/contributing.md)

---

**Congratulations!** 🎉 Your development environment is now ready.
You can start developing, testing, and contributing to the Steel Defect Prediction System.
