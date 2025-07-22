# System Requirements

This page outlines the hardware and software requirements for running the Steel Defect Prediction System in different environments.

## Minimum Requirements

### Hardware

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|--------|
| **CPU** | 2 cores | 4+ cores | Multi-core improves model training performance |
| **RAM** | 4GB | 8GB+ | LSTM training requires more memory |
| **Storage** | 2GB | 10GB+ | Includes models, data, and logs |
| **Network** | 1 Mbps | 10+ Mbps | For dashboard and data streaming |

### Software

| Requirement | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.8+ | Core runtime (3.9-3.11 recommended) |
| **pip** | 20.0+ | Package management |
| **Git** | 2.0+ | Version control and installation |

## Production Requirements

### Hardware (Production)

| Component | Minimum | Recommended | High-Performance |
|-----------|---------|-------------|------------------|
| **CPU** | 4 cores | 8+ cores | 16+ cores |
| **RAM** | 8GB | 16GB+ | 32GB+ |
| **Storage** | 50GB SSD | 100GB SSD | 500GB+ NVMe |
| **Network** | 10 Mbps | 100 Mbps | 1 Gbps |
| **GPU** | None | NVIDIA GPU | Tesla/A100 for training |

### Software Stack

#### Operating Systems

=== "Linux (Recommended)"
    - **Ubuntu**: 20.04 LTS, 22.04 LTS
    - **CentOS**: 7, 8
    - **RHEL**: 7, 8, 9
    - **Debian**: 10, 11

=== "macOS"
    - **macOS**: 10.15+ (Catalina or newer)
    - **Architecture**: Intel x64 or Apple Silicon (M1/M2)

=== "Windows"
    - **Windows**: 10, 11 
    - **WSL2**: Required for full compatibility
    - **Docker Desktop**: Recommended for containerization

#### Python Environment

```bash
# Recommended Python version
Python 3.9.7+

# Virtual environment (recommended)
python -m venv steel_defect_env
source steel_defect_env/bin/activate  # Linux/macOS
# steel_defect_env\Scripts\activate   # Windows
```

## Development Environment

### IDE Support

| IDE | Support Level | Notes |
|-----|---------------|--------|
| **VS Code** | ✅ Excellent | Recommended, includes .vscode config |
| **PyCharm** | ✅ Excellent | Professional or Community |
| **Jupyter** | ✅ Good | For notebooks and experimentation |
| **Vim/Neovim** | ✅ Good | With Python LSP |

### Development Tools

```bash
# Code formatting and linting
pip install black flake8 mypy

# Testing framework
pip install pytest pytest-cov

# Documentation
pip install -r requirements-docs.txt
```

## Browser Requirements

### Dashboard Access

| Browser | Version | Support Level |
|---------|---------|---------------|
| **Chrome** | 90+ | ✅ Recommended |
| **Firefox** | 88+ | ✅ Full support |
| **Safari** | 14+ | ✅ Full support |
| **Edge** | 90+ | ✅ Full support |

### Features Used

- **JavaScript**: ES6+ features
- **WebSockets**: For real-time updates
- **Local Storage**: For user preferences
- **Canvas/SVG**: For interactive charts

## Network Requirements

### Ports

| Port | Protocol | Purpose | Required |
|------|----------|---------|----------|
| **8050** | HTTP | Dashboard interface | Yes |
| **8051** | HTTP | API endpoints | Optional |
| **5432** | TCP | PostgreSQL (if used) | Optional |
| **6379** | TCP | Redis (if used) | Optional |

### Firewall Configuration

=== "Development"
    ```bash
    # Allow dashboard access
    sudo ufw allow 8050/tcp
    ```

=== "Production"
    ```bash
    # More restrictive rules
    sudo ufw allow from 10.0.0.0/8 to any port 8050
    sudo ufw allow from 172.16.0.0/12 to any port 8050
    sudo ufw allow from 192.168.0.0/16 to any port 8050
    ```

## Dependencies

### Core Dependencies

```bash
# Data processing
pandas>=1.5.0
numpy>=1.23.0
pyarrow>=10.0.0

# Machine learning
scikit-learn>=1.1.0
xgboost>=1.7.0
torch>=1.13.0

# Visualization
plotly>=5.15.0
dash>=3.0.0
matplotlib>=3.6.0
```

### Optional Dependencies

```bash
# GPU acceleration (CUDA)
torch[cuda]  # For NVIDIA GPUs

# Database connectors
psycopg2>=2.9.0  # PostgreSQL
redis>=4.3.0     # Redis cache

# Production monitoring
prometheus-client>=0.14.0
```

## Performance Benchmarks

### Model Training Time

| Model Type | Dataset Size | CPU (4 cores) | GPU (CUDA) |
|------------|--------------|---------------|------------|
| **XGBoost** | 10K samples | 30 seconds | N/A |
| **LSTM** | 10K sequences | 5 minutes | 1 minute |
| **Ensemble** | 50K samples | 15 minutes | 3 minutes |

### Memory Usage

| Component | Baseline | Peak | Notes |
|-----------|----------|------|--------|
| **Dashboard** | 200MB | 500MB | Depends on data volume |
| **XGBoost Training** | 300MB | 1GB | Scales with features |
| **LSTM Training** | 500MB | 2GB | Scales with sequence length |

## Cloud Deployment

### AWS Requirements

```yaml
# Minimum EC2 instance
Instance: t3.medium
vCPUs: 2
RAM: 4GB
Storage: 20GB GP2

# Recommended EC2 instance  
Instance: c5.xlarge
vCPUs: 4
RAM: 8GB
Storage: 50GB GP3
```

### Docker Requirements

```dockerfile
# Base requirements
FROM python:3.9-slim

# System packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Memory limits
--memory=2g
--memory-swap=4g
```

## Troubleshooting

### Common Issues

!!! warning "Memory Errors"
    If you encounter out-of-memory errors:
    - Increase system RAM or swap
    - Reduce batch size for LSTM training
    - Use data streaming for large datasets

!!! warning "Performance Issues"
    For slow performance:
    - Check CPU usage and available cores
    - Monitor memory usage
    - Consider GPU acceleration for training

!!! warning "Installation Failures"
    For package installation issues:
    - Update pip: `pip install --upgrade pip`
    - Use virtual environment
    - Check Python version compatibility

### Verification Commands

```bash
# Check Python version
python --version

# Verify installation
python -c "import steel_defect_demo; print('Installation OK')"

# Check memory and CPU
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().total//1e9:.1f}GB')"
python -c "import os; print(f'CPU cores: {os.cpu_count()}')"

# Test GPU (if available)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

Next: [Development Setup →](../installation/development-setup.md)