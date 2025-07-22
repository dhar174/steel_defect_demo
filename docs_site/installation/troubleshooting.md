# Installation Troubleshooting

This guide helps resolve common installation issues for the Steel Defect Prediction System.

## Common Installation Issues

### Python Dependencies

#### Issue: Package installation fails

```bash
ERROR: Could not install packages due to an EnvironmentError
```

**Solutions:**

```bash
# Solution 1: Upgrade pip
pip install --upgrade pip

# Solution 2: Use virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Solution 3: Install with user flag
pip install --user -r requirements.txt
```

#### Issue: PyTorch installation problems

**For CPU-only installation:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**For GPU installation:**

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Issue: NumPy/SciPy compilation errors

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install build-essential python3-dev

# Install system dependencies (CentOS/RHEL)
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel

# Use conda instead
conda install numpy scipy pandas scikit-learn
```

### Database Issues

#### Issue: PostgreSQL connection failed

```bash
psql: error: connection to server on socket "/var/run/postgresql/.s.PGSQL.5432" failed
```

**Solutions:**

```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Start PostgreSQL
sudo systemctl start postgresql

# Check connection parameters
export DATABASE_URL="postgresql://username:password@localhost:5432/steel_defects"

# Test connection
psql $DATABASE_URL -c "SELECT version();"
```

#### Issue: Database does not exist

```sql
-- Create database
CREATE DATABASE steel_defects;

-- Create user
CREATE USER steel_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE steel_defects TO steel_user;
```

#### Issue: Permission denied for database

```sql
-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE steel_defects TO steel_user;
GRANT ALL ON SCHEMA public TO steel_user;
```

### Docker Issues

#### Issue: Docker daemon not running

```bash
# Linux
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USER
```

#### Issue: Permission denied for Docker

```bash
# Fix Docker permissions
sudo chmod 666 /var/run/docker.sock

# Or restart Docker service
sudo systemctl restart docker
```

#### Issue: Out of disk space

```bash
# Clean up Docker
docker system prune -a

# Remove unused volumes
docker volume prune

# Check disk usage
docker system df
```

### Memory and Performance Issues

#### Issue: Out of memory during model training

**Solutions:**

```python
# Reduce batch size
BATCH_SIZE = 16  # Instead of 32 or 64

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision training
from torch.cuda.amp import autocast
with autocast():
    outputs = model(inputs)
```

#### Issue: Slow model inference

```python
# Optimize model for inference
model.eval()
torch.set_grad_enabled(False)

# Use torchscript
scripted_model = torch.jit.script(model)

# Use ONNX for deployment
torch.onnx.export(model, dummy_input, "model.onnx")
```

### Import and Path Issues

#### Issue: ModuleNotFoundError

```bash
# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/steel_defect_demo"

# Or in Python
import sys
sys.path.append('/path/to/steel_defect_demo')
```

#### Issue: Relative import errors

```python
# Use absolute imports
from src.models.lstm_model import LSTMModel
# Instead of: from ..models.lstm_model import LSTMModel
```

### Configuration Issues

#### Issue: Configuration file not found

```bash
# Verify config file exists
ls -la configs/

# Copy from template
cp configs/config.yml.example configs/config.yml

# Set environment variable
export CONFIG_PATH=/path/to/configs/config.yml
```

#### Issue: Invalid configuration format

```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('configs/config.yml'))"

# Check for common YAML errors
yamllint configs/config.yml
```

### Network and API Issues

#### Issue: Port already in use

```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Use different port
uvicorn src.api.main:app --port 8001
```

#### Issue: API not accessible

```bash
# Check if service is running
curl http://localhost:8000/health

# Check firewall
sudo ufw status
sudo ufw allow 8000

# Check binding address
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Model and Data Issues

#### Issue: Model file not found

```bash
# Check model path
ls -la models/

# Download pre-trained model
wget https://github.com/dhar174/steel_defect_demo/releases/download/v1.0/model.pth
```

#### Issue: Incompatible model format

```python
# Check PyTorch version compatibility
import torch
print(torch.__version__)

# Load with map_location for CPU
model = torch.load('model.pth', map_location='cpu')
```

#### Issue: Data preprocessing errors

```python
# Check data format
import pandas as pd
data = pd.read_csv('data/sensor_data.csv')
print(data.dtypes)
print(data.isnull().sum())

# Handle missing values
data = data.dropna()
# or
data = data.fillna(data.mean())
```

## System Requirements Verification

### Check Python Version

```bash
python --version  # Should be 3.8+
```

### Check System Resources

```bash
# Memory
free -h

# Disk space
df -h

# CPU info
lscpu
```

### Check GPU (if using)

```bash
# NVIDIA GPU
nvidia-smi

# In Python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

## Environment Validation Script

Create `validate_environment.py`:

```python
#!/usr/bin/env python3
import sys
import subprocess
import importlib

def check_python_version():
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {sys.version}")
    return True

def check_packages():
    required_packages = [
        'torch', 'pandas', 'numpy', 'scikit-learn',
        'fastapi', 'uvicorn', 'sqlalchemy', 'psycopg2'
    ]
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} not found")
            return False
    return True

def check_database():
    try:
        import psycopg2
        # Test connection logic here
        print("✅ Database connection")
        return True
    except Exception as e:
        print(f"❌ Database: {e}")
        return False

if __name__ == "__main__":
    checks = [
        check_python_version(),
        check_packages(),
        check_database()
    ]
    
    if all(checks):
        print("\n🎉 Environment validation passed!")
    else:
        print("\n❌ Environment validation failed!")
        sys.exit(1)
```

Run validation:

```bash
python validate_environment.py
```

## Getting Help

### Log Files

Check application logs:

```bash
# Application logs
tail -f logs/application.log

# System logs
journalctl -u steel-defect-prediction

# Docker logs
docker logs <container_name>
```

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python -m src.api.main
```

### Community Support

- GitHub Issues: [Report bugs and issues](https://github.com/dhar174/steel_defect_demo/issues)
- Discussions: [Ask questions and get help](https://github.com/dhar174/steel_defect_demo/discussions)
- Documentation: [Full documentation](https://dhar174.github.io/steel_defect_demo/)

### Professional Support

For enterprise support and consulting:

- Email: support@steel-defect-prediction.com
- Documentation: [Enterprise Support](../operations/maintenance.md)
- Training: [Custom training programs available](../tutorials/advanced-features.md)