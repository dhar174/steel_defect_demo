# Steel Defect Detection Demo - Documentation

## Overview

This documentation site provides comprehensive information about the Steel
Defect Detection demonstration system. The system uses machine learning
techniques to predict and detect quality issues in continuous steel casting
processes.

## Navigation

- [Dashboard Overview](user-guide/dashboard-overview.md) - Learn how to use
  the dashboard interface
- [Changelog](releases/changelog.md) - Track system updates and releases

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Required dependencies (see requirements.txt)
- Access to steel casting process data

### Installation

```bash
git clone <repository-url>
cd steel_defect_demo
pip install -r requirements.txt
```

### Basic Usage

```python
from src.models import DefectPredictor

# Initialize the predictor
predictor = DefectPredictor()

# Load and process data
data = predictor.load_data('path/to/data.csv')
results = predictor.predict(data)
```

## Architecture

The system consists of several key components:

- **Data Ingestion**: Collects real-time sensor data from casting equipment
- **Feature Engineering**: Processes raw data into meaningful features
- **ML Models**: LSTM and baseline models for defect prediction
- **Dashboard**: Web interface for monitoring and visualization
- **Alert System**: Real-time notifications for quality issues

## Documentation Sections

### User Guides

- Dashboard interface and navigation
- Data input and configuration
- Interpreting prediction results
- Alert management

### Technical Documentation

- Model architecture and training
- API reference and integration
- Deployment and configuration
- Troubleshooting guide

## Support

For technical support or questions about the system, please refer to the
main project documentation or contact the development team.

## License

This project is licensed under the terms specified in the main repository.
