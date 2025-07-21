# Steel Defect Monitoring Demo Guide

This guide provides step-by-step instructions for installing, running, and demonstrating all major components of the **steel_defect_demo** project. It consolidates details from the repository documentation so you can quickly showcase the system for interviews or presentations.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Environment Setup](#environment-setup)
3. [Generating Synthetic Data](#generating-synthetic-data)
4. [Training Models](#training-models)
   - [Baseline XGBoost Model](#baseline-xgboost-model)
   - [LSTM Model](#lstm-model)
   - [Automated Training Pipeline](#automated-training-pipeline)
5. [Running Inference](#running-inference)
6. [Dash Dashboard](#dash-dashboard)
7. [Data Quality Assessment](#data-quality-assessment)
8. [Exploring the Jupyter Notebooks](#exploring-the-jupyter-notebooks)
9. [Example Scripts](#example-scripts)
10. [API Endpoints](#api-endpoints)
11. [Additional Documentation](#additional-documentation)

---

## Project Overview
The repository implements a production-style proof of concept for predicting surface defects in continuous steel casting. Components include:
- Synthetic data generator
- Feature engineering utilities
- Baseline XGBoost and LSTM models
- A real-time prediction pipeline
- Visualization dashboards
- Data quality checks and monitoring utilities

The `docs/` folder contains detailed design documents, while the `examples/` and `demo_*.py` scripts provide runnable demonstrations.

## Environment Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/dhar174/steel_defect_demo.git
   cd steel_defect_demo
   ```
2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # For CPU-only environments you can use requirements-cpu.txt
   ```
4. **Install the package in development mode** so all modules can be imported:
   ```bash
   pip install -e .
   ```

## Generating Synthetic Data
Synthetic sensor data is required for training and demos. Use the comprehensive generator script:
```bash
python scripts/generate_synthetic_data.py \
  --config configs/data_generation.yaml \
  --output-dir data/test_synthetic \
  --num-casts 20 \
  --verbose
```
- Outputs are stored under `data/test_synthetic/` with `raw_timeseries/` parquet files and `metadata/` JSON files.
- Generation parameters can be overridden via CLI options (see `--help`).
- Data quality validation runs automatically if requested.

## Training Models
### Baseline XGBoost Model
Train an interpretable baseline model using the provided script:
```bash
python scripts/train_baseline_model.py \
  --config configs/default_training.yaml \
  --data data/test_synthetic/summary/dataset_summary.csv \
  --experiment-name demo_baseline
```
Artifacts (model, config snapshot, metrics) are saved under `results/<experiment_name>/`.

### LSTM Model
An enhanced command-line script handles deep learning training with GPU/CPU support:
```bash
python scripts/train_lstm_model.py --config configs/model_config.yaml --epochs 50
```
The script supports extensive CLI options for hyperparameters, experiment tracking, checkpointing and more. Consult `LSTM_TRAINING_IMPLEMENTATION_SUMMARY.md` for details.

### Automated Training Pipeline
The `ModelTrainer` class orchestrates end-to-end training (preprocessing, cross-validation, hyperparameter search). Example usage:
```python
from src.models import ModelTrainer
trainer = ModelTrainer(config_path='configs/training_pipeline.yaml')
results = trainer.train_pipeline(data_path='data/processed/steel_defect_features.csv', target_column='defect')
```
See `MODEL_TRAINER_GUIDE.md` for the full configuration structure.

## Running Inference
The `PredictionPipeline` provides near real‑time inference on streaming or pre-recorded data.
```bash
python scripts/run_inference_demo.py --config configs/inference_config.yaml --cast-file data/examples/steel_defect_sample.csv
```
- Supports multiple concurrent streams via `--streams`.
- Benchmark mode measures throughput and latency.
- Predictions are logged and can feed the dashboard.

## Dash Dashboard
Launch an interactive dashboard to visualize sensor trends and predictions.
```bash
python scripts/run_dashboard.py --debug          # Development mode
python scripts/run_dashboard.py --production     # Production mode
```
Key features:
- Real‑time plots of sensor data
- Risk level indicators and alerts
- Works with either live data or the mock stream simulator
Configuration options are located in `configs/inference_config.yaml` and can be overridden with command‑line flags.

## Data Quality Assessment
Assess data integrity using the dedicated module:
```bash
python demo_data_quality_assessment.py
```
This performs single‑cast and dataset‑wide checks (missing values, range violations, temporal continuity and realism). Detailed JSON reports are saved in the repository root. See `docs/DATA_QUALITY_ASSESSMENT.md` for full metrics and usage examples.

## Exploring the Jupyter Notebooks
The `notebooks/` folder contains walkthrough notebooks:
- **01_data_exploration.ipynb** – examine generated sensor data
- **02_feature_analysis.ipynb** – inspect engineered features
- **03_model_development.ipynb** – prototype baseline and LSTM models
- **04_results_analysis.ipynb** – analyze evaluation metrics
Open them with JupyterLab or VS Code to show interactive plots and code.

A supplementary `correlation_analysis_demo.ipynb` notebook mirrors `demo_correlation_analysis.py` for exploratory analysis of sensor correlations.

## Example Scripts
The repository includes many small examples illustrating API usage:
- `examples/baseline_model_example.py` – full workflow for the XGBoost model
- `examples/feature_engineering_demo.py` – demonstrates the feature engineering API
- `examples/statistical_analysis_example.py` – runs statistical tests on synthetic data
- `examples/sequence_dataset_demo.py` – prepares sequence datasets for LSTM training
- `demo_*` scripts (e.g., `demo_monitoring.py`, `demo_sensor_visualization.py`) – show individual components in action
Run any of these with `python <script_name>.py` after installing the package.

## API Endpoints
A FastAPI service (optional) exposes prediction and monitoring endpoints. Refer to `docs/api_documentation.md` for schemas. Example endpoints:
- `GET /health` – service status
- `POST /predict` – send sensor data and receive defect probabilities
- `GET /models` – list available models
Launch the API using `uvicorn` after training models:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Additional Documentation
More in‑depth guides are located in the `docs/` directory:
- `TRAINING_GUIDE.md` – step‑by‑step usage of the training scripts
- `VISUALIZATION_GUIDE.md` – explanation of plotting utilities and dashboards
- `PRODUCTION_DATA_CONNECTORS.md` – connectors for OPC UA, MQTT, REST and databases
- `deployment_guide.md` – Docker/Kubernetes deployment instructions
- `sensor_monitoring_component.md` – overview of monitoring utilities

Consult these files if you need finer implementation details during your demo.

---
With this guide you can install the project, generate demo data, train models, run the streaming inference pipeline, launch dashboards, and showcase notebooks. Combine these pieces to present a comprehensive predictive quality monitoring solution.
