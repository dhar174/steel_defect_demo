# Steel Defect Demo Usage Guide

This guide explains how to install, run, and demonstrate the various components of the **steel_defect_demo** project. It summarizes the main scripts, notebooks, and deployment options so you can showcase the system to others.

## 1. Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dhar174/steel_defect_demo.git
   cd steel_defect_demo
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Install the package in development mode** (enables imports without path hacks)
   ```bash
   pip install -e .
   ```

All demo scripts assume the package is installed in this editable mode.

## 2. Data Generation

Synthetic steel casting data is generated via `scripts/generate_synthetic_data.py`.
Typical usage:
```bash
python scripts/generate_synthetic_data.py --num-casts 500 --defect-rate 0.2
```
This produces Parquet/CSV files in `data/` along with dataset metadata.

## 3. Model Training

### Baseline XGBoost Model
Run the baseline model training pipeline:
```bash
python scripts/train_baseline_model.py --config configs/default_training.yaml --data data/processed/sample_data.csv
```
Trained models and results are stored under `results/`.

### LSTM Sequence Model
The LSTM trainer supports extensive CLI options. A quick demo:
```bash
python scripts/train_lstm_model.py --epochs 5 --batch-size 32 --gpu 0
```
For a full showcase of features see `demo_lstm_training_enhanced.sh` which runs multiple training scenarios.

## 4. Evaluation and Reporting
Use `examples/model_evaluation_example.py` to see the comprehensive evaluation framework:
```bash
python examples/model_evaluation_example.py
```
This generates ROC/PR curves, metrics tables and summary reports inside `results/example_evaluation`.

## 5. Data Quality Assessment
The project includes an automated data quality assessment system. Run the demonstration:
```bash
python demo_data_quality_assessment.py
```
Reports are saved as JSON (e.g. `data_quality_report_*_analysis_<timestamp>.json`). See `docs/DATA_QUALITY_ASSESSMENT.md` for full details.

## 6. Visualization Demos
Several scripts produce interactive dashboards and plots.

- **Sensor Visualization** – generates multiple HTML files in `demo_outputs/`:
  ```bash
  python demo_sensor_visualization.py
  ```
- **Historical Analysis Component** – run tests or a standalone Dash app:
  ```bash
  python demo_historical_analysis.py --mode demo
  ```
- **Prediction Display Components**:
  ```bash
  python demo_prediction_display.py
  ```
- **Alert Management Interface**:
  ```bash
  python demo_alert_management.py
  ```
- **Model Comparison Dashboard**:
  ```bash
  python demo_model_comparison.py
  ```
- **Real‑time Monitoring**:
  ```bash
  python demo_monitoring.py
  ```

All Dash apps run locally (default port 8050 or as indicated in the script). Open the shown URL in a browser to view the interface.

## 7. Prediction Pipeline & Inference Demo
The asynchronous prediction pipeline orchestrates multiple streams and can be benchmarked:
```bash
python scripts/run_inference_demo.py --config configs/inference_config.yaml --benchmark
```
This uses the `PredictionPipeline` to read cast files, run predictions and log performance metrics.

## 8. Integrated Dashboard
A production-ready dashboard launcher resides in `scripts/run_dashboard.py`.
Start the dashboard (with mock data) using:
```bash
python scripts/run_dashboard.py --mock-data --debug
```
Visit `http://localhost:8050` to explore real-time monitoring, predictions and historical analysis pages.

## 9. Notebooks
The `notebooks/` directory contains exploratory notebooks:
- `01_data_exploration.ipynb`
- `02_feature_analysis.ipynb`
- `03_model_development.ipynb`
- `04_results_analysis.ipynb`
- `correlation_analysis_demo.ipynb`

Launch Jupyter and open these notebooks for a step-by-step walkthrough of data loading, feature engineering, model building and results interpretation.

```bash
jupyter notebook
```

## 10. Testing
Run the unit tests to verify functionality:
```bash
python -m pytest tests/ -v
```
Additional standalone tests exist such as `test_sensor_monitoring_unit.py` and `test_alert_management.py`.

## 11. Deployment
Refer to `docs/deployment_guide.md` for containerization and production setup. Key steps:

1. **Build Docker image**
   ```bash
docker build -t steel-defect-predictor .
   ```
2. **Run container**
   ```bash
docker run -d -p 8000:8000 -p 8050:8050 --env-file .env steel-defect-predictor
   ```
3. **Kubernetes deployment** – apply manifests from the guide when scaling beyond a single host.

## 12. Additional Documentation
The `docs/` folder contains detailed guides on feature engineering, visualization utilities, configuration management, API usage and more. Review these files for in-depth explanations of each subsystem.

---
With the above commands and references you can demonstrate every major component of the project – from data generation and model training to real‑time dashboards and quality assessment. Adjust arguments as needed to tailor the demos for your audience.
