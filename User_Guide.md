# User Guide: Predictive Quality Monitoring System for Continuous Steel Casting

## 1. Introduction

This guide provides a comprehensive overview of the Predictive Quality Monitoring System for Continuous Steel Casting. It covers the setup, usage, and demonstration of the various components of the system. This guide is intended for developers, data scientists, and operators who want to use, understand, or demonstrate the system.

## 2. System Overview

The Predictive Quality Monitoring System is a proof-of-concept (PoC) that uses machine learning to predict quality outcomes in the continuous steel casting process. The system can ingest streaming sensor data, execute machine learning models (both a baseline XGBoost model and a deep learning LSTM model), and provide real-time predictions and alerts.

### Key Features:

*   **Data Generation:** A synthetic data generator that simulates realistic steel casting data.
*   **Data Quality Assessment:** A comprehensive system for validating the quality of sensor data.
*   **Feature Engineering:** A pipeline for extracting meaningful features from time-series data.
*   **Model Training:** Automated pipelines for training both baseline and LSTM models.
*   **Real-time Inference:** An inference engine for making real-time predictions on streaming data.
*   **Visualization:** A dashboard for visualizing real-time sensor data, predictions, and alerts.
*   **Alerting:** An alert system for notifying operators of potential quality issues.

## 3. Getting Started

### 3.1. Installation

To get started with the system, you need to clone the repository and install the required dependencies.

**1. Clone the repository:**

```bash
git clone https://github.com/dhar174/steel_defect_demo.git
cd steel_defect_demo
```

**2. Install dependencies:**

The project provides two `requirements.txt` files: `requirements.txt` (for GPU) and `requirements-minimal.txt` (for CPU). If you have a compatible GPU, you can use the `requirements.txt` file. Otherwise, use the `requirements-minimal.txt` file.

```bash
# For GPU
pip install -r requirements.txt

# For CPU
pip install -r requirements-cpu.txt
```

**3. Install the package in development mode:**

This allows you to import the modules directly without hardcoded path manipulation.

```bash
pip install -e .
```

### 3.2. Generating Synthetic Data

The system comes with a synthetic data generator that can create realistic steel casting data. To generate the data, run the following command:

```bash
python scripts/generate_synthetic_data.py
```

This will generate a set of Parquet files in the `data/raw` directory, as well as metadata files in the `data/synthetic` directory.

## 4. Demonstrating the System

This section provides a step-by-step guide to demonstrating the various components of the system.

### 4.1. Data Quality Assessment

The `demo_data_quality_assessment.py` script demonstrates the data quality assessment capabilities of the system. To run the demo, use the following command:

```bash
python demo_data_quality_assessment.py
```

This will run a series of checks on the generated data and print a comprehensive quality report to the console. It will also generate a JSON report with the detailed results.

### 4.2. Feature Engineering

The `feature_engineering_demo.py` script in the `examples` directory demonstrates the feature engineering pipeline. To run the demo, use the following command:

```bash
python examples/feature_engineering_demo.py
```

This will demonstrate single and batch processing, feature validation, performance assessment, and correlation analysis.

### 4.3. Model Training

The system provides separate scripts for training the baseline and LSTM models.

**4.3.1. Baseline Model Training**

The `train_baseline_model.py` script in the `scripts` directory is used to train the baseline XGBoost model. To run the training with default settings, use the following command:

```bash
python scripts/train_baseline_model.py --data data/examples/steel_defect_sample.csv
```

**4.3.2. LSTM Model Training**

The `train_lstm_model.py` script in the `scripts` directory is used to train the LSTM model. The `demo_lstm_training_enhanced.sh` script demonstrates the advanced features of the LSTM training script. To run the demo, use the following command:

```bash
bash demo_lstm_training_enhanced.sh
```

### 4.4. Model Evaluation and Comparison

The `demo_model_comparison.py` script demonstrates how to compare the performance of different models. To run the demo, use the following command:

```bash
python demo_model_comparison.py
```

This will demonstrate how to use the `ModelComparison` component to create a dashboard for comparing model performance.

### 4.5. Real-time Monitoring and Prediction

The `demo_dashboard_integration.py` script demonstrates how to integrate the various components into a real-time monitoring dashboard. To run the demo, use the following command:

```bash
python demo_dashboard_integration.py
```

This will start a web server on `http://127.0.0.1:8051`. You can open this URL in your web browser to see the real-time monitoring dashboard.

### 4.6. Alert System

The `demo_alert_system.py` script demonstrates the alert system. To run the demo, use the following command:

```bash
python demo_alert_system.py
```

This will demonstrate how to send alerts with different severity levels and how the alert suppression logic works.

## 5. Using the Notebooks

The `notebooks` directory contains a series of Jupyter notebooks that provide a more in-depth analysis of the data and models.

*   **`01_data_exploration.ipynb`:** Provides a comprehensive data exploration framework.
*   **`02_feature_analysis.ipynb` and `02_feature_analysis_updated.ipynb`:**  Provide a comprehensive analysis of the features.
*   **`03_model_development.ipynb`:** Provides a comprehensive analysis of the LSTM model development.
*   **`04_results_analysis.ipynb`:** Provides a framework for analyzing the results of the models.
*   **`correlation_analysis_demo.ipynb`:** Demonstrates the correlation analysis capabilities.

To run the notebooks, you need to have Jupyter Notebook or JupyterLab installed. You can install it using pip:

```bash
pip install jupyterlab
```

Then, you can run the notebooks by navigating to the `notebooks` directory and running the following command:

```bash
jupyter-lab
```

This will open a new tab in your web browser with the JupyterLab interface. You can then open and run the notebooks.

## 6. Conclusion

This guide has provided a comprehensive overview of the Predictive Quality Monitoring System for Continuous Steel Casting. By following the steps in this guide, you should be able to set up, use, and demonstrate the various components of the system. If you have any questions, please refer to the documentation in the `docs` directory.
