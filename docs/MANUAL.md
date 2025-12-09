# Technical Documentation

This document provides a technical overview of the Paper Trader architecture, data pipelines, and operational workflows.

---

## 🏗 System Architecture

The application follows a modular architecture designed for extensibility and reproducibility.

### 1. Data Pipeline & Logic
*   **Data Ingestion** (`src/data/loader.py`): Interfaces with the `yfinance` API to retrieve OHLCV data. optimized for bandwidth by caching requests where possible.
*   **Feature Engineering** (`src/features/indicators.py`): Computes technical derivatives (RSI, MACD, Bollinger Bands) to generate the feature vector $X$ for the model.
*   **Model Training** (`src/models/trainer.py`):
    *   Implements an **XGBoost Classifier** on a rolling window basis.
    *   Performs a time-series split to validate model generalization.
    *   Serializes the trained model artifact to `models/xgb_model.joblib`.
    *   Exports performance metrics (F1, Precision, Recall) to `results/`.
*   **Inference Engine** (`src/models/predictor.py`): Deserializes the model artifact to generate probability estimates $P(Y=1|X)$ for the upcoming trading session.
*   **Execution Engine** (`src/trading/portfolio.py`): Manages state persistence via `ledger.csv`, handling order execution, position sizing, and proper accounting.

### 2. Component Glossary

| Component | Description |
| :--- | :--- |
| **`main.py`** | **Application Entry Point**. Orchestrates the CLI arguments and delegates control to the training or trading subsystems. |
| **`Makefile`** | **Automation**. Abstraction layer for common build and execution commands. |
| **`config/settings.yaml`** | **Configuration**. Declarative configuration for the asset universe, hyperparameters, and risk constraints. |
| **`ledger.csv`** | **Persistence**. CSV-based flat-file database for transaction history and current portfolio state. |
| **`results/`** | **Telemetry**. Directory containing evaluation artifacts (confusion matrices, metric logs) from the latest training epoch. |

---

## 📊 Evaluation & Telemetry

The system generates the following artifacts post-training to verify model performance:

### 1. Metric Logs (`results/metrics.txt`)
Contains standard classification metrics:
*   **Accuracy**
*   **Precision/Recall** (Per class)
*   **F1-Score**

### 2. Confusion Matrix (`results/confusion_matrix.png`)
Visual representation of the classifier's performance, allowing for rapid assessment of Type I and Type II error rates.

---

## 🛠 Operations (Makefile)

The project utilizes `make` for standardized operations.

*   `make setup`: Provision the Conda environment and install dependencies.
*   `make trade`: Execute the standard daily trading workflow (Fetch -> Train -> Predict -> Execute).
*   `make train`: Force a manual retraining of the model without executing trades.
*   `make clean`: Remove build artifacts, pycache, and temporary results.
*   `make docker-up`: Deploy the application within the Docker container (Recommended for production consistency).
