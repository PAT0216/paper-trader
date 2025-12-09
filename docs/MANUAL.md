# 📖 Paper Trader Manual

Welcome to the comprehensive guide for the AI Paper Trader. This document explains "What Does What", how the data flows, and how to interpret the results.

---

## 🏗 System Architecture

### 1. The Workflow
1.  **Loader** (`src/data/loader.py`): Fetches raw price data from Yahoo Finance.
2.  **Featurizer** (`src/features/indicators.py`): Calculates math (RSI, MACD) to give the AI context.
3.  **Trainer** (`src/models/trainer.py`):
    *   Splits data into "Past" (Train) and "Recent" (Test).
    *   Teaches the XGBoost model patterns.
    *   Evaluates performance and saves `metrics.txt` and `confusion_matrix.png`.
4.  **Predictor** (`src/models/predictor.py`): Uses the saved model to guess *tomorrow's* move.
5.  **Portfolio** (`src/trading/portfolio.py`):
    *   Reads `ledger.csv` (Your bank account).
    *   Executes Buy/Sell orders based on predictions.

### 2. File Glossary

| File | Purpose |
| :--- | :--- |
| **`main.py`** | **The Boss**. Run this file to start everything. It decides whether to Train or Trade. |
| `Makefile` | A shortcut helper. Instead of typing long commands, you type `make train`. |
| `config/settings.yaml` | **Control Panel**. Change stocks, training years, or risk settings here. |
| `models/xgb_model.joblib` | **The Brain**. This is the saved AI file. "Joblib" is just a format for saving Python objects efficiently. |
| `ledger.csv` | **The Bank**. Records every trade and your cash balance. |
| `results/` | **The Report Card**. Contains charts and scores from the latest training run. |

---

## 📊 Interpreting Results

After a training run (which happens daily), check the `results/` folder.

### 1. `metrics.txt`
*   **Accuracy**: Overall percentage of correct guesses. (e.g., 0.55 = 55%).
*   **Precision**: When it said "BUY", how often was it right? (High precision = fewer false alarms).
*   **Recall**: Out of all the real profitable opportunities, how many did it find?
*   **F1-Score**: A balance between Precision and Recall.

### 2. `confusion_matrix.png`
This chart shows the detailed mistakes.
*   **True Positive (Bottom Right)**: Predicted UP, and it went UP. (Great!)
*   **True Negative (Top Left)**: Predicted DOWN, and it went DOWN. (Good avoidance).
*   **False Positive (Top Right)**: Predicted UP, but it went DOWN. (Ouch, lost money).
*   **False Negative (Bottom Left)**: Predicted DOWN, but it went UP. (Missed opportunity).

**Goal**: You want dark colors in Top-Left and Bottom-Right.

---

## 🛠 Automation (Makefile)

We included a `Makefile` to make your life easier. Run these commands in the terminal:

*   `make setup`: Installs all the libraries you need.
*   `make trade`: Runs the bot in trading mode (Standard daily run).
*   `make train`: Forces the AI to retrain immediately.
*   `make clean`: Deletes cache files and old results.
*   `make docker-up`: Runs the whole thing inside the Docker container.
