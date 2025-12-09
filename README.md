# Paper Trader AI

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.10-blue)
![Docker](https://img.shields.io/badge/docker-ready-blue)

**Paper Trader AI** is an autonomous, containerized algorithmic trading system designed for the US Equity Market. It leverages machine learning (XGBoost) to analyze historical price action and execute probability-based trade decisions on a simulated portfolio.

---

## 📋 Key Features

*   **Predictive Intelligence**: Utilizes an XGBoost classifier trained on **3 Years** of historical data.
*   **Containerized Architecture**: Fully Dockerized environment using `miniconda3` for reproducible, platform-independent execution.
*   **Multi-Asset Support**: Capable of managing a diverse portfolio of assets (default: **Top 30 S&P 500 Stocks**).
*   **Automated Execution**: Integrated with GitHub Actions for scheduled daily operation at market close.

## 🚀 Getting Started

### Prerequisites
*   Docker & Docker Compose
*   Git

### Installation
Clone the repository and navigate to the project root:
```bash
git clone https://github.com/PAT0216/paper-trader.git
cd paper-trader
```

### Execution
The system is designed to run via Docker Compose for maximum stability:

```bash
docker-compose up --build
```

**What happens next?**
1.  **Data Ingestion**: The system fetches 2 years of granular market data.
2.  **Model Training**: The AI model retrains on the latest data to adapt to recent market regimes.
3.  **Inference**: The model generates buy/sell probability scores for each asset.
4.  **Portfolio Rebalancing**: Trades are executed against the local ledger to align holdings with model convictions.

## ⚙️ Configuration

The system is data-driven and configurable via `config/settings.yaml`.

```yaml
# config/settings.yaml

# Define your trading universe
tickers:
  - SPY
  - AAPL
  - MSFT

# Adjust risk parameters
model:
  threshold_buy: 0.55  # Confidence required to enter position
```

## 📂 Project Structure

*   `main.py`: Application entry point and orchestrator.
*   `src/`: Core application logic.
    *   `data/`: Data fetching and preprocessing pipelines.
    *   `features/`: Technical indicator engineering.
    *   `models/`: Machine learning training and inference engines.
    *   `trading/`: Portfolio management and order execution.
*   `entrypoint.sh`: Container initialization script.

---

**Disclaimer**: *This software is for educational and research purposes only. It is not financial advice, and the authors are not responsible for any financial losses incurred.*
