# Paper Trader AI

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.10-blue)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![Tests](https://img.shields.io/badge/tests-27%20passed-brightgreen)

**Paper Trader AI** is a professional-grade, risk-managed algorithmic trading system designed for the US Equity Market. It combines machine learning (XGBoost) with institutional-level risk controls to make intelligent, probability-based trading decisions on a simulated portfolio.

---

## 📋 Key Features

### 🧠 **Predictive Intelligence**
- **XGBoost Classifier** trained on **3 years** of historical data
- **Technical Indicators**: RSI, MACD, Bollinger Bands, SMA crossovers
- **Multi-Factor Features**: Momentum, volatility, and trend indicators

### 🛡️ **Professional Risk Management** *(New in Phase 1)*
- **Volatility-Adjusted Position Sizing**: Inverse weighting by 30-day volatility
- **Portfolio Constraints**: Max 15% per position, 40% per sector
- **Value at Risk (VaR)**: Daily portfolio risk monitoring at 95% confidence
- **Pre-Trade Validation**: All trades checked against risk limits before execution
- **Sector Diversification**: Automatic tracking across Technology, Financials, Healthcare, Energy, Consumer sectors

### ✅ **Data Quality Assurance** *(New in Phase 1)*
- **10 Validation Checks**: OHLC integrity, outlier detection, missing values, data freshness
- **Automatic Rejection**: Invalid tickers filtered before model training
- **Comprehensive Logging**: Detailed validation reports for every data fetch

### 🏗️ **Production Infrastructure**
- **Containerized Architecture**: Fully Dockerized with `miniconda3`
- **Multi-Asset Support**: Manages 30+ S&P 500 stocks simultaneously
- **Automated Execution**: GitHub Actions scheduled daily at market close
- **Comprehensive Testing**: 27 unit tests covering risk management and data validation

---

## 🚀 Getting Started

### Prerequisites
*   Docker & Docker Compose
*   Git
*   Python 3.10+ (for local development)

### Installation
Clone the repository and navigate to the project root:
```bash
git clone https://github.com/PAT0216/paper-trader.git
cd paper-trader
```

### Quick Start

Use the **Makefile** for streamlined execution:

#### Run Everything (Docker - Recommended)
```bash
make docker-up
```

#### Run Locally
```bash
# One-time setup
make setup

# Execute trading bot
make trade

# Force model retraining
make train

# Run test suite
make test
```

### What Happens During Execution?

1. **📊 Data Ingestion**: Fetches 3 years of OHLCV data for 30+ tickers
2. **🔍 Data Validation**: Runs 10 quality checks, rejects invalid tickers
3. **🧠 Model Training**: XGBoost trains on technical indicators (or loads existing model)
4. **🔮 Prediction**: Generates buy/sell probability scores for each ticker
5. **⚖️ Risk-Adjusted Sizing**: Calculates position sizes based on volatility and constraints
6. **✅ Pre-Trade Validation**: Ensures trades comply with risk limits
7. **💼 Portfolio Rebalancing**: Executes validated trades and updates ledger
8. **📈 Risk Reporting**: Displays VaR, sector exposure, and portfolio metrics

---

## ⚙️ Configuration

The system is fully configurable via `config/settings.yaml`:

```yaml
# Define your trading universe
tickers:
  - SPY
  - AAPL
  - MSFT
  - NVDA
  # ... 30 total tickers

# Model parameters
model:
  training_period: "3y"      # Historical data window
  retrain_daily: true        # Retrain on each run?
  threshold_buy: 0.55        # Min probability to buy (55%)
  threshold_sell: 0.45       # Max probability to sell (45%)

# Portfolio settings
portfolio:
  initial_cash: 10000.0      # Starting capital
  min_cash_buffer: 100.0     # Reserve cash amount

# Risk management (configured in code via RiskLimits)
# - max_position_pct: 0.15   # Max 15% per position
# - max_sector_pct: 0.40     # Max 40% in any sector
# - volatility_lookback: 30  # Days for volatility calc
```

---

## 📂 Project Structure

```
paper-trader/
├── main.py                      # Application entry point
├── src/
│   ├── data/
│   │   ├── loader.py           # Market data fetching (yfinance)
│   │   └── validator.py        # Data quality checks ✨ NEW
│   ├── features/
│   │   └── indicators.py       # Technical indicator generation
│   ├── models/
│   │   ├── trainer.py          # XGBoost training pipeline
│   │   └── predictor.py        # Inference engine
│   ├── trading/
│   │   ├── portfolio.py        # Position tracking & ledger
│   │   └── risk_manager.py     # Risk controls & sizing ✨ NEW
│   └── utils/
│       └── config.py           # YAML config loader
├── tests/                       # Unit test suite ✨ NEW
│   ├── test_risk_manager.py   # Risk management tests (14)
│   └── test_validator.py      # Data validation tests (13)
├── config/
│   └── settings.yaml           # System configuration
├── docs/
│   └── MANUAL.md               # Technical documentation
├── results/                     # Training metrics & reports
│   ├── metrics.txt
│   └── confusion_matrix.png
├── models/                      # Serialized model artifacts
│   └── xgb_model.joblib
├── ledger.csv                   # Transaction history
├── Makefile                     # Build automation
└── docker-compose.yml          # Container orchestration
```

---

## 📊 Example Output

```
--- 🤖 AI Paper Trader | Mode: TRADE | 2024-12-08 23:45:12 ---
Universe: ['SPY', 'AAPL', 'MSFT', 'NVDA', ...]

Fetching market data...
Fetching data for: ['SPY', 'AAPL', 'MSFT', ...]

--- 🔍 Validating Data Quality ---
=============================================================
DATA VALIDATION SUMMARY
=============================================================
Total Tickers: 30
✅ Valid: 28
❌ Invalid: 2
=============================================================

🧠 Training Model...
Model Accuracy: 0.5234
Model saved to models/xgb_model.joblib

--- 🔮 Generating Predictions ---
🟢 AAPL: BUY  (Prob: 0.6823)
🟢 MSFT: BUY  (Prob: 0.5912)
⚪️ SPY: HOLD (Prob: 0.5123)
🔴 NVDA: SELL (Prob: 0.3456)

--- 💼 Executing Trades ---

📊 Current Sector Exposure:
   Technology: 35.2%
   Financials: 25.8%
   Healthcare: 18.5%

💵 Available Cash for Buys: $4500.00

📈 BOUGHT 15 of AAPL at $150.00 | Size: $2250 (11.2% of portfolio), vol=18.20%, scalar=1.10
📈 BOUGHT 8 of MSFT at $280.00 | Size: $2240 (11.1% of portfolio), vol=22.45%, scalar=0.89
⚪️ NVDA: Skip - Sector limit reached: Technology already at 40.1%

💰 Total Portfolio Value: $10485.50

📈 Risk Metrics:
   1-Day VaR (95%): $245.32 (2.34% of portfolio)
   Largest Sector Exposure: Technology (38.2%)
```

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
make test

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=html

# Run specific test modules
pytest tests/test_risk_manager.py -v
pytest tests/test_validator.py -v
```

**Test Coverage**: 27 tests across risk management and data validation
- ✅ Position sizing with volatility adjustment
- ✅ Sector concentration enforcement
- ✅ VaR calculation
- ✅ Data quality checks (OHLC, outliers, missing values)

---

## 📖 Documentation

- **[User Manual](docs/MANUAL.md)**: Detailed technical documentation
- **[Implementation Plan](docs/implementation_plan.md)**: Quant expert analysis & improvement roadmap *(if available)*
- **[Walkthrough](docs/walkthrough.md)**: Phase 1 implementation details *(if available)*

---

## 🚧 Roadmap

### ✅ Phase 1: Critical Foundation (Complete)
- ✅ Risk management with volatility-adjusted sizing
- ✅ Data quality validation (10 checks)
- ✅ Sector diversification controls
- ✅ VaR monitoring
- ✅ Unit test coverage

### 🔜 Phase 2: Testing & Validation (Next)
- [ ] Backtesting framework with transaction costs
- [ ] Walk-forward analysis across market regimes
- [ ] Performance analytics (Sharpe, Sortino, max drawdown)
- [ ] Historical validation (3-5 years out-of-sample)

### 🔜 Phase 3: Model Improvements
- [ ] Fix data leakage in feature generation
- [ ] Time series cross-validation
- [ ] Regression target (return magnitude vs. binary classification)
- [ ] Enhanced features (volume, macro indicators, sentiment)

### 🔜 Phase 4: Production Readiness
- [ ] Advanced logging and monitoring
- [ ] Alerting system (email/Slack)
- [ ] SQLite ledger (replace CSV)
- [ ] Multi-strategy framework

---

## ⚠️ Disclaimer

> **IMPORTANT**: This software is for **educational and research purposes only**. It is **NOT financial advice**. Do not deploy with real capital until:
> 1. ✅ Backtesting shows positive Sharpe ratio over 3+ years
> 2. ✅ Data leakage issues are resolved (Phase 3)
> 3. ✅ Live paper trading validation for 30+ days with no critical bugs
>
> The authors are **not responsible** for any financial losses incurred.

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Backtesting infrastructure
- Alternative data integration (sentiment, options flow)
- Advanced ML models (LSTMs, transformers)
- Production deployment guides

---

**Built with**: Python 3.10 | XGBoost | pandas | yfinance | Docker
