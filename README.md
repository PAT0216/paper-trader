# Paper Trader AI

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.10-blue)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![Strategy](https://img.shields.io/badge/strategy-momentum-orange)

**Paper Trader AI** is a production-grade algorithmic trading system for the US Equity Market. The **default strategy** is a **12-month momentum factor** with **15% daily stop-loss**, achieving **37.4% CAGR** and **1.97 Sharpe** in backtesting (2016-2023).

| Metric | Strategy | S&P 500 |
|--------|----------|---------|
| **CAGR** | 37.4% | 13.2% |
| **Sharpe** | 1.97 | 0.87 |
| **2022 Return** | +19% | -18.6% |

---

## 📋 Key Features

### 🧠 **Predictive Intelligence**
- **XGBoost Regressor** predicts expected returns with 5-fold TimeSeriesSplit
- **15 Technical Features** across momentum, trend, volume, and volatility
- **Dynamic Feature Selection**: Auto-filters features with <3% importance *(New in Phase 3.5)*
- **Anti-Leakage Pipeline**: Proper separation of features and target

### 🎯 **Multi-Horizon Ensemble** *(New in Phase 3.6)*
- **3 XGBoost Models**: 1-day (50%), 5-day (30%), 20-day (20%) predictions
- **Blended Signals**: More stable than single-horizon predictions
- **Horizon Weights**: Configurable for different trading styles

### 🛡️ **VIX-Based Regime Detection** *(New in Phase 3.6)*
- **Market Regimes**: NORMAL (VIX<25), ELEVATED (25-35), CRISIS (>35)
- **Defensive Mode**: Automatically reduces exposure during volatility spikes
- **Position Multipliers**: 100% → 50% → 0% based on regime

### 📦 **SQLite Data Caching** *(New in Phase 4)*
- **Local Cache**: 4.3M rows, 503 S&P 500 tickers
- **Incremental Updates**: Only fetch new bars after initial load
- **No Rate Limits**: Avoid yfinance throttling with cached data
- **GitHub Actions Cache**: Persists across workflow runs (169MB, restored in ~3s)

### 🌐 **S&P 500 Universe** *(New in Phase 4)*
- **Dynamic Universe**: Fetches current S&P 500 from Wikipedia
- **Fallback**: Mega-cap tier if fetch fails
- **Configuration**: `universe.type: sp500` or `config`

### 📊 **Enhanced Feature Engineering** *(Phase 3.5)*
- **Volume Indicators**: VWAP Deviation, OBV Momentum, Volume Ratio
- **Volatility Features**: ATR, Bollinger %B, Volatility Ratio
- **Momentum**: RSI, MACD (line, signal, histogram)
- **Trend**: SMA distances (50/200), Bollinger Width, Returns (1d/5d)

### 🛡️ **Professional Risk Management** *(Phase 1)*
- **Volatility-Adjusted Position Sizing**: Inverse weighting by 30-day volatility
- **Portfolio Constraints**: Max 15% per position, 30% per sector
- **Value at Risk (VaR)**: Daily portfolio risk monitoring at 95% confidence
- **Pre-Trade Validation**: All trades checked against risk limits before execution

### 🎯 **Advanced Risk Controls** *(New in Phase 7)*
- **15% Position Stop-Loss**: A/B tested on 75 random S&P 500 stocks
- **Portfolio Drawdown Control**: -15% warning, -20% halt, -25% emergency liquidation
- **Walk-Forward A/B Testing**: `run_unbiased_comparison.py` for strategy validation
- **Note**: Backtests show stop-loss outperforms no stop-loss; live results may vary (see disclaimers)

### ✅ **Data Quality Assurance** *(Phase 1)*
- **10 Validation Checks**: OHLC integrity, outlier detection, missing values, data freshness
- **Automatic Rejection**: Invalid tickers filtered before model training
- **Comprehensive Logging**: Detailed validation reports for every data fetch

### 📈 **Backtesting Framework** *(Phase 2)*
- **Event-Driven Engine**: 10 years of historical data (2015-2024)
- **ML Predictor Mode**: `--ml` flag uses XGBoost instead of SMA
- **Transaction Costs**: Realistic slippage (5 bps) and market impact modeling
- **Professional Quant Metrics**: Sharpe, Sortino, Calmar, VaR, CVaR, Alpha, Beta
- **Regime Analysis**: Performance split by bull/bear/crisis/sideways markets
- **ML Backtest (2015-2024)**: 748% return, 1.04 Sharpe, 67% win rate, 27% CAGR

### 🏗️ **Production Infrastructure**
- **Containerized Architecture**: Fully Dockerized with `miniconda3`
- **Multi-Asset Support**: Manages 500+ S&P 500 stocks
- **Automated Execution**: GitHub Actions scheduled daily at market close
- **Comprehensive Testing**: 55 unit tests covering risk, validation, backtesting, and ML pipeline

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

# Run backtest (2017-2024)
make backtest

# Quick backtest (2023-2024)
make backtest-quick
```

### What Happens During Execution?

1. **📊 Data Ingestion**: Fetches 3 years of OHLCV data for 30+ tickers
2. **🔍 Data Validation**: Runs 10 quality checks, rejects invalid tickers
3. **🧠 Model Training**: XGBoost trains on technical indicators (or loads existing model)
4. **🔮 Prediction**: Generates expected return predictions for each ticker
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
  # Phase 3: Now uses expected return thresholds
  # BUY threshold: +0.5% expected return
  # SELL threshold: -0.5% expected return

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
├── run_backtest.py              # Backtesting CLI runner
├── run_walkforward.py           # Walk-forward validation (Phase 5)
├── run_hyperopt.py              # Hyperparameter optimization (Phase 5)
├── src/
│   ├── backtesting/             # Backtesting framework
│   │   ├── backtester.py       # Event-driven engine
│   │   ├── performance.py      # Quant metrics (Sharpe, VaR, etc.)
│   │   └── costs.py            # Transaction cost modeling
│   ├── data/
│   │   ├── loader.py           # Market data fetching (yfinance)
│   │   ├── cache.py            # SQLite caching (Phase 4)
│   │   ├── universe.py         # S&P 500 universe (Phase 4)
│   │   ├── macro.py            # VIX and macro data
│   │   └── validator.py        # Data quality checks
│   ├── features/
│   │   └── indicators.py       # Technical indicator generation (15 features)
│   ├── models/
│   │   ├── trainer.py          # XGBoost training pipeline
│   │   └── predictor.py        # Inference engine (single + ensemble)
│   ├── trading/
│   │   ├── portfolio.py        # Position tracking & ledger
│   │   ├── risk_manager.py     # Risk controls & sizing
│   │   └── regime.py           # VIX-based regime detection (Phase 3.6)
│   └── utils/
│       └── config.py           # YAML config loader
├── tests/                       # Unit test suite (55 tests)
│   ├── test_ml_pipeline.py      # ML pipeline tests (12)
│   ├── test_backtester.py       # Backtesting tests (16)
│   ├── test_risk_manager.py     # Risk management tests (14)
│   └── test_validator.py        # Data validation tests (13)
├── config/
│   ├── settings.yaml           # Trading configuration
│   └── backtest_settings.yaml  # Backtest configuration
├── data/
│   └── market.db               # SQLite cache (4.3M rows, 503 tickers)
├── docs/
│   └── MANUAL.md               # Technical documentation
├── results/                     # Training & backtest reports
│   ├── metrics.txt             # CV metrics (RMSE, MAE, R², Dir.Acc)
│   ├── feature_importance.png  # Feature importance chart
│   ├── selected_features.txt   # Dynamically selected features
│   ├── backtest_summary.txt    # Backtest performance summary
│   ├── backtest_trades.csv     # Trade log
│   └── hyperopt_results.csv    # Hyperparameter optimization (Phase 5)
├── models/                      # Serialized model artifacts
│   ├── xgb_model.joblib        # Single-horizon model
│   └── xgb_ensemble.joblib     # Multi-horizon ensemble (Phase 3.6)
├── ledger.csv                   # Transaction history
├── Makefile                     # Build automation
├── CHANGELOG.md                 # Version history
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

**Test Coverage**: 55 tests across ML pipeline, backtesting, risk management, and data validation
- ✅ ML pipeline anti-leakage and TimeSeriesSplit
- ✅ Backtesting engine and performance metrics
- ✅ Position sizing with volatility adjustment
- ✅ VaR calculation and transaction costs
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

### ✅ Phase 2: Testing & Validation (Complete)
- ✅ Backtesting framework with transaction costs
- ✅ Walk-forward analysis across market regimes (2017-2024)
- ✅ Performance analytics (Sharpe, Sortino, Calmar, VaR, Alpha, Beta)
- ✅ Regime-based performance analysis (bull/bear/crisis)
- ✅ 16 additional unit tests

### ✅ Phase 3: Model Improvements (Complete)
- ✅ Fixed data leakage (separate features from target)
- ✅ 5-fold TimeSeriesSplit cross-validation
- ✅ Regression target (predict return magnitude)
- ✅ 12 new ML pipeline tests

### ✅ Phase 4: Data Infrastructure (Complete)
- ✅ SQLite data cache (4.3M rows, 503 tickers)
- ✅ Incremental data fetching (only new bars)
- ✅ S&P 500 dynamic universe from Wikipedia
- ✅ Enhanced features (volume, volatility, VWAP)

### ✅ Phase 5: Walk-Forward Validation (Complete)
- ✅ True out-of-sample testing (train before each test year)
- ✅ Walk-forward results: 630% return vs SPY 234% (2015-2024)
- ✅ Hyperparameter optimization with overfitting checks
- ✅ Next-day Open execution (no look-ahead bias)

### ✅ Phase 6: Deployment & Reliability (Complete)
- ✅ GitHub Actions Cache for market data persistence
- ✅ Scheduled daily cron job (1 PM PST / 9 PM UTC)
- ✅ Automated commits with ledger updates

### ✅ Phase 7: Quant Risk Enhancements (Complete)
- [x] **Portfolio Drawdown Control**: -15% warning, -20% halt, -25% liquidation
- [x] **Position-Level Stop-Loss**: 15% threshold (A/B tested: outperforms no stop-loss)
- [x] **Walk-Forward A/B Testing**: `run_unbiased_comparison.py` for unbiased validation
- [x] **Ranking Metrics**: Spearman IC and Top-10% accuracy tracking
- [x] **Realistic Disclaimers**: Documented survivorship bias and backtest limitations

### ✅ Phase 8: Comprehensive Testing & Validation (Complete)
- [x] **Pytest Suite**: 48 tests passed, 2 skipped
- [x] **Walk-Forward (2023-2024)**: 80.2% return, 34.2% CAGR (beat SPY 58%)
- [x] **Backtest (2023-2024)**: 59.4% return, 1.63 Sharpe, 10.6% max drawdown
- [x] **Bug Fixes**: Fixed 4 API compatibility issues in walkforward/backtest scripts


---

## ⚠️ Important Disclaimers

### Survivorship Bias
> **Note**: Backtests use the **current** S&P 500 constituent list applied to historical periods (2015-2024). Companies that were removed due to bankruptcy, delisting, or acquisition are not included. This may overstate historical returns by an estimated **2-4% annually**.

### Return Concentration
> **Note**: Trade analysis shows **84.1% of total P&L** comes from just 10 trades. Returns are driven by holding winners long (avg 496 days) vs cutting losers early (avg 131 days). This concentration means live performance may vary significantly.

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
