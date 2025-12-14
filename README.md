# Paper Trader AI

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.10-blue)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![Strategy](https://img.shields.io/badge/strategy-dual--portfolio-orange)

**Paper Trader AI** is a production-grade algorithmic trading system featuring a **Dual Portfolio Architecture** that runs two independent strategies simultaneously for performance comparison.

## 🎯 Dual Portfolio System

| Portfolio | Strategy | Schedule | Ledger |
|-----------|----------|----------|--------|
| **Momentum** | 12-month momentum + 15% stop-loss | Monthly (1st trading day) | `ledger_momentum.csv` |
| **ML** | XGBoost ensemble predictions | Daily (weekdays) | `ledger_ml.csv` |

### Performance Summary

| Metric | Momentum | ML | S&P 500 |
|--------|----------|-----|---------|
| **CAGR** | 37.4% | 27.0% | 13.2% |
| **Sharpe** | 1.97 | 1.04 | 0.87 |
| **2022 Return** | +19% | -18% | -18.6% |

---

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/PAT0216/paper-trader.git
cd paper-trader

# Run momentum strategy
python main.py --strategy momentum --portfolio momentum

# Run ML strategy
python main.py --strategy ml --portfolio ml

# Launch comparison dashboard
cd dashboard && streamlit run app.py
```

---

## 📋 Features

### 📈 Momentum Strategy (Primary)
- **12-month momentum factor** with Fama-French methodology
- **15% daily stop-loss** for downside protection
- **Monthly rebalancing** on first trading day
- Top 10 stocks from S&P 500 universe

### 🤖 ML Strategy (Experimental)
- **XGBoost Regressor** with 15 technical features
- **Multi-horizon ensemble** (1-day, 5-day, 20-day predictions)
- **VIX regime detection** for crisis protection
- Daily trading on weekdays

### 🛡️ Risk Management
- **Position limits**: Max 15% per stock, 30% per sector
- **Portfolio drawdown control**: Warning at -15%, halt at -20%, liquidate at -25%
- **Volatility-adjusted sizing**: Inverse weighting by 30-day volatility

### 📦 Infrastructure
- **SQLite data cache**: 4.3M rows, 503 S&P 500 tickers
- **GitHub Actions**: Automated trading workflows
- **Streamlit Dashboard**: Live portfolio comparison

---

## 📊 GitHub Actions Workflows

| Workflow | Purpose | Schedule |
|----------|---------|----------|
| **Momentum Strategy Trade** | Monthly momentum rebalance | 1st-3rd of month, 9:30 PM UTC |
| **ML Strategy Trade** | Daily ML predictions | Mon-Fri, 9:30 PM UTC |

Run manually: **Actions** → Select workflow → **Run workflow**

---

## 📁 Project Structure

```
paper-trader/
├── main.py                     # Core trading logic
├── config/
│   └── momentum_config.yaml    # Strategy configuration
├── src/
│   ├── strategies/             # Momentum strategy
│   ├── models/                 # ML models (XGBoost)
│   ├── trading/                # Portfolio management
│   ├── analytics/              # Portfolio comparison
│   └── data/                   # Data loading & caching
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── tests/                      # Test suite (15 tests)
├── docs/                       # Documentation
└── .github/workflows/          # CI/CD automation
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [COMPLETE_PROJECT_GUIDE.md](docs/COMPLETE_PROJECT_GUIDE.md) | Full system architecture |
| [MOMENTUM_STRATEGY.md](docs/MOMENTUM_STRATEGY.md) | Momentum strategy details |
| [MANUAL.md](docs/MANUAL.md) | Technical reference |

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_dual_portfolio.py -v   # Dual portfolio (7 tests)
python -m pytest tests/test_momentum_no_bias.py -v  # Momentum validation (8 tests)
```

---

## ⚠️ Disclaimer

This is a **paper trading** system for educational purposes. Past performance does not guarantee future results. Do not trade real money based on this system.

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.
