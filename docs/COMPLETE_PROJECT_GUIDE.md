# Paper Trader AI - Complete Project Guide

> **Private Documentation** - Comprehensive guide to the entire system architecture.

**Last Updated**: December 2025  
**Transaction Costs**: 5 bps slippage on all trades

---

##  System Overview

Paper Trader AI is a **dual-portfolio algorithmic trading system** that:
- Runs **two independent strategies** (Momentum + ML) in parallel
- Uses **SQLite caching** to store 4M+ rows of market data
- Applies **realistic transaction costs** (5 basis points slippage)
- Deploys via **GitHub Actions** for automated daily trading
- Displays results on a **Streamlit dashboard**

### Live Dashboard
 [paper-trader-ai.streamlit.app](https://paper-trader-ai.streamlit.app/)

---

##  Strategy 1: Momentum (Primary)

### Theory
- Based on **Jegadeesh & Titman (1993)** academic research
- Stocks that performed well continue to outperform
- 12-month lookback, skip last month to avoid reversal

### Implementation
```python
# Monthly rebalance
1. Fetch 12-month returns for S&P 500
2. Skip last month (reversal effect)
3. Rank by momentum score
4. Buy top 10 stocks (equal weight)
5. Apply 5 bps slippage to all trades
```

### Performance (Oct 1 - Dec 19, 2025 with Transaction Costs)

| Metric | Value |
|--------|-------|
| **Initial Capital** | $10,000 |
| **Final Value** | $10,720 |
| **Total Return** | +7.20% |
| **vs SPY** | +4.10% excess |
| **Total Trades** | 50 |
| **Rebalance Freq** | Monthly |

---

##  Strategy 2: ML Ensemble

### Theory
- XGBoost regression predicting next-day returns
- Multi-horizon ensemble (1d, 5d, 20d)
- 15 technical indicator features

### Implementation
```python
# Daily rebalance
1. Generate 15 technical indicators
2. Predict returns using XGBoost ensemble
3. Rank by expected return
4. Buy top 10 predicted stocks
5. Apply 5 bps slippage to all trades
```

### Performance (Oct 1 - Dec 19, 2025 with Transaction Costs)

| Metric | Value |
|--------|-------|
| **Initial Capital** | $10,000 |
| **Final Value** | $10,158 |
| **Total Return** | +1.58% |
| **vs SPY** | -1.52% underperform |
| **Total Trades** | 526 |
| **Rebalance Freq** | Daily |

### ML Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | XGBoost Regressor |
| Horizons | 1d (50%), 5d (30%), 20d (20%) |
| Features | 15 technical indicators |
| Training | TimeSeriesSplit (5-fold) |

---

##  Transaction Costs

### Cost Model
All trades include **5 basis points (0.05%) slippage**:

```python
# BUY: Pay slightly more
execution_price = quote_price × 1.0005

# SELL: Receive slightly less  
execution_price = quote_price × 0.9995
```

### Impact Analysis
| Strategy | Trades | Est. Cost Impact |
|----------|--------|------------------|
| Momentum | 50 | ~$25 (0.25%) |
| ML | 526 | ~$260 (2.6%) |

**Insight**: High-turnover ML strategy suffers more from transaction costs.

---

##  Risk Management

### Portfolio-Level Controls

| Control | Threshold | Action |
|---------|-----------|--------|
| Warning | -15% drawdown | Reduce position sizes 50% |
| Halt | -20% drawdown | No new buys, sells only |
| Liquidate | -25% drawdown | Force sell 50% |

### Position-Level Controls

| Control | Value |
|---------|-------|
| Max position | 15% of portfolio |
| Max sector | 30% of portfolio |
| Stop-loss | 8% from entry |
| Slippage | 5 bps on all trades |

---

##  Data Infrastructure

### SQLite Cache

| Stat | Value |
|------|-------|
| Database | `data/market.db` |
| Tickers | 503 (S&P 500) |
| Total Rows | 4.3M+ |
| Date Range | 2010 - Present |

### Key Functions
```python
from src.data.cache import get_cache
from src.data.loader import fetch_data, update_cache

# Fetch with caching
data = fetch_data(['AAPL', 'GOOGL'], period='2y')

# Cache-only (for backtesting)
data = fetch_from_cache_only(['AAPL'], '2024-01-01', '2025-01-01')
```

---

##  Streamlit Dashboard

### Features
- **Portfolio Overview**: Momentum vs ML vs SPY cards
- **Performance Chart**: All 3 strategies with daily values
- **Holdings Tables**: Current positions per strategy
- **Trade History**: Recent trades with prices

### Data Flow
```
GitHub Actions → main.py → ledger_*.csv → dashboard/app.py
                         → portfolio_snapshot.json
```

---

##  File Structure

```
paper-trader/
├── main.py                              # Core orchestrator
├── config/
│   └── trading.yaml                     # Strategy configuration
├── data/
│   ├── market.db                        # SQLite cache
│   ├── portfolio_snapshot.json          # Dashboard data
│   ├── spy_benchmark.json               # SPY history
│   └── sp500_tickers.txt                # Universe
├── src/
│   ├── data/                            # Cache & loader
│   ├── features/                        # Technical indicators
│   ├── models/                          # XGBoost training & prediction
│   ├── trading/                         # Portfolio & risk management
│   └── backtesting/                     # Costs & performance
├── scripts/
│   ├── backtests/                       # Walk-forward backtests
│   ├── validation/                      # PIT backtests
│   └── utils/                           # Utility scripts
├── dashboard/
│   └── app.py                           # Streamlit application
├── models/
│   ├── xgb_ensemble.joblib              # Trained model
│   └── model_metadata.json              # Features & metrics
├── ledger_ml.csv                        # ML trade history
├── ledger_momentum.csv                  # Momentum trade history
└── docs/
    ├── MANUAL.md                        # Technical reference
    └── COMPLETE_PROJECT_GUIDE.md        # This file
```

---

##  CLI Reference

```bash
# Momentum strategy (isolated portfolio)
python main.py --strategy momentum --portfolio momentum

# ML strategy (isolated portfolio)
python main.py --strategy ml --portfolio ml

# Training only
python main.py --mode train --strategy ml

# Backtesting
python main.py --mode backtest --strategy momentum
```

---

##  Make Commands

```bash
make train      # Train ML model
make trade      # Execute daily trades
make backtest   # Run historical backtest
make docker-up  # Start Docker container
make clean      # Clean artifacts
```

---

##  GitHub Actions

### Scheduled Workflows

| Workflow | Schedule | Action |
|----------|----------|--------|
| `momentum.yml` | 1st of month, 2pm PT | Monthly momentum rebalance |
| `ml_strategy.yml` | Daily, 1pm PT | Daily ML trades |
| `update_cache.yml` | Daily, 6am PT | Refresh market data |

### Secrets Required
- `GH_TOKEN` - For pushing ledger updates

---

##  Assumptions & Limitations

1. **Paper trading only** - No real broker integration
2. **End-of-day data** - Trades execute at next open
3. **5 bps slippage** - Conservative estimate for liquid stocks
4. **No dividends** - Returns based on price only
5. **S&P 500 universe** - No small caps or international

---

##  Future Improvements

- [ ] Real broker integration (Alpaca, IBKR)
- [ ] Intraday trading capability
- [ ] Sector rotation strategy
- [ ] International market support
- [ ] Mobile app for monitoring

---

*Built by Prabuddha Tamhane • December 2025*
