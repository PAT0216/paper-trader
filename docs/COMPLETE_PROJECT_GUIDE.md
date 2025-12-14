# Paper Trader AI - Complete Project Guide

> **Private Documentation** - Comprehensive guide to the entire system architecture.

---

## 🏗️ System Overview

Paper Trader AI is a **dual-portfolio algorithmic trading system** that:
1. Runs **two independent strategies** simultaneously (Momentum & ML)
2. Maintains **separate ledgers** for each strategy
3. Provides a **Streamlit dashboard** for real-time comparison
4. Automates trading via **GitHub Actions**

---

## 🎯 Dual Portfolio Architecture

### Portfolio Isolation

| Portfolio | Strategy | Ledger | Workflow |
|-----------|----------|--------|----------|
| **Momentum** | 12-month momentum + 15% SL | `ledger_momentum.csv` | `momentum_trade.yml` |
| **ML** | XGBoost ensemble | `ledger_ml.csv` | `ml_trade.yml` |

### Why Dual Portfolios?

1. **Fair comparison**: Same capital, same universe, different strategies
2. **No interference**: Momentum trades don't affect ML portfolio
3. **Performance tracking**: Clear attribution of returns
4. **Risk isolation**: One strategy failing doesn't affect the other

---

## 📈 Strategy 1: Momentum (Primary)

### Theory
- Based on Jegadeesh & Titman (1993) academic research
- Fama-French factor: stocks that went up keep going up
- 12-month lookback (skip last month to avoid reversal)

### Implementation
```python
# Monthly rebalance
1. Fetch 12-month returns for S&P 500
2. Skip last month (reversal effect)
3. Rank by momentum score
4. Buy top 10 stocks (equal weight)
5. Apply 15% stop-loss daily
```

### Performance
| Metric | Value |
|--------|-------|
| CAGR | 37.4% |
| Sharpe | 1.97 |
| 2022 Return | +19% |

---

## 🤖 Strategy 2: ML (Experimental)

### Theory
- XGBoost predicts next-day returns
- 15 technical features (momentum, volume, volatility)
- Multi-horizon ensemble (1d, 5d, 20d)

### Implementation
```python
# Daily trading
1. Generate 15 features from OHLCV
2. Predict expected return for each stock
3. Rank by prediction
4. Buy top 10%, sell bottom 10%
5. Apply risk limits
```

### Performance
| Metric | Value |
|--------|-------|
| CAGR | 27.0% |
| Sharpe | 1.04 |
| Win Rate | 67% |

---

## 📊 Feature Engineering (ML Strategy)

### Momentum Features (4)
| Feature | Calculation |
|---------|-------------|
| RSI | 14-day relative strength |
| MACD | 12/26 EMA crossover |
| MACD_signal | 9-day EMA of MACD |
| MACD_hist | MACD - Signal |

### Trend Features (5)
| Feature | Calculation |
|---------|-------------|
| BB_Width | Bollinger band width |
| Dist_SMA50 | Distance from 50-day SMA |
| Dist_SMA200 | Distance from 200-day SMA |
| Return_1d | Yesterday's return |
| Return_5d | 5-day return |

### Volume Features (3)
| Feature | Calculation |
|---------|-------------|
| OBV_Momentum | On-balance volume change |
| Volume_Ratio | 5d/20d volume ratio |
| VWAP_Dev | Deviation from VWAP |

### Volatility Features (3)
| Feature | Calculation |
|---------|-------------|
| ATR_Pct | ATR as % of price |
| BB_PctB | Position in Bollinger bands |
| Vol_Ratio | 10d/60d volatility ratio |

---

## 🛡️ Risk Management

### Portfolio-Level Controls
| Level | Trigger | Action |
|-------|---------|--------|
| Warning | -15% drawdown | Reduce position sizes 50% |
| Halt | -20% drawdown | No new buys, sells only |
| Liquidate | -25% drawdown | Force sell 50% |

### Position-Level Controls
| Control | Value |
|---------|-------|
| Max position | 15% of portfolio |
| Max sector | 30% of portfolio |
| Stop-loss | 15% from entry |
| Slippage | 10 bps assumed |

---

## 📦 Data Infrastructure

### SQLite Cache
| Stat | Value |
|------|-------|
| Database | `data/market.db` |
| Tickers | 503 (S&P 500) |
| Total Rows | 4.3M |
| Date Range | 1962-2024 |

### Cache Strategy
```
1. Check cache first
2. If cached and up-to-date → use cache (instant)
3. If cached but stale → fetch only new bars
4. If not cached → fetch full history
```

### Momentum Cache-Only Mode
For speed, momentum strategy uses **cache-only** mode:
```python
if strategy == "momentum":
    data = fetch_from_cache_only(tickers)  # No API calls
```

---

## 🔄 GitHub Actions Workflows

### Momentum Strategy Trade
```yaml
schedule: '30 21 1-3 * *'  # 1st-3rd of month, 9:30 PM UTC
```
- Runs on days 1,2,3 to catch first trading day
- Uses `--portfolio momentum --strategy momentum`
- Writes to `ledger_momentum.csv`

### ML Strategy Trade
```yaml
schedule: '30 21 * * 1-5'  # Mon-Fri, 9:30 PM UTC
```
- Runs daily on weekdays
- Uses `--portfolio ml --strategy ml`
- Writes to `ledger_ml.csv`

---

## 📊 Streamlit Dashboard

### Features
- **Portfolio comparison**: Side-by-side metrics
- **Performance chart**: Cumulative returns
- **Holdings table**: Current positions
- **Trade history**: Recent trades

### Run Locally
```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧪 Testing

### Test Suites
| Suite | Tests | Purpose |
|-------|-------|---------|
| `test_dual_portfolio.py` | 7 | Portfolio isolation |
| `test_momentum_no_bias.py` | 8 | Strategy validation |

### Run Tests
```bash
python -m pytest tests/ -v
```

---

## 📁 File Structure

```
paper-trader/
├── main.py                         # Core orchestrator
├── config/
│   ├── settings.yaml               # General settings
│   └── momentum_config.yaml        # Momentum strategy config
├── src/
│   ├── strategies/
│   │   └── momentum_strategy.py    # Momentum implementation
│   ├── models/
│   │   ├── trainer.py              # XGBoost training
│   │   └── predictor.py            # Ensemble predictions
│   ├── trading/
│   │   └── portfolio.py            # Portfolio management
│   ├── analytics/
│   │   └── portfolio_comparison.py # Comparison metrics
│   └── data/
│       ├── loader.py               # Data fetching
│       └── cache.py                # SQLite cache
├── dashboard/
│   ├── app.py                      # Streamlit app
│   └── requirements.txt
├── tests/
│   ├── test_dual_portfolio.py
│   └── test_momentum_no_bias.py
├── .github/workflows/
│   ├── momentum_trade.yml          # Monthly momentum
│   └── ml_trade.yml                # Daily ML
├── ledger_momentum.csv             # Momentum trades
├── ledger_ml.csv                   # ML trades
└── data/
    ├── market.db                   # SQLite cache
    └── sp500_tickers.txt           # Ticker list
```

---

## 🎯 CLI Reference

```bash
# Momentum strategy (isolated portfolio)
python main.py --strategy momentum --portfolio momentum

# ML strategy (isolated portfolio)
python main.py --strategy ml --portfolio ml

# Default (backward compatible)
python main.py  # Uses momentum strategy, default portfolio
```

### Arguments
| Arg | Options | Default |
|-----|---------|---------|
| `--strategy` | `momentum`, `ml` | `momentum` |
| `--portfolio` | any string | `default` |
| `--mode` | `trade`, `train`, `backtest` | `trade` |

---

## ⚠️ Known Limitations

1. **Survivorship bias**: Using current S&P 500 for historical backtest
2. **Paper trading only**: No real money integration
3. **US market only**: No international support
4. **Daily resolution**: No intraday trading

---

## 🚀 Future Improvements

- [ ] Add more strategies (mean reversion, sector rotation)
- [ ] International market support
- [ ] Real broker integration (Alpaca, IBKR)
- [ ] Intraday trading capability
- [ ] Deploy dashboard to Streamlit Cloud
