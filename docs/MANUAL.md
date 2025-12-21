# Paper Trader AI - Technical Manual

> **Complete Technical Reference** - All functions, classes, and modules documented

**Last Updated**: December 2025  
**Version**: 2.0 (with Transaction Costs)

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Data Pipeline](#data-pipeline)
3. [Feature Engineering](#feature-engineering)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
5. [Prediction & Signals](#prediction--signals)
6. [Portfolio Management](#portfolio-management)
7. [Risk Management](#risk-management)
8. [Transaction Costs](#transaction-costs)
9. [Backtesting Framework](#backtesting-framework)
10. [Dashboard](#dashboard)
11. [CLI Reference](#cli-reference)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PAPER TRADER AI                             │
│                    Dual-Portfolio Trading System                    │
└─────────────────────────────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        ▼                                                   ▼
┌───────────────────┐                           ┌───────────────────┐
│  MOMENTUM STRATEGY │                           │   ML STRATEGY     │
│  • 12-1 Momentum   │                           │  • XGBoost Ensemble│
│  • Monthly Rebal   │                           │  • Daily Rebalance │
│  • 10 Positions    │                           │  • 10 Positions    │
│  ledger_momentum.csv│                           │  ledger_ml.csv     │
└───────────────────┘                           └───────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   SHARED INFRASTRUCTURE  │
                    │  • SQLite Cache (market.db)│
                    │  • Risk Manager           │
                    │  • Transaction Cost Model │
                    │  • Streamlit Dashboard    │
                    └─────────────────────────┘
```

### Core Entry Point: `main.py`

The main orchestrator for all trading operations.

#### `main()`
**Purpose**: Entry point that routes to train, trade, or backtest modes.

**Arguments** (CLI):
| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--mode` | train, trade, backtest | trade | Operation mode |
| `--strategy` | momentum, ml | momentum | Which strategy to use |
| `--portfolio` | momentum, ml | Default | Which ledger to use |

**Flow**:
1. Load configuration from `config/trading.yaml`
2. Initialize Portfolio with specified ledger
3. Fetch market data via cache-first loading
4. Execute strategy-specific logic
5. Apply transaction costs (5 bps slippage)
6. Record trades to CSV ledger

#### `calculate_momentum_12_1(df)`
**Purpose**: Calculate Fama-French 12-1 month momentum factor.

**Parameters**:
- `df`: DataFrame with 'Close' column, minimum 252 trading days

**Returns**: Float momentum score (typically -0.5 to 3.0)

**Formula**:
```python
momentum = (price_12_months_ago / price_1_month_ago) - 1
```

---

## Strategy Architecture (NEW)

The project uses a **modular strategy pattern** allowing easy addition of new strategies.

### Class: `BaseStrategy` (`src/strategies/base.py`)

Abstract base class that all strategies must implement:

| Method | Returns | Description |
|--------|---------|-------------|
| `get_name()` | str | Strategy identifier (e.g., 'momentum') |
| `get_display_name()` | str | Human-readable name |
| `needs_training()` | bool | Whether ML training required |
| `rank_universe(data_dict)` | Dict[str, float] | Score all tickers |
| `generate_signals(data_dict)` | Dict[str, str] | Generate BUY/SELL/HOLD |
| `get_ledger_filename()` | str | Ledger file path |

### Registry: `src/strategies/registry.py`

Factory pattern for dynamic strategy loading:

```python
from src.strategies import get_strategy, list_strategies

# List available strategies
list_strategies()  # ['momentum', 'ml']

# Load strategy by name
strategy = get_strategy("momentum")
signals = strategy.generate_signals(data_dict)
```

### Adding a New Strategy

```python
# 1. Create src/strategies/my_strategy.py
class MyStrategy(BaseStrategy):
    def get_name(self): return "my_strategy"
    def needs_training(self): return False
    def rank_universe(self, data_dict): ...
    def generate_signals(self, data_dict): ...

# 2. Register in src/strategies/registry.py
from .my_strategy import MyStrategy
STRATEGIES["my_strategy"] = MyStrategy

# 3. Create workflow (optional)
# .github/workflows/my_strategy.yml

# Done! No main.py changes needed.
```

---

## Data Pipeline

### Module: `src/data/cache.py`

SQLite-based caching system to avoid API rate limits and enable fast backtesting.

#### Class: `DataCache`

**Database**: `data/market.db`

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__init__` | `db_path: str = DB_PATH` | - | Initialize database connection |
| `get_price_data` | `ticker, start_date=None, end_date=None` | DataFrame | Retrieve OHLCV data |
| `update_price_data` | `ticker, df` | - | Store new price data (INSERT OR REPLACE) |
| `get_last_date` | `ticker` | str or None | Get most recent cached date |
| `get_first_date` | `ticker` | str or None | Get earliest cached date |
| `get_cached_tickers` | - | List[str] | All tickers in cache |
| `get_cache_stats` | - | DataFrame | Stats for all cached tickers |
| `get_macro_data` | `series, start_date, end_date` | DataFrame | Fetch VIX, yields, etc. |
| `update_macro_data` | `series, df` | - | Store macro data |
| `clear_ticker` | `ticker` | - | Remove all data for a ticker |
| `clear_all` | - | - | Clear entire cache |
| `vacuum` | - | - | Optimize database file size |

**Example Usage**:
```python
from src.data.cache import DataCache, get_cache

cache = get_cache()

# Get data
df = cache.get_price_data('AAPL', '2024-01-01', '2025-01-01')

# Check freshness
last_date = cache.get_last_date('AAPL')  # '2025-12-19'

# Store new data  
cache.update_price_data('AAPL', new_df)
```

#### `get_cache()`
**Purpose**: Singleton accessor for default cache instance.
**Returns**: `DataCache` instance

---

### Module: `src/data/loader.py`

Smart data loading with cache-first strategy.

#### `fetch_data(tickers, period, interval, use_cache)`

**Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tickers` | List[str] | - | Stock symbols |
| `period` | str | "2y" | Historical period for initial fetch |
| `interval` | str | "1d" | Data granularity |
| `use_cache` | bool | True | Whether to use SQLite cache |

**Returns**: `Dict[str, pd.DataFrame]`

**Logic**:
1. Check SQLite cache first
2. If cached: fetch only new bars (incremental)
3. If not cached: fetch full history, then cache
4. Handle rate limits with exponential backoff

#### `fetch_from_cache_only(tickers, start_date, end_date)`

**Purpose**: Load data exclusively from cache (no API calls).  
**Use Case**: Backtesting without hitting rate limits.

**Returns**: `Dict[str, pd.DataFrame]`

#### `update_cache(tickers, period)`

**Purpose**: Refresh cache for all tickers.  
**Prints**: Cache statistics after update.

---

### Module: `src/data/validator.py`

#### Class: `DataValidator`

**Purpose**: Ensures data quality before model training.

| Validation Check | Threshold | Action on Fail |
|-----------------|-----------|----------------|
| Empty DataFrame | N/A | Critical Error |
| Missing OHLC Columns | Required | Critical Error |
| Data Freshness | < 48 hours | Warning |
| Missing Values | < 5% | Error if exceeded |
| Zero/Negative Prices | 0 allowed | Error |
| Duplicate Dates | 0 allowed | Error |
| Sufficient History | ≥ 200 days | Warning |

---

## Feature Engineering

### Module: `src/features/indicators.py`

Technical indicator computation with **15 features** across 4 categories.

#### Indicator Functions

**Momentum Indicators (4 features)**:

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `compute_rsi` | `series, period=14` | Series | Relative Strength Index (0-100) |
| `compute_macd` | `series, fast=12, slow=26, signal=9` | Tuple(macd, signal) | MACD and signal line |

**Volume Indicators (3 features)**:

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `compute_obv` | `close, volume` | Series | On-Balance Volume |
| `compute_obv_momentum` | `close, volume, period=10` | Series | OBV rate of change |
| `compute_volume_ratio` | `volume, short=5, long=20` | Series | Short/long volume ratio |
| `compute_vwap_deviation` | `close, volume, period=20` | Series | Price deviation from VWAP |

**Volatility Indicators (4 features)**:

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `compute_atr` | `high, low, close, period=14` | Series | Average True Range |
| `compute_bollinger_bands` | `series, window=20, num_std=2` | Tuple(upper, lower) | BB bands |
| `compute_bollinger_pctb` | `close, window=20, num_std=2` | Series | Position in BB (0-1) |
| `compute_volatility_ratio` | `close, short=10, long=60` | Series | Vol expansion/contraction |

#### `generate_features(df, include_target=False)`

**Purpose**: Generate all 15 technical indicators from OHLCV data.

**Parameters**:
- `df`: DataFrame with Open, High, Low, Close, Volume columns
- `include_target`: If True, also create target column (training only)

**Returns**: DataFrame with feature columns added, NaN rows dropped.

**Features Generated**:
```python
FEATURE_COLUMNS = [
    # Momentum (4)
    'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
    # Trend (4)
    'BB_Width', 'Dist_SMA50', 'Dist_SMA200', 'Return_1d', 'Return_5d',
    # Volume (3)
    'OBV_Momentum', 'Volume_Ratio', 'VWAP_Dev',
    # Volatility (4)
    'ATR_Pct', 'BB_PctB', 'Vol_Ratio'
]
```

#### `create_target(df, target_type, horizon)`

**Purpose**: Create target variable for ML training.

>  **IMPORTANT**: Only call in training pipeline AFTER train/test split to prevent look-ahead bias.

**Parameters**:
- `target_type`: 'regression' (return %) or 'classification' (up/down)
- `horizon`: Days ahead to predict (default: 1)

---

## Machine Learning Pipeline

### Module: `src/models/trainer.py`

XGBoost training with time-series cross-validation.

#### `select_features_better_than_noise(X, y, feature_names, n_noise=5)`

**Purpose**: Automatic feature selection using random noise baseline.

**Method**:
1. Add N random noise features to data
2. Train quick XGBoost model
3. Rank all features by importance
4. Keep only features with importance > max(noise importance)

**Returns**: List of selected feature names

#### `train_model(data_dict, n_splits=5, save_model=True)`

**Purpose**: Train single-horizon XGBoost regressor.

**Anti-Leakage Measures**:
1. Features generated WITHOUT target (no look-ahead)
2. Target added AFTER feature generation
3. TimeSeriesSplit for walk-forward validation

**Saves**:
- `models/xgb_model.joblib` - Trained model
- `models/model_metadata.json` - Feature list and metrics

#### `train_ensemble(data_dict, n_splits=5, save_model=True)`

**Purpose**: Train multi-horizon ensemble model.

**Horizons**:
| Horizon | Weight | Purpose |
|---------|--------|---------|
| 1 day | 50% | Short-term responsiveness |
| 5 days | 30% | Weekly trend |
| 20 days | 20% | Monthly trend |

**Saves**: `models/xgb_ensemble.joblib`

#### `evaluate_model(model, test_df)`

**Purpose**: Evaluate model on held-out test data.

**Returns**: Dict with RMSE, MAE, R², directional accuracy

---

## Prediction & Signals

### Module: `src/models/predictor.py`

#### Class: `Predictor`

Single-horizon prediction using trained XGBoost model.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__init__` | - | - | Load model from disk |
| `predict` | `df` | float | Expected next-day return |
| `predict_with_signal` | `df, buy_threshold=0.005, sell_threshold=-0.005` | Tuple(signal, return, confidence) | Signal generation |
| `predict_batch` | `data_dict` | Dict[str, float] | Batch prediction |

**Signal Thresholds**:
- **BUY**: Expected return > +0.5%
- **SELL**: Expected return < -0.5%
- **HOLD**: In between

#### Class: `EnsemblePredictor`

Multi-horizon ensemble for more stable predictions.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__init__` | - | - | Load ensemble from disk |
| `predict` | `df` | float | Weighted average return |
| `predict_with_regime` | `df, vix_value` | Dict | Prediction with regime adjustments |
| `predict_batch` | `data_dict, vix_value` | Dict[str, Dict] | Batch with regime awareness |

**Ensemble Blending**:
```python
prediction = (0.5 * pred_1d) + (0.3 * pred_5d/5) + (0.2 * pred_20d/20)
```

---

## Portfolio Management

### Module: `src/trading/portfolio.py`

#### Class: `Portfolio`

Manages trade recording and portfolio state.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__init__` | `portfolio_id="default", start_cash=100000` | - | Initialize with ledger file |
| `get_last_balance` | - | float | Current cash balance |
| `get_holdings` | - | Dict[str, int] | Current positions |
| `get_entry_prices` | - | Dict[str, float] | Average entry price per holding |
| `get_portfolio_value` | `current_prices` | float | Total value (cash + holdings) |
| `get_positions` | - | Dict | Detailed position info |
| `check_stop_losses` | `current_prices, stop_loss_pct=0.08` | List[Tuple] | Positions breaching stop-loss |
| `has_traded_today` | `ticker, date=None` | bool | Idempotency check |
| `record_trade` | `ticker, action, price, shares, strategy, momentum_score` | bool | Log trade to CSV |

**Ledger Files**:
- `ledger_momentum.csv` - Momentum strategy trades
- `ledger_ml.csv` - ML strategy trades

**CSV Schema**:
```
date,ticker,action,price,shares,amount,cash_balance,total_value,strategy,momentum_score
```

**Special Rows**:
- `PORTFOLIO,VALUE` - Daily portfolio snapshot for charting

---

## Risk Management

### Module: `src/trading/risk_manager.py`

#### `@dataclass RiskLimits`

Configuration for risk constraints.

| Field | Default | Description |
|-------|---------|-------------|
| `max_position_pct` | 0.15 | Max 15% per position |
| `max_sector_pct` | 0.30 | Max 30% per sector |
| `min_cash_buffer` | $200 | Minimum cash reserve |
| `max_daily_var_pct` | 0.025 | Max 2.5% daily VaR |
| `volatility_lookback` | 30 | Days for vol calculation |
| `drawdown_warning` | 0.15 | 15% drawdown warning |
| `drawdown_halt` | 0.20 | 20% drawdown halt |
| `drawdown_liquidate` | 0.25 | 25% forced liquidation |

#### Class: `DrawdownController`

Monitors portfolio drawdown and adjusts position sizing.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__init__` | `warning=0.15, halt=0.20, liquidate=0.25` | - | Set thresholds |
| `update` | `current_value` | - | Update peak and drawdown |
| `get_position_multiplier` | - | float | 1.0/0.5/0.0 based on drawdown |
| `should_liquidate` | - | bool | True if > 25% drawdown |
| `get_status` | - | str | "NORMAL", "WARNING", "HALT" |

**Drawdown Actions**:
| Drawdown | Action |
|----------|--------|
| < 15% | Normal trading |
| 15-20% | Reduce new positions by 50% |
| 20-25% | Halt all new buys |
| > 25% | Force liquidate 50% |

#### Class: `RiskManager`

Portfolio-level risk management.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__init__` | `risk_limits` | - | Initialize with limits |
| `calculate_position_size` | `ticker, price, cash, portfolio_value, ...` | Tuple(shares, reason) | Optimal position size |
| `calculate_portfolio_var` | `holdings, prices, historical_data, confidence` | float | Value at Risk |
| `validate_trade` | `ticker, action, shares, price, ...` | Tuple(bool, reason) | Pre-trade validation |

**Position Sizing Formula**:
```python
shares = min(
    shares_by_cash,
    shares_by_position_limit,  # Max 15% of portfolio
    shares_by_volatility,      # Target 20% annualized vol
    shares_by_sector_limit,    # Max 30% per sector
    shares_corrected_for_correlation  # Diversification penalty
)
```

---

## Transaction Costs

### Module: `src/backtesting/costs.py`

Realistic trading cost simulation with **5 basis points (bps) slippage**.

#### `@dataclass CostConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `commission_per_share` | $0.00 | Per-share commission |
| `min_commission` | $0.00 | Minimum commission |
| `slippage_bps` | 5.0 | Bid-ask slippage (5 bps) |
| `market_impact_enabled` | True | Enable market impact |
| `market_impact_coefficient` | 0.1 | Impact multiplier |
| `avg_daily_volume_threshold` | 0.005 | ADV threshold |

#### Class: `TransactionCostModel`

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__init__` | `config=None` | - | Initialize with defaults |
| `calculate_execution_price` | `action, price, shares, adv` | Tuple(exec_price, cost) | Apply slippage |
| `calculate_trade_costs` | `trade_value, shares, adv` | Dict | Cost breakdown |

**Execution Price Calculation**:
```python
# BUY: Pay more than quoted (slippage hurts)
execution_price = price * (1 + slippage_bps/10000)

# SELL: Receive less than quoted (slippage hurts)
execution_price = price * (1 - slippage_bps/10000)
```

**Example**:
- Stock price: $100.00
- Slippage: 5 bps = 0.05%
- BUY execution: $100.05
- SELL execution: $99.95

#### Class: `CostTracker`

Tracks cumulative costs during backtesting.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `record_trade` | `date, ticker, action, shares, price, adv` | Dict | Record and calculate costs |
| `get_summary` | - | Dict | Total costs summary |
| `get_cost_dataframe` | - | DataFrame | Full cost history |
| `reset` | - | - | Clear tracked costs |

---

## Backtesting Framework

### Module: `src/backtesting/backtester.py`

#### Class: `Backtester`

Full walk-forward backtesting with transaction costs.

**Configuration**:
```python
config = BacktestConfig(
    start_date="2020-01-01",
    end_date="2025-12-01",
    initial_capital=100000,
    benchmark_ticker="SPY",
    slippage_bps=5.0,
    max_position_pct=0.15,
    rebalance_frequency="daily"
)

backtester = Backtester(config)
metrics, trades_df, summary = backtester.run(data_dict, signal_generator)
```

### Module: `src/backtesting/performance.py`

Performance metrics calculation.

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Total Return** | (Final - Initial) / Initial | Overall performance |
| **CAGR** | (Final/Initial)^(1/years) - 1 | Annualized return |
| **Sharpe Ratio** | (Return - RiskFree) / StdDev | Risk-adjusted return |
| **Max Drawdown** | Max peak-to-trough decline | Worst loss from peak |
| **Win Rate** | Winning trades / Total trades | Trade success rate |
| **Sortino Ratio** | Return / Downside StdDev | Downside risk-adjusted |

---

## Dashboard

### Module: `dashboard/app.py`

Streamlit-based live dashboard.

**URL**: [paper-trader-ai.streamlit.app](https://paper-trader-ai.streamlit.app/)

**Features**:
- Portfolio Overview cards (Momentum, ML, SPY)
- Performance chart with all 3 strategies
- Current Holdings tables
- Recent Trades display
- Detailed Metrics table

**Data Sources**:
- `data/portfolio_snapshot.json` - Summary metrics
- `ledger_momentum.csv` - Trade history
- `ledger_ml.csv` - Trade history
- `data/spy_benchmark.json` - SPY comparison

---

## CLI Reference

### Trading Commands

```bash
# Momentum Strategy (monthly rebalance)
python main.py --strategy momentum --portfolio momentum

# ML Strategy (daily rebalance)
python main.py --strategy ml --portfolio ml

# Training only
python main.py --mode train --strategy ml

# Backtest
python main.py --mode backtest --strategy momentum
```

### Utility Scripts

```bash
# Update market data cache
python -c "from src.data.loader import update_cache; update_cache(['AAPL','TSLA'])"

# Compute portfolio snapshot
python scripts/utils/compute_portfolio_snapshot.py

# Run PIT backtest
python scripts/validation/pit_backtest_oct_dec.py
python scripts/validation/pit_momentum_oct_dec.py
```

### Make Commands

```bash
make train      # Train ML model
make trade      # Execute trades
make backtest   # Run backtest
make docker-up  # Start Docker container
make clean      # Clean artifacts
```

---

## Current Performance (Dec 2025)

### With 5 bps Transaction Costs (Oct 1 - Dec 19, 2025)

| Metric | Momentum | ML | SPY |
|--------|----------|-----|-----|
| **Final Value** | $10,720 | $10,158 | $10,310 |
| **Return** | +7.20% | +1.58% | +3.10% |
| **Excess Return** | +4.10% | -1.52% | — |
| **Total Trades** | 50 | 526 | — |

**Key Insight**: Momentum outperforms with transaction costs due to lower turnover (50 vs 526 trades).

---

## File Structure

```
paper-trader/
├── main.py                              # Core orchestrator
├── config/
│   └── trading.yaml                     # Strategy configuration
├── data/
│   ├── market.db                        # SQLite cache (4M+ rows)
│   ├── portfolio_snapshot.json          # Dashboard data
│   └── sp500_tickers.txt                # Universe
├── src/
│   ├── data/
│   │   ├── cache.py                     # DataCache class
│   │   ├── loader.py                    # fetch_data, update_cache
│   │   └── validator.py                 # DataValidator
│   ├── features/
│   │   └── indicators.py                # 15 technical indicators
│   ├── models/
│   │   ├── trainer.py                   # train_model, train_ensemble
│   │   └── predictor.py                 # Predictor, EnsemblePredictor
│   ├── trading/
│   │   ├── portfolio.py                 # Portfolio class
│   │   ├── risk_manager.py              # RiskManager, DrawdownController
│   │   └── regime.py                    # Market regime detection
│   └── backtesting/
│       ├── costs.py                     # TransactionCostModel
│       ├── backtester.py                # Backtester class
│       └── performance.py               # Metrics calculation
├── scripts/
│   ├── backtests/                       # Backtest scripts
│   ├── validation/                      # PIT validation
│   └── utils/                           # Utility scripts
├── dashboard/
│   └── app.py                           # Streamlit dashboard
├── ledger_ml.csv                        # ML trade history
└── ledger_momentum.csv                  # Momentum trade history
```

---

*Built by Prabuddha Tamhane • 2025*
