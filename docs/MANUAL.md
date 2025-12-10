# Paper Trader AI - Technical Manual

This document provides comprehensive technical documentation for the Paper Trader AI system, including architecture, risk management, data pipelines, and operational workflows.

---

## 🏗 System Architecture

The application follows a modular, production-ready architecture designed for extensibility, testability, and risk control.

### Core Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Main Trading Loop                                │
│                           (main.py)                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│   Data Layer     │       │   Model Layer    │       │  Trading Layer   │
│                  │       │                  │       │                  │
│ • loader.py      │       │ • trainer.py     │       │ • portfolio.py   │
│ • cache.py ✨    │──────▶│ • predictor.py   │──────▶│ • risk_manager   │
│ • universe.py ✨ │       │   (Ensemble) ✨  │       │ • regime.py ✨   │
│ • validator.py   │       │                  │       │                  │
│ • macro.py       │       │                  │       │                  │
└──────────────────┘       └──────────────────┘       └──────────────────┘
        │                           │                           │
        ▼                           ▼                           ▼
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│ market.db (SQLite)│       │ XGBoost Models   │       │ ledger.csv       │
│ • 503 tickers ✨  │       │ • 1d/5d/20d ✨   │       │ Trade History    │
│ • 4.3M rows       │       │ • Ensemble       │       │                  │
└──────────────────┘       └──────────────────┘       └──────────────────┘
        │
        ▼
┌──────────────────┐
│ S&P 500 Universe │
│ (Wikipedia) ✨   │
└──────────────────┘

✨ = New in Phase 3.6 / Phase 4
```

---

## 📦 Component Reference

### 1. Data Pipeline

#### **Data Loader** (`src/data/loader.py`)
- **Purpose**: Fetches historical OHLCV data from Yahoo Finance
- **API**: `yfinance` library
- **Caching**: Optimized to minimize redundant API calls
- **Output**: Dictionary of `{ticker: DataFrame}` with DatetimeIndex

**Key Function**:
```python
fetch_data(tickers, period="3y", interval="1d") -> Dict[str, pd.DataFrame]
```

#### **Data Validator** (`src/data/validator.py`) ✨ *New in Phase 1*
- **Purpose**: Ensures data quality before model training
- **Checks Performed**: 10 comprehensive validations
- **Output**: `ValidationResult` with errors and warnings

**Validation Checks**:

| Check | Threshold | Action on Fail |
|-------|-----------|----------------|
| Empty DataFrame | N/A | Critical Error |
| Missing Columns | Must have OHLC, Volume | Critical Error |
| Data Freshness | < 48 hours old | Warning |
| Missing Values | < 5% missing | Error if > 5% |
| Price Validity | Prices > $0.01 | Critical Error |
| OHLC Relationships | High ≥ Low, etc. | Critical Error |
| Outlier Detection | Returns > 10σ | Warning |
| Volume Validity | No negative volume | Warning |
| Duplicates | No duplicate dates | Critical Error |
| DateTime Index | Must be DatetimeIndex | Critical Error |

**Usage Example**:
```python
from src.data.validator import DataValidator

validator = DataValidator(
    max_missing_pct=0.05,       # Max 5% missing
    outlier_std_threshold=10.0, # Flag >10σ returns
    max_data_age_hours=48       # Data < 48h old
)

results = validator.validate_data_dict(data_dict)
validator.print_validation_summary(results)
```

---

### 2. Feature Engineering ✨ *Updated in Phase 3.5*

#### **Technical Indicators** (`src/features/indicators.py`)

The model uses **15 features** across 4 categories. All features use only past data (no look-ahead bias).

---

**MOMENTUM INDICATORS** (4 features)

| Feature | Formula | Trading Interpretation |
|---------|---------|----------------------|
| **RSI** | 100 - 100/(1 + RS) where RS = AvgGain/AvgLoss | >70 = overbought, <30 = oversold |
| **MACD** | EMA(12) - EMA(26) | Positive = bullish momentum |
| **MACD_signal** | EMA(9) of MACD | Crossover signals |
| **MACD_hist** | MACD - Signal | Histogram divergence |

---

**TREND INDICATORS** (5 features)

| Feature | Formula | Trading Interpretation |
|---------|---------|----------------------|
| **BB_Width** | (Upper - Lower) / Close | Low = consolidation, high = volatility |
| **Dist_SMA50** | Close / SMA_50 - 1 | >0 = above trend, <0 = below |
| **Dist_SMA200** | Close / SMA_200 - 1 | Long-term trend direction |
| **Return_1d** | Close.pct_change() | Yesterday's return |
| **Return_5d** | Close.pct_change(5) | Week momentum |

---

**VOLUME INDICATORS** (3 features) *(New in Phase 3.5)*

| Feature | Formula | Trading Interpretation |
|---------|---------|----------------------|
| **OBV_Momentum** | pct_change(OBV, 10) | Rising OBV = buying pressure |
| **Volume_Ratio** | Vol(5d avg) / Vol(20d avg) | >1 = unusual volume |
| **VWAP_Dev** | Close / VWAP - 1 | >0 = trading above fair value |

---

**VOLATILITY INDICATORS** (3 features) *(New in Phase 3.5)*

| Feature | Formula | Trading Interpretation |
|---------|---------|----------------------|
| **ATR_Pct** | ATR(14) / Close | Normalized volatility measure |
| **BB_PctB** | (Close - Lower) / (Upper - Lower) | 0-1 position in bands |
| **Vol_Ratio** | Vol(10d) / Vol(60d) | >1 = volatility expansion |

---

**Target Variable** *(Phase 3)*:
- **Regression**: Next-day return (percentage)
- Created SEPARATELY from features to prevent look-ahead bias

---

### 3. Machine Learning Pipeline ✨ *Updated in Phase 3.5*

#### **Model Trainer** (`src/models/trainer.py`)
- **Algorithm**: XGBoost Regressor (predicts return magnitude)
- **Features**: 15 technical indicators (dynamically filtered)
- **Target**: Next-day return (continuous)
- **Cross-Validation**: 5-fold TimeSeriesSplit (proper walk-forward)
- **Anti-Leakage**: Target created AFTER feature generation

**Hyperparameters**:
- `n_estimators=100`
- `learning_rate=0.05`
- `max_depth=5`
- `objective='reg:squarederror'`

**Dynamic Feature Selection** *(New in Phase 3.5)*:
1. Train initial model with all 15 features
2. Calculate feature importance scores
3. Drop features with importance < 3%
4. Retrain with only useful features
5. Save selected feature list with model

**Cross-Validation Metrics**:
- RMSE: Root Mean Squared Error
- MAE: Mean Absolute Error
- R²: Coefficient of determination
- Directional Accuracy: % of correct up/down predictions

**Output Artifacts**:
- `models/xgb_model.joblib`: Model + metadata (selected features)
- `results/metrics.txt`: CV metrics across all folds
- `results/feature_importance.png`: Feature importance with threshold line
- `results/selected_features.txt`: Features used for inference

#### **Predictor** (`src/models/predictor.py`)
- **Input**: Raw OHLC DataFrame
- **Output**: Expected next-day return (e.g., +1.2% or -0.5%)
- **Process**:
  1. Load model + selected features from metadata
  2. Transform OHLC → Features via `generate_features()`
  3. Extract only selected features for latest row
  4. Predict with XGBoost regressor
  5. Return expected return value

**Trading Thresholds**:
- BUY: Expected return > +0.5%
- SELL: Expected return < -0.5%
- HOLD: otherwise

---

### 4. Risk Management ✨ *New in Phase 1*

#### **Risk Manager** (`src/trading/risk_manager.py`)

Professional-grade risk control system preventing catastrophic losses.

**Risk Limits** (Configurable via `RiskLimits` dataclass):

```python
max_position_pct: 0.15      # Maximum 15% per position
max_sector_pct: 0.40        # Maximum 40% in any sector
min_cash_buffer: 100.0      # Minimum cash reserve
max_daily_var_pct: 0.025    # Maximum 2.5% VaR
volatility_lookback: 30     # Days for volatility calculation
correlation_threshold: 0.7  # Reduce size if corr > 70%
```

**Position Sizing Algorithm**:

1. **Volatility Adjustment**:
   ```
   volatility = std(returns) * sqrt(252)  # Annualized
   vol_scalar = min(target_vol / volatility, 1.5)
   adjusted_shares = base_shares * vol_scalar
   ```
   - Higher volatility → Smaller position
   - Target volatility: 20% annualized

2. **Constraint Checks**:
   - Max position value: `portfolio_value × 0.15`
   - Max sector value: `portfolio_value × 0.40 - current_sector_exposure`
   - Available cash constraint

3. **Correlation Penalty**:
   - If same sector already > 25% of portfolio → Apply penalty (up to 50% reduction)

4. **Final Position**:
   ```
   shares = min(
       shares_by_cash,
       shares_by_position_limit,
       shares_by_volatility,
       shares_by_sector_limit
   ) × (1 - correlation_penalty)
   ```

**Value at Risk (VaR)**:
- **Method**: Historical simulation
- **Confidence**: 95%
- **Horizon**: 1 day
- **Calculation**:
  1. Calculate daily returns for each holding over last 30 days
  2. Weight returns by portfolio allocation
  3. Compute aggregate portfolio returns
  4. VaR = 5th percentile of return distribution × portfolio_value

**Sector Classification**:
| Sector        | Tickers |
|---------------|---------|
| Technology    | AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, AVGO, AMD, CRM, NFLX |
| Financials    | JPM, V, MA, BAC, BRK-B |
| Consumer      | HD, COST, WMT, PG, KO, PEP, DIS |
| Healthcare    | UNH, JNJ, LLY, MRK |
| Energy        | XOM, CVX |
| Industrials   | BA, CAT |
| Index         | SPY, QQQ, IWM, DIA |

**Pre-Trade Validation**:
```python
is_valid, reason = risk_mgr.validate_trade(
    ticker, action, shares, price,
    current_holdings, current_prices,
    cash_balance, portfolio_value
)
```

Checks:
- ✅ Sufficient cash (for BUY)
- ✅ Position size within limits
- ✅ Sector concentration within limits
- ✅ Minimum cash buffer maintained
- ✅ Sufficient shares owned (for SELL)

---

### 5. Portfolio Management

#### **Portfolio** (`src/trading/portfolio.py`)
- **Ledger**: CSV-based transaction log (`ledger.csv`)
- **State Tracking**: Cash balance, holdings by ticker
- **Idempotency**: Prevents duplicate trades on same ticker/day

**Key Methods**:
- `get_holdings()`: Returns `{ticker: shares}` dictionary
- `get_last_balance()`: Current cash balance
- `get_portfolio_value(prices)`: Total value (cash + holdings)
- `record_trade(ticker, action, price, shares)`: Executes and logs trade
- `has_traded_today(ticker)`: Checks if already traded

---

### 6. Backtesting Framework ✨ *New in Phase 2*

The backtesting module (`src/backtesting/`) provides event-driven strategy validation with professional quant metrics.

#### **Backtester** (`src/backtesting/backtester.py`)

**Architecture**: Event-driven simulation engine that walks through historical data day-by-day, simulating trades with realistic costs and risk controls.

```python
from src.backtesting import Backtester, BacktestConfig

config = BacktestConfig(
    start_date="2017-01-01",
    end_date="2024-12-31",
    initial_cash=100000.0,
    benchmark_ticker="SPY",
    slippage_bps=5.0,           # 5 basis points slippage
    max_position_pct=0.15,      # Max 15% per position
    rebalance_frequency="daily" # daily, weekly, monthly
)

backtester = Backtester(config)
metrics, trades_df, summary = backtester.run(
    data_dict=data_dict,
    signal_generator=my_strategy_function,
    benchmark_data=spy_data
)
```

#### **Performance Metrics** (`src/backtesting/performance.py`)

**Professional Quant Metrics**:

| Metric | Formula | Target |
|--------|---------|--------|
| **Sharpe Ratio** | `(Return - Rf) / Volatility` | > 1.0 |
| **Sortino Ratio** | `(Return - Rf) / Downside Vol` | > 1.5 |
| **Calmar Ratio** | `CAGR / Max Drawdown` | > 0.5 |
| **Information Ratio** | `Active Return / Tracking Error` | > 0.5 |
| **VaR (95%)** | 5th percentile of returns | < 2% daily |
| **CVaR (Expected Shortfall)** | Average loss beyond VaR | < 3% daily |
| **Alpha** | Excess return after beta adjustment | > 0% |
| **Beta** | Market sensitivity | 0.5 - 1.5 |

**Trade Quality Metrics**:
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Average Holding Period**: Days per position
- **Turnover**: Annual portfolio turnover

#### **Transaction Costs** (`src/backtesting/costs.py`)

**Cost Components**:

| Component | Default | Description |
|-----------|---------|-------------|
| **Slippage** | 5 bps | Bid-ask spread impact |
| **Commission** | $0.00 | Per-share fee (zero for modern brokers) |
| **Market Impact** | √(participation) × price | For orders > 0.5% of ADV |

**Execution Price Calculation**:
```python
# BUY: Pay more than quoted price
execution_price = price × (1 + slippage_bps/10000) + market_impact

# SELL: Receive less than quoted price  
execution_price = price × (1 - slippage_bps/10000) - market_impact
```

#### **Regime Classification**

The backtester automatically classifies market regimes for stratified performance analysis:

| Regime | Detection Criteria |
|--------|-------------------|
| **Bull** | SPY > SMA_200, Volatility < 1.5× average |
| **Bear** | SPY < SMA_200 |
| **Crisis** | Volatility > 2× average (e.g., COVID, 2008) |
| **Sideways** | No clear trend |

**Running a Backtest**:
```bash
# Full backtest (2017-2024)
make backtest

# Quick backtest (2023-2024)  
make backtest-quick

# CLI with custom dates
python run_backtest.py --start 2020-01-01 --end 2023-12-31 --cash 50000
```

**Output Files**:
- `results/backtest_summary.txt`: Human-readable performance summary
- `results/backtest_metrics.json`: Machine-readable metrics
- `results/backtest_trades.csv`: Complete trade log

---

## 📊 Understanding the Metrics

### **Model Performance Metrics** (`results/metrics.txt`) ✨ *Updated in Phase 3*

**RMSE (Root Mean Squared Error)**:
- Average prediction error in return terms
- Example: RMSE = 0.0197 means ~2% average error
- **Target**: < 0.025 (2.5%)

**MAE (Mean Absolute Error)**:
- Similar to RMSE but less sensitive to outliers
- **Target**: < 0.015 (1.5%)

**R² (Coefficient of Determination)**:
- How much variance the model explains
- R² = 0.0 means model = mean prediction
- R² < 0 means model is worse than mean
- **For daily returns**: Negative R² is common (markets are efficient)

**Directional Accuracy**:
- Percentage of correct up/down predictions
- **Target**: > 52% (edge over random)
- Example: 52.03% = slight but consistent edge

### **Feature Importance** (`results/feature_importance.png`)

Shows which features drive predictions:
- Higher importance = more predictive power
- Helps understand model behavior
- Useful for feature engineering decisions

### **Risk Metrics** (Displayed at runtime)

**1-Day VaR (95%)**:
- Expected maximum loss over 1 day with 95% confidence
- Example: `VaR = $245.32 (2.45% of portfolio)` means there's a 5% chance of losing more than $245 tomorrow

**Sector Exposure**:
- Percentage of portfolio allocated to each sector
- **Target**: No sector > 40%, diversification across 3+ sectors

**Volatility Scalar**:
- Adjustment factor for position sizing
- `scalar > 1.0`: Low volatility → Increase position
- `scalar < 1.0`: High volatility → Decrease position

---

## 🛠 Operations (Makefile)

The project uses `make` for standardized operations.

### Available Commands

#### **Setup Environment**
```bash
make setup
```
- Creates Conda environment from `environment.yml`
- Installs all dependencies (pandas, yfinance, xgboost, pytest, etc.)

#### **Run Trading Bot**
```bash
make trade
```
- Executes full trading workflow:
  1. Fetch market data (3 years)
  2. Validate data quality
  3. Train/load XGBoost model
  4. Generate predictions
  5. Calculate risk-adjusted position sizes
  6. Execute validated trades
  7. Display portfolio summary and risk metrics

#### **Force Model Retraining**
```bash
make train
```
- Trains model without executing trades
- Useful for testing model performance
- Saves new model to `models/xgb_model.joblib`

#### **Run Test Suite**
```bash
make test
```
- Executes all 55 unit tests
- Covers ML pipeline (12), backtesting (16), risk manager (14), validator (13)
- Displays test results and coverage

#### **Docker Deployment**
```bash
make docker-up    # Start containerized system
make docker-down  # Stop containers
```
- Runs system in isolated Docker environment
- Ensures reproducibility across platforms

#### **Clean Build Artifacts**
```bash
make clean
```
- Removes `__pycache__` directories
- Clears `results/` folder
- Fresh start for debugging

---

## 🧪 Testing Guide

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_risk_manager.py -v
pytest tests/test_validator.py -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html  # View coverage
```

### Test Coverage

**Risk Manager Tests** (`tests/test_risk_manager.py`):
- Position sizing with volatility adjustment
- Sector concentration limits
- VaR calculation accuracy
- Trade validation (BUY/SELL)
- Correlation penalty logic
- Edge cases (insufficient cash, oversized positions)

**Data Validator Tests** (`tests/test_validator.py`):
- Empty DataFrame detection
- Missing column detection
- Invalid OHLC relationships
- Outlier detection (>10σ returns)
- Stale data detection
- Duplicate date handling

---

## ⚙️ Configuration Guide

### Main Configuration (`config/settings.yaml`)

```yaml
# Trading universe (30 tickers default)
tickers:
  - SPY    # S&P 500 Index
  - AAPL   # Technology
  - JPM    # Financials
  # ... add more

# Model settings
model:
  training_period: "3y"      # Data window: 1y, 2y, 3y, 5y, max
  retrain_daily: true        # Retrain on each run? (true/false)
  threshold_buy: 0.55        # Min probability to BUY (0.0-1.0)
  threshold_sell: 0.45       # Max probability to SELL (0.0-1.0)

# Portfolio settings
portfolio:
  initial_cash: 10000.0      # Starting capital ($)
  min_cash_buffer: 100.0     # Reserve cash ($)
```

### Risk Configuration (in code)

Edit `main.py` to customize risk limits:

```python
risk_limits = RiskLimits(
    max_position_pct=0.15,      # 15% max per position
    max_sector_pct=0.40,        # 40% max per sector
    min_cash_buffer=100.0,      # $100 reserve
    max_daily_var_pct=0.025,    # 2.5% max VaR
    volatility_lookback=30,     # 30-day volatility window
    correlation_threshold=0.7   # Penalize if corr > 70%
)
```

---

## 🔍 Troubleshooting

### Common Issues

**1. Data Fetch Fails**
```
❌ Failed to fetch data.
```
**Solution**: Check internet connection, yfinance may be rate-limited. Wait 1-5 minutes and retry.

**2. Invalid Tickers**
```
⚠️ Removing 2 invalid tickers: ['BRK-B', 'XYZ']
```
**Cause**: Ticker may be delisted, have insufficient data, or contain data errors.
**Action**: Review validation summary, remove problematic tickers from `config/settings.yaml`.

**3. Model Not Found**
```
Warning: No model found. Run training first.
```
**Solution**: Run `make train` to create `models/xgb_model.joblib`.

**4. Trade Rejected**
```
⚠️ AAPL: Trade rejected - Position too large: 25.0% > 15.0% limit
```
**Cause**: Risk manager blocked trade exceeding constraints.
**Action**: This is working as intended. Adjust `max_position_pct` if needed.

**5. Test Failures**
```
FAILED tests/test_risk_manager.py::test_name
```
**Solution**: Ensure dependencies are installed (`make setup`). Check pytest output for specific error.

---

## 📈 Interpreting Results

### Good Model Performance
- ✅ Accuracy: 0.52-0.58 (anything > 0.50 is better than random)
- ✅ F1 Score: > 0.50
- ✅ Balanced precision/recall (both > 0.48)

### Poor Model Performance
- ❌ Accuracy: ~0.50 (coin flip)
- ❌ Extremely imbalanced confusion matrix (all predictions one class)
- ❌ Very low precision or recall (< 0.40)

### Healthy Portfolio Metrics
- ✅ VaR < 3% of portfolio daily
- ✅ Sector exposure: No sector > 45%
- ✅ Largest position < 18% of portfolio
- ✅ 5+ holdings across 3+ sectors

### Risky Portfolio State
- ❌ VaR > 5% of portfolio
- ❌ Single sector > 60%
- ❌ Single position > 25% of portfolio
- ❌ < 3 holdings (under-diversified)

---

## 🚀 Completed Phases

### ✅ Phase 1: Risk Management & Data Validation
- Volatility-adjusted position sizing
- Portfolio constraints (15% max position, 40% max sector)
- 10 data quality validation checks

### ✅ Phase 2: Backtesting Framework
- Event-driven backtesting engine
- Transaction cost modeling (slippage, market impact)
- Regime-based performance analysis

### ✅ Phase 3: ML Improvements
- TimeSeriesSplit cross-validation
- Regression target (return prediction)
- Anti-leakage feature pipeline

### ✅ Phase 3.5: Enhanced Features
- 15 technical indicators (volume, volatility, momentum, trend)
- Dynamic feature selection (auto-filter <3% importance)

### ✅ Phase 3.6: Regime Detection & Multi-Horizon *(New)*
- **VIX-based regime detection**: NORMAL/ELEVATED/CRISIS
- **Position multipliers**: 100% → 50% → 0% based on VIX
- **Multi-horizon ensemble**: 1-day (50%), 5-day (30%), 20-day (20%)
- **Blended signals**: More stable than single-horizon

### ✅ Phase 4: Data Caching & Universe *(New)*
- **SQLite cache**: 4.3M rows, 503 tickers
- **Incremental fetching**: Only new bars after initial load
- **S&P 500 universe**: Dynamic from Wikipedia
- **GitHub Artifacts**: Cache persists across workflow runs

### ✅ Phase 5: Walk-Forward Validation & Hyperopt *(NEW!)*
- **Walk-forward validation**: True out-of-sample testing
- **Results**: 630% return vs SPY 234% (2015-2024)
- **2010+ data filter**: 35/35 tickers now pass validation
- **Hyperparameter optimization**: Current params already optimal
- **Overfitting check**: Negative train/val gap = model generalizes well

---

## 🎯 VIX Regime Detection *(Phase 3.6)*

**File**: `src/trading/regime.py`

**Thresholds**:
| VIX Level | Regime | Position Multiplier |
|-----------|--------|---------------------|
| < 25 | NORMAL | 100% |
| 25-35 | ELEVATED | 50% |
| > 35 | CRISIS | 0% (hold cash) |

**Usage**:
```python
from src.trading.regime import RegimeDetector

detector = RegimeDetector()
info = detector.get_regime_info(vix_value=22)
# Returns: {'regime': 'NORMAL', 'multiplier': 1.0}
```

---

## 🎯 Multi-Horizon Ensemble *(Phase 3.6)*

**File**: `src/models/predictor.py` → `EnsemblePredictor`

**Horizons**:
| Model | Weight | Purpose |
|-------|--------|---------|
| 1-day | 50% | Responsive to short-term moves |
| 5-day | 30% | Weekly trend |
| 20-day | 20% | Monthly direction |

**Usage**:
```python
from src.models.predictor import EnsemblePredictor

predictor = EnsemblePredictor()
result = predictor.predict_with_regime(df, vix_value=22)
# Returns: {'expected_return': 0.012, 'signal': 'BUY', 'regime': 'NORMAL'}
```

---

## 📦 SQLite Data Cache *(Phase 4)*

**File**: `src/data/cache.py`

**Database**: `data/market.db`

**Tables**:
- `price_data`: OHLCV data (ticker, date, open, high, low, close, volume)
- `macro_data`: VIX and other macro series
- `cache_metadata`: Last update timestamps

**Usage**:
```python
from src.data.cache import DataCache

cache = DataCache()
df = cache.get_price_data('AAPL', '2020-01-01', '2024-12-01')
stats = cache.get_cache_stats()  # Shows rows per ticker
```

---

## 🌐 S&P 500 Universe *(Phase 4)*

**File**: `src/data/universe.py`

**Configuration** (`config/settings.yaml`):
```yaml
universe:
  type: "sp500"   # or "config" for manual tickers
```

**Functions**:
- `fetch_sp500_tickers()`: Gets ~503 tickers from Wikipedia
- `get_mega_caps()`: Returns top 50 mega-cap stocks (fallback)
- `filter_by_liquidity()`: Filters by average daily volume

---

## 🚀 Walk-Forward Validation *(Phase 5 - NEW)*

**File**: `run_walkforward.py`

**Purpose**: True out-of-sample testing to eliminate look-ahead bias

**Process**:
```
Year 1: Train on 2010-2014 → Test on 2015
Year 2: Train on 2010-2015 → Test on 2016
...
Year 10: Train on 2010-2023 → Test on 2024
```

**Results (vs SPY Buy-and-Hold)**:
| Metric | WALK-FORWARD | SPY | Winner |
|--------|--------------|-----|--------|
| Total Return | 630% | 234% | 🏆 MODEL |
| CAGR | 22.0% | 12.8% | 🏆 MODEL |
| Sharpe Ratio | 0.73 | 0.49 | 🏆 MODEL |
| Max Drawdown | -35% | -34% | SPY |

**Run it**:
```bash
python run_walkforward.py --start 2015 --end 2024
```

---

## 🔧 Hyperparameter Optimization *(Phase 5)*

**File**: `run_hyperopt.py`

**Purpose**: Ensure model isn't overfitting

**Safeguards**:
- TimeSeriesSplit (not random split)
- Early stopping on validation set
- Overfitting check (train/val gap)

**Results**: Current params already optimal (+0.03% improvement only)

**Optimal Parameters**:
```python
n_estimators: 50
max_depth: 4
learning_rate: 0.05
min_child_weight: 5
subsample: 0.8
reg_lambda: 1
```

---

## 📚 Additional Resources

- **XGBoost Documentation**: https://xgboost.readthedocs.io/
- **yfinance GitHub**: https://github.com/ranaroussi/yfinance
- **Technical Analysis Library**: https://technical-analysis-library-in-python.readthedocs.io/
- **Quantitative Finance**: "Advances in Financial Machine Learning" by Marcos Lopez de Prado

---

**Last Updated**: December 2024 (Phase 5 Complete - Walk-Forward + Hyperopt)

