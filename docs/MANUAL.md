# Paper Trader AI - Technical Manual

This document provides comprehensive technical documentation for the Paper Trader AI system, including architecture, risk management, data pipelines, and operational workflows.

---

## 🏗 System Architecture

The application follows a modular, production-ready architecture designed for extensibility, testability, and risk control.

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                    Main Trading Loop                     │
│                      (main.py)                          │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Data Layer   │   │ Model Layer  │   │Trading Layer │
│              │   │              │   │              │
│ • loader.py  │   │ • trainer.py │   │• portfolio.py│
│ • validator  │──▶│ • predictor  │──▶│• risk_mgr.py │
│              │   │              │   │              │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ yfinance API │   │ XGBoost Model│   │ ledger.csv   │
│ Market Data  │   │ Predictions  │   │ Transactions │
└──────────────┘   └──────────────┘   └──────────────┘
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

### 2. Feature Engineering

#### **Technical Indicators** (`src/features/indicators.py`)
Generates predictive features from OHLC data:

**Momentum Indicators**:
- **RSI (14)**: Relative Strength Index for overbought/oversold detection
- **MACD**: Moving Average Convergence Divergence
  - Fast EMA: 12 days
  - Slow EMA: 26 days
  - Signal line: 9 days
- **MACD Histogram**: MACD - Signal

**Volatility Indicators**:
- **Bollinger Bands**: 20-day SMA ± 2 standard deviations
- **Bollinger Band Width**: `(Upper - Lower) / Close`

**Trend Indicators**:
- **SMA 50**: 50-day simple moving average
- **SMA 200**: 200-day simple moving average
- **Distance to SMA**: `(Close / SMA) - 1` (percentage)

**Returns**:
- **1-Day Return**: `Close.pct_change()`
- **5-Day Return**: `Close.pct_change(5)`

**Target Variable**:
- Binary classification: `1 if Next_Close > Current_Close else 0`

---

### 3. Machine Learning Pipeline

#### **Model Trainer** (`src/models/trainer.py`)
- **Algorithm**: XGBoost Classifier
- **Features**: 9 technical indicators
- **Target**: Binary (price up/down next day)
- **Train/Test Split**: 80/20, time-series aware (no shuffle)
- **Hyperparameters**:
  - `n_estimators=100`
  - `learning_rate=0.05`
  - `max_depth=5`

**Output Artifacts**:
- `models/xgb_model.joblib`: Serialized model
- `results/metrics.txt`: Accuracy, precision, recall, F1
- `results/confusion_matrix.png`: Visual classification performance

#### **Predictor** (`src/models/predictor.py`)
- **Input**: Raw OHLC DataFrame
- **Output**: Probability `P(Price_Up | Features)`
- **Process**:
  1. Transform OHLC → Features via `generate_features()`
  2. Extract latest row
  3. Predict with loaded XGBoost model
  4. Return probability for class 1 (price increase)

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

## 📊 Understanding the Metrics

### **Model Performance Metrics** (`results/metrics.txt`)

**Accuracy**: `(TP + TN) / Total`
- Percentage of correct predictions
- **Caveat**: Can be misleading if data is imbalanced

**Precision**: `TP / (TP + FP)`
- Of all predicted "price up", what % were actually correct?
- High precision → Few false alarms

**Recall**: `TP / (TP + FN)`
- Of all actual "price up" days, what % did we predict?
- High recall → Don't miss many opportunities

**F1 Score**: `2 × (Precision × Recall) / (Precision + Recall)`
- Harmonic mean balancing precision and recall
- **Target**: > 0.50 (better than random)

### **Confusion Matrix** (`results/confusion_matrix.png`)

```
                Predicted
              Down    Up
Actual Down   TN     FP   ← False positive (false alarm)
       Up     FN     TP   ← False negative (missed opportunity)
```

- **True Positive (TP)**: Correctly predicted price increase
- **True Negative (TN)**: Correctly predicted price decrease
- **False Positive (FP)**: Predicted up, but went down (entered bad trade)
- **False Negative (FN)**: Predicted down, but went up (missed opportunity)

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
- Executes all 27 unit tests
- Covers risk manager (14 tests) and data validator (13 tests)
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

## 🚀 Next Steps

### Phase 2: Backtesting & Validation
Before deploying real capital, implement:
- Historical backtesting (2018-2024)
- Transaction cost modeling (slippage, fees)
- Walk-forward analysis
- Sharpe ratio, max drawdown calculation

### Phase 3: Model Improvements
- Fix data leakage in feature engineering
- Time series cross-validation
- Regression target (predict return magnitude)
- Additional features (volume, macro indicators)

### Phase 4: Production Deployment
- Advanced logging (JSON structured logs)
- Alerting system (email, Slack)
- SQLite ledger (replace CSV)
- Live monitoring dashboard

---

## 📚 Additional Resources

- **XGBoost Documentation**: https://xgboost.readthedocs.io/
- **yfinance GitHub**: https://github.com/ranaroussi/yfinance
- **Technical Analysis Library**: https://technical-analysis-library-in-python.readthedocs.io/
- **Quantitative Finance**: "Advances in Financial Machine Learning" by Marcos Lopez de Prado

---

**Last Updated**: December 2024 (Phase 1 Complete)
