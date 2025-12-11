# Paper Trader AI - Complete Project Guide

> **Private Documentation** - Comprehensive guide to understand the entire system.

---

## 🏗️ System Overview

Paper Trader AI is an automated trading system that:
1. Fetches market data for 30+ S&P 500 stocks
2. Validates data quality (10 checks)
3. Generates 15 technical features from OHLCV data
4. Trains XGBoost regressor to predict next-day returns
5. Generates BUY/SELL/HOLD signals based on predictions
6. Applies risk management before executing trades
7. Records trades to ledger.csv

---

## 📊 How the Model Works

### Step 1: Data Flow
```
Yahoo Finance API → Raw OHLCV Data → Validation → Feature Engineering → Model → Predictions → Risk Check → Trade
```

### Step 2: Feature Generation

The model uses **15 features** to predict next-day returns:

#### **MOMENTUM FEATURES** (4)

| Feature | What It Measures | How It's Calculated | Trading Signal |
|---------|-----------------|---------------------|----------------|
| **RSI** | Overbought/oversold | `100 - 100/(1 + AvgGain/AvgLoss)` over 14 days | >70 = sell, <30 = buy |
| **MACD** | Momentum direction | Fast EMA(12) - Slow EMA(26) | Positive = bullish |
| **MACD_signal** | Smoothed MACD | EMA(9) of MACD line | MACD crossing above = buy |
| **MACD_hist** | Momentum strength | MACD - Signal line | Growing histogram = strengthening trend |

#### **TREND FEATURES** (5)

| Feature | What It Measures | How It's Calculated | Trading Signal |
|---------|-----------------|---------------------|----------------|
| **BB_Width** | Volatility squeeze | (Upper - Lower) / Close | Low width = breakout incoming |
| **Dist_SMA50** | Short-term trend | Close / SMA_50 - 1 | >0 = above trend |
| **Dist_SMA200** | Long-term trend | Close / SMA_200 - 1 | >0 = bull market |
| **Return_1d** | Yesterday's move | Close.pct_change() | Momentum continuation |
| **Return_5d** | Week momentum | Close.pct_change(5) | 5-day trend |

#### **VOLUME FEATURES** (3) - *New in Phase 3.5*

| Feature | What It Measures | How It's Calculated | Trading Signal |
|---------|-----------------|---------------------|----------------|
| **OBV_Momentum** | Buying/selling pressure | Rate of change of On-Balance Volume | Rising = accumulation |
| **Volume_Ratio** | Unusual activity | 5-day avg volume / 20-day avg | >1.5 = unusual interest |
| **VWAP_Dev** | Fair value deviation | Close / VWAP - 1 | >0 = expensive, <0 = cheap |

#### **VOLATILITY FEATURES** (3) - *New in Phase 3.5*

| Feature | What It Measures | How It's Calculated | Trading Signal |
|---------|-----------------|---------------------|----------------|
| **ATR_Pct** | Normalized volatility | ATR(14) / Close | High = risky, reduce position |
| **BB_PctB** | Position in bands | (Close - Lower) / (Upper - Lower) | >1 = overbought, <0 = oversold |
| **Vol_Ratio** | Volatility regime | Vol(10d) / Vol(60d) | >1 = volatility expansion |

---

### Step 3: Model Training

**Algorithm**: XGBoost Regressor

**What it predicts**: Next-day percentage return (e.g., +1.2% or -0.5%)

**How it's trained**:
1. Combine all ticker data (sorted by date)
2. Split into 5 time-series folds (no shuffling!)
3. Train on historical data, test on future data
4. Average metrics across all folds
5. Train final model on all data

**Dynamic Feature Selection** (Phase 3.5):
1. Train with all 15 features
2. Calculate feature importance (how much each contributes)
3. Drop features contributing < 3% 
4. Retrain with remaining features
5. Save selected features list with model

**Why this matters**:
- Noisy features hurt predictions
- Market regimes change which features are useful
- Model adapts automatically

---

### Step 4: Making Predictions

```python
# In predictor.py
def predict(df):
    # Generate all 15 features from raw OHLCV
    features = generate_features(df)
    
    # Use only features the model was trained on
    selected = features[self.selected_features]
    
    # Get model prediction
    expected_return = model.predict(selected)[-1]
    
    return expected_return  # e.g., 0.0158 = +1.58%
```

**Signal Generation (Fixed Thresholds - A/B Tested)**:

After A/B testing (Fixed vs Z-Score), **Fixed Thresholds** performed better (Sharpe 3.41 vs 0.99):

```python
if expected_return > 0.005:   # +0.5%
    signal = "BUY"
elif expected_return < -0.005:  # -0.5%
    signal = "SELL"
else:
    signal = "HOLD"
```

**Why Fixed Thresholds Won**:
- Z-Score method was too aggressive (16% buy/16% sell daily)
- Fixed thresholds are more selective (fewer, higher-conviction trades)
- Strategy works by letting winners run; fewer trades = less disruption

---

### Step 5: Risk Management (Phase 7 - A/B Tested)

Before any trade executes, it passes through **Double-Layer Protection**:

#### Layer 1: Portfolio Circuit Breakers (`DrawdownController`)
Monitors total portfolio drawdown from peak equity.
- **-15% Drawdown**: ⚠️ WARNING. New position sizes reduced to **50%**.
- **-20% Drawdown**: 🛑 HALT. No new buy orders allowed. Sells only.
- **-25% Drawdown**: 🚨 LIQUIDATE. Forced selling to raise 50% cash.

#### Layer 2: Position Controls (A/B Tested)
- **Stop-Loss**: **15%** exit threshold (A/B tested: outperforms no stop-loss)
- **Max Position**: 15% of portfolio.
- **Max Sector**: 30% per sector (e.g., max 30% in Tech).
- **Slippage**: Assumed 10 bps cost per trade.
- **Volatility Sizing**: Positions sized inversely to 30-day volatility.

#### Double Holdout Validation Results
Tested on 20 tickers the model **NEVER saw** (2023-2024):
- **15% Stop-Loss**: 43.8% return (~20% annual)
- **No Stop-Loss**: 35.3% return (~16% annual)
- **Stop-loss wins** by +8.5% absolute

## 📈 Feature Importance (Current Model) *(Updated Phase 7)*

```
BB_PctB         ████████████ 11.6%  ← TOP (Volatility - position in Bollinger Bands)
VWAP_Dev        ██████████   9.4%   ← #2 (Volume - deviation from VWAP)
Vol_Ratio       █████████    9.4%   ← Volatility regime
Return_1d       ████████     8.5%   ← Yesterday's return
Return_5d       ████████     8.3%   ← Weekly momentum
MACD_hist       ██████       7.0%
Volume_Ratio    ██████       6.3%   ← Unusual volume
MACD            ██████       6.3%
RSI             ██████       6.0%
ATR_Pct         █████        5.6%   ← Volatility
MACD_signal     █████        5.1%
Dist_SMA50      █████        5.0%
OBV_Momentum    ████         4.3%   ← Volume
Dist_SMA200     ███          3.6%
BB_Width        ███          3.6%
```

**Key Insight**: BB_PctB (position in Bollinger Bands) is now the top feature, followed by volume features!

---

## 🧪 Backtest Results (ML Predictor 2015-2024) *(Updated Phase 4)*

| Metric | Value | Assessment |
|--------|-------|------------|
| **Total Return** | 748.15% | Excellent |
| **CAGR** | 27.0% | Very Strong |
| **Sharpe Ratio** | 1.04 | Good (>1 ideal) ✅ |
| **Win Rate** | 67% | Strong Edge |
| **Alpha** | 22.9% | Beat benchmark significantly |
| **Max Drawdown** | 29.1% | Moderate |
| **Total Trades** | 354 | ~7/week |

### Performance by Market Regime

| Regime | Sharpe | Return | Notes |
|--------|--------|--------|-------|
| **Bull** | 1.89 | +543% | Excellent |
| **Sideways** | 3.19 | +22% | Very strong |
| **Crisis** | 0.13 | +1.4% | Survived |
| **Bear** | -1.88 | -35% | ⚠️ Weakness |

---

## 🎯 VIX Regime Detection *(Phase 3.6)*

**Purpose**: Automatically reduce exposure during volatility spikes

| VIX Level | Regime | Position Multiplier |
|-----------|--------|---------------------|
| < 25 | NORMAL | 100% |
| 25-35 | ELEVATED | 50% |
| > 35 | CRISIS | 0% (hold cash) |

---

## 🎯 Multi-Horizon Ensemble *(Phase 3.6)*

**Purpose**: More stable signals by blending multiple prediction horizons

| Model | Weight | Purpose |
|-------|--------|---------|
| 1-day | 50% | Short-term responsiveness |
| 5-day | 30% | Weekly trend |
| 20-day | 20% | Monthly direction |

---

## 📦 SQLite Data Cache *(Phase 4)*

**Purpose**: Avoid yfinance rate limits, faster data access

| Stat | Value |
|------|-------|
| Database | `data/market.db` |
| Tickers | 503 (S&P 500) |
| Total Rows | 4.3M |
| Date Range | 1962-2024 |

---

## 🌐 S&P 500 Universe *(Phase 4)*

**Purpose**: Trade full S&P 500 instead of 35 manual tickers

```yaml
# config/settings.yaml
universe:
  type: "sp500"   # Fetches ~503 tickers from Wikipedia
```

---

## 🚀 Walk-Forward Validation *(Phase 5 - NEW)*

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
| Total Return | 497% | 234% | 🏆 MODEL |
| CAGR | 19.6% | 12.8% | 🏆 MODEL |
| Sharpe Ratio | 0.53 | 0.49 | 🏆 MODEL |
| Max Drawdown | -48% | -34% | SPY |

**Note**: Model wins on returns but has higher drawdown during bear markets.

**Run it**:
```bash
python run_walkforward.py --start 2015 --end 2024
```

---

## 🔧 Hyperparameter Optimization *(Phase 5)*

**Purpose**: Ensure model isn't overfitting

**Safeguards**:
- TimeSeriesSplit (not random split)
- Early stopping on validation set
- Overfitting check (train/val gap)

**Results**: Current params already optimal (+0.03% improvement only)

**Best params found**:
```python
n_estimators: 50
max_depth: 4
learning_rate: 0.05
min_child_weight: 5
subsample: 0.8
reg_lambda: 1
```

**Run it**:
```bash
python run_hyperopt.py
```

---

## 🔄 Daily Execution Flow

```
1. GitHub Actions triggers at 9 PM UTC (4 PM EST)
2. Docker container starts
3. Download cache artifact (if exists)
4. Fetch only NEW data for 500+ tickers
5. Filter to 2010+ data (avoid old data issues)
6. Validate data quality (reject bad tickers)
7. Check VIX regime (reduce exposure if elevated)
8. Generate multi-horizon ensemble predictions
9. Apply risk management filters
10. Execute trades (update ledger.csv)
11. Upload updated cache artifact
12. Push changes back to GitHub
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `main.py` | Orchestrates entire trading workflow |
| `run_backtest.py` | Backtest with SMA or ML predictor |
| `run_walkforward.py` | True out-of-sample validation *(Phase 5)* |
| `run_hyperopt.py` | Hyperparameter optimization *(Phase 5)* |
| `src/data/loader.py` | Fetches data from Yahoo Finance (cache-first) |
| `src/data/cache.py` | SQLite cache manager *(Phase 4)* |
| `src/data/universe.py` | S&P 500 universe management *(Phase 4)* |
| `src/data/validator.py` | 10 data quality checks + backtest_mode |
| `src/features/indicators.py` | All 15 feature calculations |
| `src/models/trainer.py` | XGBoost training + feature selection |
| `src/models/predictor.py` | Ensemble predictor + regime awareness |
| `src/trading/regime.py` | VIX regime detection *(Phase 3.6)* |
| `ledger.csv` | Complete trade history |
| `data/market.db` | SQLite cache (4.3M rows) *(Phase 4)* |

---

## 🎯 Current Status

- ✅ Phase 1: Risk Management & Data Validation
- ✅ Phase 2: Backtesting Framework  
- ✅ Phase 3: ML Improvements (TimeSeriesSplit, Regression)
- ✅ Phase 3.5: Enhanced Features + Dynamic Selection
- ✅ Phase 3.6: VIX Regime Detection + Multi-Horizon Ensemble
- ✅ Phase 4: SQLite Cache + S&P 500 Universe
- ✅ **Phase 5: Walk-Forward Validation + Hyperopt**
- ✅ **Phase 7: Quant Risk Enhancements (A/B Tested)**
  - Portfolio Drawdown Control (-15%/-20%/-25%)
  - Position Stop-Loss: **15%** (A/B tested: outperforms no stop-loss)
  - Signal Method: Fixed Thresholds (A/B proved better than Z-Score)
  - Ranking Metrics (Spearman IC)
  - Double Holdout Validation (gold standard testing)

---

## ⚠️ Important Caveats

1. **Backtest returns are likely inflated**:
   - Survivorship bias (using current S&P 500)
   - Position concentration (few trades drive returns)

2. **Realistic expectation**: ~15-25% annual returns (not 100%+)

3. **Double holdout test** (on unseen tickers): ~20% annual return

---

## 💡 Why This Works

1. **Walk-forward validation** - True out-of-sample testing
2. **Volume features capture smart money** - Institutional flows show up in volume
3. **VIX-based defense** - Automatically reduces exposure in crisis
4. **15% Stop-Loss** - A/B tested: protects capital without cutting winners
5. **2010+ data filter** - Avoids old data quality issues
6. **Double holdout validation** - Tests on tickers model never saw
7. **Full S&P 500 universe** - More diversification opportunities
8. **Ranking > R²** - Model ranks stocks well despite negative R²

### 🧠 Deep Dive: The R² Paradox
In Phase 7, we discovered a critical insight: **A model can have negative R² (predicting return magnitude poorly) but high profitability.**

**Why?**
- **R²** penalizes large errors in magnitude (e.g., predicting +5% when it was +10%).
- **Trading** only cares about **Ranking** (e.g., predicting Stock A > Stock B).
- Our model captures the *relative order* of returns correctly (Spearman IC > 0.10), even if the exact numbers are noisy.
- **Evidence**: Top-10% predictions yield 67% win rate, while R² remains negative.

**Conclusion**: We optimize for **Spearman Rank Correlation**, not RMSE/R².

---

## ⚠️ Known Limitations (Quant Perspective)

1. **Survivorship bias**: Using today's S&P 500 for 2015 (estimated 2-4% annual overstatement)
2. **Return concentration**: 91.5% of P&L from top 10 trades (addressed by z-score ranking reliability)
3. **Market Impact**: Assumed infinite liquidity (partially mitigated by 10 bps slippage)
5. **Best test**: Paper trade daily for 6-12 months (already doing!)


