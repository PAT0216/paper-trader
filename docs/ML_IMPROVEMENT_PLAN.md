# ML Model Improvement Plan
## Comprehensive Testing & Implementation Guide

**Created:** December 2024  
**Status:** Phase 1 Experiments Running  
**Goal:** Fix ML ensemble strategy that's underperforming (-79% vs SPY +80%)

---

## Table of Contents
1. [Problem Summary](#problem-summary)
2. [Root Cause Analysis](#root-cause-analysis)
3. [Experiment Design](#experiment-design)
4. [Implementation Details](#implementation-details)
5. [How to Run Tests](#how-to-run-tests)
6. [Interpreting Results](#interpreting-results)
7. [Validation Protocol](#validation-protocol)
8. [Production Deployment](#production-deployment)

---

## Problem Summary

### Current State
- **ML Ensemble Return:** -79% (2017-2022 partial run)
- **SPY Benchmark:** +80% same period
- **Momentum Strategy:** +390% (separate backtest)
- **Directional Accuracy:** 51-52% (barely better than coin flip)

### Key Issues Identified
1. **Mean-Reversion Features** - RSI, Bollinger Bands work against momentum
2. **Short-Horizon Dominance** - 1-day predictions weighted 50%, too noisy
3. **No Risk Controls** - No stop-losses, no diversification limits
4. **Possible Signal Inversion** - Model may be predicting opposite direction

---

## Root Cause Analysis

### Feature Problem
Current features are mean-reversion indicators:
```python
FEATURE_COLUMNS = [
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Upper', 'BB_Lower', 'BB_Width', 'ATR',
    'OBV', 'VWAP_Ratio', 'ROC', 'Stoch_K', 'Stoch_D',
    'ADX', 'CCI'
]
```

**Problem:** These indicators signal "buy low, sell high" which conflicts with momentum (winners keep winning).

### Horizon Weighting Problem
```python
HORIZON_WEIGHTS = {1: 0.5, 5: 0.3, 20: 0.2}  # Current
```

**Problem:** 1-day predictions are very noisy (52% accuracy). Academic research shows 3-12 month momentum is strongest.

### Missing Risk Controls
- No stop-loss → Single bad pick can destroy returns (SMCI -48%)
- No sector limits → Concentrated in volatile tech
- No drawdown circuit breaker

---

## Experiment Design

### Experiment 0: Baseline (Control)
**Hypothesis:** Establish current performance baseline  
**Changes:** None - current production logic  
**Expected:** Negative returns, ~52% accuracy

### Experiment 1: Signal Inversion
**Hypothesis:** Model predicts opposite direction; inverting improves returns  
**Changes:** 
```python
# Instead of buying top predictions, buy bottom predictions
if config.invert_signal:
    sorted_preds = sorted_preds[::-1]  # Reverse order
```
**Expected:** If model is backwards, this should dramatically improve returns

### Experiment 2: Momentum Features
**Hypothesis:** Replace mean-reversion with momentum indicators  
**Changes:**
```python
MOMENTUM_FEATURE_COLUMNS = [
    'MOM_12_1',      # 12-1 month momentum (academic standard)
    'MOM_6_1',       # 6-1 month momentum
    'MOM_3_1',       # 3-1 month momentum
    'MOM_1M',        # 1 month momentum
    'ROC_20',        # 20-day rate of change
    'ROC_60',        # 60-day rate of change
    'TREND_STRENGTH', # Above/below 200 SMA
    'RELATIVE_STRENGTH', # Normalized rank
    'VOLUME_TREND',  # Volume vs average
    'VOLATILITY',    # 20-day volatility
]
```
**Implementation:**
```python
def generate_momentum_features(df):
    close = df['Close']
    # 12-1 month momentum (skip last month to avoid reversal)
    df['MOM_12_1'] = close.shift(21) / close.shift(252) - 1
    df['MOM_6_1'] = close.shift(21) / close.shift(126) - 1
    df['MOM_3_1'] = close.shift(21) / close.shift(63) - 1
    # ... etc
```
**Expected:** Aligns with proven momentum factor

### Experiment 3: Long Horizon Weighting
**Hypothesis:** Weight longer horizons more; they're more predictable  
**Changes:**
```python
# Old: {1: 0.5, 5: 0.3, 20: 0.2}
# New: {1: 0.1, 5: 0.3, 20: 0.6}
config.horizon_weights = {1: 0.1, 5: 0.3, 20: 0.6}
```
**Expected:** Less noise, smoother predictions

### Experiment 4: ML + Momentum Filter (Hybrid)
**Hypothesis:** Only buy ML picks that also have positive momentum  
**Changes:**
```python
def calculate_12_1_momentum(df):
    if len(df) < 252:
        return None
    close = df['Close']
    return (close.iloc[-21] / close.iloc[-252]) - 1

# Filter: only buy stocks with positive 12-1 momentum
if config.use_momentum_filter:
    mom = momentum_scores.get(ticker, -999)
    if mom < config.momentum_threshold:  # threshold = 0.0
        continue  # Skip negative momentum stocks
```
**Expected:** Filters out value traps, keeps momentum winners

### Experiment 5: Stop-Loss Risk Control
**Hypothesis:** Cutting losses early prevents catastrophic drawdowns  
**Changes:**
```python
if config.use_stop_loss:
    for idx in range(len(df_period)):
        low = df_period['Low'].iloc[idx]
        if (low / entry - 1) < -config.stop_loss_pct:  # -8%
            # Exit at stop price
            exit_price = entry * (1 - config.stop_loss_pct)
            break
```
**Expected:** Limits max loss per position, smoother equity curve

### Experiment 6: Combined Improvements
**Hypothesis:** Best improvements together should compound  
**Changes:**
```python
config = ExperimentConfig(
    name="6_Combined_Improvements",
    invert_signal=True,           # If inversion works
    use_momentum_filter=True,     # Filter by momentum
    momentum_threshold=0.0,
    use_stop_loss=True,
    stop_loss_pct=0.08,
    horizon_weights={1: 0.2, 5: 0.3, 20: 0.5}  # Weight long
)
```
**Expected:** If individual improvements work, combination should be best

### Experiment 7: Pure Momentum ML
**Hypothesis:** ML on momentum features with risk controls  
**Changes:**
```python
config = ExperimentConfig(
    name="7_Pure_Momentum_ML",
    use_momentum_features=True,   # Momentum indicators
    use_stop_loss=True,
    stop_loss_pct=0.10
)
```
**Expected:** Clean momentum approach with ML ranking

---

## Implementation Details

### File Structure
```
scripts/
├── ml_improvement_experiments.py   # Main experiment runner
├── walkforward_production_mirror.py # Original production mirror
│
src/
├── features/
│   └── indicators.py               # Feature generation
├── models/
│   ├── trainer.py                  # Model training
│   └── predictor.py                # Production predictor
│
results/
├── ml_experiments_results.json     # Experiment results
├── ml_experiments_log.txt          # Detailed logs
└── QUANT_ANALYSIS_REPORT.md        # Analysis report
```

### Key Functions

#### Momentum Feature Generation
```python
def generate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate momentum-based features.
    
    Academic basis:
    - Jegadeesh & Titman (1993): 12-1 month momentum
    - Fama-French: Momentum factor
    """
    df = df.copy()
    close = df['Close']
    
    # Core momentum signals (look-back only, no look-ahead)
    df['MOM_12_1'] = close.shift(21) / close.shift(252) - 1
    df['MOM_6_1'] = close.shift(21) / close.shift(126) - 1
    df['MOM_3_1'] = close.shift(21) / close.shift(63) - 1
    df['MOM_1M'] = close.pct_change(21)
    
    # Trend strength
    sma_200 = close.rolling(200).mean()
    df['TREND_STRENGTH'] = (close / sma_200) - 1
    
    return df.dropna()
```

#### Walk-Forward Engine
```python
def run_experiment(config, all_price_data, spy_data):
    """
    Run walk-forward backtest with given config.
    
    Process:
    1. For each week in test period
    2. Every N weeks, retrain model on past data
    3. Generate predictions for all stocks
    4. Select top N stocks (or bottom if inverted)
    5. Apply filters (momentum, sector)
    6. Calculate returns with slippage and stop-loss
    7. Track portfolio value
    """
    for week_idx in range(len(mondays) - 1):
        # Retrain periodically
        if week_idx % config.retrain_weeks == 0:
            model = train_model(train_data, config)
        
        # Generate predictions
        predictions = {}
        for ticker, df in all_price_data.items():
            predictions[ticker] = predict(model, df, config)
        
        # Select stocks
        sorted_preds = sorted(predictions.items(), 
                             key=lambda x: x[1], 
                             reverse=not config.invert_signal)
        
        # Apply filters and calculate returns
        # ...
```

---

## How to Run Tests

### Run All Experiments
```bash
cd /Users/pat0216/Documents/codin/paper-trader
python -u scripts/ml_improvement_experiments.py 2>&1 | tee results/ml_experiments_log.txt
```

### Check Progress
```bash
# View last 50 lines of log
tail -50 results/ml_experiments_log.txt

# Check if still running
ps aux | grep ml_improvement | grep -v grep

# View results (when complete)
cat results/ml_experiments_results.json | python -m json.tool
```

### Run Single Experiment (for debugging)
```python
# In Python
from scripts.ml_improvement_experiments import *

# Load data
all_price_data, spy_data = load_data()

# Run specific experiment
config = ExperimentConfig(
    name="test_inversion",
    invert_signal=True,
    start_year=2024,
    end_year=2025,
    retrain_weeks=4
)
result = run_experiment(config, all_price_data, spy_data)
print(result)
```

---

## Interpreting Results

### Key Metrics
| Metric | Good | Bad | Notes |
|--------|------|-----|-------|
| Total Return | > SPY | < 0% | Beat benchmark |
| Sharpe Ratio | > 1.0 | < 0 | Risk-adjusted return |
| Max Drawdown | < 20% | > 40% | Worst peak-to-trough |
| Win Rate | > 55% | < 45% | % profitable weeks |

### Decision Matrix
```
If Signal Inversion > Baseline by >20%:
    → Model is predicting backwards
    → Consider inverting in production

If Momentum Features > Baseline:
    → Replace current features with momentum
    → Update src/features/indicators.py

If Stop-Loss improves Sharpe:
    → Add stop-loss to production
    → Update src/backtesting/backtester.py

If Combined is best:
    → Implement all improvements
    → Full validation required
```

### Example Output
```
================================================================================
EXPERIMENT RESULTS SUMMARY
================================================================================
Experiment                       Return    vs SPY   Sharpe    MaxDD   WinRate
--------------------------------------------------------------------------------
0_Baseline_Current               -25.0%    -50.0%    -1.20    45.0%    42.0%
1_Signal_Inversion               +15.0%    -10.0%    +0.45    25.0%    52.0%  ← Better!
2_Momentum_Features              +22.0%     -3.0%    +0.75    20.0%    55.0%  ← Best!
...
--------------------------------------------------------------------------------
 Best Sharpe: 2_Momentum_Features
 Best Return: 2_Momentum_Features
```

---

## Validation Protocol

### Phase 1: Quick Test (CURRENT)
- Period: 2024-2025 (1 year)
- Retraining: Monthly (4 weeks)
- Purpose: Rapid iteration, find promising approaches
- Time: ~6-7 hours for all 8 experiments

### Phase 2: Full Validation (NEXT)
For top 2-3 experiments from Phase 1:
```python
config = ExperimentConfig(
    name="validation_best_approach",
    start_year=2017,
    end_year=2025,
    retrain_weeks=1,  # Weekly retraining
)
```
- Period: 2017-2025 (8 years)
- Retraining: Weekly
- Purpose: Confirm performance across market regimes
- Time: ~2-3 hours per experiment

### Phase 3: Robustness Tests
```python
# Test different parameters
for top_n in [5, 10, 15, 20]:
    for stop_loss in [0.05, 0.08, 0.10, 0.15]:
        config = ExperimentConfig(
            top_n=top_n,
            stop_loss_pct=stop_loss,
            # ... best settings from Phase 2
        )
        run_experiment(config, ...)
```

### Phase 4: Out-of-Sample Test
- Hold out 2024-2025 data completely
- Train on 2017-2023
- Test on 2024-2025
- No retraining during test period

---

## Production Deployment

### If Momentum Features Win
Update `src/features/indicators.py`:
```python
# Add momentum features
def generate_features(df, include_target=False):
    # Keep existing indicators or replace...
    
    # Add momentum
    df['MOM_12_1'] = close.shift(21) / close.shift(252) - 1
    df['MOM_6_1'] = close.shift(21) / close.shift(126) - 1
    # ...
```

### If Signal Inversion Needed
Update `src/models/predictor.py`:
```python
class EnsemblePredictor:
    def predict(self, df):
        pred = self._raw_predict(df)
        return -pred  # Invert signal
```

### If Stop-Loss Needed
Update `src/backtesting/backtester.py`:
```python
class BacktestConfig:
    stop_loss_pct: float = 0.08  # Add stop-loss parameter
```

### If Horizon Weights Need Change
Update `src/models/predictor.py`:
```python
HORIZON_WEIGHTS = {1: 0.1, 5: 0.3, 20: 0.6}  # New weights
```

---

## Checklist

### Before Running
- [ ] Verify data cache exists (`data/cache/`)
- [ ] Check available disk space (results can be large)
- [ ] Ensure no other heavy processes running

### During Run
- [ ] Monitor with `tail -f results/ml_experiments_log.txt`
- [ ] Check CPU usage isn't thermal throttling
- [ ] Verify experiments are progressing

### After Completion
- [ ] Review `results/ml_experiments_results.json`
- [ ] Identify best 2-3 experiments
- [ ] Run full validation on winners
- [ ] Document findings
- [ ] Plan production changes

---

## Quick Reference Commands

```bash
# Check experiment status
tail -50 results/ml_experiments_log.txt

# View results JSON
cat results/ml_experiments_results.json | python -m json.tool

# Kill if needed
pkill -f ml_improvement_experiments.py

# Restart experiments
python -u scripts/ml_improvement_experiments.py 2>&1 | tee results/ml_experiments_log.txt

# Run validation on best experiment
python -c "
from scripts.ml_improvement_experiments import *
all_price_data, spy_data = load_data()
config = ExperimentConfig(
    name='validation_BEST_EXPERIMENT_NAME',
    start_year=2017,
    end_year=2025,
    retrain_weeks=1,
    # ... copy settings from best experiment
)
result = run_experiment(config, all_price_data, spy_data)
print(result)
"
```

---

## Contact / Notes

- Original analysis in `results/QUANT_ANALYSIS_REPORT.md`
- Experiment script: `scripts/ml_improvement_experiments.py`
- Production predictor: `src/models/predictor.py`

**Key Insight:** The model likely learned mean-reversion patterns from RSI/Bollinger features, but momentum dominates in equity markets. Switching to momentum features should align the model with market dynamics.

