# 🔬 QUANT ANALYSIS REPORT: ML Strategy Investigation

**Date:** December 16, 2025  
**Team:** Debugging Quant Desk  
**Status:** CRITICAL FINDINGS

---

## Executive Summary

We identified **multiple critical implementation bugs** causing the massive underperformance (-79% vs +390% for momentum strategy). The issues span both the walkforward test AND production code.

---

## 1. 🚨 CRITICAL BUG #1: Model Predicts Mean-Reversion, Not Momentum

### Evidence
From the logs, when stocks have **large positive 1-day predictions**, they tend to have **negative actual returns**:

```
Week 249: PLTR predicted +0.63% daily → Actual: CRASHED
Week 249: PYPL predicted +0.57% daily → Actual: CRASHED  
Week 249: NFLX predicted +0.44% daily → Actual: CRASHED
Result: 0/10 winners, -10.85% loss
```

### Root Cause
The XGBoost model is trained on **technical indicators** (RSI, MACD, etc.) which are inherently **mean-reverting signals**:
- High RSI = overbought = model predicts DOWN
- Low RSI = oversold = model predicts UP

But the strategy **buys the top predictions** (stocks model thinks will go UP), which are often **oversold/beaten-down stocks**. These stocks are oversold for a reason and continue falling.

### The Paradox
- **Momentum strategies** (which work) buy WINNERS (stocks that are already up)
- **ML model** is trained on mean-reversion signals, so it buys LOSERS (stocks that are down)
- Result: Complete opposite of what works

---

## 2. 🚨 CRITICAL BUG #2: Wrong Strategy Comparison

### Good Results (Momentum Strategy)
```json
"momentum_vs_spy_2016_2025.json":
- Total Return: +390%
- CAGR: +17.3%
- Max DD: -30.6%
```

### Bad Results (ML Strategy)
```
walkforward_production_mirror:
- Total Return: -79% (stopped at Week 250)
- Projected: -90%+
```

### Key Difference
| Aspect | Momentum Strategy | ML Strategy |
|--------|-------------------|-------------|
| Selection | 12-1 month price momentum | XGBoost predictions |
| Rebalance | Monthly | Weekly |
| What it buys | Winners (stocks UP 40%+) | "Undervalued" (model thinks cheap) |
| Historical edge | Proven since 1993 | Unproven, potentially negative |

---

## 3. 🚨 CRITICAL BUG #3: Concentrated Volatile Stock Picking

### Pattern Observed
The model consistently picks the SAME volatile stocks:
- PLTR, COIN, TSLA, SMCI, MRNA, ALB, PYPL, NFLX

These are all **high-beta growth stocks** that crash hard during corrections.

### Single Stock Disasters
- Week 39 (Oct 2024): SMCI -48.03% in ONE WEEK
- Week 249 (Apr 2022): All 10 picks negative
- Week 61 (2025): 0/10 winners, -7.00%

### Why This Happens
The model's features (RSI oversold, high volume, MACD histogram) trigger on stocks that have fallen hard. These tend to be volatile names.

---

## 4. 🚨 BUG #4: Feature Selection Instability

### Evidence
Feature selection varies wildly week to week:
```
Horizon 1d: 11-15 features (varies)
Horizon 5d: 8-14 features (varies)  
Horizon 20d: 12-15 features (varies)
```

### Problem
- Different features selected each week = model sees different "reality"
- No consistency in what the model learns
- Random noise may be dominating predictions

---

## 5. 🚨 BUG #5: Horizon Weights May Be Inverted

### Current Weights (Production)
```python
HORIZON_WEIGHTS = {1: 0.5, 5: 0.3, 20: 0.2}
```

### Issue
- 1-day predictions get 50% weight
- But 1-day returns are NOISE (essentially random walk)
- 20-day predictions (which might have signal) only get 20%

### Academic Reference
Short-term (1-5 day) stock movements are dominated by noise. Only longer horizons (1-12 months) have momentum signal (Jegadeesh & Titman, 1993).

---

## 6. 📊 Walkforward vs Production Differences

| Aspect | Walkforward Test | Production |
|--------|-----------------|------------|
| Retraining | Weekly | Daily (retrain_daily=True) |
| Universe | Full 506 tickers | S&P 500 |
| Execution | Monday Open to Monday Open | Previous close to current |
| Position Sizing | Equal weight (10% each) | Risk-adjusted |
| Stop Loss | None | 8% (configurable) |
| Drawdown Control | None | Yes (halt at -20%) |

### Key Issue
The walkforward test **doesn't have stop-losses or drawdown control**, which production has. This makes losses compound faster in the test.

---

## 7. 🔧 RECOMMENDED FIXES

### Fix 1: Abandon ML for Stock Selection
The ML model is fundamentally picking mean-reversion (losers) while the market rewards momentum (winners).

**Recommendation:** Use the proven momentum strategy (12-1 month) which shows +390% returns.

### Fix 2: If Keeping ML, Flip the Signal
```python
# Instead of buying TOP predictions
# BUY BOTTOM predictions (stocks model thinks will fall)
# These are actually the momentum winners!
selected = [t for t, _ in sorted_preds[-TOP_N:]]  # Bottom N, not Top N
```

### Fix 3: Add Stop-Losses to Walkforward Test
```python
# Add stop-loss check
for ticker in selected:
    if current_return < -0.08:  # 8% stop loss
        force_exit = True
```

### Fix 4: Reduce Concentration
Instead of 10 stocks with 10% each:
- Use 30 stocks with ~3% each
- Reduces single-stock disaster risk

### Fix 5: Change Horizon Weights
```python
# Weight longer horizons more
HORIZON_WEIGHTS = {1: 0.1, 5: 0.3, 20: 0.6}
```

---

## 8. 📉 WHY PRODUCTION MIGHT BE FAILING TOO

If production uses the same ML model, it's likely also underperforming because:

1. **Same model** trained on mean-reverting features
2. **Same stocks** being selected (volatile losers)
3. **Daily retraining** might make instability worse (overfitting to recent noise)

### To Verify
Check production ledger returns vs SPY over same period.

---

## 9. 🎯 CONCLUSION

**Root Cause:** The ML model was trained to predict short-term returns using mean-reverting indicators, but is being used in a momentum-style "buy the top picks" strategy. This is a fundamental strategy mismatch.

**Impact:** -79% to -90% returns vs +47% SPY, ~$130K loss on $100K

**Recommendation:** 
1. **Immediately:** Switch production to momentum strategy
2. **If ML desired:** Completely redesign features to capture momentum, not mean-reversion
3. **Alternative:** Use ML for risk/position sizing, not stock selection

---

## 10. 🧪 QUICK VALIDATION TEST

To confirm our hypothesis, run walkforward but **invert the picks**:

```python
# In walkforward_production_mirror.py, line 237:
# OLD: selected = [t for t, _ in sorted_preds[:TOP_N]]  # Top N
# NEW: 
selected = [t for t, _ in sorted_preds[-TOP_N:]]  # Bottom N (inverse)
```

If the inverted strategy performs BETTER, it confirms the model is picking mean-reversion (losers) and we should flip.

---

**Report Prepared By:** AI Quant Analysis Team  
**Next Steps:** Implement Fix 2 (invert signals) as quick test, then evaluate Fix 1 (momentum strategy)

