# Momentum Strategy

## Overview

The **primary production strategy** using 12-month momentum factor with 15% daily stop-loss.

Based on academic research by Jegadeesh & Titman (1993) and Fama-French factor investing.

---

## Performance (Backtested 2016-2023)

| Metric | Momentum | S&P 500 |
|--------|----------|---------|
| **CAGR** | 37.4% | 13.2% |
| **Sharpe Ratio** | 1.97 | 0.87 |
| **Total Return** | 1,169% | 169% |

### Bear Market (2022)
- **Momentum**: +19.0%
- **S&P 500**: -18.6%
- **Outperformance**: +37.6%

---

## How It Works

### 1. Stock Selection (Monthly)
```
1. Calculate 12-month return for each S&P 500 stock
   (excluding last month to avoid short-term reversal)
2. Rank all stocks by momentum
3. Select top 10 stocks
4. Equal-weight allocation (~10% each)
```

### 2. Risk Control (Daily)
```
For each position:
  - Track entry price
  - If current price drops 15% below entry → SELL immediately
```

---

## Configuration

Edit `config/momentum_config.yaml`:

```yaml
strategy:
  lookback_days: 252  # 12-month momentum
  n_stocks: 10        # Number of positions
  
capital:
  initial: 100000     # Starting capital ($)
  
risk:
  stop_loss_pct: 0.15 # 15% daily stop-loss
```

---

## Usage

### Command Line
```bash
# Run momentum strategy with isolated portfolio
python main.py --strategy momentum --portfolio momentum
```

### GitHub Actions
1. Go to **Actions** tab
2. Select **"Momentum Strategy Trade"**
3. Click **"Run workflow"**

---

## Workflow Schedule

| Workflow | Schedule | Action |
|----------|----------|--------|
| **Momentum Strategy Trade** | 1st-3rd of month, 9:30 PM UTC | Monthly rebalance |

The workflow runs on days 1-3 to catch the first trading day regardless of weekends/holidays.

---

## Files

| File | Purpose |
|------|---------|
| `src/strategies/momentum_strategy.py` | Strategy implementation |
| `config/momentum_config.yaml` | Configuration |
| `ledger_momentum.csv` | Trade history |
| `.github/workflows/momentum_trade.yml` | Automation |

---

## Dual Portfolio Integration

This strategy runs independently alongside the ML strategy:

| Portfolio | Ledger | Strategy |
|-----------|--------|----------|
| Momentum | `ledger_momentum.csv` | This strategy |
| ML | `ledger_ml.csv` | XGBoost predictions |

Compare performance in the **Streamlit Dashboard**:
```bash
cd dashboard && streamlit run app.py
```
