# Momentum Strategy

## Overview

The **primary production strategy** using 12-month momentum factor with 15% daily stop-loss.

Based on academic research by Jegadeesh & Titman (1993) and Fama-French factor investing.

---

## Performance (Unbiased Point-in-Time Backtest 2016-2025)

| Metric | Momentum | S&P 500 |
|--------|----------|---------|
| **CAGR** | 17.3% | 15.0% |
| **Total Return** | 391% | 303% |
| **Best Year** | +40.0% (2017) | +31.1% (2019) |
| **Worst Year** | +4.2% (2018) | -18.6% (2022) |

> ⚠️ **Note**: Results use **point-in-time S&P 500 universe** (survivorship bias removed). Strategy beat S&P 500 in 6/10 years.

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
