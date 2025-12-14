# Momentum Strategy

## Overview

This is a **long-only momentum strategy** based on academic research by Jegadeesh & Titman (1993) and Fama-French factor investing.

**Configuration:** 12-month momentum lookback with 15% daily stop-loss

## Performance (Backtested 2016-2023)

| Metric | Value |
|--------|-------|
| **CAGR** | 37.4% |
| **Sharpe Ratio** | 1.97 |
| **Total Return** | 1,169% |
| vs SPY | 169% |

### Bear Market Performance (2022)
- Strategy: **+19.0%**
- SPY: **-18.6%**
- Outperformance: **+37.6%**

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
  - This prevents catastrophic losses
```

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

## Usage

```bash
# Run with momentum strategy
python main.py --strategy momentum

# Or use make
make trade-momentum
```

## Schedule

| Job | When | Action |
|-----|------|--------|
| Daily Monitor | Mon-Fri 4:30 PM ET | Check stop-loss, update PnL |
| Monthly Rebalance | 1st trading day of month | Execute trades |

## Files

| File | Purpose |
|------|---------|
| `src/strategies/momentum_strategy.py` | Strategy implementation |
| `config/momentum_config.yaml` | Configuration |
| `results/full_universe_comparison.json` | Backtest results |
