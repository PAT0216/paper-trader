# Results Directory Structure

This directory contains both **production** (live trading) and **research** (backtest) outputs.

## 📁 Directory Organization

### Production (Live Trading)
These files are updated by the live trading workflow and represent **real paper trading**:

- **`metrics.txt`** - Live model performance metrics
- **`feature_importance.png`** - Current production model feature importance
- **`selected_features.txt`** - Features used by production model
- **`trade_analysis.txt`** - Analysis of recent live trades
- **`live/`** - Live trading analysis and logs

### Research (Backtests)
Backtest results are stored in **dated subdirectories** for comparison:

- **`backtests/YYYY-MM-DD_description/`** - Archived backtest runs
  - Each folder contains a complete backtest: metrics, trades, equity curve
  - Allows comparing different strategies and time periods

### Latest Backtest (Dashboard Access)
These are **symlinks or copies** of the most recent backtest for quick dashboard access:

- **`backtest_metrics.json`** - Latest backtest performance metrics
- **`backtest_summary.txt`** - Latest backtest summary
- **`backtest_trades.csv`** - Latest backtest trade log

## 🔍 Quick Reference

| File | Type | Updated By | Purpose |
|------|------|------------|---------|
| `metrics.txt` | Production | Workflow | Live performance |
| `backtest_metrics.json` | Research | Manual | Latest backtest |
| `backtests/2025-12-11/` | Research | Manual | Archived backtest |
| `live/` | Production | Workflow | Live trading logs |

## 📊 Backtest Naming Convention

Backtest folders follow this pattern:
```
backtests/YYYY-MM-DD_description/
```

Examples:
- `backtests/2025-12-11_full_universe/`
- `backtests/2025-12-10_phase7_test/`
- `backtests/2025-12-09_no_stoploss/`

## ⚠️ Important Notes

1. **Never delete production files** (`metrics.txt`, `live/`) - these track real trading
2. **Backtest folders can be deleted** if you need space - they're for research only
3. **Dashboard reads from symlinks** (`backtest_*.json`) which point to latest backtest
4. **Production model** is in `models/xgb_model.joblib` (updated by workflow)

## 🎯 Workflow

1. **Run backtest** → Creates `backtests/YYYY-MM-DD/`
2. **Automatically updates** → `backtest_metrics.json` (symlink to latest)
3. **Dashboard reads** → `backtest_metrics.json`
4. **Live trading runs** → Updates `metrics.txt`, `ledger.csv`

This structure keeps research experiments organized while maintaining backward compatibility with existing dashboards and scripts.
