# Development Guidelines

> **FOR AI AGENTS AND DEVELOPERS**: Read this entire file before making ANY changes to this project.

---

## 🚨 Critical Rules (Non-Negotiable)

### 1. Data Freshness
**NEVER run scripts that modify ledgers/snapshots with stale local data.**

```bash
# ALWAYS do this first
git pull origin main

# Check if your local cache is current
sqlite3 data/market.db "SELECT MAX(date) FROM price_data;"
# If date is old, DO NOT run snapshot scripts locally
```

**Why**: Local `market.db` is often weeks old. Workflows use fresh cache from GitHub Actions. Pushing locally-generated snapshots will corrupt dashboard values.

### 2. Never Duplicate Trades
Workflows execute real trades. Running them multiple times on the same day may cause duplicate positions.

```bash
# Check last trade date before triggering
git log --oneline -5  # Look for "Daily trade YYYY-MM-DD"
```

### 3. Git Commit Conventions
```
🤖 ML: Daily trade YYYY-MM-DD
🧠 LSTM: Daily trade YYYY-MM-DD  
📈 Momentum: Monthly rebalance YYYY-MM-DD
📦 Cache + snapshot updated YYYY-MM-DD
fix: description
feat: description
docs: description
```

---

## Architecture Overview (v2.0.0)

```
paper-trader/
├── main.py                 # Entry point: --mode train|trade --strategy ml|lstm|momentum
├── src/
│   ├── models/             # ML models
│   │   ├── trainer.py      # XGBoost training
│   │   ├── training_utils.py   # Shared training utilities
│   │   └── lstm/           # LSTM model
│   ├── trading/            # Portfolio, execution
│   │   ├── portfolio.py    # Ledger management
│   │   ├── ledger_utils.py # Ledger parsing utilities
│   │   └── risk_manager.py # Position sizing
│   ├── data/               # Data layer
│   │   ├── cache.py        # SQLite cache
│   │   └── price_utils.py  # Price fetching utilities
│   └── backtesting/        # Walk-forward validation
├── scripts/
│   ├── utils/
│   │   └── compute_portfolio_snapshot.py  # --strategy ml|lstm|momentum
│   └── simulate_production.py  # 3-day trade simulation
├── data/
│   ├── ledgers/            # Trade ledgers (v2.0.0 - moved from root)
│   │   ├── ledger_ml.csv
│   │   ├── ledger_lstm.csv
│   │   └── ledger_momentum.csv
│   ├── snapshots/          # Per-strategy snapshots
│   ├── market.db           # SQLite price cache (NOT committed)
│   └── portfolio_snapshot.json  # Consolidated values for dashboard
├── tests/                  # 42 unit tests
└── .github/workflows/      # Automated trading
```

---

## ⚙️ Workflow System

### Schedule (UTC, Mon-Fri)

| Workflow | Time | Purpose |
|----------|------|---------|
| `cache_refresh.yml` | 21:00 | Update market.db with latest prices |
| `ml_trade.yml` | 21:30 | Execute ML strategy trades |
| `lstm_trade.yml` | 21:45 | Execute LSTM strategy trades |
| `momentum_trade.yml` | 22:00 (1st-3rd only) | Monthly rebalance |

### Why Staggered?
Prevents merge conflicts. Each workflow modifies shared files (ledgers, snapshots).

### Per-Strategy Snapshots (v1.9.0+)
Each workflow writes to its own file:
- `data/snapshots/ml.json` 
- `data/snapshots/lstm.json`
- `data/snapshots/momentum.json`

Then consolidates into `data/portfolio_snapshot.json`.

---

## 🧪 Testing Guidelines

### Safe Local Testing
```bash
# 1. Read-only operations (safe)
python main.py --mode predict --strategy ml  # Just predictions, no trades

# 2. Snapshot viewing (safe, but DON'T commit output)
python scripts/utils/compute_portfolio_snapshot.py --strategy ml

# 3. Backtest (safe - uses historical data only)
python run_backtest.py
```

### Dangerous Operations (Use Workflows Instead)
```bash
# ❌ DON'T run locally - use GitHub Actions
python main.py --mode trade --strategy ml  # Modifies ledger
```

### Unit Tests
```bash
pytest tests/ -v
```

---

## 📊 Ledger Format

Ledgers are now in `data/ledgers/ledger_*.csv`:
```csv
date,ticker,action,price,shares,amount,cash_balance,total_value,strategy,...
2026-01-02,AAPL,BUY,185.50,10,1855.00,8145.00,10000.00,ml,...
2026-01-02,PORTFOLIO,VALUE,1.0,0,0.0,0.0,10272.62,ml,...
```

**Important rows:**
- `BUY`/`SELL`: Actual trades
- `PORTFOLIO,VALUE`: End-of-day portfolio valuation (used by dashboard chart)
- `DEPOSIT`: Initial capital

---

## 🔧 Common Tasks

### Add a New Strategy
1. Create `src/strategies/new_strategy.py` extending `BaseStrategy`
2. Register in `src/strategies/registry.py`
3. Create `data/ledgers/ledger_newstrategy.csv` with DEPOSIT row
4. Create `.github/workflows/newstrategy_trade.yml` using `strategy-trade.yml` template
5. Update `compute_portfolio_snapshot.py` ledger dict
6. Stagger schedule to avoid conflicts

### Fix Dashboard Showing Wrong Values
1. Check `data/portfolio_snapshot.json` values
2. Compare against `data/ledgers/ledger_*.csv` PORTFOLIO,VALUE rows
3. Dashboard reads VALUE entries from ledgers for accurate display
4. If mismatch, wait for next workflow run (do not manually edit)

### Debug Workflow Failure
1. Check GitHub Actions logs for error
2. Common issues:
   - Disk space → Add cleanup step
   - Merge conflict → Stagger schedule or manual fix
   - Push failed → Check retry logs

---

## ⚠️ Known Gotchas

1. **tensorflow-macos vs tensorflow**: Use `tensorflow-cpu` in `environment.yml` for Docker builds
2. **yfinance rate limits**: Workflows may fail if called too frequently
3. **Streamlit caching**: Dashboard may show stale data - reboot app to refresh
4. **Market holidays**: Workflows run but may find no new data

---

## 📝 Before Submitting Changes

- [ ] `git pull origin main` first
- [ ] Run `pytest tests/ -v`
- [ ] Don't commit `data/market.db`, `data/snapshots/*.json` from local runs
- [ ] Check `.gitignore` for sensitive files
- [ ] Use conventional commit messages
- [ ] If modifying workflows, test with `workflow_dispatch` before merging

---

## 🔐 Secrets & Environment

Required in `.env` (local) or GitHub Secrets:
```
# Currently no external API keys required for core trading
# Future integrations may require:
# SEC_EDGAR_EMAIL=...      # For SEC filings (if enabled)
```

---

**Maintained by**: PAT0216  
**Last Updated**: 2026-01-17 (v2.0.0)
