# Paper Trader AI - Debug Session Progress & Complete Project Context

**Session Date:** January 21-26, 2026  
**Conversation ID:** 37d88143-60a0-462e-ae59-7cf61c3dcaf5  
**Author:** Prabuddha Tamhane  
**Version:** v2.0.0

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Bugs Fixed](#bugs-fixed)
3. [Multi-Consensus Analysis](#multi-consensus-analysis)
4. [Project Architecture](#project-architecture)
5. [Strategies Deep Dive](#strategies-deep-dive)
6. [Workflow System](#workflow-system)
7. [Data Pipeline](#data-pipeline)
8. [Configuration](#configuration)
9. [Current Portfolio State](#current-portfolio-state)
10. [Key Code Files](#key-code-files)
11. [Testing](#testing)
12. [Deferred Issues](#deferred-issues)
13. [Key Learnings](#key-learnings)

---

## Executive Summary

This session focused on diagnosing and resolving **5 critical bugs** preventing ledger updates and causing the dashboard to display stale data. Through multi-consensus analysis (5-role engineering review), we identified and fixed all issues across workflows and core Python code. The system is now fully operational with daily trades executing correctly.

---

## Session: February 8, 2026 - AWS Lambda Integration

**Session ID:** c326d96c-f45b-4fe6-8ba8-c0a499fa7197

### Overview

This session completed AWS Lambda integration and fixed critical data integrity issues that arose during testing.

### Changes Made

#### 1. AWS Lambda Integration (PR #12: feature/aws-integration)

| File | Change |
|------|--------|
| `lambda_handler.py` | Complete handler for serverless trading |
| `Dockerfile.lambda` | Container build for Lambda |
| `.github/workflows/aws-ecr-push.yml` | Automatic ECR push on merge to main |
| `requirements-lambda.txt` | Lightweight dependencies |

**Lambda Architecture:**
- EventBridge: Mon-Fri 4:30 PM PT
- S3: `paper-trader-data-pat0216` for market.db
- ECR: `paper-trader-lambda:latest`
- GitHub: Source of truth for ledgers

#### 2. Data Integrity Fixes

**Problem**: Lambda test run corrupted historical data by downloading from empty S3, creating fresh ledger, and overwriting GitHub.

**Fix 1: Lambda downloads from GitHub first**
```python
# lambda_handler.py
def download_from_github():
    """Download ledger and portfolio_snapshot from GitHub (source of truth)."""
```

**Fix 2: SPY benchmark preserves start_date**
```python
# compute_portfolio_snapshot.py - add_spy_benchmark()
# Priority 1: Preserve existing start_date from spy_benchmark.json
# Priority 2: Use ledger min date
# Priority 3: Fallback to TRADING_START_DATE constant
```

**Fix 3: Added fallback constant**
```python
TRADING_START_DATE = "2025-10-01"  # Known first trading day
```

#### 3. Data Restoration

Restored from feature branch to main:
- `ledger_momentum.csv`: 15 -> 141 lines
- `spy_benchmark.json`: 13 -> 365 lines

#### 4. Documentation

- Updated README.md with AWS Lambda section
- Added architecture, data flow, environment variables, build/deploy, cost estimates

### Verification

| Check | Status |
|-------|--------|
| CI tests | Passed |
| ECR build | Complete |
| Dashboard momentum graph | Restored (Oct 2025-Feb 2026) |
| SPY benchmark | Correct (+3.62%) |
| Line counts | Correct (141/365) |

---

## Bugs Fixed

### Bug 1: Docker Container Missing GITHUB_ACTIONS Env
**File:** `.github/workflows/strategy-trade.yml`  
**Symptom:** `compute_portfolio_snapshot.py` aborted with "LOCAL EXECUTION DETECTED" in GitHub Actions  
**Root Cause:** Docker containers don't inherit environment variables automatically. The `GITHUB_ACTIONS` var wasn't passed to containers.  
**Fix:** Added `-e GITHUB_ACTIONS=true` to docker run commands  
**Commit:** `d5d86d1`

```yaml
# Before
docker run --rm ... paper-trader-dev ...

# After
docker run --rm -e GITHUB_ACTIONS=true ... paper-trader-dev ...
```

### Bug 2: Incorrect Ledger Paths in update_strategy_value()
**File:** `scripts/utils/compute_portfolio_snapshot.py`  
**Symptom:** Momentum and LSTM ledgers not receiving daily VALUE updates  
**Root Cause:** During v2.0.0 refactoring, ledger paths were moved to `data/ledgers/` but this function wasn't updated  
**Fix:** Updated `ledger_paths` dictionary  
**Commit:** `88e7d7b`

```python
# Before
ledger_paths = {
    'momentum': 'ledger_momentum.csv',
    'ml': 'ledger_ml.csv',
    'lstm': 'ledger_lstm.csv',
}

# After
ledger_paths = {
    'momentum': 'data/ledgers/ledger_momentum.csv',
    'ml': 'data/ledgers/ledger_ml.csv',
    'lstm': 'data/ledgers/ledger_lstm.csv',
}
```

### Bug 3: Snapshot JSONs Not Committed
**File:** `.github/workflows/cache_refresh.yml`  
**Symptom:** Dashboard showing stale `price_date` despite workflow running  
**Root Cause:** `git add` command excluded `data/snapshots/*.json`  
**Fix:** Added snapshot files to commit  
**Commit:** `4e5a82d`

```yaml
# Before
git add data/portfolio_snapshot.json data/ledgers/ledger_*.csv

# After
git add data/portfolio_snapshot.json data/ledgers/ledger_*.csv data/snapshots/*.json
```

### Bug 4: Cache Skipping Today's Data
**File:** `src/data/loader.py`  
**Symptom:** Market data stuck on previous day's date  
**Root Cause:** Date comparison used `>=` instead of `>`. When `fetch_start == today`, it returned cached data instead of fetching.  
**Fix:** Changed comparison operator  
**Commit:** `1b43659`

```python
# Before
if fetch_start.date() >= today.date():
    return cache_df

# After
if fetch_start.date() > today.date():
    return cache_df
```

### Bug 5: Missing GITHUB_ACTIONS Env in Monthly Retrain
**File:** `.github/workflows/monthly_retrain.yml`  
**Symptom:** Same "LOCAL EXECUTION" issue could occur during model retraining  
**Fix:** Added `-e GITHUB_ACTIONS=true` to docker run commands  
**Part of commit:** `ffbcfb7`

---

## Multi-Consensus Analysis

We performed a 5-role engineering consensus analysis:

| Role | Focus Area |
|------|------------|
| Workflow Engineer | YAML syntax, scheduling, dependencies |
| Data Engineer | Data flows, caching, storage |
| QA Engineer | Test coverage, edge cases |
| DevOps Engineer | Docker, secrets, deployment |
| Code Analyst | Python logic, imports, design |

### Additional Fixes Implemented

| Fix | File | Change |
|-----|------|--------|
| Timezone consistency | `loader.py` | Use NYC timezone for date comparison |
| Syntax error | `ledger_utils.py` | Remove broken conditional dict column |

### Deferred Issues

| Priority | Issue | Mitigation |
|----------|-------|------------|
| Medium | Cache key race condition | Sequential scheduling |
| Medium | No market holiday detection | Acceptable behavior |
| Low | Complex retry logic | Low priority |
| Low | Hardcoded paths | Future refactor |

---

## Project Architecture

### Directory Structure
```
paper-trader/
├── main.py                         # Core trading logic (17KB)
├── config/
│   ├── settings.yaml               # Main configuration
│   ├── momentum_config.yaml        # Momentum strategy settings
│   └── backtest_settings.yaml      # Backtest parameters
├── src/
│   ├── strategies/                 # Strategy implementations
│   │   ├── base.py                 # BaseStrategy ABC
│   │   ├── momentum_strategy.py    # 12-1 momentum factor
│   │   ├── ml_strategy.py          # XGBoost ensemble
│   │   ├── lstm_strategy.py        # Neural network
│   │   └── registry.py             # Factory pattern
│   ├── models/
│   │   ├── trainer.py              # XGBoost training
│   │   ├── training_utils.py       # Shared utilities
│   │   └── lstm/                   # LSTM model files
│   ├── trading/
│   │   ├── portfolio.py            # Portfolio & ledger mgmt
│   │   ├── ledger_utils.py         # Ledger utilities
│   │   └── risk_manager.py         # Position sizing
│   ├── data/
│   │   ├── loader.py               # Data fetching (yfinance)
│   │   ├── cache.py                # SQLite cache
│   │   └── price_utils.py          # Price utilities
│   ├── features/                   # Technical indicators
│   └── backtesting/                # Backtest engine
├── scripts/
│   ├── utils/
│   │   └── compute_portfolio_snapshot.py  # Snapshot updates
│   └── simulate_production.py      # Production simulation
├── dashboard/
│   └── app.py                      # Streamlit dashboard
├── data/
│   ├── ledgers/                    # Trade ledgers
│   │   ├── ledger_ml.csv
│   │   ├── ledger_lstm.csv
│   │   └── ledger_momentum.csv
│   ├── snapshots/                  # Per-strategy snapshots
│   │   ├── ml.json
│   │   ├── lstm.json
│   │   └── momentum.json
│   ├── market.db                   # SQLite cache (1.9M lines)
│   └── portfolio_snapshot.json     # Consolidated snapshot
├── models/
│   ├── xgb_model.joblib            # Trained XGBoost model
│   ├── xgb_model.json              # Model metadata
│   └── model_metadata.json         # Training metadata
├── tests/                          # 75 unit tests
└── .github/workflows/              # CI/CD automation (10 files)
```

### Data Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   yfinance      │───▶│   market.db     │───▶│    loader.py    │
│   (Data Source) │    │   (SQLite)      │    │   (Fetcher)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                       ┌───────────────────────────────┘
                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Strategies    │───▶│   portfolio.py  │───▶│   ledger_*.csv  │
│   (Signals)     │    │   (Execution)   │    │   (History)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                       ┌───────────────────────────────┘
                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   snapshot.json │◀───│   cache_refresh │◀───│   GitHub        │
│   (Dashboard)   │    │   (Workflow)    │    │   Actions       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## Strategies Deep Dive

### 1. Momentum Strategy
**File:** `src/strategies/momentum_strategy.py`  
**Schedule:** Monthly (1st trading day)  
**Universe:** S&P 500 (~503 stocks)

**Algorithm:**
1. Fetch 12-month price history for all S&P 500 stocks
2. Calculate momentum = (P_now / P_12months_ago) - 1
3. Skip the most recent month (momentum reversal effect)
4. Rank stocks by 12-1 month momentum
5. Select top 10 stocks
6. Apply 15% per-stock, 30% per-sector limits
7. Execute trades with 5 bps slippage
8. Check 15% stop-loss daily

**Current Holdings (Jan 23, 2026):**
| Ticker | Shares | Description |
|--------|--------|-------------|
| LRCX | 6 | Lam Research |
| WBD | 41 | Warner Bros Discovery |
| APP | 1 | AppLovin |
| NEM | 10 | Newmont Mining |
| HOOD | 8 | Robinhood |
| MU | 4 | Micron Technology |
| STX | 3 | Seagate |
| WDC | 6 | Western Digital |
| PLTR | 5 | Palantir |
| AVGO | 2 | Broadcom |

### 2. ML Strategy (XGBoost Ensemble)
**File:** `src/strategies/ml_strategy.py`  
**Schedule:** Daily (weekdays)  
**Model:** XGBoost Regressor

**Features (15 total):**
- RSI (7, 14, 21 day)
- MACD, MACD Signal, MACD Histogram
- Bollinger Bands (Upper, Lower, Width)
- ATR (14 day)
- Volume SMA ratio
- Returns (1d, 5d, 20d)
- Volatility (20d)

**Ensemble Weights:**
| Horizon | Weight | Purpose |
|---------|--------|---------|
| 1-day | 50% | Short-term signals |
| 5-day | 30% | Weekly trends |
| 20-day | 20% | Monthly context |

**Trading Logic:**
- Top 10% ranked → BUY
- Bottom 10% ranked → SELL
- Middle 80% → HOLD

### 3. LSTM Strategy
**File:** `src/strategies/lstm_strategy.py`  
**Schedule:** Daily (weekdays)  
**Model:** TensorFlow LSTM Neural Network

**Architecture:**
- Sequence length: 60 days
- LSTM layers with dropout
- Dense output layer
- Trained monthly (monthly_retrain.yml)

---

## Workflow System

### Workflow Files
| File | Purpose | Schedule |
|------|---------|----------|
| `cache_refresh.yml` | Update market data + snapshots | Daily 9:00 PM UTC |
| `ml_trade.yml` | Trigger ML strategy | Mon-Fri 9:30 PM UTC |
| `lstm_trade.yml` | Trigger LSTM strategy | Mon-Fri 9:45 PM UTC |
| `momentum_trade.yml` | Trigger Momentum strategy | 1st-3rd of month, 9:30 PM UTC |
| `strategy-trade.yml` | Reusable trade workflow | Called by strategy workflows |
| `monthly_retrain.yml` | Retrain LSTM model | 2nd of month, 10 PM UTC |
| `universe_refresh.yml` | Update S&P 500 list | 1st of month, 8 PM UTC |
| `ci_tests.yml` | Run test suite | On push/PR |

### Execution Order (Daily)
```
9:00 PM UTC   → cache_refresh.yml (prices + snapshots)
9:30 PM UTC   → ml_trade.yml → strategy-trade.yml
9:45 PM UTC   → lstm_trade.yml → strategy-trade.yml
```

### Key Workflow Details

**cache_refresh.yml:**
- Updates SQLite cache with latest prices
- Runs `compute_portfolio_snapshot.py --mode cache_refresh`
- Updates VALUE entries in all ledgers
- Commits: `portfolio_snapshot.json`, `ledger_*.csv`, `snapshots/*.json`

**strategy-trade.yml:**
- Reusable workflow called by individual strategies
- Runs Docker container with GITHUB_ACTIONS=true
- Executes: `python main.py --strategy $STRATEGY`
- Handles git push with 3-retry loop

---

## Data Pipeline

### SQLite Cache (`data/market.db`)
- **Size:** ~1.9 million lines
- **Tickers:** 503 S&P 500 stocks
- **History:** 8+ years of daily OHLCV data
- **Source:** yfinance API

### Ledger Format
```csv
date,ticker,action,price,shares,value,cash,portfolio_value,strategy,signal
2026-01-23,GLW,BUY,48.00,11,528.00,6476.15,10676.42,ml,
```

| Column | Description |
|--------|-------------|
| date | Trade date (YYYY-MM-DD) |
| ticker | Stock symbol |
| action | BUY, SELL, or VALUE (daily snapshot) |
| price | Execution price |
| shares | Number of shares |
| value | Trade value (price × shares) |
| cash | Remaining cash |
| portfolio_value | Total portfolio value |
| strategy | momentum, ml, or lstm |
| signal | Model signal (optional) |

### Snapshot Format (`portfolio_snapshot.json`)
```json
{
  "timestamp": "2026-01-23T21:59:58.562445Z",
  "price_date": "2026-01-23",
  "initial_capital": 10000,
  "portfolios": {
    "momentum": { "value": 11801.98, "return_pct": 18.02, ... },
    "ml": { "value": 10676.42, "return_pct": 6.76, ... },
    "lstm": { "value": 10769.21, "return_pct": 7.69, ... }
  },
  "benchmark": {
    "ticker": "SPY",
    "return_pct": 3.41,
    "value": 10341.34
  }
}
```

---

## Configuration

### settings.yaml
```yaml
universe:
  type: "sp500"  # Full S&P 500

model:
  training_period: "max"
  retrain_daily: true
  threshold_buy: 0.55
  threshold_sell: 0.45

portfolio:
  initial_cash: 10000.0
  min_cash_buffer: 200.0

risk:
  stop_loss_pct: 0.15        # 15% stop-loss
  drawdown_warning: 0.15     # Reduce sizes at -15%
  drawdown_halt: 0.20        # Stop buys at -20%
  drawdown_liquidate: 0.25   # Liquidate at -25%
  max_position_pct: 0.15     # Max 15% per stock
  max_sector_pct: 0.30       # Max 30% per sector
```

---

## Current Portfolio State

**As of:** January 23, 2026  
**Initial Capital:** $10,000

| Portfolio | Value | Return | vs SPY |
|-----------|-------|--------|--------|
| **Momentum** | $11,801.98 | +18.02% | +14.61% |
| **LSTM** | $10,769.21 | +7.69% | +4.28% |
| **ML** | $10,676.42 | +6.76% | +3.35% |
| **SPY (Benchmark)** | $10,341.34 | +3.41% | — |

**Winner:** Momentum strategy, outperforming by 14.61%

---

## Key Code Files

### main.py (17KB)
Entry point for all trading operations:
- Argument parsing (--strategy, --portfolio, --mode)
- Strategy instantiation via registry
- Trade execution with risk management
- Ledger updates

### src/trading/portfolio.py
Portfolio management:
- `execute_trades()` - Execute buy/sell orders
- `update_ledger()` - Log trades to CSV
- `apply_slippage()` - 5 bps transaction costs
- `get_portfolio_value()` - Calculate current value

### scripts/utils/compute_portfolio_snapshot.py
Snapshot management:
- `update_strategy_value()` - Add VALUE rows to ledgers
- `consolidate_snapshots()` - Merge per-strategy JSONs
- `is_github_actions()` - Safety check for CI/CD

### src/data/loader.py
Data fetching:
- `fetch_data()` - Main data retrieval
- `_fetch_single_ticker()` - Cache-aware fetching
- NYC timezone for consistent date handling

---

## Testing

### Test Suite
- **Total Tests:** 75
- **Passing:** 72
- **Known Failures:** 3 (pre-existing, unrelated to bugs fixed)

### Test Files
| File | Coverage |
|------|----------|
| `test_risk.py` | Risk manager |
| `test_dual_portfolio.py` | Portfolio isolation |
| `test_transactions.py` | Transaction costs |
| `test_momentum_no_bias.py` | Momentum strategy |

### Running Tests
```bash
python -m pytest tests/ -v
```

---

## Deferred Issues

### Medium Priority
1. **Cache Key Race Condition**
   - Risk: Concurrent workflows could corrupt cache
   - Mitigation: Sequential scheduling (9:00, 9:30, 9:45 PM UTC)
   - Future: Implement cache locking

2. **Market Holiday Detection**
   - Risk: Workflows run on holidays, find no new data
   - Mitigation: Currently logs "already has entry"
   - Future: Add holiday calendar check

### Low Priority
1. **Hardcoded Paths**
   - Issue: Ledger paths scattered across 10+ files
   - Solution: Create `config.py` with path constants

2. **Complex Git Retry Logic**
   - Issue: 3-retry loop with exponential backoff is verbose
   - Solution: Extract to reusable action

---

## Key Learnings

1. **Docker containers don't inherit env vars** - Must explicitly pass with `-e`

2. **Date comparisons are subtle** - `>=` vs `>` can cause off-by-one errors in caching

3. **Git add patterns must be explicit** - Wildcards in workflows need full paths

4. **Multi-consensus analysis catches hidden bugs** - 5 perspectives found issues a single reviewer missed

5. **Timezone matters** - NYC timezone ensures consistent date handling with market hours

6. **Safety mechanisms are critical** - `is_github_actions()` check prevents local execution from corrupting production data

---

## Timeline

| Date | Action |
|------|--------|
| Jan 21 | Initial diagnosis, Docker env issue found |
| Jan 21 | Fixed strategy-trade.yml (GITHUB_ACTIONS) |
| Jan 21 | Fixed compute_portfolio_snapshot.py (ledger paths) |
| Jan 21 | Fixed cache_refresh.yml (snapshot commit) |
| Jan 21 | Fixed loader.py (date comparison) |
| Jan 21 | Multi-consensus analysis (7 issues found) |
| Jan 21 | Implemented 3 additional fixes |
| Jan 21 | Final production readiness check |
| Jan 22-23 | Trades executed successfully |
| Jan 24 | Verified momentum flat values = coincidence |
| Jan 26 | Documentation update, README enhancements |
| Jan 29 | Fixed LSTM crash: NaN price handling (Bug 6) |
| Jan 29 | Fixed KeyError: missing ticker in buy/sell loops (Bug 7) |
| Feb 2 | Diagnosed GitHub infrastructure outage (not code) |
| Feb 2 | Fixed git merge conflict in concurrent workflows (Bug 8) |
| Feb 2 | FAANG 6-role team consensus review: APPROVED |

---

## Commits Summary

| Commit | Description |
|--------|-------------|
| `d5d86d1` | Add GITHUB_ACTIONS env to strategy-trade.yml |
| `88e7d7b` | Fix ledger paths in update_strategy_value() |
| `4e5a82d` | Add snapshots/*.json to git add |
| `1b43659` | Fix date comparison in loader.py |
| `ffbcfb7` | Multi-consensus approved fixes |
| `e94bfe1` | README: add Sharpe/MaxDD, Docker docs, fix Makefile |
| `f411b17` | Add NaN price guards to risk_manager.py |
| `3a4b315` | Guard buy/sell loops against missing price data |
| `dd3ae1e` | Conflict-resilient push for concurrent workflows |

---

## Bugs Fixed (Continued)

### Bug 6: NaN Price Crash in LSTM Workflow (Jan 29)
**File:** `src/trading/risk_manager.py`, `main.py`  
**Symptom:** `ValueError: cannot convert float NaN to integer` during position sizing  
**Root Cause:** yfinance returned NaN for some ticker Close prices. NaN propagated to `int(max_position_value / current_price)`.  
**Fix:** 
1. Added NaN guards at start of `calculate_position_size()`
2. Filter NaN prices when building `current_prices` dict
**Commits:** `f411b17`

### Bug 7: KeyError for Filtered Tickers (Jan 29)
**File:** `main.py`  
**Symptom:** `KeyError: 'MMC'` when accessing `current_prices[ticker]`  
**Root Cause:** Ticker MMC was filtered from `current_prices` (due to NaN) but still appeared in `buy_candidates` list.  
**Fix:** Added existence checks in buy/sell loops: `if ticker not in current_prices: continue`  
**Commits:** `3a4b315`

### Bug 8: Git Merge Conflict in Concurrent Workflows (Feb 2)
**File:** `.github/workflows/strategy-trade.yml`  
**Symptom:** All trade workflows (ML, LSTM, Momentum) fail at "Push Changes" with `CONFLICT (content): Merge conflict in data/portfolio_snapshot.json`  
**Root Cause:** When workflows run concurrently, `git stash pop` fails because the stashed changes conflict with recently pushed changes.  
**Fix:** Replaced fragile stash-based retry with conflict-resilient logic:
1. Try simple push first
2. On conflict: accept remote's `portfolio_snapshot.json`, keep our strategy files
3. Re-run `--consolidate` to merge all strategy data
4. Commit and retry push
**Commits:** `dd3ae1e`

**Verification:**
- LSTM Strategy Trade #41: SUCCESS (after fix)
- ML Strategy Trade #51: SUCCESS (after fix)
- LSTM Strategy Trade #40: FAILED (before fix)
- ML Strategy Trade #50: FAILED (before fix)

---

## Session: February 2, 2026

### Issues Diagnosed and Fixed

1. **GitHub Infrastructure Outage:** Daily Cache Refresh #57 failed with "The job was not acquired by Runner of type hosted even after multiple attempts" - this was a GitHub-side issue, not code.

2. **Git Merge Conflict Race Condition:** All 3 trade workflows (ML, LSTM, Momentum) were failing at push step. Implemented conflict-resilient push logic with FAANG 6-role engineering review consensus.

### FAANG Team Consensus Review

| Role | Verdict |
|------|---------|
| Staff SWE (System Design) | APPROVED |
| SRE (Reliability) | APPROVED |
| Senior Backend (Code Review) | APPROVED |
| ML Platform (Data Integrity) | APPROVED |
| Security Engineer | APPROVED |
| QA Engineer | APPROVED |

---

## System Status

**Current State:** Fully Operational

All workflows running correctly. Dashboard updating daily with fresh market data. No manual intervention required.

**Live Dashboard:** [paper-trader-ai.streamlit.app](https://paper-trader-ai.streamlit.app/)

---

*Generated: February 2, 2026*
