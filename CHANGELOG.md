# Changelog

All notable changes to the Paper Trader AI project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.9.3] - 2026-01-16

### Fix Momentum Missing from Chart (Daily VALUE Rows for Monthly Strategies)

Momentum strategy was missing from the dashboard chart because it only got VALUE rows when trading (monthly), not daily.

#### Root Cause
- `update_strategy_value()` was reading existing VALUE rows from ledgers
- Momentum only trades monthly, so its ledger stopped getting updates after the last trade
- No new VALUE rows = no data points on the chart

#### Fixed
Rewrote `update_strategy_value()` to COMPUTE values fresh from holdings + prices:
1. Read holdings/cash from strategy's JSON snapshot
2. Fetch latest prices from database
3. Compute portfolio value
4. Append VALUE row to ledger (for chart)
5. Update JSON snapshot (for cell values)

Now ALL strategies get daily VALUE rows via cache_refresh, regardless of trading frequency.

#### Files Changed
- `scripts/utils/compute_portfolio_snapshot.py`: Rewrote `update_strategy_value()`

---

## [1.9.2] - 2026-01-15

### Fix Missing Daily VALUE Rows in Ledgers

Dashboard graph was stuck at Jan 12/13 because no new VALUE rows were being created in ledgers.

#### Root Cause
- `process_single_strategy()` used `get_latest_date(DB_PATH)` to determine the date for VALUE rows
- Database MAX(date) was Jan 13 (cache_refresh ran before Jan 14 market data was available)
- When trading workflows ran on Jan 14, they tried to write VALUE rows for Jan 13
- Jan 13 VALUE rows already existed, so nothing was appended

#### Fixed
Changed `process_single_strategy()` to use `datetime.now().strftime('%Y-%m-%d')` (today's date) instead of database MAX(date). Since workflows only run on trading days (Mon-Fri cron at market close), today is always a valid trading date.

#### Files Changed
- `scripts/utils/compute_portfolio_snapshot.py`: Line ~240, use today's date for VALUE rows

---

## [1.9.1] - 2026-01-14

### Fix Stale Monthly Strategy Values in Dashboard

This release fixes an issue where monthly strategies (momentum) showed stale portfolio values in the dashboard between monthly rebalances.

#### Root Cause
- Dashboard reads values from `data/portfolio_snapshot.json`
- This file consolidates per-strategy snapshots from `data/snapshots/*.json`
- Monthly strategies like momentum only update their snapshot when they run (1st-3rd of month)
- Result: Stale values displayed for ~28 days each month

#### Fixed

**Modular Value Refresh (`--update-value-only`)**
- Added `update_strategy_value()` function to refresh JSON snapshot values from ledger VALUE rows
- Added `update_all_strategy_values()` to update all strategies at once
- Added `--update-value-only` CLI flag to `compute_portfolio_snapshot.py`
- `cache_refresh.yml` now calls this daily to keep all strategy values fresh

**How It Works**
- Ledger VALUE rows are authoritative (written by trading logic)
- Daily cache_refresh updates JSON snapshot values from ledgers
- Holdings/cash preserved from last actual strategy run
- No changes needed to individual strategy workflows

#### Files Changed
- `scripts/utils/compute_portfolio_snapshot.py`: Added value update functions
- `.github/workflows/cache_refresh.yml`: Uses `--update-value-only` flag

> [!TIP]  
> **To revert**: Change `--update-value-only` back to no argument in `cache_refresh.yml`, and remove the new functions from `compute_portfolio_snapshot.py`.

---

## [1.9.0] - 2026-01-03

### 🔧 Workflow Reliability & Conflict Prevention

This release fixes critical issues with GitHub Actions workflows that were causing merge conflicts and failed pushes when multiple strategies ran simultaneously.

#### Root Cause
- All 3 trading workflows (ML, LSTM, Momentum) were scheduled at the same time (21:30 UTC)
- All workflows wrote to the same `data/portfolio_snapshot.json` file
- Concurrent git pushes caused merge conflicts
- Push failures were silent (`echo "Push failed"` instead of `exit 1`)

#### Fixed

**Per-Strategy Snapshots (Conflict-Free Architecture)**
- Each workflow now writes to its own file: `data/snapshots/{strategy}.json`
- Automatic consolidation into `data/portfolio_snapshot.json` after each workflow
- No more merge conflicts - each strategy only modifies its own files

**Staggered Workflow Schedules**
| Workflow | Old Schedule | New Schedule |
|----------|--------------|--------------|
| ML | 21:30 UTC | 21:30 UTC |
| LSTM | 21:30 UTC | **21:45 UTC** (+15 min) |
| Momentum | 21:30 UTC | **22:00 UTC** (+30 min) |

**Robust Push Logic**
- 3 retry attempts with 10-second delays
- Fallback to merge if rebase fails
- Explicit `exit 1` on failure (no silent failures)
- Full git history fetch (`fetch-depth: 0`) for proper rebasing

**Disk Space Fixes**
- Added disk cleanup step (frees ~10GB before Docker build)
- Changed `tensorflow-macos` → `tensorflow-cpu` (smaller package)
- Moved Sentinull dependencies to separate `sentinull/requirements.txt`

#### Changed
- `scripts/utils/compute_portfolio_snapshot.py`: Added `--strategy` argument
- All trade workflows: Now pass `--strategy {ml|lstm|momentum}` flag
- Created `data/snapshots/` directory for per-strategy files

> [!CAUTION]
> **Local Testing Warning**: Never run snapshot scripts locally and commit the results without verifying your local `market.db` has current prices. Local caches are often stale. See `docs/DEVELOPMENT.md` for safe testing patterns.

---

## [1.8.0] - 2025-12-11

### 🧪 Phase 8: Comprehensive Testing & Validation

#### Test Results
- **Pytest Suite**: 48 tests passed, 2 skipped (optional deps)
- **Walk-Forward (2023-2024)**: 80.2% return, 34.2% CAGR beat SPY (58%)
- **Backtest (2023-2024)**: 59.4% return, 1.63 Sharpe, 10.6% max drawdown

### Fixed
- **Predictor API**: Fixed `model_path` argument (use `__new__` bypass)
- **BacktestConfig**: Added missing `use_dated_folders` attribute
- **Backtester API**: Moved `signal_generator` from `__init__` to `run()`
- **run_walkforward.py**: Added missing `json` import
- **test_explainability.py**: Added `pytest.importorskip` for `shap` module

---

## [1.7.1] - 2025-12-11

### Fixed
- **Workflow Crash**: Fixed `AttributeError: 'Portfolio' object has no attribute 'get_positions'` in `record_trade` by implementing the missing method and improving trade recording logic.
- **Ledger Calculation**: Fixed `total_value` double-counting bug on SELL orders (was adding sold value instead of subtracting).
- **Ledger Repair**: Added `repair_ledger.py` utility and repaired historical `total_value` column in `ledger.csv`.


## [1.7.0] - 2025-12-10

### 🛡️ Phase 7: Quant Risk Enhancements - Complete

This release implements institutional-grade risk controls validated through rigorous A/B testing.

### Added
- **DrawdownController**: Automatic position reduction (-15%), trading halt (-20%), emergency liquidation (-25%)
- **Position-Level Stop-Loss**: 15% stop-loss threshold (A/B tested)
- **Unbiased Walk-Forward A/B Test**: `run_unbiased_comparison.py` for proper strategy validation
- **Stop-Loss Threshold Sweep**: `run_stoploss_test.py` for comparing stop-loss levels
- **Ranking Metrics**: Spearman Rank Correlation and Top-10% Accuracy in model evaluation

### A/B Testing Findings
- **15% stop-loss outperforms no stop-loss** in walk-forward validation
- Tested on 75 random S&P 500 tickers (2018-2024)
- **Caveat**: Backtest returns may be inflated due to survivorship bias and position concentration

### Changed
- **Stop-Loss Threshold**: 8% → **15%** (A/B tested with diverse stocks - banks, energy, cyclicals)
- **Signal Method**: Z-Score → Fixed Threshold (A/B proved Fixed better: Sharpe 3.41 vs 0.99)
- **Risk Limits**: Max sector exposure 40% → 30%, cash buffer $100 → $200
- **Slippage**: 5 bps → 10 bps for realistic backtests

---

## [1.6.0] - 2025-12-10

### 🎉 Phase 6: Deployment & Reliability - Complete

This release fixes the GitHub Actions cache persistence and establishes reliable daily automated trading.

### Fixed
- **GitHub Actions Caching**: Switched from `actions/upload-artifact` to `actions/cache@v4`
  - Artifacts don't persist across workflow runs; cache does
  - Market data (169MB) now restored in ~3 seconds
  - Eliminates need to re-fetch 10+ years of S&P 500 data each run
- **Shell Script Exit Code**: Fixed `Check Cache Status` step failing when `universe_cache.csv` missing

### Changed
- **Workflow Duration**: ~25 minutes → ~5 minutes (with cache hit)
- **Cron Schedule**: Daily at 9 PM UTC / 1 PM PST (market close)

---

## [1.5.0] - 2025-12-10

### 🎉 Phase 5: Walk-Forward Validation & Hyperopt - Complete

True out-of-sample validation with model trained BEFORE each test period.

### Added
- **Walk-Forward Validation** (`run_walkforward.py`):
  - Trains on 2010 to (Year-1), tests on Year
  - No look-ahead bias: model trained before each test period
  - Results: 630% return vs SPY 234% (2015-2024)
- **Next-Day Open Execution**: Signals at T close, execute at T+1 open
- **Priority Ranking**: Stocks sorted by expected return (highest first gets capital)
- **Hyperparameter Optimization** (`run_hyperopt.py`):
  - TimeSeriesSplit cross-validation
  - Overfitting check (train/val gap)
  - Current params already optimal

### Fixed
- **Look-Ahead Bias**: Execution now uses T+1 Open, not T Close
- **Data Validation Filter**: 2010+ filter applied before validation (35/35 tickers pass)

---

## [1.4.0] - 2025-12-09

### 🎉 Phase 4: Data Infrastructure - Complete

SQLite caching and S&P 500 universe support.

### Added
- **SQLite Data Cache** (`src/data/cache.py`): 4.3M rows, 503 tickers
- **Incremental Fetching**: Only fetch new bars after initial load
- **S&P 500 Universe** (`src/data/universe.py`): Dynamic from Wikipedia
- **Macro Data** (`src/data/macro.py`): VIX and yield curve support

---

## [1.3.0] - 2025-12-09

### 🎉 Phase 3.6: Regime Detection & Multi-Horizon Ensemble

VIX-based defensive positioning and ensemble predictions.

### Added
- **VIX Regime Detection**: NORMAL (<25), ELEVATED (25-35), CRISIS (>35)
- **Position Multipliers**: 100% → 50% → 0% based on regime
- **Multi-Horizon Ensemble**: 1-day (50%), 5-day (30%), 20-day (20%) blend

---

## [1.2.0] - 2025-12-09

### 🎉 Phase 3.5: Enhanced Feature Engineering

Expanded from 9 to 15 technical indicators.

### Added
- **Volume Features**: OBV Momentum, Volume Ratio, VWAP Deviation
- **Volatility Features**: ATR %, Bollinger %B, Volatility Ratio
- **Dynamic Feature Selection**: Auto-filter features with <3% importance

---


## [1.1.0] - 2025-12-09

### 🎉 Phase 2: Testing & Validation - Complete

This release adds a comprehensive backtesting framework with professional quant metrics, transaction cost modeling, and regime-based performance analysis.

### Added

#### Backtesting Framework (`src/backtesting/`)
- **Event-Driven Engine** (`backtester.py`):
  - Walk-forward simulation over 7+ years of historical data
  - Multi-ticker portfolio backtesting
  - Risk manager integration for position sizing
  - Configurable rebalance frequency (daily/weekly/monthly)
- **Performance Metrics** (`performance.py`):
  - Sharpe ratio, Sortino ratio, Calmar ratio
  - Value at Risk (VaR) and CVaR (Expected Shortfall)
  - Alpha, Beta, Information Ratio
  - Win rate, profit factor, average holding period
- **Transaction Cost Modeling** (`costs.py`):
  - Slippage simulation (5 bps default)
  - Market impact for large orders (sqrt law)
  - Cost tracking and breakdown

#### Regime-Based Analysis
- Automatic market regime classification (bull/bear/crisis/sideways)
- Performance metrics split by regime
- Uses SMA_200 and volatility thresholds

#### Configuration (`config/backtest_settings.yaml`)
- Date range: 2017-2024 (covers all market regimes)
- Cost parameters (slippage, commission)
- Risk settings (position limits, sector limits)
- Benchmark ticker configuration

#### Makefile Commands
- `make backtest`: Full 2017-2024 backtest
- `make backtest-quick`: Quick 2023-2024 backtest

#### Unit Tests
- 16 new tests for backtesting module (`tests/test_backtester.py`)
- Total test count: 43 (27 Phase 1 + 16 Phase 2)

### Changed

#### Codebase Cleanup
- **Removed** legacy files: `run_bot.py`, `strategy.py`
- **Removed** empty `data/` directory
- **Updated** `.gitignore` with pytest cache and coverage files
- **Fixed** timezone handling in backtester (yfinance tz-aware dates)

#### Documentation
- **README.md**: Added Phase 2 features, updated project structure, marked Phase 2 complete in roadmap
- **MANUAL.md**: Added 99 lines of backtesting documentation
- **CHANGELOG.md**: This entry

## [1.0.0] - 2025-12-08

### 🎉 Phase 1: Critical Foundation - Complete

This release transforms the paper trader from a basic prototype into a professional-grade trading system with institutional-level risk controls and data quality assurance.

### Added

#### Risk Management System
- **Volatility-Adjusted Position Sizing**: Inverse relationship between asset volatility and position size
  - 30-day historical volatility calculation
  - Target volatility: 20% annualized
  - Scalar range: 0.67x to 1.5x of base position
- **Portfolio Constraints**:
  - Maximum position size: 15% of portfolio per asset
  - Maximum sector exposure: 40% per sector
  - Minimum cash buffer: $100 (configurable)
- **Value at Risk (VaR) Calculation**:
  - 1-day VaR at 95% confidence level
  - Historical simulation method over 30-day window
  - Portfolio-level aggregation
- **Pre-Trade Validation**:
  - Validates all BUY/SELL orders against risk limits
  - Prevents oversized positions and sector concentration
  - Ensures sufficient cash and shares before execution
- **Sector Classification**: 
  - Automatic categorization across 7 sectors
  - Real-time sector exposure tracking and reporting
- **Correlation Penalty**:
  - Reduces position size when sector > 25% of portfolio
  - Up to 50% reduction for highly concentrated sectors

#### Data Quality Validation
- **DataValidator Module**: Comprehensive data quality checks before model training
- **10 Validation Checks**:
  1. Empty DataFrame detection
  2. Required column verification (OHLC, Volume)
  3. DatetimeIndex validation
  4. Data freshness check (< 48 hours old)
  5. Missing value detection (< 5% threshold)
  6. Price validity (positive, > $0.01)
  7. OHLC relationship integrity (High ≥ Low, etc.)
  8. Outlier detection (> 10σ daily returns)
  9. Volume validity (no negative volumes)
  10. Duplicate date detection
- **Automatic Filtering**: Invalid tickers removed before model training
- **Detailed Reporting**: Summary with errors and warnings for each ticker

#### Testing Infrastructure
- **Unit Test Suite**: 27 comprehensive tests
  - 14 tests for `RiskManager` (position sizing, VaR, validation)
  - 13 tests for `DataValidator` (all 10 checks + edge cases)
- **Test Coverage**: pytest integration with coverage reporting
- **Edge Case Coverage**: Insufficient cash, oversized positions, invalid data, stale data

#### Enhanced Documentation
- **Updated README.md**:
  - Phase 1 feature highlights
  - Example output with risk metrics
  - Testing section
  - Updated roadmap showing Phase 1 complete
- **Comprehensive MANUAL.md**:
  - System architecture diagram
  - Complete risk management documentation
  - Data validation specification
  - Metrics interpretation guide
  - Testing guide
  - Troubleshooting section
- **Enhanced Makefile**:
  - `make test`: Run test suite
  - `make test-coverage`: Generate coverage report
  - `make validate`: Quick data quality check
  - `make status`: View recent transactions
  - `make results`: Display model metrics
  - `make help`: Command documentation
- **CHANGELOG.md**: This file

### Changed

#### Main Trading Pipeline (`main.py`)
- **Data Validation Step**: Added mandatory validation before model training
- **Risk-Adjusted Sizing**: Replaced naive equal-weight allocation with sophisticated position sizing
  - Considers volatility, sector exposure, correlation
  - Dynamically adjusts to portfolio state
- **Pre-Trade Checks**: All trades validated for compliance before execution
- **Enhanced Logging**:
  - Current sector exposure display
  - Risk-adjusted sizing rationale for each trade
  - Portfolio VaR reporting
  - Largest sector exposure tracking

#### Requirements (`requirements.txt`)
- Added `pytest` for test suite
- Added explicit dependencies: `scikit-learn`, `numpy`, `pandas`, `pyyaml`

### Performance Improvements
- **Smarter Position Sizing**: Reduces risk exposure to volatile assets
- **Data Quality Gates**: Prevents model training on corrupted data
- **Diversification**: Automatic sector balancing prevents concentrated risk
- **Transparency**: Clear logging of why each trade was accepted/rejected

### Security
- **Risk Controls**: Cannot allocate >15% to single asset or >40% to single sector
- **Data Validation**: Detects and rejects compromised market data
- **VaR Monitoring**: Real-time portfolio risk assessment

---

## [0.1.0] - 2024-11-XX

### Initial Release (Pre-Phase 1)

#### Added
- XGBoost classifier for price prediction
- Technical indicator generation (RSI, MACD, Bollinger Bands, SMA)
- Basic portfolio management with CSV ledger
- yfinance integration for market data
- Docker containerization
- GitHub Actions for scheduled execution
- Basic Makefile commands (setup, train, trade)

#### Known Issues (Fixed in 1.0.0)
- ❌ No risk management (could allocate 100% to one asset)
- ❌ No data validation (garbage in, garbage out)
- ❌ Equal-weight position sizing (ignores volatility)
- ❌ No sector awareness (could create 90% tech exposure)
- ❌ No pre-trade validation
- ❌ No test coverage

---

## Version History Summary

| Version | Date | Phase | Key Features |
|---------|------|-------|--------------|
| **1.6.0** | 2025-12-10 | Phase 6 | GitHub Actions Cache, reliable deployment |
| **1.5.0** | 2025-12-10 | Phase 5 | Walk-forward validation, T+1 execution |
| **1.4.0** | 2025-12-09 | Phase 4 | SQLite cache, S&P 500 universe |
| **1.3.0** | 2025-12-09 | Phase 3.6 | VIX regime, multi-horizon ensemble |
| **1.2.0** | 2025-12-09 | Phase 3.5 | 15 features, dynamic selection |
| **1.1.0** | 2025-12-09 | Phase 2 | Backtesting framework |
| **1.0.0** | 2025-12-08 | Phase 1 | Risk management, data validation |
| 0.1.0 | 2024-11-XX | Initial | Basic ML trading, no risk controls |

---

## Upgrade Guide

### From 0.1.0 to 1.0.0

**Breaking Changes**: None - fully backward compatible

**New Dependencies**:
```bash
# Update environment
conda env update -f environment.yml

# Or with pip
pip install -r requirements.txt
```

**Configuration Changes**:
- Risk limits now hardcoded in `main.py` (can be made configurable in future)
- No changes required to `config/settings.yaml`

**Behavioral Changes**:
- Trades may be rejected if they violate risk limits (expected behavior)
- Invalid tickers automatically filtered from universe
- Position sizes may be smaller due to volatility adjustment
- Cash buffer ($100) reserved and not available for trading

**Recommended Actions After Upgrade**:
1. Run `make test` to verify installation
2. Run `make validate` to check data quality for your ticker universe
3. Review risk limits in `main.py` (adjust if needed)
4. Monitor first few trades to understand new sizing logic

---

## Contributing

See areas of focus in README.md "Contributing" section.

---

**Maintained by**: PAT0216  
**License**: MIT
