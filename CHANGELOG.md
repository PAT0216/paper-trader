# Changelog

All notable changes to the Paper Trader AI project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.7.0] - 2025-12-10

### 🛡️ Phase 7: Quant Risk Enhancements - Complete

This release implements institutional-grade risk controls validated through rigorous A/B testing.

### Added
- **DrawdownController**: Automatic position reduction (-15%), trading halt (-20%), emergency liquidation (-25%)
- **Position-Level Stop-Loss**: 15% stop-loss threshold (A/B tested)
- **Unbiased Walk-Forward A/B Test**: `run_unbiased_comparison.py` for proper strategy validation
- **Stop-Loss Threshold Sweep**: `run_stoploss_test.py` for comparing stop-loss levels
- **Ranking Metrics**: Spearman Rank Correlation and Top-10% Accuracy in model evaluation

### Key A/B Testing Results (2018-2024, 75 Random S&P 500 Tickers)
| Strategy | Avg Yearly Return | Sharpe |
|----------|-------------------|--------|
| **15% Stop-Loss** | **104.8%** | **1.30** |
| No Stop-Loss | 87.2% | 1.26 |
| S&P 500 (SPY) | 15.3% | ~0.7 |

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
