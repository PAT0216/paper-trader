# Changelog

All notable changes to the Paper Trader AI project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for Phase 3
- Fix data leakage in feature generation
- Time series cross-validation
- Regression target (predict return magnitude)
- Enhanced features (volume indicators, macro data, sentiment)

### Planned for Phase 4
- Advanced structured logging (JSON format)
- Alerting system (email, Slack webhooks)
- SQLite ledger to replace CSV
- Multi-strategy framework with weighted voting

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
| **1.0.0** | 2024-12-08 | Phase 1 Complete | Risk management, data validation, testing |
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
