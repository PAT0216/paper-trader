#!/usr/bin/env python3
"""
Backtest Runner Script

Executes backtesting over historical data and generates performance reports.
Usage: python run_backtest.py [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--output results/]
"""

import argparse
import os
import sys
import yaml
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.loader import fetch_data
from src.data.validator import DataValidator
from src.backtesting import (
    Backtester,
    BacktestConfig,
    PerformanceCalculator,
    create_simple_signal_generator,
    create_ml_signal_generator,
    create_cross_sectional_signal_generator,
    generate_performance_summary
)
from src.models.predictor import Predictor
from src.utils.config import load_config


def get_sp500_tickers() -> list:
    """
    Get list of S&P 500 tickers.
    
    Strategy:
    1. Try loading from cached file (data/sp500_tickers.txt)
    2. Fallback to Wikipedia scrape
    3. Filter out invalid/problematic tickers
    """
    import os
    
    cache_file = "data/sp500_tickers.txt"
    
    # Try cache first
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
        print(f"   Loaded {len(tickers)} tickers from cache")
        return tickers
    
    # Fallback to Wikipedia scrape
    try:
        import pandas as pd
        import requests
        from io import StringIO
        
        # Fetch with user agent
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse HTML table
        table = pd.read_html(StringIO(response.text))[0]
        tickers = table['Symbol'].str.replace('.', '-').tolist()
        
        # Filter out problematic tickers
        exclude = {'BRK.B', 'BF.B'}  # Dual class stocks
        tickers = [t for t in tickers if t not in exclude]
        
        # Cache for next time
        os.makedirs('data', exist_ok=True)
        with open(cache_file, 'w') as f:
            f.write('\n'.join(tickers))
        
        print(f"   Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers
        
    except Exception as e:
        print(f"   ⚠️ Failed to fetch S&P 500 list: {e}")
        print(f"   Falling back to config tickers")
        return []


def load_backtest_config(config_path: str = "config/backtest_settings.yaml") -> BacktestConfig:
    """Load backtest configuration from YAML."""
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        bt = cfg.get('backtest', {})
        costs = cfg.get('costs', {})
        risk = cfg.get('risk', {})
        execution = cfg.get('execution', {})
        
        return BacktestConfig(
            start_date=bt.get('start_date', '2017-01-01'),
            end_date=bt.get('end_date', '2024-12-31'),
            initial_cash=bt.get('initial_cash', 100000.0),
            benchmark_ticker=bt.get('benchmark_ticker', 'SPY'),
            max_position_pct=risk.get('max_position_pct', 0.15),
            max_sector_pct=risk.get('max_sector_pct', 0.30),
            min_cash_buffer=risk.get('min_cash_buffer', 200.0),
            slippage_bps=costs.get('slippage_bps', 10.0),
            commission_per_share=costs.get('commission_per_share', 0.0),
            rebalance_frequency=execution.get('rebalance_frequency', 'daily'),
            use_risk_manager=execution.get('use_risk_manager', True),
            # Phase 7: Stop-loss and drawdown controls
            stop_loss_pct=risk.get('stop_loss_pct', 0.08),
            drawdown_warning=risk.get('drawdown_warning', 0.15),
            drawdown_halt=risk.get('drawdown_halt', 0.20),
            drawdown_liquidate=risk.get('drawdown_liquidate', 0.25),
            use_stop_loss=risk.get('use_stop_loss', True),
            use_drawdown_control=risk.get('use_drawdown_control', True),
        )
    except FileNotFoundError:
        print(f"Config file not found: {config_path}. Using defaults.")
        return BacktestConfig()


def run_backtest(
    tickers: list,
    config: BacktestConfig,
    output_dir: str = "results",
    use_ml: bool = False
) -> dict:
    """
    Run backtesting on specified tickers.
    
    Args:
        tickers: List of ticker symbols
        config: BacktestConfig with simulation parameters
        output_dir: Directory to save results
        
    Returns:
        Summary dictionary with results
    """
    print("=" * 60)
    print("PAPER TRADER BACKTEST")
    print("=" * 60)
    print(f"Date Range: {config.start_date} to {config.end_date}")
    print(f"Initial Capital: ${config.initial_cash:,.2f}")
    print(f"Tickers: {len(tickers)}")
    print("=" * 60)
    
    # Fetch historical data
    print("\n📊 Fetching historical data...")
    
    # Determine data period
    start_year = int(config.start_date[:4])
    current_year = datetime.now().year
    years_needed = current_year - start_year + 1
    period = f"{min(years_needed + 1, 10)}y"  # yfinance max ~10y reliable
    
    # Include benchmark ticker
    all_tickers = list(set(tickers + [config.benchmark_ticker]))
    
    # fetch_data uses smart caching: loads from cache, then fetches only missing recent dates
    data_dict = fetch_data(all_tickers, period=period, use_cache=True)
    
    if not data_dict:
        raise RuntimeError("Failed to fetch data")
    
    print(f"   Fetched data for {len(data_dict)} tickers")
    
    # Filter to 2010+ data BEFORE validation (matches training filter)
    # This avoids rejecting tickers for old data issues we'll never use
    MIN_DATE = '2010-01-01'
    print(f"   Filtering to {MIN_DATE}+ (matching training data filter)")
    for ticker in data_dict:
        df = data_dict[ticker]
        if hasattr(df.index, 'tz_localize'):
            data_dict[ticker] = df[df.index >= MIN_DATE]
    
    # Validate data (now only on recent data we'll actually use)
    print("\n🔍 Validating data quality (2010+ data only)...")
    validator = DataValidator(backtest_mode=True)
    validation_results = validator.validate_data_dict(data_dict)
    
    # Remove invalid tickers
    valid_tickers = [t for t, r in validation_results.items() if r.is_valid]
    invalid_tickers = [t for t, r in validation_results.items() if not r.is_valid]
    
    if invalid_tickers:
        print(f"   ⚠️  Removing {len(invalid_tickers)} invalid tickers: {invalid_tickers}")
        for t in invalid_tickers:
            if t in data_dict:
                del data_dict[t]
    
    print(f"   ✅ {len(valid_tickers)} valid tickers")
    
    # Extract benchmark
    benchmark_data = data_dict.pop(config.benchmark_ticker, None)
    if benchmark_data is None:
        print(f"   ⚠️  Benchmark {config.benchmark_ticker} not available")
    
    # Initialize backtester
    print("\n⚙️  Initializing backtester...")
    backtester = Backtester(config)
    
    # Create signal generator
    if use_ml:
        print("   Loading ML model...")
        predictor = Predictor()
        if predictor.model is None:
            print("   ⚠️ No ML model found, falling back to SMA")
            signal_generator = create_simple_signal_generator()
            strategy_name = "Simple SMA Crossover (fallback)"
        else:
            # Use cross-sectional ranking: BUY top 10%, SELL bottom 10%
            # This works better than fixed thresholds when model predicts close to mean
            from src.backtesting import create_cross_sectional_signal_generator
            signal_generator = create_cross_sectional_signal_generator(
                predictor, 
                buy_pct=0.10,   # Top 10%
                sell_pct=0.10  # Bottom 10%
            )
            strategy_name = "ML XGBoost + Cross-Sectional Ranking (Top/Bottom 10%)"
    else:
        signal_generator = create_simple_signal_generator()
        strategy_name = "Simple SMA Crossover"
    
    print(f"   Using: {strategy_name}")
    print(f"   Rebalance: {config.rebalance_frequency}")
    print(f"   Risk Manager: {'Enabled' if config.use_risk_manager else 'Disabled'}")
    
    # Run backtest
    print("\n🚀 Running backtest...")
    metrics, trades_df, summary = backtester.run(
        data_dict=data_dict,
        signal_generator=signal_generator,
        benchmark_data=benchmark_data
    )
    
    # Generate reports
    print("\n📈 Generating reports...")
    
    # Create dated output directory if use_dated_folders is enabled
    from datetime import datetime as dt
    if config.use_dated_folders:
        date_str = dt.now().strftime("%Y-%m-%d")
        # Find description from command line or config
        description = "full_universe" if len(tickers) > 100 else "custom"
        dated_dir = os.path.join(output_dir, f"{date_str}_{description}")
        os.makedirs(dated_dir, exist_ok=True)
        actual_output_dir = dated_dir
        print(f"   Saving to: {dated_dir}")
    else:
        actual_output_dir = output_dir
        os.makedirs(actual_output_dir, exist_ok=True)
    
    backtester.generate_report(metrics, actual_output_dir)
    
    # Save trades
    if len(trades_df) > 0:
        trades_df.to_csv(os.path.join(actual_output_dir, "backtest_trades.csv"), index=False)
        print(f"   Trades saved to {actual_output_dir}/backtest_trades.csv")
    
    # Create symlinks for dashboard access (latest backtest)
    if config.use_dated_folders:
        results_root = "results"
        symlink_files = [
            ("backtest_metrics.json", "backtest_metrics.json"),
            ("backtest_summary.txt", "backtest_summary.txt"),
            ("backtest_trades.csv", "backtest_trades.csv")
        ]
        
        for src_file, link_name in symlink_files:
            src_path = os.path.join(actual_output_dir, src_file)
            link_path = os.path.join(results_root, link_name)
            
            if os.path.exists(src_path):
                # Remove old symlink/file
                if os.path.exists(link_path) or os.path.islink(link_path):
                    os.remove(link_path)
                # Create symlink (relative path for portability)
                rel_path = os.path.relpath(src_path, results_root)
                try:
                    os.symlink(rel_path, link_path)
                except OSError:
                    # Fallback: copy if symlink fails (Windows)
                    import shutil
                    shutil.copy2(src_path, link_path)
        
        print(f"   ✅ Symlinks created in results/ for dashboard access")
    
    # Save summary
    summary_text = generate_performance_summary(metrics)
    print("\n" + summary_text)
    
    return summary


def main():
    """Main entry point for backtest runner."""
    parser = argparse.ArgumentParser(description="Run Paper Trader Backtest")
    parser.add_argument(
        '--start', 
        type=str, 
        default=None,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end', 
        type=str, 
        default=None,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--cash', 
        type=float, 
        default=None,
        help='Initial cash amount'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='results',
        help='Output directory for reports'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/backtest_settings.yaml',
        help='Path to backtest config file'
    )
    parser.add_argument(
        '--ml',
        action='store_true',
        help='Use ML predictor instead of SMA strategy'
    )
    parser.add_argument(
        '--full-universe',
        action='store_true',
        help='Run backtest on full S&P 500 universe'
    )
    
    args = parser.parse_args()
    
    # Load configurations
    main_config = load_config()
    backtest_config = load_backtest_config(args.config)
    
    # Override with command line args
    if args.start:
        backtest_config.start_date = args.start
    if args.end:
        backtest_config.end_date = args.end
    if args.cash:
        backtest_config.initial_cash = args.cash
    
    # Get tickers - either full universe or config list
    if args.full_universe:
        tickers = get_sp500_tickers()
        print(f"🌐 Using full S&P 500 universe: {len(tickers)} tickers")
    else:
        tickers = main_config.get('tickers', ['SPY', 'AAPL', 'MSFT'])
        print(f"📋 Using config ticker list: {len(tickers)} tickers")
    
    try:
        summary = run_backtest(tickers, backtest_config, args.output, use_ml=args.ml)
        
        print("\n" + "=" * 60)
        print("✅ BACKTEST COMPLETE")
        print("=" * 60)
        
        # Quick summary
        results = summary.get('results', {})
        print(f"   Final Value: ${results.get('final_value', 0):,.2f}")
        print(f"   Total Return: {results.get('total_return', 0):.2%}")
        print(f"   Sharpe Ratio: {results.get('sharpe', 0):.3f}")
        print(f"   Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        print(f"   Total Trades: {results.get('total_trades', 0)}")
        
        costs = summary.get('costs', {})
        print(f"\n   Transaction Costs:")
        print(f"      Total: ${costs.get('total_cost', 0):.2f}")
        print(f"      Avg per Trade: ${costs.get('avg_cost_per_trade', 0):.2f}")
        
        print(f"\n   Reports saved to: {args.output}/")
        
    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
