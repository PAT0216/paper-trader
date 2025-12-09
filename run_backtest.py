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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import fetch_data
from src.data.validator import DataValidator
from src.backtesting import (
    Backtester,
    BacktestConfig,
    PerformanceCalculator,
    create_simple_signal_generator,
    generate_performance_summary
)
from src.utils.config import load_config


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
            max_sector_pct=risk.get('max_sector_pct', 0.40),
            min_cash_buffer=risk.get('min_cash_buffer', 100.0),
            slippage_bps=costs.get('slippage_bps', 5.0),
            commission_per_share=costs.get('commission_per_share', 0.0),
            rebalance_frequency=execution.get('rebalance_frequency', 'daily'),
            use_risk_manager=execution.get('use_risk_manager', True),
        )
    except FileNotFoundError:
        print(f"Config file not found: {config_path}. Using defaults.")
        return BacktestConfig()


def run_backtest(
    tickers: list,
    config: BacktestConfig,
    output_dir: str = "results"
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
    
    data_dict = fetch_data(all_tickers, period=period)
    
    if not data_dict:
        raise RuntimeError("Failed to fetch data")
    
    print(f"   Fetched data for {len(data_dict)} tickers")
    
    # Validate data
    print("\n🔍 Validating data quality...")
    validator = DataValidator()
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
    # Using simple SMA crossover for initial testing
    # TODO: Replace with ML predictor for production use
    signal_generator = create_simple_signal_generator()
    
    print(f"   Using: Simple SMA Crossover Strategy")
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
    os.makedirs(output_dir, exist_ok=True)
    backtester.generate_report(metrics, output_dir)
    
    # Save trades
    if len(trades_df) > 0:
        trades_df.to_csv(os.path.join(output_dir, "backtest_trades.csv"), index=False)
        print(f"   Trades saved to {output_dir}/backtest_trades.csv")
    
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
    
    # Get tickers from main config
    tickers = main_config.get('tickers', ['SPY', 'AAPL', 'MSFT'])
    
    try:
        summary = run_backtest(tickers, backtest_config, args.output)
        
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
