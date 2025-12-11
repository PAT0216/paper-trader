#!/usr/bin/env python3
"""
Walk-Forward Backtesting Script - FIXED VERSION

True out-of-sample validation using proper Backtester infrastructure.
Now includes Phase 7 risk controls for valid comparison.

Process:
    Year 1: Train on 2010-2014, Test on 2015 (with risk controls)
    Year 2: Train on 2010-2015, Test on 2016 (with risk controls)
    ...
    Year 10: Train on 2010-2023, Test on 2024 (with risk controls)

Usage: python run_walkforward.py [--start 2015] [--end 2024]
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import fetch_data, fetch_from_cache_only
from src.data.validator import DataValidator
from src.data.cache import DataCache
from src.features.indicators import generate_features
from src.models.trainer import create_target, FEATURE_COLUMNS
from src.utils.config import load_config
from src.backtesting import (
    Backtester,
    BacktestConfig,
    create_ml_signal_generator
)
import xgboost as xgb


def train_model_for_period(data_dict, end_date):
    """
    Train XGBoost model using data up to end_date.
    
    Args:
        data_dict: Dict of {ticker: DataFrame}
        end_date: Last date to use for training (YYYY-MM-DD)
        
    Returns:
        Trained XGBoost model
    """
    all_features = []
    
    for ticker, df in data_dict.items():
        # Filter to training period only
        train_df = df[df.index < end_date].copy()
        if len(train_df) < 100:  # Need minimum data
            continue
            
        # Generate features and target
        processed = generate_features(train_df, include_target=False)
        processed = create_target(processed, target_type='regression')
        processed = processed.dropna(subset=['Target'])
        
        if len(processed) > 50:
            all_features.append(processed)
    
    if not all_features:
        return None
    
    full_df = pd.concat(all_features).sort_index()
    
    # Clean data
    X = full_df[FEATURE_COLUMNS].values
    y = full_df['Target'].values
    X = np.where(np.isinf(X), np.nan, X)
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[valid], y[valid]
    
    if len(X) < 100:
        return None
    
    # Train model
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(X, y)
    
    return model


def load_backtest_config():
    """Load backtest configuration from YAML."""
    config_path = "config/backtest_settings.yaml"
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        bt = cfg.get('backtest', {})
        costs = cfg.get('costs', {})
        risk = cfg.get('risk', {})
        
        return BacktestConfig(
            start_date="2015-01-01",  # Will be overridden per year
            end_date="2024-12-31",
            initial_cash=bt.get('initial_cash', 100000.0),
            benchmark_ticker=bt.get('benchmark_ticker', 'SPY'),
            max_position_pct=risk.get('max_position_pct', 0.15),
            max_sector_pct=risk.get('max_sector_pct', 0.30),
            min_cash_buffer=risk.get('min_cash_buffer', 200.0),
            stop_loss_pct=risk.get('stop_loss_pct', 0.15),
            use_stop_loss=risk.get('use_stop_loss', False),
            use_drawdown_control=risk.get('use_drawdown_control', True),
            drawdown_warning=risk.get('drawdown_warning', 0.15),
            drawdown_halt=risk.get('drawdown_halt', 0.20),
            drawdown_liquidate=risk.get('drawdown_liquidate', 0.25),
            use_risk_manager=True,
            slippage_bps=costs.get('slippage_bps', 10.0),
            use_dated_folders=False  # Walk-forward manages own output
        )
    except Exception as e:
        print(f"Warning: Could not load config: {e}, using defaults")
        return BacktestConfig()


def run_year_with_backtester(model, data_dict, year, config_template, cumulative_portfolio):
    """
    Run backtest for a single year using trained model and Backtester infrastructure.
    
    Args:
        model: Trained XGBoost model for this year
        data_dict: Price data for all tickers
        year: Year to test
        config_template: BacktestConfig template
        cumulative_portfolio: Portfolio state from previous years
        
    Returns:
        Updated portfolio with this year's trades
    """
    # Create config for this year only
    year_config = BacktestConfig(
        start_date=f"{year}-01-01",
        end_date=f"{year}-12-31",
        initial_cash=cumulative_portfolio['cash'],  # Continue from previous year
        **{k: v for k, v in config_template.__dict__.items() 
           if k not in ['start_date', 'end_date', 'initial_cash']}
    )
    
    # Create ML signal generator with trained model
    # create_ml_signal_generator expects a Predictor object, not raw model
    from src.models.predictor import Predictor
    
    # Create predictor wrapper for model
    predictor = Predictor(model_path=None)  # Don't load from file
    predictor.model = model  # Set model directly
    
    signal_generator = create_ml_signal_generator(
        predictor=predictor,
        threshold_buy=0.005,  # 0.5% expected return (regression model, not classification)
        threshold_sell=-0.005
    )
    
    # Run backtest for this year
    backtester = Backtester(
        config=year_config,
        signal_generator=signal_generator,
        risk_manager=None  # Let Backtester create it
    )
    
    try:
        metrics, trades_df, summary = backtester.run(data_dict)
        
        # Update cumulative portfolio
        final_value = metrics.get('ending_value', year_config.initial_cash)
        cumulative_portfolio['cash'] = final_value
        cumulative_portfolio['trades'].extend(trades_df.to_dict('records') if not trades_df.empty else [])
        cumulative_portfolio['yearly_values'].append({
            'year': year,
            'value': final_value,
            'return': metrics.get('total_return', 0.0),
            'start_value': metrics.get('starting_value', year_config.initial_cash),
            'sharpe': metrics.get('sharpe_ratio', 0.0),
            'max_drawdown': metrics.get('max_drawdown', 0.0),
            'cagr': metrics.get('cagr', 0.0)
        })
        
        # Append daily portfolio history for this year
        if 'portfolio_history' in summary and not summary['portfolio_history'].empty:
            cumulative_portfolio['daily_portfolio_history'].append(summary['portfolio_history'])
        
        print(f"   Final value: ${final_value:,.0f}")
        
    except Exception as e:
        print(f"   ⚠️  Backtest failed for {year}: {e}")
    
    return cumulative_portfolio


def get_spy_returns(start_year, end_year):
    """Get SPY buy-and-hold returns for comparison."""
    cache = DataCache()
    spy = cache.get_price_data('SPY', f'{start_year}-01-01', f'{end_year}-12-31')
    
    if spy is None or len(spy) == 0:
        from src.data.loader import fetch_data
        data = fetch_data(['SPY'], period='max')
        spy = data.get('SPY')
    
    spy = spy[spy.index >= f'{start_year}-01-01']
    spy = spy[spy.index <= f'{end_year}-12-31']
    
    return spy


def run_walkforward(start_year=2015, end_year=2024, initial_cash=100000, use_full_universe=True):
    """
    Run walk-forward validation.
    
    Args:
        start_year: First year to test
        end_year: Last year to test
        initial_cash: Starting portfolio value
        use_full_universe: If True, use all S&P 500 tickers from cache. If False, use config tickers.
    """
    print("=" * 60)
    print("WALK-FORWARD VALIDATION")
    print("=" * 60)
    print(f"Test Period: {start_year} to {end_year}")
    print(f"Training: Expanding window from 2010")
    print(f"Initial Capital: ${initial_cash:,}")
    print("=" * 60)
    
    # Get tickers - use full S&P 500 universe or config
    if use_full_universe:
        print("\n📊 Loading full S&P 500 universe from cache...")
        cache = DataCache()
        stats = cache.get_cache_stats()
        tickers = stats['ticker'].tolist() if len(stats) > 0 else []
        
        # Exclude the benchmark
        tickers = [t for t in tickers if t != 'SPY']
        print(f"   {len(tickers)} tickers in cache")
    else:
        config = load_config()
        tickers = config.get('tickers', ['SPY', 'AAPL', 'MSFT'])
        print(f"\n📊 Using {len(tickers)} config tickers")
    
    # Fetch all data from cache
    print("\n📊 Fetching data from cache...")
    data_dict = fetch_data(tickers, period='max')
    
    # Filter to 2010+
    MIN_DATE = '2010-01-01'
    for ticker in list(data_dict.keys()):
        df = data_dict[ticker]
        data_dict[ticker] = df[df.index >= MIN_DATE]
        if len(data_dict[ticker]) < 500:  # Need ~2 years minimum
            del data_dict[ticker]
    
    print(f"   {len(data_dict)} tickers with sufficient data (2010+)")
    
    
    # Load backtest config template
    config_template = load_backtest_config()
    
    # Initialize cumulative portfolio tracking
    cumulative_portfolio = {
        'cash': initial_cash,
        'trades': [],
        'yearly_values': [],
        'daily_portfolio_history': []
    }
    
    # Run walk-forward for each year with proper Backtester
    for test_year in range(start_year, end_year + 1):
        train_end = f"{test_year}-01-01"
        
        print(f"\n🔄 Year {test_year}: Train on 2010-{test_year-1}, Test on {test_year}")
        
        # Train model on historical data only (before test year)
        model = train_model_for_period(data_dict, train_end)
        
        if model is None:
            print(f"   ⚠️ Insufficient training data, skipping {test_year}")
            continue
        
        # Run backtest for this year with trained model and risk controls
        cumulative_portfolio = run_year_with_backtester(
            model=model,
            data_dict=data_dict,
            year=test_year,
            config_template=config_template,
            cumulative_portfolio=cumulative_portfolio
        )
    
    
    # Calculate metrics from cumulative portfolio
    if not cumulative_portfolio['yearly_values']:
        print("   ❌ No yearly data generated")
        return
    
    print("\n📊 Calculating cumulative metrics...")
    
    # Calculate overall metrics from yearly values
    start_value = initial_cash
    end_value = cumulative_portfolio['cash']
    total_return = (end_value - start_value) / start_value
    
    years = end_year - start_year + 1
    cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    # Calculate Sharpe from yearly returns
    yearly_returns = [y['return'] for y in cumulative_portfolio['yearly_values']]
    if yearly_returns:
        avg_return = np.mean(yearly_returns)
        volatility = np.std(yearly_returns)
        sharpe = (avg_return - 0.04) / volatility if volatility > 0 else 0
    else:
        sharpe = 0
    
    # Max drawdown from yearly values
    cumulative_values = [initial_cash]
    for y in cumulative_portfolio['yearly_values']:
        cumulative_values.append(y['value'])
    
    rolling_max = np.maximum.accumulate(cumulative_values)
    drawdowns = [(val - peak) / peak for val, peak in zip(cumulative_values, rolling_max)]
    max_dd = min(drawdowns)
    
    # Create portfolio DataFrame for saving
    portfolio = pd.DataFrame(cumulative_portfolio['yearly_values'])
    portfolio.index = portfolio['year']
    
    # Get SPY comparison
    print("\n📈 Comparing to S&P 500 (SPY)...")
    spy = get_spy_returns(start_year, end_year)
    
    spy_start = spy['Close'].iloc[0]
    spy_end = spy['Close'].iloc[-1]
    spy_total_return = (spy_end - spy_start) / spy_start
    spy_cagr = (1 + spy_total_return) ** (1/years) - 1 if years > 0 else 0
    spy_daily = spy['Close'].pct_change().dropna()
    spy_vol = spy_daily.std() * np.sqrt(252)
    spy_sharpe = (spy_cagr - 0.04) / spy_vol if spy_vol > 0 else 0
    spy_rolling_max = spy['Close'].cummax()
    spy_dd = ((spy['Close'] - spy_rolling_max) / spy_rolling_max).min()
    
    # Print results
    print("\n" + "=" * 60)
    print("WALK-FORWARD RESULTS (True Out-of-Sample)")
    print("=" * 60)
    
    print(f"\n{'Metric':<25} {'WALK-FORWARD':>15} {'SPY':>15} {'Winner':>12}")
    print("-" * 70)
    print(f"{'Total Return':<25} {total_return*100:>14.2f}% {spy_total_return*100:>14.2f}% {'MODEL' if total_return > spy_total_return else 'SPY':>12}")
    print(f"{'CAGR':<25} {cagr*100:>14.2f}% {spy_cagr*100:>14.2f}% {'MODEL' if cagr > spy_cagr else 'SPY':>12}")
    print(f"{'Sharpe Ratio':<25} {sharpe:>15.3f} {spy_sharpe:>15.3f} {'MODEL' if sharpe > spy_sharpe else 'SPY':>12}")
    print(f"{'Max Drawdown':<25} {max_dd*100:>14.2f}% {spy_dd*100:>14.2f}% {'MODEL' if max_dd > spy_dd else 'SPY':>12}")
    print(f"{'Final Value ($100k)':<25} ${end_value:>13,.0f} ${100000*(1+spy_total_return):>13,.0f} {'MODEL' if end_value > 100000*(1+spy_total_return) else 'SPY':>12}")
    
    print("\n" + "=" * 60)
    print("KEY: Walk-forward means model was trained BEFORE each test year")
    print("     This is the most realistic estimate of live performance")
    print("=" * 60)
    
    return {
        'portfolio': portfolio,
        'total_return': total_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'spy_return': spy_total_return,
        'spy_cagr': spy_cagr,
        'spy_sharpe': spy_sharpe
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-Forward Validation")
    parser.add_argument('--start', type=int, default=2015, help='Start year for testing')
    parser.add_argument('--end', type=int, default=2024, help='End year for testing')
    parser.add_argument('--cash', type=float, default=100000, help='Initial cash')
    
    args = parser.parse_args()
    
    results = run_walkforward(args.start, args.end, args.cash)
    
    # Save results to separate directory
    output_dir = f"results/walkforward/{datetime.now().strftime('%Y-%m-%d')}_y{args.start}-{args.end}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics = {
        'total_return': float(results['total_return']),
        'cagr': float(results['cagr']),
        'sharpe': float(results['sharpe']),
        'max_drawdown': float(results['max_dd']),
        'spy_return': float(results['spy_return']),
        'spy_cagr': float(results['spy_cagr']),
        'spy_sharpe': float(results['spy_sharpe']),
        'start_year': args.start,
        'end_year': args.end,
        'initial_cash': args.cash
    }
    
    with open(f"{output_dir}/walkforward_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save portfolio history
    results['portfolio'].to_csv(f"{output_dir}/walkforward_portfolio.csv", index=False)
    
    print(f"\n✅ Results saved to: {output_dir}/")
