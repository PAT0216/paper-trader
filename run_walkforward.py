#!/usr/bin/env python3
"""
Walk-Forward Backtesting Script

True out-of-sample validation where the model is trained BEFORE each test period.
This eliminates look-ahead bias and provides realistic performance estimates.

Process:
    Year 1: Train on 2010-2014, Test on 2015
    Year 2: Train on 2010-2015, Test on 2016
    ...
    Year 10: Train on 2010-2023, Test on 2024

Usage: python run_walkforward.py [--start 2015] [--end 2024]
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import fetch_data
from src.data.validator import DataValidator
from src.data.cache import DataCache
from src.features.indicators import generate_features
from src.models.trainer import create_target, FEATURE_COLUMNS
from src.utils.config import load_config
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


def predict_signals(model, data_dict, start_date, end_date, buy_threshold=0.005, sell_threshold=-0.005):
    """
    Generate trading signals for a period using trained model.
    
    Returns:
        Tuple of:
        - signals: Dict of {date: {ticker: signal}}
        - expected_returns: Dict of {date: {ticker: expected_return}}
    """
    signals = {}
    expected_returns = {}  # For priority ranking
    
    for ticker, df in data_dict.items():
        # Get test period data
        test_df = df[(df.index >= start_date) & (df.index < end_date)].copy()
        
        for date in test_df.index:
            # Use data up to this date for prediction (no look-ahead)
            hist = df[df.index <= date].copy()
            if len(hist) < 50:
                continue
            
            try:
                features = generate_features(hist, include_target=False)
                if len(features) == 0:
                    continue
                    
                X = features[FEATURE_COLUMNS].iloc[-1:].values
                X = np.where(np.isinf(X), 0, X)
                X = np.nan_to_num(X, 0)
                
                pred = model.predict(X)[0]
                
                if date not in signals:
                    signals[date] = {}
                    expected_returns[date] = {}
                
                # Store expected return for priority ranking
                expected_returns[date][ticker] = pred
                
                if pred > buy_threshold:
                    signals[date][ticker] = 'BUY'
                elif pred < sell_threshold:
                    signals[date][ticker] = 'SELL'
                else:
                    signals[date][ticker] = 'HOLD'
            except Exception:
                pass
    
    return signals, expected_returns


def simulate_portfolio(signals, expected_returns, data_dict, initial_cash=100000, max_position_pct=0.15):
    """
    Portfolio simulation with proper execution timing.
    
    CRITICAL: Executes at NEXT day's Open, not current day's Close
    This eliminates look-ahead bias.
    
    Args:
        signals: Dict of {date: {ticker: signal}}
        expected_returns: Dict of {date: {ticker: expected_return}} for priority
        data_dict: Dict of {ticker: DataFrame}
        initial_cash: Starting capital
        max_position_pct: Max position as % of portfolio
        
    Returns:
        DataFrame with portfolio values over time
    """
    cash = initial_cash
    holdings = {}  # {ticker: shares}
    portfolio_values = []
    
    # Get all dates sorted
    all_dates = sorted(set(signals.keys()))
    
    for i, date in enumerate(all_dates):
        day_signals = signals[date]
        day_expected_returns = expected_returns.get(date, {})
        
        # Calculate current portfolio value (using current Close for valuation only)
        portfolio_value = cash
        for ticker, shares in holdings.items():
            if ticker in data_dict and date in data_dict[ticker].index:
                price = data_dict[ticker].loc[date, 'Close']
                portfolio_value += shares * price
        
        # Get NEXT day for execution (no look-ahead)
        next_day_idx = i + 1
        if next_day_idx >= len(all_dates):
            portfolio_values.append({
                'date': date,
                'value': portfolio_value,
                'cash': cash,
                'n_positions': len([h for h in holdings.values() if h > 0])
            })
            continue
            
        next_date = all_dates[next_day_idx]
        
        # Get NEXT day's OPEN prices for execution
        execution_prices = {}
        for ticker in day_signals.keys():
            if ticker in data_dict and next_date in data_dict[ticker].index:
                df = data_dict[ticker]
                if 'Open' in df.columns:
                    execution_prices[ticker] = df.loc[next_date, 'Open']
                else:
                    execution_prices[ticker] = df.loc[next_date, 'Close']
        
        # Process SELL signals first (to free up cash)
        for ticker, signal in day_signals.items():
            if signal == 'SELL' and holdings.get(ticker, 0) > 0:
                price = execution_prices.get(ticker)
                if price:
                    cash += holdings[ticker] * price
                    holdings[ticker] = 0
        
        # Process BUY signals - SORTED BY EXPECTED RETURN (highest first)
        buy_candidates = [
            (ticker, day_expected_returns.get(ticker, 0.0))
            for ticker, signal in day_signals.items()
            if signal == 'BUY' and holdings.get(ticker, 0) == 0
        ]
        buy_candidates.sort(key=lambda x: x[1], reverse=True)  # Highest expected return first
        
        for ticker, exp_return in buy_candidates:
            price = execution_prices.get(ticker)
            if not price or price <= 0:
                continue
            
            # Buy position (max position size)
            max_invest = portfolio_value * max_position_pct
            shares_to_buy = int(max_invest / price)
            if shares_to_buy > 0 and cash >= shares_to_buy * price:
                holdings[ticker] = shares_to_buy
                cash -= shares_to_buy * price
        
        portfolio_values.append({
            'date': date,
            'value': portfolio_value,
            'cash': cash,
            'n_positions': len([h for h in holdings.values() if h > 0])
        })
    
    return pd.DataFrame(portfolio_values).set_index('date')


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
    
    # Run walk-forward for each year
    all_signals = {}
    all_expected_returns = {}  # For priority ranking
    
    for test_year in range(start_year, end_year + 1):
        train_end = f"{test_year}-01-01"
        test_start = f"{test_year}-01-01"
        test_end = f"{test_year + 1}-01-01"
        
        print(f"\n🔄 Year {test_year}: Train on 2010-{test_year-1}, Test on {test_year}")
        
        # Train model on historical data only
        model = train_model_for_period(data_dict, train_end)
        
        if model is None:
            print(f"   ⚠️ Insufficient training data, skipping {test_year}")
            continue
        
        # Generate signals for test period (using model trained BEFORE this period)
        year_signals, year_expected_returns = predict_signals(model, data_dict, test_start, test_end)
        all_signals.update(year_signals)
        all_expected_returns.update(year_expected_returns)
        
        print(f"   ✅ Generated {len(year_signals)} days of signals")
    
    # Simulate portfolio (with next-day Open execution and priority ranking)
    print("\n💼 Simulating portfolio (next-day Open execution)...")
    portfolio = simulate_portfolio(all_signals, all_expected_returns, data_dict, initial_cash)
    
    if len(portfolio) == 0:
        print("   ❌ No portfolio data generated")
        return
    
    # Calculate metrics
    start_value = initial_cash
    end_value = portfolio['value'].iloc[-1]
    total_return = (end_value - start_value) / start_value
    
    years = (portfolio.index[-1] - portfolio.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    daily_returns = portfolio['value'].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe = (cagr - 0.04) / volatility if volatility > 0 else 0
    
    # Max drawdown
    rolling_max = portfolio['value'].cummax()
    drawdown = (portfolio['value'] - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
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
