#!/usr/bin/env python
"""
Run Zhang 2019 Strategy Backtest

Implements the complete Zhang 2019 methodology:
1. 20 PCA-optimized lagged return features
2. XGBoost classification (beat cross-sectional median)
3. Vertical Ensemble (3-day weighted probabilities)
4. Long 10, short 10 strategy

Based on: "Investment Strategy Based on Machine Learning Models" by Jiayue Zhang (2019)
Paper Results: 117% excess return, 2.85 Sharpe ratio
"""

import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf

# Add project root to path
sys.path.insert(0, '.')

from src.data.universe import fetch_sp500_tickers
from src.models.zhang_trainer import ZhangStrategy
from src.features.zhang_features import ZHANG_PERIODS, ZHANG_FEATURE_COLUMNS



def run_zhang_backtest(start_date: str, end_date: str, n_tickers: int = 100):
    """
    Run Zhang 2019 strategy backtest.
    
    Args:
        start_date: Backtest start date (YYYY-MM-DD)
        end_date: Backtest end date (YYYY-MM-DD)
        n_tickers: Number of tickers to use (for speed)
    """
    print("=" * 60)
    print("ZHANG 2019 STRATEGY BACKTEST")
    print("=" * 60)
    print(f"\nPeriod: {start_date} to {end_date}")
    print(f"Features: {len(ZHANG_FEATURE_COLUMNS)} lagged returns")
    print(f"Strategy: Long 10, Short 10")
    print(f"Vertical Ensemble: (0.1, 0.3, 0.6)")
    
    # Get universe
    print(f"\n📊 Getting S&P 500 universe...")
    tickers = fetch_sp500_tickers()[:n_tickers]
    print(f"   Using {len(tickers)} tickers")
    
    # Fetch data (need extra history for features)
    train_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=500)).strftime('%Y-%m-%d')
    
    print(f"\n📥 Loading data from cache (from {train_start})...")
    
    # Use existing cache
    from src.data.cache import DataCache
    cache = DataCache()
    
    data_dict = {}
    for ticker in tickers:
        df = cache.get_price_data(ticker, start_date=train_start, end_date=end_date)
        if df is not None and len(df) > 250:
            data_dict[ticker] = df
    
    print(f"   Loaded {len(data_dict)} tickers from cache")

    
    # Split into train and test
    train_end = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"\n🎯 Training period: {train_start} to {train_end}")
    print(f"🎯 Testing period: {start_date} to {end_date}")
    
    # Train Zhang strategy
    strategy = ZhangStrategy(n_top=10)
    
    # Filter training data
    train_data = {}
    for ticker, df in data_dict.items():
        train_df = df[df.index <= train_end]
        if len(train_df) > 250:  # Need at least 1 year
            train_data[ticker] = train_df
    
    print(f"\n   Training on {len(train_data)} tickers...")
    strategy.train(train_data, min_date='2014-01-01')
    
    # Backtest on test period
    print(f"\n📈 Running backtest...")
    
    # Get trading days in test period
    test_dates = pd.bdate_range(start=start_date, end=end_date)
    
    portfolio_values = [100000]  # Start with $100k
    daily_returns = []
    
    for i, date in enumerate(test_dates):
        date_str = date.strftime('%Y-%m-%d')
        
        # Get data up to this date for each ticker
        current_data = {}
        next_day_returns = {}
        
        for ticker, df in data_dict.items():
            df_up_to_date = df[df.index <= date_str]
            if len(df_up_to_date) > 220:  # Need enough history
                current_data[ticker] = df_up_to_date
                
                # Get next day return for P&L calculation
                df_after = df[df.index > date_str]
                if len(df_after) > 0:
                    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                    if isinstance(df.columns, pd.MultiIndex):
                        # Handle multi-index
                        current_close = df_up_to_date.iloc[-1]['Close']
                        if isinstance(current_close, pd.Series):
                            current_close = current_close.iloc[0]
                        next_close = df_after.iloc[0]['Close']
                        if isinstance(next_close, pd.Series):
                            next_close = next_close.iloc[0]
                    else:
                        current_close = df_up_to_date[price_col].iloc[-1]
                        next_close = df_after[price_col].iloc[0]
                    next_day_returns[ticker] = (next_close - current_close) / current_close
        
        if not current_data:
            continue
        
        # Generate signals
        signals = strategy.generate_signals(current_data)
        
        # Calculate portfolio return
        long_tickers = [t for t, s in signals.items() if s == 'BUY' and t in next_day_returns]
        short_tickers = [t for t, s in signals.items() if s == 'SELL' and t in next_day_returns]
        
        if long_tickers or short_tickers:
            # Equal weight within long/short
            long_return = np.mean([next_day_returns.get(t, 0) for t in long_tickers]) if long_tickers else 0
            short_return = -np.mean([next_day_returns.get(t, 0) for t in short_tickers]) if short_tickers else 0
            
            # 50% long, 50% short
            portfolio_return = 0.5 * long_return + 0.5 * short_return
            daily_returns.append(portfolio_return)
            
            new_value = portfolio_values[-1] * (1 + portfolio_return)
            portfolio_values.append(new_value)
        
        # Progress
        if (i + 1) % 50 == 0:
            print(f"   Day {i+1}/{len(test_dates)}: Portfolio = ${portfolio_values[-1]:,.2f}")
    
    # Calculate metrics
    if len(daily_returns) > 0:
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
        volatility = np.std(daily_returns) * np.sqrt(252)
        sharpe = (annualized_return - 0.02) / volatility if volatility > 0 else 0
        max_dd = min(0, min([(v / max(portfolio_values[:i+1]) - 1) for i, v in enumerate(portfolio_values)]))
        
        print(f"\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"   Total Return: {total_return:.2%}")
        print(f"   Annualized Return: {annualized_return:.2%}")
        print(f"   Volatility: {volatility:.2%}")
        print(f"   Sharpe Ratio: {sharpe:.3f}")
        print(f"   Max Drawdown: {max_dd:.2%}")
        print(f"   Final Value: ${portfolio_values[-1]:,.2f}")
        print(f"   Trading Days: {len(daily_returns)}")
        print("=" * 60)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'final_value': portfolio_values[-1]
        }
    
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Zhang 2019 Strategy Backtest')
    parser.add_argument('--start', type=str, default='2023-01-01', help='Start date')
    parser.add_argument('--end', type=str, default='2024-12-01', help='End date')
    parser.add_argument('--tickers', type=int, default=50, help='Number of tickers')
    
    args = parser.parse_args()
    
    run_zhang_backtest(args.start, args.end, args.tickers)
