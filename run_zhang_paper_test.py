#!/usr/bin/env python
"""
Zhang 2019 Paper Exact Replication Test

Implements the EXACT test conditions from:
"Investment Strategy Based on Machine Learning Models" by Jiayue Zhang (2019)

Parameters (from paper):
- Training: Jan 1, 2015 - Dec 31, 2017
- Testing: Jan 1, 2018 - Dec 31, 2018
- Transaction cost: 0.05% per half-turn
- Portfolio: Long 10, Short 10 (equal weight)
- Vertical Ensemble: (0.1, 0.3, 0.6)

Expected Results (Table 4.6):
- Excess Return (after fees): 0.8312 (Only t) / 0.9327 (Best Weights)
- Sharpe Ratio (after fees): 2.2042 (Only t) / 2.4532 (Best Weights)
"""

import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.universe import fetch_sp500_tickers
from src.data.cache import DataCache
from src.models.zhang_trainer import ZhangStrategy
from src.features.zhang_features import (
    ZHANG_FEATURE_COLUMNS,
    apply_vertical_ensemble,
    VERTICAL_ENSEMBLE_WEIGHTS
)


# ============================================================================
# EXACT PAPER PARAMETERS
# ============================================================================
TRAIN_START = "2015-01-01"
TRAIN_END = "2017-12-31"
TEST_START = "2018-01-01"
TEST_END = "2018-12-31"

# Transaction cost: 0.05% per half-turn (from paper)
TRANSACTION_COST_HALF_TURN = 0.0005  # 0.05% = 5 bps

# Portfolio size
N_LONG = 10
N_SHORT = 10

# Risk-free rate (for excess return calculation)
RISK_FREE_RATE = 0.02  # ~2% as typical for 2018


def run_zhang_paper_test():
    """
    Run exact Zhang 2019 paper test conditions.
    """
    print("=" * 70)
    print("ZHANG 2019 PAPER EXACT REPLICATION TEST")
    print("=" * 70)
    print(f"\nPaper: 'Investment Strategy Based on Machine Learning Models'")
    print(f"Author: Jiayue Zhang (University of Waterloo, 2019)")
    print()
    print("PARAMETERS:")
    print(f"  Training Period:     {TRAIN_START} to {TRAIN_END}")
    print(f"  Testing Period:      {TEST_START} to {TEST_END}")
    print(f"  Transaction Cost:    {TRANSACTION_COST_HALF_TURN * 100:.2f}% per half-turn")
    print(f"  Portfolio:           Long {N_LONG}, Short {N_SHORT}")
    print(f"  Vertical Ensemble:   {VERTICAL_ENSEMBLE_WEIGHTS}")
    print()
    
    # ========================================================================
    # Load Data from Cache
    # ========================================================================
    print("📥 Loading S&P 500 data from cache...")
    
    cache = DataCache()
    tickers = fetch_sp500_tickers()
    print(f"   Universe: {len(tickers)} S&P 500 tickers")
    
    # Need extra history for feature generation (220 days lookback)
    data_start = "2014-01-01"
    
    data_dict = {}
    for ticker in tickers:
        df = cache.get_price_data(ticker, start_date=data_start, end_date=TEST_END)
        if df is not None and len(df) > 300:  # Need enough history
            data_dict[ticker] = df
    
    print(f"   Loaded: {len(data_dict)} tickers with sufficient history")
    
    # ========================================================================
    # Prepare Training Data
    # ========================================================================
    print(f"\n🎓 Preparing training data ({TRAIN_START} to {TRAIN_END})...")
    
    train_data = {}
    for ticker, df in data_dict.items():
        train_df = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]
        # Need lookback data too
        full_df = df[df.index <= TRAIN_END]
        if len(train_df) > 200 and len(full_df) > 300:
            train_data[ticker] = full_df
    
    print(f"   Training tickers: {len(train_data)}")
    
    # ========================================================================
    # Train Model
    # ========================================================================
    print(f"\n🔧 Training XGBoost model...")
    
    strategy = ZhangStrategy(n_top=N_LONG)
    strategy.train(train_data, min_date=TRAIN_START)
    
    print(f"   Model trained successfully")
    
    # ========================================================================
    # Test Period Backtest
    # ========================================================================
    print(f"\n📈 Running backtest ({TEST_START} to {TEST_END})...")
    
    # Get trading days in test period
    test_dates = pd.bdate_range(start=TEST_START, end=TEST_END)
    print(f"   Trading days: {len(test_dates)}")
    
    daily_returns = []
    daily_returns_after_fees = []
    
    for i, date in enumerate(test_dates):
        date_str = date.strftime('%Y-%m-%d')
        
        # Get data up to this date for prediction
        current_data = {}
        next_day_returns = {}
        
        for ticker, df in data_dict.items():
            df_up_to_date = df[df.index <= date_str]
            if len(df_up_to_date) > 220:  # Need enough for features
                current_data[ticker] = df_up_to_date
                
                # Get next day return
                df_after = df[df.index > date_str]
                if len(df_after) > 0:
                    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                    
                    # Handle multi-index columns
                    if isinstance(df.columns, pd.MultiIndex):
                        current_close = df_up_to_date.iloc[-1]['Close']
                        next_close = df_after.iloc[0]['Close']
                        if isinstance(current_close, pd.Series):
                            current_close = current_close.iloc[0]
                        if isinstance(next_close, pd.Series):
                            next_close = next_close.iloc[0]
                    else:
                        current_close = df_up_to_date[price_col].iloc[-1]
                        next_close = df_after[price_col].iloc[0]
                    
                    next_day_returns[ticker] = (next_close - current_close) / current_close
        
        if not current_data:
            continue
        
        # Generate signals using vertical ensemble
        signals = strategy.generate_signals(current_data)
        
        # Get long and short positions
        long_tickers = [t for t, s in signals.items() if s == 'BUY' and t in next_day_returns]
        short_tickers = [t for t, s in signals.items() if s == 'SELL' and t in next_day_returns]
        
        if long_tickers or short_tickers:
            # Equal weight (1/N) within long/short legs
            long_return = np.mean([next_day_returns[t] for t in long_tickers]) if long_tickers else 0
            short_return = -np.mean([next_day_returns[t] for t in short_tickers]) if short_tickers else 0
            
            # 50% long, 50% short (market neutral)
            portfolio_return = 0.5 * long_return + 0.5 * short_return
            daily_returns.append(portfolio_return)
            
            # Transaction cost: 0.05% per half-turn on both legs
            # Daily rebalancing = full turnover = 2 half-turns per leg
            n_positions = len(long_tickers) + len(short_tickers)
            if n_positions > 0:
                # Cost = 0.05% * 2 (for buy and sell) * 2 legs
                transaction_cost = TRANSACTION_COST_HALF_TURN * 2  # Round trip per day
                portfolio_return_after_fees = portfolio_return - transaction_cost
            else:
                portfolio_return_after_fees = portfolio_return
            
            daily_returns_after_fees.append(portfolio_return_after_fees)
            
            # Debug output every 50 days
            if (i + 1) % 50 == 0:
                # Check probability spread
                all_probs = [strategy.predict_probability(t, current_data[t]) for t in list(current_data.keys())[:20]]
                print(f"   Day {i+1}: prob range [{min(all_probs):.3f}, {max(all_probs):.3f}], "
                      f"long_ret={long_return:.4f}, short_ret={short_return:.4f}")
    
    # ========================================================================
    # Calculate Metrics (matching paper format)
    # ========================================================================
    print(f"\n🧮 Calculating metrics...")
    
    # Convert to numpy
    returns_before = np.array(daily_returns)
    returns_after = np.array(daily_returns_after_fees)
    
    n_days = len(returns_before)
    
    # Annualized Return (compounded)
    annualized_before = np.prod(1 + returns_before) - 1
    annualized_after = np.prod(1 + returns_after) - 1
    
    # Excess Return (over risk-free rate)
    excess_before = annualized_before - RISK_FREE_RATE
    excess_after = annualized_after - RISK_FREE_RATE
    
    # Standard Deviation (annualized)
    std_before = np.std(returns_before) * np.sqrt(252)
    std_after = np.std(returns_after) * np.sqrt(252)
    
    # Sharpe Ratio
    sharpe_before = excess_before / std_before if std_before > 0 else 0
    sharpe_after = excess_after / std_after if std_after > 0 else 0
    
    # ========================================================================
    # Print Results
    # ========================================================================
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"{'Metric':<25} {'Before Fees':>15} {'After Fees':>15}")
    print("-" * 55)
    print(f"{'Mean Return':<25} {annualized_before:>15.4f} {annualized_after:>15.4f}")
    print(f"{'Excess Return':<25} {excess_before:>15.4f} {excess_after:>15.4f}")
    print(f"{'St.D':<25} {std_before:>15.4f} {std_after:>15.4f}")
    print(f"{'Sharpe Ratio':<25} {sharpe_before:>15.4f} {sharpe_after:>15.4f}")
    print()
    print(f"Trading Days: {n_days}")
    print()
    
    # ========================================================================
    # Compare with Paper
    # ========================================================================
    print("=" * 70)
    print("COMPARISON WITH PAPER (Table 4.6, XGBoost n=10)")
    print("=" * 70)
    print()
    print(f"{'Metric':<20} {'Paper (After Fees)':>18} {'Ours (After Fees)':>18} {'Diff':>10}")
    print("-" * 66)
    
    paper_excess = 0.8312  # Without vertical ensemble ("Only t")
    paper_sharpe = 2.2042
    
    diff_excess = excess_after - paper_excess
    diff_sharpe = sharpe_after - paper_sharpe
    
    print(f"{'Excess Return':<20} {paper_excess:>18.4f} {excess_after:>18.4f} {diff_excess:>+10.4f}")
    print(f"{'Sharpe Ratio':<20} {paper_sharpe:>18.4f} {sharpe_after:>18.4f} {diff_sharpe:>+10.4f}")
    print()
    
    # Also compare with Best Weights results
    print("With Vertical Ensemble (Best Weights 0.1, 0.3, 0.6):")
    paper_excess_best = 0.9327
    paper_sharpe_best = 2.4532
    print(f"  Paper Excess Return: {paper_excess_best:.4f}")
    print(f"  Paper Sharpe Ratio:  {paper_sharpe_best:.4f}")
    print()
    print("=" * 70)
    
    return {
        'annualized_return': annualized_after,
        'excess_return': excess_after,
        'sharpe_ratio': sharpe_after,
        'std_dev': std_after,
        'trading_days': n_days,
        'paper_excess': paper_excess,
        'paper_sharpe': paper_sharpe
    }


if __name__ == '__main__':
    run_zhang_paper_test()
