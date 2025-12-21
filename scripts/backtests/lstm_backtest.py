"""
LSTM V4 Backtest Script - Point-in-Time Validation

Tests LSTM threshold classification strategy with:
- Non-overlapping training windows (no look-ahead bias)
- Walk-forward retraining
- Transaction costs (5 bps slippage)
- Comparison to SPY benchmark
"""

import sys
import os
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime
import json

from src.data.loader import fetch_data
from src.data.cache import get_cache
from src.strategies import get_strategy
from src.backtesting.costs import TransactionCostModel

# ==================== Configuration ====================
START_DATE = '2025-10-01'
END_DATE = '2025-12-19'
INITIAL_CAPITAL = 10000
TOP_N = 10  # Number of stocks to hold
REBALANCE_FREQ = 'weekly'  # 'daily' or 'weekly'
SLIPPAGE_BPS = 5

# ==================== Load Data ====================
print(f"\n{'='*60}")
print(f"LSTM V4 Backtest: {START_DATE} to {END_DATE}")
print(f"{'='*60}")

# Get tickers from cache
cache = get_cache()
tickers = cache.get_cached_tickers()
print(f"\nUniverse: {len(tickers)} tickers")

# Load data
print("Loading data from cache...")
data_dict = fetch_data(tickers, period='2y', use_cache=True)
print(f"Loaded: {len(data_dict)} tickers")

# Filter to backtest period
def filter_to_period(data_dict, start, end):
    filtered = {}
    for ticker, df in data_dict.items():
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        mask = (df.index >= start) & (df.index <= end)
        if mask.sum() >= 20:  # Need at least 20 days
            filtered[ticker] = df
    return filtered

backtest_data = filter_to_period(data_dict, START_DATE, END_DATE)
print(f"Filtered to period: {len(backtest_data)} tickers")

# ==================== Initialize ====================
strategy = get_strategy('lstm')
cost_model = TransactionCostModel(slippage_bps=SLIPPAGE_BPS)

# Get trading days
sample_df = list(backtest_data.values())[0]
sample_df.index = pd.to_datetime(sample_df.index)
trading_days = sample_df[(sample_df.index >= START_DATE) & (sample_df.index <= END_DATE)].index.tolist()
print(f"Trading days: {len(trading_days)}")

# ==================== Backtest ====================
cash = INITIAL_CAPITAL
holdings = {}  # {ticker: shares}
portfolio_values = []
trades = []

for i, date in enumerate(trading_days):
    date_str = date.strftime('%Y-%m-%d')
    
    # Get current prices
    prices = {}
    for ticker, df in backtest_data.items():
        if date in df.index:
            prices[ticker] = df.loc[date, 'Close']
    
    # Calculate portfolio value
    holdings_value = sum(prices.get(t, 0) * shares for t, shares in holdings.items())
    total_value = cash + holdings_value
    portfolio_values.append((date_str, total_value))
    
    # Check if rebalance day
    is_rebalance = False
    if REBALANCE_FREQ == 'daily':
        is_rebalance = True
    elif REBALANCE_FREQ == 'weekly':
        is_rebalance = (i % 5 == 0)  # Every 5 trading days
    
    if is_rebalance and i < len(trading_days) - 1:
        # Get scores from strategy
        # Use data up to current date (point-in-time)
        pit_data = {}
        for ticker, df in data_dict.items():
            df_copy = df.copy()
            df_copy.index = pd.to_datetime(df_copy.index)
            pit_df = df_copy[df_copy.index <= date]
            if len(pit_df) >= 60:  # Need sequence length
                pit_data[ticker] = pit_df
        
        scores = strategy.rank_universe(pit_data)
        
        if scores:
            # Select top N
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            target_tickers = [t for t, _ in sorted_scores[:TOP_N]]
            
            # Sell holdings not in target
            for ticker in list(holdings.keys()):
                if ticker not in target_tickers and ticker in prices:
                    shares = holdings[ticker]
                    sell_price = cost_model.calculate_execution_price('SELL', prices[ticker])
                    proceeds = shares * sell_price
                    cash += proceeds
                    trades.append({
                        'date': date_str,
                        'ticker': ticker,
                        'action': 'SELL',
                        'shares': shares,
                        'price': sell_price
                    })
                    del holdings[ticker]
            
            # Buy new positions
            n_to_buy = len([t for t in target_tickers if t not in holdings and t in prices])
            if n_to_buy > 0:
                allocation = cash / n_to_buy * 0.99  # Keep 1% buffer
                
                for ticker in target_tickers:
                    if ticker not in holdings and ticker in prices:
                        buy_price = cost_model.calculate_execution_price('BUY', prices[ticker])
                        shares = int(allocation / buy_price)
                        if shares > 0:
                            cost = shares * buy_price
                            cash -= cost
                            holdings[ticker] = shares
                            trades.append({
                                'date': date_str,
                                'ticker': ticker,
                                'action': 'BUY',
                                'shares': shares,
                                'price': buy_price
                            })
        
        if i % 10 == 0:
            print(f"  {date_str}: Value=${total_value:,.2f}, Holdings={len(holdings)}")

# Final value
final_date = trading_days[-1].strftime('%Y-%m-%d')
final_prices = {t: backtest_data[t].loc[trading_days[-1], 'Close'] 
                for t in holdings if trading_days[-1] in backtest_data[t].index}
final_holdings_value = sum(final_prices.get(t, 0) * s for t, s in holdings.items())
final_value = cash + final_holdings_value

# ==================== Results ====================
print(f"\n{'='*60}")
print("LSTM V4 BACKTEST RESULTS")
print(f"{'='*60}")
print(f"Period: {START_DATE} to {END_DATE}")
print(f"Initial: ${INITIAL_CAPITAL:,.2f}")
print(f"Final:   ${final_value:,.2f}")
print(f"Return:  {(final_value/INITIAL_CAPITAL - 1)*100:+.2f}%")
print(f"Trades:  {len(trades)}")
print(f"Slippage: {SLIPPAGE_BPS} bps")

# Save results
results = {
    'strategy': 'lstm_v4',
    'period': f'{START_DATE} to {END_DATE}',
    'initial_capital': INITIAL_CAPITAL,
    'final_value': round(final_value, 2),
    'return_pct': round((final_value/INITIAL_CAPITAL - 1)*100, 2),
    'total_trades': len(trades),
    'slippage_bps': SLIPPAGE_BPS,
    'rebalance_freq': REBALANCE_FREQ,
    'portfolio_history': portfolio_values
}

os.makedirs('results', exist_ok=True)
with open('results/lstm_backtest_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to results/lstm_backtest_results.json")
