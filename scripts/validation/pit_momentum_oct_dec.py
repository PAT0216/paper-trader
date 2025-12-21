#!/usr/bin/env python3
"""
Point-in-Time Backtest: Momentum Strategy Oct 1, 2025 - Dec 19, 2025
Monthly rebalancing with 12-1 month momentum factor + transaction costs
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.cache import DataCache
from src.backtesting.costs import TransactionCostModel

# Configuration
START_DATE = "2025-10-01"
END_DATE = "2025-12-19"
INITIAL_CAPITAL = 10000
TOP_N = 10  # Match production


def calculate_momentum(ticker_data, lookback=252, skip=21):
    """Calculate 12-1 month momentum (skip most recent month)."""
    if len(ticker_data) < lookback:
        return np.nan
    
    price_col = 'Adj_Close' if 'Adj_Close' in ticker_data.columns else 'Close'
    prices = ticker_data[price_col].values
    
    # 12-1 month momentum: return from 12 months ago to 1 month ago
    if len(prices) >= lookback:
        return (prices[-skip] / prices[-lookback]) - 1
    return np.nan


def run_backtest():
    print("=" * 70)
    print("MOMENTUM PIT BACKTEST: Oct 1 - Dec 19, 2025")
    print("Monthly rebalancing with 5 bps slippage")
    print("=" * 70)
    
    cache = DataCache()
    cost_model = TransactionCostModel()  # 5 bps slippage
    
    # Get tickers
    with open('data/sp500_tickers.txt', 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]
    
    print(f"Using {len(tickers)} tickers")
    
    # Get rebalance dates (first of each month in range)
    rebalance_dates = ['2025-10-01', '2025-11-01', '2025-12-01']
    
    # Track portfolio
    cash = INITIAL_CAPITAL
    holdings = {}
    portfolio_history = [(START_DATE, INITIAL_CAPITAL)]
    all_trades = []
    
    for rebalance_date in rebalance_dates:
        print(f"\n📅 Rebalancing: {rebalance_date}")
        
        # Get 1 year of history for momentum calculation
        start = (datetime.strptime(rebalance_date, '%Y-%m-%d') - timedelta(days=400)).strftime('%Y-%m-%d')
        
        # Load price data
        price_data = []
        for ticker in tickers:
            df = cache.get_price_data(ticker, start, rebalance_date)
            if not df.empty:
                df['ticker'] = ticker
                price_data.append(df)
        
        if not price_data:
            continue
            
        price_df = pd.concat(price_data)
        price_col = 'Adj_Close' if 'Adj_Close' in price_df.columns else 'Close'
        
        # Calculate momentum for each ticker
        momentum_scores = {}
        latest_prices = {}
        
        for ticker in price_df['ticker'].unique():
            ticker_data = price_df[price_df['ticker'] == ticker].sort_values('date')
            if len(ticker_data) >= 252:
                score = calculate_momentum(ticker_data)
                if not np.isnan(score) and -0.5 < score < 3.0:
                    momentum_scores[ticker] = score
                    latest_prices[ticker] = ticker_data[price_col].iloc[-1]
        
        print(f"   Valid momentum scores: {len(momentum_scores)}")
        
        if not momentum_scores:
            continue
        
        # Top N stocks
        top_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
        print(f"   Top picks: {[t[0] for t in top_stocks[:5]]}...")
        
        # Calculate current portfolio value
        portfolio_value = cash
        for ticker, shares in holdings.items():
            if ticker in latest_prices:
                portfolio_value += shares * latest_prices[ticker]
        
        # SELL all holdings (with slippage)
        for ticker, shares in list(holdings.items()):
            if ticker in latest_prices and shares > 0:
                raw_price = latest_prices[ticker]
                exec_price, _ = cost_model.calculate_execution_price('SELL', raw_price, shares)
                amount = shares * exec_price
                cash += amount
                all_trades.append({
                    'date': rebalance_date,
                    'ticker': ticker,
                    'action': 'SELL',
                    'price': exec_price,
                    'shares': shares
                })
        
        holdings = {}
        
        # BUY new positions (with slippage)
        position_size = portfolio_value / len(top_stocks)
        
        for ticker, score in top_stocks:
            if ticker not in latest_prices:
                continue
            
            raw_price = latest_prices[ticker]
            exec_price, _ = cost_model.calculate_execution_price('BUY', raw_price, 1)
            shares = int(position_size / exec_price)
            
            if shares > 0 and cash >= shares * exec_price:
                cost = shares * exec_price
                cash -= cost
                holdings[ticker] = shares
                all_trades.append({
                    'date': rebalance_date,
                    'ticker': ticker,
                    'action': 'BUY',
                    'price': exec_price,
                    'shares': shares,
                    'momentum_score': score
                })
        
        # Update portfolio value
        portfolio_value = cash
        for ticker, shares in holdings.items():
            if ticker in latest_prices:
                portfolio_value += shares * latest_prices[ticker]
        
        portfolio_history.append((rebalance_date, portfolio_value))
        ret = (portfolio_value / INITIAL_CAPITAL - 1) * 100
        print(f"   Portfolio: ${portfolio_value:,.2f} ({ret:+.2f}%)")
    
    # Final value on END_DATE
    final_prices = {}
    for ticker in holdings:
        df = cache.get_price_data(ticker, END_DATE, END_DATE)
        if not df.empty:
            price_col = 'Adj_Close' if 'Adj_Close' in df.columns else 'Close'
            final_prices[ticker] = df[price_col].iloc[-1]
        elif ticker in latest_prices:
            final_prices[ticker] = latest_prices[ticker]
    
    final_value = cash
    for ticker, shares in holdings.items():
        if ticker in final_prices:
            final_value += shares * final_prices[ticker]
    
    portfolio_history.append((END_DATE, final_value))
    
    # Results
    total_return = (final_value / INITIAL_CAPITAL - 1) * 100
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Initial Value:  ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Value:    ${final_value:,.2f}")
    print(f"Total Return:   {total_return:+.2f}%")
    print(f"Total Trades:   {len(all_trades)}")
    print(f"Final Holdings: {len(holdings)} positions")
    
    # Save results
    results = {
        'strategy': 'momentum',
        'start_date': START_DATE,
        'end_date': END_DATE,
        'initial_value': INITIAL_CAPITAL,
        'final_value': round(final_value, 2),
        'total_return': round(total_return, 2),
        'transaction_costs': '5 bps slippage',
        'portfolio_history': portfolio_history,
        'total_trades': len(all_trades),
        'final_holdings': holdings
    }
    
    with open('results/pit_momentum_oct_dec_2025.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to results/pit_momentum_oct_dec_2025.json")
    
    # Save ledger
    ledger_rows = [{'date': START_DATE, 'ticker': 'CASH', 'action': 'DEPOSIT', 
                    'price': 1.0, 'shares': INITIAL_CAPITAL, 'amount': INITIAL_CAPITAL,
                    'cash_balance': INITIAL_CAPITAL, 'total_value': INITIAL_CAPITAL,
                    'strategy': 'momentum', 'momentum_score': ''}]
    
    running_cash = INITIAL_CAPITAL
    for trade in all_trades:
        if trade['action'] == 'SELL':
            running_cash += trade['shares'] * trade['price']
        else:
            running_cash -= trade['shares'] * trade['price']
        
        ledger_rows.append({
            'date': trade['date'],
            'ticker': trade['ticker'],
            'action': trade['action'],
            'price': trade['price'],
            'shares': trade['shares'],
            'amount': trade['shares'] * trade['price'],
            'cash_balance': running_cash,
            'total_value': final_value,  # Approximate
            'strategy': 'momentum',
            'momentum_score': trade.get('momentum_score', '')
        })
    
    ledger_df = pd.DataFrame(ledger_rows)
    ledger_df.to_csv('ledger_momentum.csv', index=False)
    print(f"💾 Ledger saved to ledger_momentum.csv ({len(all_trades)} trades)")


if __name__ == "__main__":
    run_backtest()
