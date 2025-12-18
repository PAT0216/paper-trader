"""
Backfill ledgers with walk-forward simulation from October 1st, 2025.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.cache import DataCache

# Configuration
START_DATE = "2025-10-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 10000

def get_trading_days(start: str, end: str) -> list:
    dates = pd.date_range(start=start, end=end, freq='B')
    return [d.strftime('%Y-%m-%d') for d in dates]

def get_all_price_data(cache, tickers, start_date, end_date):
    """Get price data for multiple tickers."""
    all_data = []
    for ticker in tickers:
        try:
            df = cache.get_price_data(ticker, start_date=start_date, end_date=end_date)
            if not df.empty:
                df = df.reset_index()
                df['ticker'] = ticker
                # Normalize column names
                df.columns = [c.lower().replace('_', '') for c in df.columns]
                all_data.append(df)
        except Exception:
            continue
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

def calculate_momentum(ticker_data, lookback=252, skip=21):
    """Calculate 12-1 month momentum using available data."""
    if len(ticker_data) < 50:  # Need at least some history
        return np.nan
    
    # Use 'adjclose' or 'close' column
    price_col = 'adjclose' if 'adjclose' in ticker_data.columns else 'close'
    
    # For shorter periods (Oct-Dec), use available data
    start_idx = max(0, len(ticker_data) - min(lookback, len(ticker_data)))
    end_idx = max(0, len(ticker_data) - min(skip, len(ticker_data) // 4) - 1)
    
    if start_idx >= end_idx:
        return np.nan
        
    start_price = ticker_data[price_col].iloc[start_idx]
    end_price = ticker_data[price_col].iloc[end_idx]
    
    if start_price <= 0:
        return np.nan
        
    return (end_price / start_price) - 1

def run_momentum_backfill():
    """Simulate momentum strategy from Oct 1st."""
    print("=" * 60)
    print("MOMENTUM STRATEGY BACKFILL")
    print("=" * 60)
    
    cache = DataCache()
    
    trading_days = get_trading_days(START_DATE, END_DATE)
    
    # Get first of each month for rebalancing
    rebalance_dates = []
    current_month = None
    for d in trading_days:
        month = d[:7]
        if month != current_month:
            rebalance_dates.append(d)
            current_month = month
    
    print(f"Trading period: {START_DATE} to {END_DATE}")
    print(f"Rebalance dates: {rebalance_dates}")
    
    # Initialize
    ledger = []
    cash = INITIAL_CAPITAL
    holdings = {}
    
    ledger.append({
        'date': START_DATE,
        'ticker': 'CASH',
        'action': 'DEPOSIT',
        'price': 1.0,
        'shares': INITIAL_CAPITAL,
        'amount': INITIAL_CAPITAL,
        'cash_balance': cash,
        'total_value': INITIAL_CAPITAL,
        'strategy': '',
        'momentum_score': ''
    })
    
    # Use verified S&P 500 universe
    if os.path.exists('data/sp500_current.csv'):
        sp500_df = pd.read_csv('data/sp500_current.csv')
        tickers = sp500_df['Symbol'].tolist()
        print(f"Using verified S&P 500: {len(tickers)} tickers")
    else:
        tickers = cache.get_cached_tickers()[:150]
        print(f"Fallback to cache: {len(tickers)} tickers")
    
    for rebalance_date in rebalance_dates:
        print(f"\n📅 Rebalancing on {rebalance_date}...")
        
        try:
            # Get 1 year of history for momentum calculation
            start = (datetime.strptime(rebalance_date, '%Y-%m-%d') - timedelta(days=400)).strftime('%Y-%m-%d')
            price_data = get_all_price_data(cache, tickers, start, rebalance_date)
            
            if price_data.empty:
                print(f"  No data, skipping")
                continue
            
            print(f"  Got data for {price_data['ticker'].nunique()} tickers")
            
            # Calculate momentum for each ticker
            momentum_scores = {}
            latest_prices = {}
            
            price_col = 'adjclose' if 'adjclose' in price_data.columns else 'close'
            
            for ticker in price_data['ticker'].unique():
                ticker_data = price_data[price_data['ticker'] == ticker].sort_values('date')
                if len(ticker_data) >= 50:
                    score = calculate_momentum(ticker_data)
                    if not np.isnan(score):  # Removed artificial cap
                        momentum_scores[ticker] = score
                        latest_prices[ticker] = ticker_data[price_col].iloc[-1]
            
            if not momentum_scores:
                print(f"  No valid momentum scores")
                continue
            
            print(f"  Valid scores: {len(momentum_scores)}")
            
            # Top 12 stocks
            top_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[:12]
            print(f"  Top picks: {[t[0] for t in top_stocks[:5]]}...")
            
            # Calculate current value
            portfolio_value = cash
            for ticker, shares in holdings.items():
                if ticker in latest_prices:
                    portfolio_value += shares * latest_prices[ticker]
            
            # Sell all current holdings
            for ticker, shares in list(holdings.items()):
                if ticker in latest_prices and shares > 0:
                    price = latest_prices[ticker]
                    amount = shares * price
                    cash += amount
                    
                    ledger.append({
                        'date': rebalance_date,
                        'ticker': ticker,
                        'action': 'SELL',
                        'price': round(price, 2),
                        'shares': shares,
                        'amount': round(amount, 2),
                        'cash_balance': round(cash, 2),
                        'total_value': round(portfolio_value, 2),
                        'strategy': 'momentum',
                        'momentum_score': ''
                    })
            
            holdings = {}
            
            # Buy new positions (equal weight)
            position_size = portfolio_value / len(top_stocks)
            
            for ticker, score in top_stocks:
                if ticker not in latest_prices:
                    continue
                    
                price = latest_prices[ticker]
                if price <= 0:
                    continue
                    
                shares = int(position_size / price)
                
                if shares > 0 and cash >= shares * price:
                    amount = shares * price
                    cash -= amount
                    holdings[ticker] = shares
                    
                    total_value = cash + sum(s * latest_prices.get(t, 0) for t, s in holdings.items())
                    
                    ledger.append({
                        'date': rebalance_date,
                        'ticker': ticker,
                        'action': 'BUY',
                        'price': round(price, 2),
                        'shares': shares,
                        'amount': round(amount, 2),
                        'cash_balance': round(cash, 2),
                        'total_value': round(total_value, 2),
                        'strategy': 'momentum',
                        'momentum_score': round(score, 4)
                    })
            
            print(f"  Holdings: {len(holdings)} positions, Value: ${portfolio_value:,.0f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save the ledger (trades only - dashboard computes daily values dynamically)
    df = pd.DataFrame(ledger)
    df.to_csv('ledger_momentum.csv', index=False)
    print(f"\n✅ Saved ledger_momentum.csv with {len(df)} entries")
    return df


def run_ml_backfill():
    """Simulate ML strategy (weekly trades, short-term momentum)."""
    print("\n" + "=" * 60)
    print("ML STRATEGY BACKFILL")
    print("=" * 60)
    
    cache = DataCache()
    
    trading_days = get_trading_days(START_DATE, END_DATE)
    trade_dates = [d for i, d in enumerate(trading_days) if i % 5 == 0]
    
    print(f"Trading period: {START_DATE} to {END_DATE}")
    print(f"Trade dates: {len(trade_dates)} weeks")
    
    ledger = []
    cash = INITIAL_CAPITAL
    holdings = {}
    
    ledger.append({
        'date': START_DATE,
        'ticker': 'CASH',
        'action': 'DEPOSIT',
        'price': 1.0,
        'shares': INITIAL_CAPITAL,
        'amount': INITIAL_CAPITAL,
        'cash_balance': cash,
        'total_value': INITIAL_CAPITAL,
        'strategy': '',
        'momentum_score': ''
    })
    
    tickers = cache.get_cached_tickers()[:100]
    
    for trade_date in trade_dates:
        try:
            start = (datetime.strptime(trade_date, '%Y-%m-%d') - timedelta(days=60)).strftime('%Y-%m-%d')
            price_data = get_all_price_data(cache, tickers, start, trade_date)
            
            if price_data.empty:
                continue
            
            price_col = 'adjclose' if 'adjclose' in price_data.columns else 'close'
            
            # Short-term momentum (20-day)
            momentum_scores = {}
            latest_prices = {}
            
            for ticker in price_data['ticker'].unique():
                ticker_data = price_data[price_data['ticker'] == ticker].sort_values('date')
                if len(ticker_data) >= 20:
                    ret = (ticker_data[price_col].iloc[-1] / ticker_data[price_col].iloc[-20]) - 1
                    if -0.5 < ret < 1.0:
                        momentum_scores[ticker] = ret
                        latest_prices[ticker] = ticker_data[price_col].iloc[-1]
            
            if not momentum_scores:
                continue
            
            # Top 10
            top_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            top_tickers = [t[0] for t in top_stocks]
            
            # Portfolio value
            portfolio_value = cash
            for ticker, shares in holdings.items():
                if ticker in latest_prices:
                    portfolio_value += shares * latest_prices[ticker]
            
            # Sell if not in top 10
            for ticker in list(holdings.keys()):
                if ticker not in top_tickers and holdings[ticker] > 0:
                    if ticker in latest_prices:
                        price = latest_prices[ticker]
                        shares = holdings[ticker]
                        amount = shares * price
                        cash += amount
                        
                        ledger.append({
                            'date': trade_date,
                            'ticker': ticker,
                            'action': 'SELL',
                            'price': round(price, 2),
                            'shares': shares,
                            'amount': round(amount, 2),
                            'cash_balance': round(cash, 2),
                            'total_value': round(portfolio_value, 2),
                            'strategy': 'ml',
                            'momentum_score': ''
                        })
                        del holdings[ticker]
            
            # Buy new
            position_size = portfolio_value / 10
            
            for ticker, score in top_stocks:
                if ticker in holdings:
                    continue
                if ticker not in latest_prices:
                    continue
                    
                price = latest_prices[ticker]
                if price <= 0:
                    continue
                    
                shares = int(position_size / price)
                
                if shares > 0 and cash >= shares * price:
                    amount = shares * price
                    cash -= amount
                    holdings[ticker] = shares
                    
                    total_value = cash + sum(s * latest_prices.get(t, 0) for t, s in holdings.items())
                    
                    ledger.append({
                        'date': trade_date,
                        'ticker': ticker,
                        'action': 'BUY',
                        'price': round(price, 2),
                        'shares': shares,
                        'amount': round(amount, 2),
                        'cash_balance': round(cash, 2),
                        'total_value': round(total_value, 2),
                        'strategy': 'ml',
                        'momentum_score': round(score, 4)
                    })
            
            print(f"  {trade_date}: {len(holdings)} positions, Value: ${portfolio_value:,.0f}")
            
        except Exception as e:
            print(f"  Error on {trade_date}: {e}")
            continue
    
    df = pd.DataFrame(ledger)
    df.to_csv('ledger_ml.csv', index=False)
    print(f"\n✅ Saved ledger_ml.csv with {len(df)} entries")
    return df


if __name__ == "__main__":
    print(f"Backfilling ledgers from {START_DATE} to {END_DATE}")
    print(f"Initial capital: ${INITIAL_CAPITAL:,}")
    print()
    
    mom_df = run_momentum_backfill()
    ml_df = run_ml_backfill()
    
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    
    # Check momentum ledger
    mom_df = pd.read_csv('ledger_momentum.csv')
    print(f"\nMomentum Ledger:")
    print(f"  Entries: {len(mom_df)}")
    print(f"  Date range: {mom_df['date'].min()} to {mom_df['date'].max()}")
    print(f"  Unique dates: {mom_df['date'].nunique()}")
    print(f"  Actions: {mom_df['action'].value_counts().to_dict()}")
    if len(mom_df) > 1:
        print(f"  Final value: ${mom_df.iloc[-1]['total_value']:,.0f}")
    
    # Check ML ledger
    ml_df = pd.read_csv('ledger_ml.csv')
    print(f"\nML Ledger:")
    print(f"  Entries: {len(ml_df)}")
    print(f"  Date range: {ml_df['date'].min()} to {ml_df['date'].max()}")
    print(f"  Unique dates: {ml_df['date'].nunique()}")
    print(f"  Actions: {ml_df['action'].value_counts().to_dict()}")
    if len(ml_df) > 1:
        print(f"  Final value: ${ml_df.iloc[-1]['total_value']:,.0f}")
