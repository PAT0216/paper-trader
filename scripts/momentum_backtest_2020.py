"""
Walk-Forward Backtest: Momentum Strategy 2020-2025 YTD
Monthly rebalancing with 12-1 month momentum factor
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cache import DataCache

# Configuration
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100000
TOP_N = 12  # Number of stocks to hold

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
                df.columns = [c.lower().replace('_', '') for c in df.columns]
                all_data.append(df)
        except Exception:
            continue
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

def calculate_momentum(ticker_data, lookback=252, skip=21):
    """Calculate 12-1 month momentum."""
    if len(ticker_data) < lookback:
        return np.nan
    
    price_col = 'adjclose' if 'adjclose' in ticker_data.columns else 'close'
    
    start_price = ticker_data[price_col].iloc[-(lookback)]
    end_price = ticker_data[price_col].iloc[-(skip+1)]
    
    if start_price <= 0:
        return np.nan
        
    return (end_price / start_price) - 1

def run_backtest():
    """Run the full momentum backtest."""
    print("=" * 70)
    print("MOMENTUM STRATEGY WALK-FORWARD BACKTEST")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,}")
    print(f"Top N Stocks: {TOP_N}")
    print("=" * 70)
    
    cache = DataCache()
    tickers = cache.get_cached_tickers()
    print(f"Using {len(tickers)} tickers from cache")
    
    trading_days = get_trading_days(START_DATE, END_DATE)
    
    # Get first of each month for rebalancing
    rebalance_dates = []
    current_month = None
    for d in trading_days:
        month = d[:7]
        if month != current_month:
            rebalance_dates.append(d)
            current_month = month
    
    print(f"Rebalance dates: {len(rebalance_dates)} months")
    print()
    
    # Track portfolio
    cash = INITIAL_CAPITAL
    holdings = {}  # ticker -> shares
    portfolio_history = []  # List of (date, value)
    trades = []
    
    for i, rebalance_date in enumerate(rebalance_dates):
        try:
            # Get 1 year of history for momentum calculation
            start = (datetime.strptime(rebalance_date, '%Y-%m-%d') - timedelta(days=400)).strftime('%Y-%m-%d')
            price_data = get_all_price_data(cache, tickers, start, rebalance_date)
            
            if price_data.empty:
                continue
            
            price_col = 'adjclose' if 'adjclose' in price_data.columns else 'close'
            
            # Calculate momentum for each ticker
            momentum_scores = {}
            latest_prices = {}
            
            for ticker in price_data['ticker'].unique():
                ticker_data = price_data[price_data['ticker'] == ticker].sort_values('date')
                if len(ticker_data) >= 252:  # Need full year of history
                    score = calculate_momentum(ticker_data)
                    if not np.isnan(score) and -0.5 < score < 3.0:
                        momentum_scores[ticker] = score
                        latest_prices[ticker] = ticker_data[price_col].iloc[-1]
            
            if not momentum_scores:
                continue
            
            # Top N stocks
            top_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
            
            # Calculate current portfolio value
            portfolio_value = cash
            for ticker, shares in holdings.items():
                if ticker in latest_prices:
                    portfolio_value += shares * latest_prices[ticker]
            
            portfolio_history.append((rebalance_date, portfolio_value))
            
            # Print monthly summary
            if i % 6 == 0 or i == len(rebalance_dates) - 1:
                ret = (portfolio_value / INITIAL_CAPITAL - 1) * 100
                print(f"{rebalance_date}: ${portfolio_value:,.0f} ({ret:+.1f}%)")
            
            # Sell all holdings
            for ticker, shares in list(holdings.items()):
                if ticker in latest_prices and shares > 0:
                    price = latest_prices[ticker]
                    amount = shares * price
                    cash += amount
                    trades.append({
                        'date': rebalance_date,
                        'ticker': ticker,
                        'action': 'SELL',
                        'shares': shares,
                        'price': price,
                        'amount': amount
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
                    trades.append({
                        'date': rebalance_date,
                        'ticker': ticker,
                        'action': 'BUY',
                        'shares': shares,
                        'price': price,
                        'amount': amount,
                        'momentum': score
                    })
            
        except Exception as e:
            print(f"Error on {rebalance_date}: {e}")
            continue
    
    # Final values
    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    if portfolio_history:
        final_value = portfolio_history[-1][1]
        total_return = (final_value / INITIAL_CAPITAL - 1) * 100
        
        # Calculate CAGR
        years = (datetime.strptime(END_DATE, '%Y-%m-%d') - datetime.strptime(START_DATE, '%Y-%m-%d')).days / 365
        cagr = ((final_value / INITIAL_CAPITAL) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Calculate max drawdown
        values = [v[1] for v in portfolio_history]
        peak = values[0]
        max_dd = 0
        for v in values:
            if v > peak:
                peak = v
            dd = (v - peak) / peak
            if dd < max_dd:
                max_dd = dd
        
        # Calculate yearly returns
        yearly_returns = {}
        for date, value in portfolio_history:
            year = date[:4]
            if year not in yearly_returns:
                yearly_returns[year] = {'start': value, 'end': value}
            yearly_returns[year]['end'] = value
        
        print(f"Initial Capital:    ${INITIAL_CAPITAL:,}")
        print(f"Final Value:        ${final_value:,.0f}")
        print(f"Total Return:       {total_return:+.2f}%")
        print(f"CAGR:               {cagr:.2f}%")
        print(f"Max Drawdown:       {max_dd*100:.2f}%")
        print(f"Total Trades:       {len(trades)}")
        print()
        
        print("YEARLY RETURNS:")
        prev_end = INITIAL_CAPITAL
        for year in sorted(yearly_returns.keys()):
            yr_ret = (yearly_returns[year]['end'] / prev_end - 1) * 100
            print(f"  {year}: {yr_ret:+.2f}%")
            prev_end = yearly_returns[year]['end']
        
        # Save results
        results = {
            'period': f"{START_DATE} to {END_DATE}",
            'initial_capital': INITIAL_CAPITAL,
            'final_value': final_value,
            'total_return_pct': total_return,
            'cagr_pct': cagr,
            'max_drawdown_pct': max_dd * 100,
            'total_trades': len(trades),
            'yearly_returns': {y: (d['end']/d['start']-1)*100 for y, d in yearly_returns.items()}
        }
        
        # Save to file
        import json
        with open('results/momentum_backtest_2020_2025.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Saved results to results/momentum_backtest_2020_2025.json")
        
        # Save portfolio history
        pd.DataFrame(portfolio_history, columns=['date', 'value']).to_csv(
            'results/momentum_portfolio_history.csv', index=False
        )
        print("✅ Saved portfolio history to results/momentum_portfolio_history.csv")
        
        return results
    
    return None


if __name__ == "__main__":
    run_backtest()
