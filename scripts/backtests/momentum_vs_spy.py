"""
Walk-Forward Backtest: Momentum Strategy 2016-2025 vs S&P 500
Monthly rebalancing with 12-1 month momentum factor
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.cache import DataCache
from src.data.universe import UniverseManager

# Configuration
START_DATE = "2016-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100000
TOP_N = 10
CHANGES_CSV = 'data/sp500_changes_2019_plus.csv'

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

def get_spy_returns(cache, start_date, end_date):
    """Get SPY returns for comparison."""
    spy_data = cache.get_price_data('SPY', start_date=start_date, end_date=end_date)
    if spy_data.empty:
        return {}
    
    spy_data = spy_data.reset_index()
    spy_data.columns = [c.lower().replace('_', '') for c in spy_data.columns]
    price_col = 'adjclose' if 'adjclose' in spy_data.columns else 'close'
    
    spy_data['year'] = pd.to_datetime(spy_data['date']).dt.year
    
    yearly_returns = {}
    for year in spy_data['year'].unique():
        year_data = spy_data[spy_data['year'] == year].sort_values('date')
        if len(year_data) >= 2:
            start_px = year_data[price_col].iloc[0]
            end_px = year_data[price_col].iloc[-1]
            yearly_returns[int(year)] = ((end_px / start_px) - 1) * 100
    
    # Total return
    start_px = spy_data[price_col].iloc[0]
    end_px = spy_data[price_col].iloc[-1]
    total_return = ((end_px / start_px) - 1) * 100
    
    years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365
    cagr = ((end_px / start_px) ** (1/years) - 1) * 100 if years > 0 else 0
    
    return {
        'yearly': yearly_returns,
        'total_return': total_return,
        'cagr': cagr,
        'start_value': INITIAL_CAPITAL,
        'end_value': INITIAL_CAPITAL * (end_px / start_px)
    }

def run_backtest():
    """Run the full momentum backtest with SPY comparison."""
    print("=" * 70)
    print("MOMENTUM STRATEGY WALK-FORWARD BACKTEST + SPY COMPARISON")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,}")
    print(f"Top N Stocks: {TOP_N}")
    print("=" * 70)
    
    # Load current verified universe from CSV
    print("🔍 Loading verified S&P 500 list from data/sp500_current.csv...")
    if os.path.exists('data/sp500_current.csv'):
        sp500_df = pd.read_csv('data/sp500_current.csv')
        current_tickers = sp500_df['Symbol'].tolist()
        print(f"Current Verified Universe: {len(current_tickers)} tickers")
    else:
        print("⚠️ Warning: data/sp500_current.csv not found, falling back to cache")
        cache = DataCache()
        current_tickers = cache.get_cached_tickers()
    
    # Initialize Universe Manager
    universe_mgr = UniverseManager(current_tickers, CHANGES_CSV)
    
    cache = DataCache() # Re-init cache for price data
    
    # Get SPY benchmark
    print("\n📊 Loading SPY benchmark...")
    spy_results = get_spy_returns(cache, START_DATE, END_DATE)
    
    trading_days = get_trading_days(START_DATE, END_DATE)
    
    # Monthly rebalance dates
    rebalance_dates = []
    current_month = None
    for d in trading_days:
        month = d[:7]
        if month != current_month:
            rebalance_dates.append(d)
            current_month = month
    
    print(f"Rebalance dates: {len(rebalance_dates)} months\n")
    
    # Track portfolio
    cash = INITIAL_CAPITAL
    holdings = {}
    portfolio_history = []
    yearly_values = {}  # {year: {'start': val, 'end': val}}
    
    for i, rebalance_date in enumerate(rebalance_dates):
        year = int(rebalance_date[:4])
        
        try:
            # UNBIASED UNIVERSE SELECTION
            valid_universe = universe_mgr.get_universe_at(rebalance_date)
            # Only trade tickers we have data for AND were in the index
            active_tickers = [t for t in valid_universe if t in current_tickers]
            
            # DEBUG: Check if SMCI is in Jan 2024
            if rebalance_date.startswith("2024-01") and "SMCI" in active_tickers:
                 print(f"⚠️ WAIT: SMCI found in valid_universe for {rebalance_date}! (Should be removed)")
            if rebalance_date.startswith("2024-01") and "SMCI" not in active_tickers:
                 print(f"✅ CONFIRMED: SMCI excluded from universe for {rebalance_date}")

            start = (datetime.strptime(rebalance_date, '%Y-%m-%d') - timedelta(days=400)).strftime('%Y-%m-%d')
            # Using active_tickers instead of full list
            price_data = get_all_price_data(cache, active_tickers, start, rebalance_date)
            
            if price_data.empty:
                continue
            
            price_col = 'adjclose' if 'adjclose' in price_data.columns else 'close'
            
            # Calculate momentum
            momentum_scores = {}
            latest_prices = {}
            
            for ticker in price_data['ticker'].unique():
                ticker_data = price_data[price_data['ticker'] == ticker].sort_values('date')
                if len(ticker_data) >= 252:
                    score = calculate_momentum(ticker_data)
                    if not np.isnan(score):
                        momentum_scores[ticker] = score
                        latest_prices[ticker] = ticker_data[price_col].iloc[-1]
            
            if not momentum_scores:
                continue
            
            # Top N stocks
            top_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
            
            # Calculate portfolio value
            portfolio_value = cash
            for ticker, shares in holdings.items():
                if ticker in latest_prices:
                    portfolio_value += shares * latest_prices[ticker]
            
            # Track yearly values
            if year not in yearly_values:
                yearly_values[year] = {'start': portfolio_value, 'end': portfolio_value}
            yearly_values[year]['end'] = portfolio_value
            
            # DEBUG: Analysis of 2024 performance
            if year == 2024:
                print(f"\\n🔍 2024 DEBUG [{rebalance_date}] Value: ${portfolio_value:,.0f}")
                print(f"   Top Holdings (Momentum):")
                for t, score in top_stocks[:5]:
                    curr_px = latest_prices[t]
                    print(f"   - {t}: Score {score:.2f}, Price ${curr_px:.2f}")
                
                # Check performance of sold positions if we had any
                if holdings:
                    print("   Selling Returns:")
                    for t, shares in holdings.items():
                        if t in latest_prices:
                            sell_px = latest_prices[t]
                            # We don't track buy price easily here, but we can verify reasonable prices
                            print(f"   - Sold {t}: ${sell_px:.2f} ({shares} shares)")

            portfolio_history.append((rebalance_date, portfolio_value))
            
            # Sell all
            for ticker, shares in list(holdings.items()):
                if ticker in latest_prices and shares > 0:
                    cash += shares * latest_prices[ticker]
            holdings = {}
            
            # Buy new positions
            position_size = portfolio_value / len(top_stocks)
            
            for ticker, score in top_stocks:
                if ticker not in latest_prices:
                    continue
                price = latest_prices[ticker]
                if price <= 0:
                    continue
                shares = int(position_size / price)
                if shares > 0 and cash >= shares * price:
                    cash -= shares * price
                    holdings[ticker] = shares
            
        except Exception as e:
            continue
    
    # Calculate results
    print("\n" + "=" * 70)
    print("RESULTS: MOMENTUM vs S&P 500")
    print("=" * 70)
    
    if portfolio_history:
        final_value = portfolio_history[-1][1]
        total_return = (final_value / INITIAL_CAPITAL - 1) * 100
        
        years = (datetime.strptime(END_DATE, '%Y-%m-%d') - datetime.strptime(START_DATE, '%Y-%m-%d')).days / 365
        cagr = ((final_value / INITIAL_CAPITAL) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Max drawdown
        values = [v[1] for v in portfolio_history]
        peak = values[0]
        max_dd = 0
        for v in values:
            if v > peak:
                peak = v
            dd = (v - peak) / peak
            if dd < max_dd:
                max_dd = dd
        
        # Yearly returns
        yearly_returns = {}
        prev_end = INITIAL_CAPITAL
        for year in sorted(yearly_values.keys()):
            yr_ret = (yearly_values[year]['end'] / prev_end - 1) * 100
            yearly_returns[year] = yr_ret
            prev_end = yearly_values[year]['end']
        
        # Print comparison table
        print(f"\n{'Year':<8} {'Momentum':>12} {'S&P 500':>12} {'Diff':>10}")
        print("-" * 45)
        
        for year in sorted(yearly_returns.keys()):
            mom_ret = yearly_returns[year]
            spy_ret = spy_results['yearly'].get(year, 0)
            diff = mom_ret - spy_ret
            diff_str = f"{diff:+.1f}%" if diff != 0 else "-"
            
            mom_color = "🟢" if mom_ret > spy_ret else "🔴"
            print(f"{year:<8} {mom_ret:>11.1f}% {spy_ret:>11.1f}% {mom_color} {diff_str:>8}")
        
        print("-" * 45)
        
        # Summary
        print(f"\n{'SUMMARY':<20}")
        print(f"{'Initial Capital:':<20} ${INITIAL_CAPITAL:>15,}")
        print(f"{'─' * 40}")
        print(f"{'':20} {'Momentum':>15} {'S&P 500':>15}")
        print(f"{'Final Value:':<20} ${final_value:>14,.0f} ${spy_results['end_value']:>14,.0f}")
        print(f"{'Total Return:':<20} {total_return:>14.1f}% {spy_results['total_return']:>14.1f}%")
        print(f"{'CAGR:':<20} {cagr:>14.1f}% {spy_results['cagr']:>14.1f}%")
        print(f"{'Max Drawdown:':<20} {max_dd*100:>14.1f}%")
        
        # Winner
        if cagr > spy_results['cagr']:
            print(f"\n🏆 MOMENTUM WINS by {cagr - spy_results['cagr']:.1f}% CAGR")
        else:
            print(f"\n📉 S&P 500 WINS by {spy_results['cagr'] - cagr:.1f}% CAGR")
        
        # Save results
        results = {
            'period': f"{START_DATE} to {END_DATE}",
            'momentum': {
                'initial': INITIAL_CAPITAL,
                'final': final_value,
                'total_return_pct': total_return,
                'cagr_pct': cagr,
                'max_drawdown_pct': max_dd * 100,
                'yearly_returns': yearly_returns
            },
            'spy': spy_results
        }
        
        os.makedirs('results', exist_ok=True)
        with open('results/momentum_vs_spy_2016_2025.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✅ Saved to results/momentum_vs_spy_2016_2025.json")
        
        return results
    
    return None


if __name__ == "__main__":
    run_backtest()
