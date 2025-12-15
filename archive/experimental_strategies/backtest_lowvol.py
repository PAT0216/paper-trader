"""
Backtest: Low Volatility Factor Strategy

Buy the 50 least volatile S&P 500 stocks (based on 3-year weekly volatility).
Rebalance monthly.
"""

import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime


def load_sp500_tickers():
    """Load S&P 500 tickers from cache."""
    txt_path = 'data/sp500_tickers.txt'
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    return []


def load_price_data(db_path: str, tickers: list, start: str, end: str) -> dict:
    """Load price data from SQLite cache."""
    print(f"\n📥 Loading data from {db_path}")
    
    if not os.path.exists(db_path):
        print(f"   ❌ Database not found: {db_path}")
        return {}
    
    con = sqlite3.connect(db_path)
    data = {}
    
    for ticker in tickers:
        query = """
            SELECT date, COALESCE(adj_close, close) as close
            FROM price_data
            WHERE ticker = ?
              AND date >= ?
              AND date <= ?
            ORDER BY date
        """
        df = pd.read_sql_query(query, con, params=(ticker, start, end))
        if len(df) > 100:  # Need sufficient history
            df['date'] = pd.to_datetime(df['date'])
            data[ticker] = df.set_index('date')
    
    con.close()
    print(f"   Loaded {len(data)} tickers with sufficient data")
    return data


def calculate_volatility(prices: pd.Series, lookback_weeks: int = 156) -> float:
    """
    Calculate annualized volatility from weekly returns.
    
    Args:
        prices: Daily price series
        lookback_weeks: Number of weeks to look back (156 = 3 years)
    """
    # Resample to weekly (Friday close)
    weekly = prices.resample('W-FRI').last().dropna()
    
    if len(weekly) < lookback_weeks:
        return np.nan
    
    # Use last N weeks
    weekly = weekly.iloc[-lookback_weeks:]
    
    # Weekly returns
    returns = weekly.pct_change().dropna()
    
    # Annualized volatility (52 weeks per year)
    return returns.std() * np.sqrt(52)


def run_backtest(data: dict, start_date: str, end_date: str, 
                 n_stocks: int = 50, initial_capital: float = 10000) -> pd.DataFrame:
    """
    Run low volatility backtest.
    
    Args:
        data: Dictionary of {ticker: DataFrame with 'close' column}
        start_date: Backtest start
        end_date: Backtest end
        n_stocks: Number of low-vol stocks to hold
        initial_capital: Starting capital
    """
    print(f"\n📊 BACKTESTING LOW VOLATILITY STRATEGY")
    print(f"   Period: {start_date} to {end_date}")
    print(f"   Holding {n_stocks} lowest volatility stocks")
    print("=" * 50)
    
    # Use SPY dates as the trading calendar (most liquid, has all dates)
    if 'SPY' not in data:
        print("   ❌ SPY not in data - needed for calendar")
        return pd.DataFrame()
    
    all_dates = list(data['SPY'].index)
    all_dates = sorted([d for d in all_dates if start_date <= str(d)[:10] <= end_date])
    
    if not all_dates:
        print("   ❌ No dates found in range")
        return pd.DataFrame()
    
    print(f"   Trading days: {len(all_dates)}")
    
    # Get monthly rebalance dates
    rebalance_dates = set()
    for d in all_dates:
        month_key = (d.year, d.month)
        if month_key not in {(rd.year, rd.month) for rd in rebalance_dates}:
            rebalance_dates.add(d)
    
    print(f"   Rebalance dates: {len(rebalance_dates)}")
    
    # Initialize
    cash = initial_capital
    holdings = {}  # {ticker: shares}
    portfolio_values = []
    
    selected_tickers = []
    
    for i, date in enumerate(all_dates):
        # Calculate portfolio value
        total_value = cash
        for ticker, shares in holdings.items():
            if date in data[ticker].index:
                price = data[ticker].loc[date, 'close']
                total_value += shares * price
        
        portfolio_values.append({
            'date': date,
            'value': total_value
        })
        
        # Rebalance at month start
        if date in rebalance_dates:
            # Calculate volatility for all tickers (need 3 years of history)
            volatilities = {}
            for ticker, df in data.items():
                # Get data up to this date
                df_hist = df[df.index <= date]
                if len(df_hist) >= 756:  # ~3 years of daily data
                    vol = calculate_volatility(df_hist['close'])
                    if not np.isnan(vol):
                        volatilities[ticker] = vol
            
            if len(volatilities) < n_stocks:
                continue  # Not enough data
            
            # Select lowest volatility stocks
            sorted_vols = sorted(volatilities.items(), key=lambda x: x[1])
            selected_tickers = [t for t, v in sorted_vols[:n_stocks]]
            
            # Sell everything
            for ticker, shares in holdings.items():
                if date in data[ticker].index:
                    price = data[ticker].loc[date, 'close']
                    cash += shares * price
            holdings = {}
            
            # Buy new portfolio (equal weight)
            weight = 1.0 / n_stocks
            for ticker in selected_tickers:
                if date in data[ticker].index:
                    price = data[ticker].loc[date, 'close']
                    target_value = total_value * weight
                    shares = int(target_value / price)
                    if shares > 0:
                        holdings[ticker] = shares
                        cash -= shares * price
    
    return pd.DataFrame(portfolio_values)


def calculate_metrics(df: pd.DataFrame, name: str) -> dict:
    """Calculate and print performance metrics."""
    if df.empty or len(df) < 2:
        return {}
    
    df['return'] = df['value'].pct_change()
    
    total_return = (df['value'].iloc[-1] / df['value'].iloc[0]) - 1
    years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    volatility = df['return'].std() * np.sqrt(252)
    sharpe = cagr / volatility if volatility > 0 else 0
    
    rolling_max = df['value'].cummax()
    drawdown = (df['value'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    metrics = {
        'total_return': round(total_return * 100, 2),
        'cagr': round(cagr * 100, 2),
        'volatility': round(volatility * 100, 2),
        'sharpe': round(sharpe, 2),
        'max_drawdown': round(max_drawdown * 100, 2),
        'final_value': round(df['value'].iloc[-1], 2)
    }
    
    print(f"\n📈 {name}:")
    for key, value in metrics.items():
        print(f"   {key}: {value}")
    
    return metrics


def run_spy_benchmark(data: dict, start_date: str, end_date: str, 
                      initial_capital: float) -> pd.DataFrame:
    """Run SPY buy-and-hold benchmark."""
    if 'SPY' not in data:
        return pd.DataFrame()
    
    df = data['SPY'].reset_index()
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    if df.empty:
        return pd.DataFrame()
    
    start_price = df['close'].iloc[0]
    df['value'] = initial_capital * (df['close'] / start_price)
    
    return df[['date', 'value']]


def main():
    print("=" * 60)
    print("LOW VOLATILITY FACTOR - BACKTEST")
    print("=" * 60)
    
    # Configuration
    DB_PATH = 'data/market.db'
    START_DATE = '2016-01-01'  # Need 3 years prior for volatility calculation
    END_DATE = '2024-11-01'
    N_STOCKS = 50
    INITIAL_CAPITAL = 10000
    
    # Load tickers
    tickers = load_sp500_tickers()
    if not tickers:
        print("❌ No tickers found")
        return
    
    # Add SPY for benchmark
    if 'SPY' not in tickers:
        tickers.append('SPY')
    
    # Load data (need extra history for volatility calculation)
    data = load_price_data(DB_PATH, tickers, '2013-01-01', END_DATE)
    
    if len(data) < N_STOCKS:
        print(f"❌ Insufficient data: {len(data)} tickers")
        return
    
    # Run backtest
    results = run_backtest(data, START_DATE, END_DATE, N_STOCKS, INITIAL_CAPITAL)
    
    if results.empty:
        print("❌ Backtest failed")
        return
    
    # Run benchmark
    spy_results = run_spy_benchmark(data, START_DATE, END_DATE, INITIAL_CAPITAL)
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    lowvol_metrics = calculate_metrics(results, "LOW VOLATILITY")
    spy_metrics = calculate_metrics(spy_results, "SPY BUY-AND-HOLD")
    
    # Comparison
    print("\n🆚 COMPARISON:")
    cagr_diff = lowvol_metrics.get('cagr', 0) - spy_metrics.get('cagr', 0)
    dd_diff = spy_metrics.get('max_drawdown', 0) - lowvol_metrics.get('max_drawdown', 0)
    sharpe_diff = lowvol_metrics.get('sharpe', 0) - spy_metrics.get('sharpe', 0)
    
    print(f"   CAGR Difference: {cagr_diff:+.2f}%")
    print(f"   Drawdown Improvement: {dd_diff:+.2f}%")
    print(f"   Sharpe Improvement: {sharpe_diff:+.2f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    import json
    output = {
        'strategy': 'low_volatility',
        'period': f"{START_DATE} to {END_DATE}",
        'n_stocks': N_STOCKS,
        'lowvol_metrics': lowvol_metrics,
        'spy_metrics': spy_metrics
    }
    with open('results/lowvol_backtest.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved to results/lowvol_backtest.json")


if __name__ == "__main__":
    main()
