#!/usr/bin/env python3
"""
Fetch and cache Indian market data for Paper Trader India.

This script downloads OHLCV data for NIFTY 50 stocks and saves
it to india/data/india_cache.db

Usage:
    python india/fetch_india_data.py

The data is then used by the Indian market backtesting scripts.
"""

import os
import sys
import time
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from india.nifty50_universe import get_nifty50_tickers, NIFTY_INDEX, SENSEX_INDEX

# India-specific paths
INDIA_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INDIA_CACHE_DB = os.path.join(INDIA_DATA_DIR, "india_cache.db")


def fetch_and_cache_india_data(period="10y"):
    """
    Fetch NIFTY 50 stocks and indices, save to SQLite cache.
    
    Args:
        period: yfinance period string (e.g., "10y", "5y", "max")
    """
    print("=" * 60)
    print("INDIAN MARKET DATA FETCHER")
    print("=" * 60)
    
    # Ensure data directory exists
    os.makedirs(INDIA_DATA_DIR, exist_ok=True)
    
    # Get tickers
    tickers = get_nifty50_tickers()
    # Add benchmarks
    tickers.extend([NIFTY_INDEX, SENSEX_INDEX])
    
    print(f"Fetching {len(tickers)} tickers (NIFTY 50 + indices)")
    print(f"Period: {period}")
    print(f"Cache: {INDIA_CACHE_DB}")
    print()
    
    # Connect to SQLite
    conn = sqlite3.connect(INDIA_CACHE_DB)
    
    success = 0
    failed = []
    
    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] {ticker}...", end=" ")
        
        try:
            # Fetch data
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            
            if len(df) < 100:
                print(f"❌ Only {len(df)} rows")
                failed.append(ticker)
                continue
            
            # Handle multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Reset index to make Date a column
            df = df.reset_index()
            df['ticker'] = ticker
            
            # Standardize column names
            df.columns = [str(c).lower().replace(' ', '_') for c in df.columns]
            
            # Ensure we have required columns
            required = ['date', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required):
                print(f"❌ Missing columns: {df.columns.tolist()}")
                failed.append(ticker)
                continue
            
            # Select only needed columns
            df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']]
            
            # Save to SQLite (append)
            df.to_sql(
                name='price_data',
                con=conn,
                if_exists='append',
                index=False
            )
            
            print(f"✅ {len(df)} rows")
            success += 1
            
            # Small delay to avoid rate limiting
            time.sleep(0.3)
            
        except Exception as e:
            print(f"❌ {e}")
            failed.append(ticker)
    
    conn.close()
    
    print()
    print("=" * 60)
    print(f"COMPLETE: {success}/{len(tickers)} tickers cached")
    if failed:
        print(f"FAILED: {failed}")
    print(f"Cache saved to: {INDIA_CACHE_DB}")
    print("=" * 60)


def get_india_data(ticker):
    """Load cached data for a ticker."""
    if not os.path.exists(INDIA_CACHE_DB):
        return None
    
    conn = sqlite3.connect(INDIA_CACHE_DB)
    query = f"SELECT * FROM price_data WHERE ticker = '{ticker}'"
    df = pd.read_sql(query, conn, parse_dates=['date'])
    conn.close()
    
    if len(df) == 0:
        return None
    
    df = df.set_index('date')
    # Rename to standard format
    df = df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume'
    })
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def get_all_india_tickers():
    """Get list of all tickers in the cache."""
    if not os.path.exists(INDIA_CACHE_DB):
        return []
    
    conn = sqlite3.connect(INDIA_CACHE_DB)
    query = "SELECT DISTINCT ticker FROM price_data"
    tickers = pd.read_sql(query, conn)['ticker'].tolist()
    conn.close()
    return tickers


if __name__ == "__main__":
    fetch_and_cache_india_data(period="10y")
