"""
Data Loader for Paper Trader - Phase 4

Smart data loading with SQLite caching:
1. Check cache first
2. Fetch only new bars (incremental update)
3. Rate limit handling with retries
4. Graceful fallbacks
"""

import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

from src.data.cache import DataCache, get_cache

logger = logging.getLogger(__name__)

# Rate limit settings
FETCH_DELAY = 0.3  # Seconds between API calls
MAX_RETRIES = 3
RETRY_DELAY = 5  # Seconds between retries


def fetch_data(
    tickers: List[str], 
    period: str = "2y", 
    interval: str = "1d",
    use_cache: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Fetches historical data for multiple tickers with caching.
    
    Cache-first strategy:
    1. Check if data exists in SQLite cache
    2. If cached, only fetch new bars (yesterday to today)
    3. If not cached, fetch full history and cache it
    
    Args:
        tickers: List of ticker symbols
        period: Period for initial fetch (e.g., '2y', '3y', 'max')
        interval: Data interval (e.g., '1d', '1h')
        use_cache: Whether to use SQLite cache
        
    Returns:
        Dictionary of {ticker: DataFrame}
    """
    data = {}
    cache = get_cache() if use_cache else None
    
    print(f"Fetching data for: {tickers}")
    if use_cache:
        print(f"   Cache enabled: data/market.db")
    
    for i, ticker in enumerate(tickers):
        try:
            df = _fetch_single_ticker(ticker, period, interval, cache)
            
            if df is not None and not df.empty:
                data[ticker] = df
            else:
                print(f"   ⚠️ No data for {ticker}")
            
            # Rate limiting between API calls
            if i < len(tickers) - 1:
                time.sleep(FETCH_DELAY)
                
        except Exception as e:
            print(f"   ❌ Error fetching {ticker}: {e}")
    
    print(f"   ✅ Loaded {len(data)} tickers")
    return data


def _fetch_single_ticker(
    ticker: str, 
    period: str, 
    interval: str, 
    cache: Optional[DataCache]
) -> Optional[pd.DataFrame]:
    """
    Fetch data for a single ticker with caching logic.
    """
    # Check cache first
    if cache:
        last_cached = cache.get_last_date(ticker)
        
        if last_cached:
            # We have cached data - only fetch new bars
            last_date = datetime.strptime(last_cached, '%Y-%m-%d')
            fetch_start = last_date + timedelta(days=1)
            today = datetime.now()
            
            if fetch_start.date() > today.date():
                # Cache is fully up to date (has today's data already)
                cached_df = cache.get_price_data(ticker)
                if not cached_df.empty:
                    return cached_df
            
            # Fetch only new data
            new_data = _fetch_with_retry(ticker, start=fetch_start.strftime('%Y-%m-%d'))
            
            if new_data is not None and not new_data.empty:
                cache.update_price_data(ticker, new_data)
            
            # Return full cached data
            return cache.get_price_data(ticker)
    
    # No cache or not cached - fetch full history
    full_data = _fetch_with_retry(ticker, period=period)
    
    if full_data is not None and not full_data.empty:
        if cache:
            cache.update_price_data(ticker, full_data)
        return full_data
    
    return None


def _fetch_with_retry(
    ticker: str, 
    period: str = None, 
    start: str = None,
    end: str = None
) -> Optional[pd.DataFrame]:
    """
    Fetch data from yfinance with retry logic.
    
    Handles rate limits with exponential backoff.
    """
    for attempt in range(MAX_RETRIES):
        try:
            stock = yf.Ticker(ticker)
            
            if start:
                df = stock.history(start=start, end=end)
            else:
                df = stock.history(period=period)
            
            if df.empty:
                return None
            
            # Ensure proper format
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            error_str = str(e).lower()
            
            if 'rate limit' in error_str or 'too many requests' in error_str:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    print(f"   ⚠️ Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"   ❌ Rate limit exceeded for {ticker}")
                    return None
            else:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    raise
    
    return None


def fetch_from_cache_only(
    tickers: List[str],
    start_date: str = None,
    end_date: str = None
) -> Dict[str, pd.DataFrame]:
    """
    Load data from cache only, no API calls.
    
    Useful for backtesting to avoid rate limits entirely.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        Dictionary of {ticker: DataFrame}
    """
    cache = get_cache()
    data = {}
    
    for ticker in tickers:
        df = cache.get_price_data(ticker, start_date, end_date)
        if not df.empty:
            data[ticker] = df
    
    print(f"📦 Loaded {len(data)}/{len(tickers)} tickers from cache")
    return data


def update_cache(tickers: List[str], period: str = "3y"):
    """
    Update cache with latest data for all tickers.
    
    Use this to populate the cache initially or refresh all data.
    
    Args:
        tickers: List of ticker symbols
        period: Period for initial fetch
    """
    print(f"📥 Updating cache for {len(tickers)} tickers...")
    fetch_data(tickers, period=period, use_cache=True)
    
    cache = get_cache()
    stats = cache.get_cache_stats()
    print(f"📊 Cache stats:")
    print(f"   Tickers cached: {len(stats)}")
    if not stats.empty:
        print(f"   Date range: {stats['first_date'].min()} to {stats['last_date'].max()}")
        print(f"   Total rows: {stats['rows_cached'].sum():,}")

