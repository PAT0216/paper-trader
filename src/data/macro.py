"""
Macro Data Module for Paper Trader - Phase 3.6

Fetches macroeconomic indicators from FRED (Federal Reserve Economic Data).
These features provide market-wide context for individual stock predictions.

Features:
- VIX (fear index) for regime detection
- Retry logic with exponential backoff
- File-based caching (24-hour TTL)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import warnings
import time
import os
import json


# FRED Series IDs for macro indicators
MACRO_SERIES = {
    'VIX': 'VIXCLS',           # CBOE Volatility Index (fear gauge)
    'Yield_10Y': 'DGS10',      # 10-Year Treasury Constant Maturity Rate
    'Yield_2Y': 'DGS2',        # 2-Year Treasury Constant Maturity Rate
    'Yield_Spread': 'T10Y2Y',  # 10Y-2Y Treasury Spread (recession indicator)
    'Fed_Funds': 'DFF',        # Effective Federal Funds Rate
}

# Cache settings
CACHE_DIR = "data/cache"
CACHE_TTL_HOURS = 24


def _ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)


def _get_cache_path(series_name: str) -> str:
    """Get cache file path for a series."""
    return os.path.join(CACHE_DIR, f"{series_name}_cache.json")


def _is_cache_valid(cache_path: str) -> bool:
    """Check if cache file exists and is within TTL."""
    if not os.path.exists(cache_path):
        return False
    
    mtime = os.path.getmtime(cache_path)
    age_hours = (time.time() - mtime) / 3600
    return age_hours < CACHE_TTL_HOURS


def _save_to_cache(series_name: str, value: float, timestamp: str):
    """Save a value to cache."""
    _ensure_cache_dir()
    cache_path = _get_cache_path(series_name)
    cache_data = {
        'value': value,
        'timestamp': timestamp,
        'cached_at': datetime.now().isoformat()
    }
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f)


def _load_from_cache(series_name: str) -> Optional[float]:
    """Load a value from cache if valid."""
    cache_path = _get_cache_path(series_name)
    if _is_cache_valid(cache_path):
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                return data.get('value')
        except:
            return None
    return None


def fetch_with_retry(func, max_retries: int = 3, delay_base: float = 1.0):
    """
    Execute function with retry logic and exponential backoff.
    
    Args:
        func: Callable to execute
        max_retries: Maximum number of attempts
        delay_base: Base delay in seconds (doubles each retry)
        
    Returns:
        Result of func() or None if all retries fail
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                delay = delay_base * (2 ** attempt)  # 1s, 2s, 4s
                print(f"   ⚠️ Retry {attempt + 1}/{max_retries} after {delay}s: {e}")
                time.sleep(delay)
            else:
                print(f"   ❌ All retries failed: {e}")
                return None


def fetch_fred_data(
    series: Optional[list] = None,
    start_date: str = None,
    end_date: str = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch macro data from FRED API with retry logic and caching.
    
    Args:
        series: List of series names from MACRO_SERIES keys. Default: ['VIX', 'Yield_10Y']
        start_date: Start date (YYYY-MM-DD). Default: 3 years ago
        end_date: End date (YYYY-MM-DD). Default: today
        use_cache: Whether to use file-based caching
        
    Returns:
        DataFrame with macro indicators, DatetimeIndex
    """
    try:
        import pandas_datareader as pdr
    except ImportError:
        warnings.warn(
            "pandas-datareader not installed. Macro features will be unavailable. "
            "Install with: pip install pandas-datareader"
        )
        return pd.DataFrame()
    
    if series is None:
        series = ['VIX', 'Yield_10Y']  # Default: most useful indicators
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Map friendly names to FRED series IDs
    fred_ids = [MACRO_SERIES.get(s, s) for s in series]
    
    def _fetch():
        data = pdr.DataReader(fred_ids, 'fred', start_date, end_date)
        # Rename columns back to friendly names
        rename_map = {v: k for k, v in MACRO_SERIES.items()}
        data = data.rename(columns=rename_map)
        # Forward-fill missing values (FRED has gaps on weekends/holidays)
        data = data.ffill()
        return data
    
    result = fetch_with_retry(_fetch, max_retries=3)
    
    if result is not None:
        # Cache the latest values
        if use_cache and not result.empty:
            for col in result.columns:
                latest_value = result[col].dropna().iloc[-1]
                _save_to_cache(col, float(latest_value), result.index[-1].strftime('%Y-%m-%d'))
        return result
    
    return pd.DataFrame()


def get_current_vix(use_cache: bool = True) -> float:
    """
    Get the current (or most recent) VIX value.
    
    Uses caching to avoid repeated API calls.
    Falls back to cache if API fails.
    
    Args:
        use_cache: Whether to check cache first
        
    Returns:
        VIX value (float), or 20.0 as default if unavailable
    """
    DEFAULT_VIX = 20.0  # Historical average
    
    # Check cache first
    if use_cache:
        cached_vix = _load_from_cache('VIX')
        if cached_vix is not None:
            print(f"📊 Using cached VIX: {cached_vix:.2f}")
            return cached_vix
    
    # Fetch fresh data
    print("📊 Fetching current VIX from FRED...")
    vix_data = fetch_fred_data(series=['VIX'], use_cache=use_cache)
    
    if not vix_data.empty and 'VIX' in vix_data.columns:
        current_vix = vix_data['VIX'].dropna().iloc[-1]
        print(f"   VIX: {current_vix:.2f}")
        return float(current_vix)
    
    # Fallback to cache even if expired
    cached_vix = _load_from_cache('VIX')
    if cached_vix is not None:
        print(f"   ⚠️ Using expired cache: VIX = {cached_vix:.2f}")
        return cached_vix
    
    print(f"   ⚠️ VIX unavailable, using default: {DEFAULT_VIX}")
    return DEFAULT_VIX


def merge_macro_features(
    stock_df: pd.DataFrame,
    macro_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge macro features with stock data.
    
    Aligns macro data to stock trading dates using forward-fill
    to ensure no look-ahead bias (uses last known macro value).
    
    Args:
        stock_df: Stock OHLCV DataFrame with DatetimeIndex
        macro_df: Macro DataFrame from fetch_fred_data()
        
    Returns:
        Stock DataFrame with macro columns added
    """
    if macro_df.empty:
        return stock_df
    
    # Ensure both have DatetimeIndex
    stock_df = stock_df.copy()
    
    # Make sure index is timezone-naive for merge
    if hasattr(stock_df.index, 'tz') and stock_df.index.tz is not None:
        stock_df.index = stock_df.index.tz_localize(None)
    if hasattr(macro_df.index, 'tz') and macro_df.index.tz is not None:
        macro_df.index = macro_df.index.tz_localize(None)
    
    # Align macro to stock dates using merge_asof (forward-looking prevented)
    # This ensures we only use macro data available at time t
    merged = pd.merge_asof(
        stock_df.reset_index(),
        macro_df.reset_index(),
        left_on=stock_df.index.name or 'Date',
        right_on=macro_df.index.name or 'Date',
        direction='backward'  # Use most recent available macro data
    )
    
    # Restore index
    if 'Date' in merged.columns:
        merged = merged.set_index('Date')
    elif stock_df.index.name and stock_df.index.name in merged.columns:
        merged = merged.set_index(stock_df.index.name)
    
    return merged


# Macro feature columns to use
MACRO_FEATURE_COLUMNS = ['VIX', 'Yield_10Y']
