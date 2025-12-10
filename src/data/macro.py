"""
Macro Data Module for Paper Trader - Phase 3.5

Fetches macroeconomic indicators from FRED (Federal Reserve Economic Data).
These features provide market-wide context for individual stock predictions.

FRED is free to use and requires no API key for basic access via pandas-datareader.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import warnings


# FRED Series IDs for macro indicators
MACRO_SERIES = {
    'VIX': 'VIXCLS',           # CBOE Volatility Index (fear gauge)
    'Yield_10Y': 'DGS10',      # 10-Year Treasury Constant Maturity Rate
    'Yield_2Y': 'DGS2',        # 2-Year Treasury Constant Maturity Rate
    'Yield_Spread': 'T10Y2Y',  # 10Y-2Y Treasury Spread (recession indicator)
    'Fed_Funds': 'DFF',        # Effective Federal Funds Rate
}


def fetch_fred_data(
    series: Optional[list] = None,
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """
    Fetch macro data from FRED API.
    
    Args:
        series: List of series names from MACRO_SERIES keys. Default: ['VIX', 'Yield_10Y']
        start_date: Start date (YYYY-MM-DD). Default: 3 years ago
        end_date: End date (YYYY-MM-DD). Default: today
        
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
    
    try:
        data = pdr.DataReader(fred_ids, 'fred', start_date, end_date)
        
        # Rename columns back to friendly names
        rename_map = {v: k for k, v in MACRO_SERIES.items()}
        data = data.rename(columns=rename_map)
        
        # Forward-fill missing values (FRED has gaps on weekends/holidays)
        data = data.ffill()
        
        return data
        
    except Exception as e:
        warnings.warn(f"Failed to fetch FRED data: {e}")
        return pd.DataFrame()


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
