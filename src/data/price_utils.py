"""
Price Utilities - Functions for fetching prices from the market database.

Extracted from compute_portfolio_snapshot.py for reusability.
"""

import os
import sqlite3
import pandas as pd
from typing import Dict, List
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo


def get_market_date() -> str:
    """Get today's date in New York timezone (market time)."""
    ny_tz = ZoneInfo('America/New_York')
    return datetime.now(ny_tz).strftime('%Y-%m-%d')


def get_latest_prices(tickers: List[str], db_path: str) -> Dict[str, float]:
    """
    Fetch latest prices for given tickers from the database.
    
    Args:
        tickers: List of ticker symbols
        db_path: Path to market database
        
    Returns:
        Dictionary of {ticker: price}
    """
    if not tickers or not os.path.exists(db_path):
        return {}
    
    con = sqlite3.connect(db_path)
    try:
        ticker_list = ','.join([f"'{t}'" for t in tickers])
        query = f"""
            SELECT ticker, date, COALESCE(adj_close, close) AS price
            FROM price_data
            WHERE ticker IN ({ticker_list})
            ORDER BY date DESC
        """
        df = pd.read_sql_query(query, con)
        
        # Get latest price for each ticker
        latest = df.groupby('ticker').first().reset_index()
        return dict(zip(latest['ticker'], latest['price']))
    finally:
        con.close()


def get_latest_date(db_path: str) -> str:
    """
    Get the most recent date in the database.
    
    Args:
        db_path: Path to market database
        
    Returns:
        Date string (YYYY-MM-DD)
    """
    if not os.path.exists(db_path):
        return get_market_date()
    
    con = sqlite3.connect(db_path)
    try:
        result = pd.read_sql_query("SELECT MAX(date) as max_date FROM price_data", con)
        return result['max_date'].iloc[0] if not result.empty else get_market_date()
    finally:
        con.close()


def compute_portfolio_value(
    holdings: Dict[str, int], 
    prices: Dict[str, float], 
    cash: float
) -> float:
    """
    Calculate total portfolio value.
    
    Args:
        holdings: Dictionary of {ticker: shares}
        prices: Dictionary of {ticker: price}
        cash: Cash balance
        
    Returns:
        Total portfolio value
    """
    total = cash
    for ticker, shares in holdings.items():
        if ticker in prices:
            total += shares * prices[ticker]
    return total
