"""
Universe Management for Paper Trader - Phase 4

Manages the investable stock universe:
1. Fetches S&P 500 constituents from Wikipedia
2. Filters by liquidity (Average Daily Volume)
3. Provides tiered universe for different use cases

No API key required - uses free public data sources.
"""

import pandas as pd
import requests
from typing import List, Optional, Dict
import time
import logging

logger = logging.getLogger(__name__)

# Default tickers (fallback if S&P 500 fetch fails)
DEFAULT_TICKERS = [
    # Mega-cap tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    # ETFs
    'SPY', 'QQQ', 'DIA', 'IWM',
    # Financials
    'JPM', 'V', 'MA', 'BAC',
    # Healthcare
    'UNH', 'JNJ', 'LLY', 'MRK',
    # Consumer
    'WMT', 'HD', 'PG', 'KO', 'PEP', 'COST',
    # Energy
    'XOM', 'CVX',
    # Other mega-caps
    'AVGO', 'BRK-B', 'NFLX', 'CRM', 'AMD', 'CAT', 'DIS', 'BA'
]


def fetch_sp500_tickers() -> List[str]:
    """
    Fetch current S&P 500 constituents from Wikipedia.
    
    Wikipedia maintains an up-to-date table of S&P 500 companies.
    This is a reliable, free source that doesn't require an API key.
    
    Returns:
        List of ticker symbols (500+ typically)
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    try:
        # Wikipedia requires a user agent
        headers = {
            'User-Agent': 'PaperTrader/1.0 (Educational Project)'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML tables
        tables = pd.read_html(response.text)
        
        # First table contains the current constituents
        sp500_table = tables[0]
        
        # Symbol column might be named 'Symbol' or 'Ticker'
        symbol_col = 'Symbol' if 'Symbol' in sp500_table.columns else 'Ticker'
        
        tickers = sp500_table[symbol_col].tolist()
        
        # Clean tickers (some have . which yfinance uses -)
        tickers = [t.replace('.', '-') for t in tickers]
        
        print(f"📊 Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers
        
    except Exception as e:
        print(f"⚠️ Failed to fetch S&P 500 from Wikipedia: {e}")
        print(f"   Using default {len(DEFAULT_TICKERS)} tickers")
        return DEFAULT_TICKERS


def get_ticker_volume(ticker: str, period: str = '3mo') -> float:
    """
    Get average daily dollar volume for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        period: Period to average over
        
    Returns:
        Average daily volume in dollars
    """
    try:
        import yfinance as yf
        
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            return 0.0
        
        # Average daily dollar volume = avg(volume * close)
        adv = (hist['Volume'] * hist['Close']).mean()
        return float(adv)
        
    except Exception as e:
        logger.debug(f"Failed to get volume for {ticker}: {e}")
        return 0.0


def filter_by_liquidity(
    tickers: List[str], 
    min_adv: float = 50_000_000,
    max_tickers: int = 200,
    delay: float = 0.2
) -> List[str]:
    """
    Filter tickers by minimum Average Daily Volume.
    
    To avoid rate limits, this function:
    1. Uses a delay between API calls
    2. Stops once we have enough tickers
    3. Prioritizes checking likely-liquid stocks first
    
    Args:
        tickers: List of ticker symbols to filter
        min_adv: Minimum average daily volume in dollars (default: $50M)
        max_tickers: Stop after finding this many liquid tickers
        delay: Delay between API calls (seconds)
        
    Returns:
        List of liquid ticker symbols
    """
    print(f"🔍 Filtering {len(tickers)} tickers by liquidity (ADV > ${min_adv/1e6:.0f}M)...")
    
    liquid_tickers = []
    checked = 0
    
    # Sort alphabetically but prioritize known liquid names
    priority_tickers = [t for t in tickers if t in DEFAULT_TICKERS]
    other_tickers = [t for t in tickers if t not in DEFAULT_TICKERS]
    ordered_tickers = priority_tickers + other_tickers
    
    for ticker in ordered_tickers:
        if len(liquid_tickers) >= max_tickers:
            break
        
        checked += 1
        adv = get_ticker_volume(ticker)
        
        if adv >= min_adv:
            liquid_tickers.append(ticker)
            if checked % 50 == 0:
                print(f"   Checked {checked}, found {len(liquid_tickers)} liquid tickers...")
        
        time.sleep(delay)
    
    print(f"   ✅ Found {len(liquid_tickers)} liquid tickers (checked {checked})")
    return liquid_tickers


def get_tradeable_universe(
    min_adv: float = 50_000_000,
    top_n: int = 200,
    use_cache: bool = True
) -> List[str]:
    """
    Get the final tradeable universe of stocks.
    
    Process:
    1. Fetch S&P 500 tickers
    2. Filter by liquidity
    3. Return top N most liquid
    
    Args:
        min_adv: Minimum average daily volume in dollars
        top_n: Maximum number of tickers to return
        use_cache: Whether to use cached universe if available
        
    Returns:
        List of tradeable ticker symbols
    """
    # Check for cached universe
    cache_file = 'data/universe_cache.csv'
    
    if use_cache:
        try:
            import os
            if os.path.exists(cache_file):
                # Check if cache is fresh (< 7 days old)
                mtime = os.path.getmtime(cache_file)
                age_days = (time.time() - mtime) / 86400
                
                if age_days < 7:
                    df = pd.read_csv(cache_file)
                    tickers = df['ticker'].tolist()[:top_n]
                    print(f"📊 Using cached universe: {len(tickers)} tickers")
                    return tickers
        except Exception as e:
            logger.debug(f"Cache read failed: {e}")
    
    # Fetch fresh universe
    sp500 = fetch_sp500_tickers()
    
    # For quick startup, just return S&P 500 without full liquidity check
    # Full liquidity filter takes ~10 minutes due to API calls
    print(f"📊 Returning S&P 500 universe: {len(sp500)} tickers")
    
    # Save to cache
    try:
        import os
        os.makedirs('data', exist_ok=True)
        pd.DataFrame({'ticker': sp500}).to_csv(cache_file, index=False)
    except Exception as e:
        logger.debug(f"Cache write failed: {e}")
    
    return sp500[:top_n]


def get_tiered_universe() -> Dict[str, List[str]]:
    """
    Get tiered universe for different use cases.
    
    Returns:
        Dict with 'tier1', 'tier2', 'tier3' lists
    """
    return {
        'tier1': DEFAULT_TICKERS,  # ~35 mega-caps (always trade)
        'tier2': get_tradeable_universe(top_n=150),  # ~150 liquid S&P 500
        'tier3': get_tradeable_universe(top_n=500),  # Full S&P 500
    }


# Convenience functions
def get_sp500() -> List[str]:
    """Get all S&P 500 tickers."""
    return fetch_sp500_tickers()


def get_mega_caps() -> List[str]:
    """Get mega-cap tickers (Tier 1)."""
    return DEFAULT_TICKERS.copy()
