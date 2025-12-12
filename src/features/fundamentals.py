"""
Fundamental Factors Data Fetcher

Implements Fama-French 5-Factor model data collection:
- Size (SMB): Market capitalization
- Value (HML): Price-to-Book ratio
- Profitability (RMW): ROE, profit margins
- Investment (CMA): Capital allocation metrics
- Momentum: Already in indicators.py

Uses yfinance API to fetch fundamental data with caching
to avoid excessive API calls.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Cache directory for fundamental data
FUNDAMENTALS_CACHE_DIR = "data/cache/fundamentals"


def get_fundamentals_for_ticker(ticker: str, use_cache: bool = True) -> Dict:
    """
    Fetch fundamental data for a single ticker.
    
    Args:
        ticker: Stock ticker symbol
        use_cache: Whether to use cached data if available
        
    Returns:
        Dict with fundamental metrics
    """
    cache_file = os.path.join(FUNDAMENTALS_CACHE_DIR, f"{ticker}.json")
    
    # Check cache first
    if use_cache and os.path.exists(cache_file):
        cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
        if cache_age < timedelta(days=7):  # Cache valid for 7 days
            with open(cache_file, 'r') as f:
                return json.load(f)
    
    # Fetch from yfinance
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        fundamentals = {
            'ticker': ticker,
            'fetch_date': datetime.now().isoformat(),
            
            # Size Factor (SMB)
            'marketCap': info.get('marketCap'),
            'enterpriseValue': info.get('enterpriseValue'),
            
            # Value Factors (HML)
            'trailingPE': info.get('trailingPE'),
            'forwardPE': info.get('forwardPE'),
            'priceToBook': info.get('priceToBook'),
            'priceToSales': info.get('priceToSalesTrailing12Months'),
            'enterpriseToEbitda': info.get('enterpriseToEbitda'),
            'enterpriseToRevenue': info.get('enterpriseToRevenue'),
            
            # Profitability Factors (RMW)
            'returnOnEquity': info.get('returnOnEquity'),
            'returnOnAssets': info.get('returnOnAssets'),
            'profitMargins': info.get('profitMargins'),
            'operatingMargins': info.get('operatingMargins'),
            'grossMargins': info.get('grossMargins'),
            
            # Growth Factors
            'earningsGrowth': info.get('earningsGrowth'),
            'revenueGrowth': info.get('revenueGrowth'),
            'earningsQuarterlyGrowth': info.get('earningsQuarterlyGrowth'),
            
            # Quality/Leverage Factors
            'debtToEquity': info.get('debtToEquity'),
            'currentRatio': info.get('currentRatio'),
            'quickRatio': info.get('quickRatio'),
            
            # Dividend Factors
            'dividendYield': info.get('dividendYield'),
            'payoutRatio': info.get('payoutRatio'),
            
            # Trading Metrics
            'beta': info.get('beta'),
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh'),
            'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow'),
            
            # Sector for sector-relative analysis
            'sector': info.get('sector'),
            'industry': info.get('industry'),
        }
        
        # Save to cache
        os.makedirs(FUNDAMENTALS_CACHE_DIR, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(fundamentals, f, indent=2)
        
        return fundamentals
        
    except Exception as e:
        print(f"⚠️ Error fetching fundamentals for {ticker}: {e}")
        return {'ticker': ticker, 'error': str(e)}


def get_fundamentals_for_universe(tickers: List[str], use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch fundamental data for entire universe.
    
    Args:
        tickers: List of ticker symbols
        use_cache: Whether to use cached data
        
    Returns:
        DataFrame with all fundamentals, indexed by ticker
    """
    print(f"📊 Fetching fundamentals for {len(tickers)} tickers...")
    
    all_fundamentals = []
    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            print(f"   Progress: {i+1}/{len(tickers)}")
        
        data = get_fundamentals_for_ticker(ticker, use_cache)
        all_fundamentals.append(data)
    
    df = pd.DataFrame(all_fundamentals)
    df = df.set_index('ticker')
    
    print(f"✅ Fetched fundamentals. Valid data for {df.dropna(subset=['marketCap']).shape[0]}/{len(tickers)} tickers")
    
    return df


def compute_factor_scores(fundamentals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-sectional z-scores for each factor.
    
    This is the key step: rank stocks relative to each other,
    not by absolute values.
    
    Args:
        fundamentals_df: DataFrame with raw fundamental data
        
    Returns:
        DataFrame with z-scored factor values
    """
    from scipy.stats import zscore
    
    factor_scores = pd.DataFrame(index=fundamentals_df.index)
    
    # Size Factor: Smaller is better (SMB = Small Minus Big)
    # Invert so higher score = smaller cap
    if 'marketCap' in fundamentals_df.columns:
        valid = fundamentals_df['marketCap'].dropna()
        if len(valid) > 1:
            # Log transform to handle extreme values
            log_cap = np.log(valid + 1)
            factor_scores.loc[valid.index, 'size_score'] = -zscore(log_cap)  # Invert
    
    # Value Factor: Lower P/E, P/B is better (HML = High book-to-market Minus Low)
    for col, name in [('trailingPE', 'pe_score'), ('priceToBook', 'pb_score')]:
        if col in fundamentals_df.columns:
            valid = fundamentals_df[col].dropna()
            valid = valid[(valid > 0) & (valid < 1000)]  # Filter outliers
            if len(valid) > 1:
                factor_scores.loc[valid.index, name] = -zscore(valid)  # Invert (lower is better)
    
    # Profitability Factor: Higher is better (RMW = Robust Minus Weak)
    for col, name in [('returnOnEquity', 'roe_score'), 
                       ('profitMargins', 'margin_score'),
                       ('returnOnAssets', 'roa_score')]:
        if col in fundamentals_df.columns:
            valid = fundamentals_df[col].dropna()
            valid = valid[(valid > -1) & (valid < 2)]  # Filter outliers
            if len(valid) > 1:
                factor_scores.loc[valid.index, name] = zscore(valid)
    
    # Growth Factor: Higher is better
    for col, name in [('earningsGrowth', 'earnings_growth_score'),
                       ('revenueGrowth', 'revenue_growth_score')]:
        if col in fundamentals_df.columns:
            valid = fundamentals_df[col].dropna()
            valid = valid[(valid > -1) & (valid < 5)]  # Filter outliers
            if len(valid) > 1:
                factor_scores.loc[valid.index, name] = zscore(valid)
    
    # Quality Factor: Lower leverage is better
    if 'debtToEquity' in fundamentals_df.columns:
        valid = fundamentals_df['debtToEquity'].dropna()
        valid = valid[(valid >= 0) & (valid < 500)]  # Filter outliers
        if len(valid) > 1:
            factor_scores.loc[valid.index, 'leverage_score'] = -zscore(valid)  # Invert
    
    # Composite Score: Equal-weighted combination of all factors
    score_cols = [c for c in factor_scores.columns if c.endswith('_score')]
    if score_cols:
        factor_scores['composite_score'] = factor_scores[score_cols].mean(axis=1)
    
    return factor_scores


def get_factor_features_for_ticker(ticker: str, factor_scores: pd.DataFrame) -> Dict:
    """
    Get factor features for a single ticker to add to ML feature set.
    
    Args:
        ticker: Stock ticker symbol
        factor_scores: DataFrame with computed factor scores
        
    Returns:
        Dict with factor features
    """
    if ticker not in factor_scores.index:
        # Return defaults for missing tickers
        return {
            'size_score': 0.0,
            'pe_score': 0.0,
            'pb_score': 0.0,
            'roe_score': 0.0,
            'margin_score': 0.0,
            'earnings_growth_score': 0.0,
            'composite_score': 0.0,
        }
    
    row = factor_scores.loc[ticker]
    return {
        'size_score': row.get('size_score', 0.0) if pd.notna(row.get('size_score')) else 0.0,
        'pe_score': row.get('pe_score', 0.0) if pd.notna(row.get('pe_score')) else 0.0,
        'pb_score': row.get('pb_score', 0.0) if pd.notna(row.get('pb_score')) else 0.0,
        'roe_score': row.get('roe_score', 0.0) if pd.notna(row.get('roe_score')) else 0.0,
        'margin_score': row.get('margin_score', 0.0) if pd.notna(row.get('margin_score')) else 0.0,
        'earnings_growth_score': row.get('earnings_growth_score', 0.0) if pd.notna(row.get('earnings_growth_score')) else 0.0,
        'composite_score': row.get('composite_score', 0.0) if pd.notna(row.get('composite_score')) else 0.0,
    }


# Fundamental factor columns for ML model
FUNDAMENTAL_COLUMNS = [
    'size_score',       # SMB factor
    'pe_score',         # Value: P/E
    'pb_score',         # Value: P/B
    'roe_score',        # Profitability: ROE
    'margin_score',     # Profitability: margins
    'earnings_growth_score',  # Growth
    'composite_score',  # Combined quality score
]


if __name__ == "__main__":
    # Test with a few tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    print("\n=== Testing Fundamental Data Fetcher ===\n")
    
    # Fetch fundamentals
    df = get_fundamentals_for_universe(test_tickers, use_cache=False)
    print("\nRaw fundamentals:")
    print(df[['marketCap', 'trailingPE', 'priceToBook', 'returnOnEquity', 'profitMargins']].to_string())
    
    # Compute factor scores
    scores = compute_factor_scores(df)
    print("\nFactor scores (z-scored):")
    print(scores.to_string())
