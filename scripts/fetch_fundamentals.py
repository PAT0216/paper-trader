"""
Fetch Historical Fundamental Data from Financial Modeling Prep (FMP)

Fetches quarterly fundamentals going back to 2010 for S&P 500 stocks.
Designed to run on GitHub Actions to maximize API calls.

FMP Free Tier: 250 calls/day
Each ticker needs ~2-3 calls, so process ~80-100 tickers per run.

Output: data/fundamentals_historical.csv
"""

import pandas as pd
import requests
import os
import time
import json
from datetime import datetime
from typing import Optional, List


# FMP API base URL
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


def get_api_key() -> str:
    """Get API key from environment or raise error."""
    key = os.environ.get('FMP_API_KEY')
    if not key:
        raise ValueError(
            "FMP_API_KEY not found in environment. "
            "Set it with: export FMP_API_KEY=your_key"
        )
    return key


def load_tickers() -> List[str]:
    """Load S&P 500 tickers from file."""
    txt_path = 'data/sp500_tickers.txt'
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
        print(f"📋 Loaded {len(tickers)} tickers from {txt_path}")
        return tickers
    
    # Fallback
    print("⚠️ No ticker file found, using sample")
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']


def fetch_key_metrics(ticker: str, api_key: str, limit: int = 60) -> pd.DataFrame:
    """
    Fetch quarterly key metrics (P/E, P/B, ROE, etc.) from FMP.
    
    Returns DataFrame with quarterly data going back ~15 years.
    """
    url = f"{FMP_BASE_URL}/key-metrics/{ticker}"
    params = {
        'period': 'quarter',
        'limit': limit,  # 60 quarters = 15 years
        'apikey': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['ticker'] = ticker
        return df
        
    except Exception as e:
        print(f"   ⚠️ key-metrics failed: {e}")
        return pd.DataFrame()


def fetch_ratios(ticker: str, api_key: str, limit: int = 60) -> pd.DataFrame:
    """
    Fetch quarterly financial ratios from FMP.
    """
    url = f"{FMP_BASE_URL}/ratios/{ticker}"
    params = {
        'period': 'quarter',
        'limit': limit,
        'apikey': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['ticker'] = ticker
        return df
        
    except Exception as e:
        print(f"   ⚠️ ratios failed: {e}")
        return pd.DataFrame()


def fetch_fundamentals_for_ticker(ticker: str, api_key: str) -> pd.DataFrame:
    """
    Fetch and combine all fundamental data for a ticker.
    
    Returns DataFrame with date, ticker, and all metrics.
    """
    # Fetch both endpoints
    metrics_df = fetch_key_metrics(ticker, api_key)
    ratios_df = fetch_ratios(ticker, api_key)
    
    if metrics_df.empty and ratios_df.empty:
        return pd.DataFrame()
    
    # Merge on date if both exist
    if not metrics_df.empty and not ratios_df.empty:
        # Remove duplicate columns before merge
        common_cols = set(metrics_df.columns) & set(ratios_df.columns) - {'date', 'ticker'}
        ratios_df = ratios_df.drop(columns=list(common_cols), errors='ignore')
        
        df = pd.merge(metrics_df, ratios_df, on=['date', 'ticker'], how='outer')
    elif not metrics_df.empty:
        df = metrics_df
    else:
        df = ratios_df
    
    return df


def fetch_all_fundamentals(
    tickers: List[str], 
    api_key: str, 
    max_tickers: Optional[int] = None,
    start_idx: int = 0
) -> pd.DataFrame:
    """
    Fetch fundamentals for multiple tickers with rate limiting.
    
    Args:
        tickers: List of ticker symbols
        api_key: FMP API key
        max_tickers: Max tickers to process (for partial runs)
        start_idx: Starting index (for resume)
    """
    all_data = []
    
    # Slice tickers for partial run
    tickers_to_process = tickers[start_idx:]
    if max_tickers:
        tickers_to_process = tickers_to_process[:max_tickers]
    
    total = len(tickers_to_process)
    print(f"\n🚀 Fetching fundamentals for {total} tickers")
    print(f"   Starting at index {start_idx}")
    print("=" * 60)
    
    for i, ticker in enumerate(tickers_to_process):
        print(f"\n[{i+1}/{total}] {ticker}...")
        
        try:
            df = fetch_fundamentals_for_ticker(ticker, api_key)
            
            if not df.empty:
                all_data.append(df)
                quarters = len(df)
                earliest = df['date'].min() if 'date' in df.columns else 'N/A'
                print(f"   ✅ Got {quarters} quarters (back to {earliest})")
            else:
                print(f"   ⚠️ No data returned")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Rate limit: ~1 request per second to stay under limits
        time.sleep(1)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def main():
    print("=" * 60)
    print("HISTORICAL FUNDAMENTAL DATA FETCH (FMP)")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Get config from environment
    api_key = get_api_key()
    max_tickers = int(os.environ.get('MAX_TICKERS', 100))  # Default 100 per run
    start_idx = int(os.environ.get('START_INDEX', 0))
    
    # Load tickers
    tickers = load_tickers()
    
    # Fetch fundamentals
    df = fetch_all_fundamentals(
        tickers,
        api_key,
        max_tickers=max_tickers,
        start_idx=start_idx
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if df.empty:
        print("❌ No data fetched")
        return
    
    unique_tickers = df['ticker'].nunique() if 'ticker' in df.columns else 0
    total_rows = len(df)
    
    print(f"✅ Tickers processed: {unique_tickers}")
    print(f"✅ Total rows: {total_rows}")
    
    if 'date' in df.columns:
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Print columns
    print(f"\n📊 Columns ({len(df.columns)}):")
    for col in sorted(df.columns):
        print(f"   - {col}")
    
    # Save
    os.makedirs('data', exist_ok=True)
    
    output_path = f"data/fundamentals_historical_{start_idx}_{start_idx + max_tickers}.csv"
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to {output_path}")
    
    # Also save a combined file if it exists
    combined_path = 'data/fundamentals_historical.csv'
    if os.path.exists(combined_path):
        existing = pd.read_csv(combined_path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['ticker', 'date'])
        combined.to_csv(combined_path, index=False)
        print(f"💾 Updated {combined_path} (total: {len(combined)} rows)")
    else:
        df.to_csv(combined_path, index=False)
        print(f"💾 Created {combined_path}")
    
    # Save metadata
    metadata = {
        'fetch_date': datetime.now().isoformat(),
        'tickers_processed': unique_tickers,
        'total_rows': total_rows,
        'start_index': start_idx,
        'max_tickers': max_tickers,
        'date_range': {
            'min': str(df['date'].min()) if 'date' in df.columns else None,
            'max': str(df['date'].max()) if 'date' in df.columns else None
        }
    }
    with open('data/fundamentals_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n🏁 Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
