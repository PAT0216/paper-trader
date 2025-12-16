"""
Fetch Historical Fundamental Data via yfinance

Fetches quarterly financials (income statement, balance sheet) from yfinance.
Gets ~4-5 years of history which is enough for backtesting.

Output: data/fundamentals_historical.csv
"""

import pandas as pd
import yfinance as yf
import os
import time
import json
from datetime import datetime
from typing import List
import traceback


def load_tickers() -> List[str]:
    """Load S&P 500 tickers from file."""
    txt_path = 'data/sp500_tickers.txt'
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
        print(f"📋 Loaded {len(tickers)} tickers from {txt_path}")
        return tickers
    
    # Fallback: fetch from Wikipedia
    print("📋 Fetching tickers from Wikipedia...")
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        tickers = tables[0]['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        print(f"   Got {len(tickers)} tickers")
        return tickers
    except Exception as e:
        print(f"❌ Failed to get tickers: {e}")
        return []


def fetch_fundamentals_for_ticker(ticker: str, max_retries: int = 2) -> pd.DataFrame:
    """
    Fetch quarterly fundamentals from yfinance.
    
    Returns DataFrame with quarterly data (income + balance sheet).
    """
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            
            # Get quarterly income statement
            income = stock.quarterly_income_stmt
            if income is None or income.empty:
                return pd.DataFrame()
            
            # Get quarterly balance sheet
            balance = stock.quarterly_balance_sheet
            
            # Transpose so dates are rows
            income_t = income.T
            income_t.index.name = 'date'
            income_t = income_t.reset_index()
            income_t['ticker'] = ticker
            
            if balance is not None and not balance.empty:
                balance_t = balance.T
                balance_t.index.name = 'date'
                balance_t = balance_t.reset_index()
                balance_t['ticker'] = ticker
                
                # Merge on date
                common_cols = set(income_t.columns) & set(balance_t.columns) - {'date', 'ticker'}
                balance_t = balance_t.drop(columns=list(common_cols), errors='ignore')
                df = pd.merge(income_t, balance_t, on=['date', 'ticker'], how='outer')
            else:
                df = income_t
            
            # Calculate key ratios
            if not df.empty:
                # Profit Margin
                if 'Total Revenue' in df.columns and 'Net Income' in df.columns:
                    df['profitMargin'] = df['Net Income'] / df['Total Revenue'].replace(0, pd.NA)
                
                # Gross Margin
                if 'Total Revenue' in df.columns and 'Gross Profit' in df.columns:
                    df['grossMargin'] = df['Gross Profit'] / df['Total Revenue'].replace(0, pd.NA)
                
                # ROE
                if 'Net Income' in df.columns and 'Stockholders Equity' in df.columns:
                    df['roe'] = df['Net Income'] / df['Stockholders Equity'].replace(0, pd.NA)
                
                # Debt to Equity
                if 'Total Debt' in df.columns and 'Stockholders Equity' in df.columns:
                    df['debtToEquity'] = df['Total Debt'] / df['Stockholders Equity'].replace(0, pd.NA)
                
                # Current Ratio
                if 'Current Assets' in df.columns and 'Current Liabilities' in df.columns:
                    df['currentRatio'] = df['Current Assets'] / df['Current Liabilities'].replace(0, pd.NA)
                
                # EPS (if we have shares outstanding)
                if 'Net Income' in df.columns and 'Ordinary Shares Number' in df.columns:
                    df['eps'] = df['Net Income'] / df['Ordinary Shares Number'].replace(0, pd.NA)
            
            # Convert date column
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            return df
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return pd.DataFrame()
    
    return pd.DataFrame()


def fetch_all_fundamentals(tickers: List[str], batch_size: int = 20) -> pd.DataFrame:
    """
    Fetch fundamentals for all tickers with progress logging.
    """
    all_data = []
    total = len(tickers)
    success = 0
    failed = 0
    
    print(f"\n🚀 Fetching fundamentals for {total} tickers via yfinance...")
    print("=" * 60)
    
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_tickers = tickers[batch_start:batch_end]
        
        print(f"\n📦 Batch {batch_start//batch_size + 1}: "
              f"Tickers {batch_start+1}-{batch_end} of {total}")
        
        for ticker in batch_tickers:
            try:
                df = fetch_fundamentals_for_ticker(ticker)
                
                if not df.empty:
                    all_data.append(df)
                    quarters = len(df)
                    print(f"   ✅ {ticker}: {quarters} quarters")
                    success += 1
                else:
                    print(f"   ⚠️ {ticker}: No data")
                    failed += 1
                    
            except Exception as e:
                print(f"   ❌ {ticker}: {str(e)[:50]}")
                failed += 1
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        # Pause between batches
        if batch_end < total:
            print(f"   ⏳ Pausing 3s...")
            time.sleep(3)
    
    print(f"\n📊 Results: {success} success, {failed} failed")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def main():
    print("=" * 60)
    print("HISTORICAL FUNDAMENTAL DATA FETCH (yfinance)")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Config from environment
    max_tickers = int(os.environ.get('MAX_TICKERS', 500))
    start_idx = int(os.environ.get('START_INDEX', 0))
    
    # Load tickers
    tickers = load_tickers()
    tickers = tickers[start_idx:start_idx + max_tickers]
    
    if not tickers:
        print("❌ No tickers to process")
        return
    
    # Fetch fundamentals
    df = fetch_all_fundamentals(tickers)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if df.empty:
        print("❌ No data fetched")
        return
    
    unique_tickers = df['ticker'].nunique()
    total_rows = len(df)
    
    print(f"✅ Tickers with data: {unique_tickers}")
    print(f"✅ Total rows: {total_rows}")
    
    if 'date' in df.columns:
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Save
    os.makedirs('data', exist_ok=True)
    
    output_path = 'data/fundamentals_historical.csv'
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to {output_path}")
    
    # Save metadata
    metadata = {
        'fetch_date': datetime.now().isoformat(),
        'tickers_processed': unique_tickers,
        'total_rows': total_rows,
        'date_range': {
            'min': str(df['date'].min()) if 'date' in df.columns else None,
            'max': str(df['date'].max()) if 'date' in df.columns else None
        },
        'columns': list(df.columns)[:20]  # First 20 columns
    }
    with open('data/fundamentals_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Metadata saved")
    print(f"\n🏁 Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
