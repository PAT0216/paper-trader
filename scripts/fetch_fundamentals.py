"""
Fetch Fundamental Data for S&P 500 Stocks

Fetches quarterly fundamentals from yfinance for all S&P 500 stocks.
Designed to run on GitHub Actions to avoid local rate limiting.

Output: data/fundamentals.csv
"""

import pandas as pd
import yfinance as yf
import os
import time
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback


def load_tickers():
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
        tickers = [t.replace('.', '-') for t in tickers]  # BRK.B -> BRK-B
        print(f"   Got {len(tickers)} tickers")
        return tickers
    except Exception as e:
        print(f"❌ Failed to get tickers: {e}")
        return []


def fetch_fundamentals_for_ticker(ticker: str, max_retries: int = 3) -> dict:
    """
    Fetch fundamental data for a single ticker.
    
    Returns dict with fundamental metrics or None if failed.
    """
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info or 'symbol' not in info:
                return {'ticker': ticker, 'error': 'No data available'}
            
            # Extract key fundamentals
            fundamentals = {
                'ticker': ticker,
                'fetch_date': datetime.now().strftime('%Y-%m-%d'),
                
                # Valuation
                'trailingPE': info.get('trailingPE'),
                'forwardPE': info.get('forwardPE'),
                'priceToBook': info.get('priceToBook'),
                'priceToSales': info.get('priceToSalesTrailing12Months'),
                'enterpriseToEbitda': info.get('enterpriseToEbitda'),
                'enterpriseToRevenue': info.get('enterpriseToRevenue'),
                
                # Profitability
                'profitMargins': info.get('profitMargins'),
                'operatingMargins': info.get('operatingMargins'),
                'returnOnAssets': info.get('returnOnAssets'),
                'returnOnEquity': info.get('returnOnEquity'),
                'grossMargins': info.get('grossMargins'),
                
                # Growth
                'earningsGrowth': info.get('earningsGrowth'),
                'revenueGrowth': info.get('revenueGrowth'),
                'earningsQuarterlyGrowth': info.get('earningsQuarterlyGrowth'),
                
                # Financial Health
                'debtToEquity': info.get('debtToEquity'),
                'currentRatio': info.get('currentRatio'),
                'quickRatio': info.get('quickRatio'),
                
                # Dividends
                'dividendYield': info.get('dividendYield'),
                'payoutRatio': info.get('payoutRatio'),
                
                # Size
                'marketCap': info.get('marketCap'),
                'enterpriseValue': info.get('enterpriseValue'),
                
                # Trading
                'beta': info.get('beta'),
                'averageVolume': info.get('averageVolume'),
                '52WeekChange': info.get('52WeekChange'),
                
                # Per Share
                'trailingEps': info.get('trailingEps'),
                'forwardEps': info.get('forwardEps'),
                'bookValue': info.get('bookValue'),
                
                # Sector/Industry
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                
                # Status
                'error': None
            }
            
            return fundamentals
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))  # Exponential backoff
            else:
                return {
                    'ticker': ticker,
                    'error': str(e)[:100],
                    'fetch_date': datetime.now().strftime('%Y-%m-%d')
                }
    
    return {'ticker': ticker, 'error': 'Max retries exceeded'}


def fetch_all_fundamentals(tickers: list, max_workers: int = 5, 
                            batch_size: int = 50) -> pd.DataFrame:
    """
    Fetch fundamentals for all tickers with rate limiting.
    
    Uses threading with careful rate limiting to avoid blocks.
    """
    all_data = []
    total = len(tickers)
    
    print(f"\n🚀 Fetching fundamentals for {total} tickers...")
    print(f"   Workers: {max_workers}, Batch size: {batch_size}")
    print("=" * 60)
    
    # Process in batches to add pauses
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_tickers = tickers[batch_start:batch_end]
        
        print(f"\n📦 Batch {batch_start//batch_size + 1}: "
              f"Tickers {batch_start+1}-{batch_end} of {total}")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fetch_fundamentals_for_ticker, ticker): ticker 
                for ticker in batch_tickers
            }
            
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    all_data.append(result)
                    
                    if result.get('error'):
                        print(f"   ⚠️ {ticker}: {result['error'][:50]}")
                    else:
                        pe = result.get('trailingPE', 'N/A')
                        print(f"   ✅ {ticker}: P/E={pe}")
                        
                except Exception as e:
                    print(f"   ❌ {ticker}: Unexpected error: {e}")
                    all_data.append({
                        'ticker': ticker,
                        'error': str(e)[:100],
                        'fetch_date': datetime.now().strftime('%Y-%m-%d')
                    })
        
        # Pause between batches to avoid rate limiting
        if batch_end < total:
            print(f"   ⏳ Pausing 5s between batches...")
            time.sleep(5)
    
    return pd.DataFrame(all_data)


def main():
    print("=" * 60)
    print("FUNDAMENTAL DATA FETCH")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Load tickers
    tickers = load_tickers()
    if not tickers:
        print("❌ No tickers to process")
        return
    
    # Fetch fundamentals
    df = fetch_all_fundamentals(
        tickers, 
        max_workers=5,   # Conservative to avoid rate limits
        batch_size=50    # Process in batches
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    success_count = df[df['error'].isna()].shape[0]
    error_count = df[df['error'].notna()].shape[0]
    
    print(f"✅ Successful: {success_count}")
    print(f"⚠️ Errors: {error_count}")
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    output_path = 'data/fundamentals.csv'
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to {output_path}")
    
    # Also save a smaller "clean" version
    clean_df = df[df['error'].isna()].drop(columns=['error'])
    clean_path = 'data/fundamentals_clean.csv'
    clean_df.to_csv(clean_path, index=False)
    print(f"💾 Clean data saved to {clean_path}")
    
    # Save metadata
    metadata = {
        'fetch_date': datetime.now().isoformat(),
        'total_tickers': len(tickers),
        'successful': success_count,
        'failed': error_count,
        'columns': list(df.columns)
    }
    with open('data/fundamentals_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n🏁 Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
