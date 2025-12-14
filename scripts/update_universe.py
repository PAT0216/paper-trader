"""
Update S&P 500 Universe

Fetches current S&P 500 constituents from Wikipedia and updates:
- data/sp500_tickers.txt (used by main.py for trading)
- data/sp500_current.csv (used for backtesting)

Run monthly before trading to ensure universe is current.
"""

import pandas as pd
import os
from datetime import datetime


def update_universe():
    print("=" * 50)
    print("S&P 500 UNIVERSE UPDATE")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Fetch current S&P 500 from Wikipedia
    print("\n📊 Fetching S&P 500 from Wikipedia...")
    try:
        # Need user-agent to avoid 403 Forbidden
        import urllib.request
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            html = response.read()
        tables = pd.read_html(html)
        df = tables[0]
        
        # Extract and clean tickers
        tickers = df['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]  # BRK.B -> BRK-B
        tickers = sorted(list(set(tickers)))
        
        print(f"   Found {len(tickers)} tickers")
        
    except Exception as e:
        print(f"❌ Error fetching from Wikipedia: {e}")
        return False
    
    # Load existing tickers
    txt_path = 'data/sp500_tickers.txt'
    current_tickers = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            current_tickers = sorted([line.strip() for line in f if line.strip()])
    
    # Compare
    current_set = set(current_tickers)
    new_set = set(tickers)
    
    added = new_set - current_set
    removed = current_set - new_set
    
    if added or removed:
        print(f"\n🔄 Changes detected:")
        if added:
            print(f"   ✅ ADDED ({len(added)}): {', '.join(sorted(added))}")
        if removed:
            print(f"   ❌ REMOVED ({len(removed)}): {', '.join(sorted(removed))}")
        
        # Update sp500_tickers.txt
        os.makedirs('data', exist_ok=True)
        with open(txt_path, 'w') as f:
            f.write('\n'.join(tickers))
        print(f"\n   Updated {txt_path}")
        
        # Update sp500_current.csv
        csv_path = 'data/sp500_current.csv'
        df_clean = df.copy()
        df_clean['Symbol'] = df_clean['Symbol'].str.replace('.', '-', regex=False)
        df_clean.to_csv(csv_path, index=False)
        print(f"   Updated {csv_path}")
        
    else:
        print(f"\n✅ No changes - universe is up to date ({len(tickers)} tickers)")
    
    return True


if __name__ == "__main__":
    success = update_universe()
    exit(0 if success else 1)
