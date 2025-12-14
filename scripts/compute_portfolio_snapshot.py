"""
Compute current portfolio values from ledger holdings and latest prices.
Outputs a JSON snapshot for the dashboard to read.
"""

import pandas as pd
import sqlite3
import json
import os
from datetime import datetime

# Paths
DB_PATH = 'data/market.db'
OUTPUT_PATH = 'data/portfolio_snapshot.json'
INITIAL_CAPITAL = 10000


def get_current_holdings(ledger_path: str) -> tuple[dict, float]:
    """
    Parse ledger to get current holdings and cash balance.
    Returns: (holdings_dict, cash_balance)
    """
    if not os.path.exists(ledger_path):
        return {}, 0
    
    df = pd.read_csv(ledger_path)
    if df.empty:
        return {}, 0
    
    holdings = {}
    cash = 0
    
    # Sort by date and process in order
    df = df.sort_values('date')
    
    # Group by date and sort actions within each day
    action_order = {'DEPOSIT': 0, 'SELL': 1, 'BUY': 2}
    
    for date_str, day_group in df.groupby(df['date'].astype(str).str[:10], sort=False):
        day_sorted = day_group.copy()
        day_sorted['_order'] = day_sorted['action'].map(action_order).fillna(3)
        day_sorted = day_sorted.sort_values('_order')
        
        for _, row in day_sorted.iterrows():
            action = row.get('action', '')
            ticker = row.get('ticker', '')
            
            if action == 'DEPOSIT':
                cash = row.get('cash_balance', 0)
            elif action == 'BUY':
                shares = row.get('shares', 0)
                holdings[ticker] = holdings.get(ticker, 0) + shares
                cash = row.get('cash_balance', 0)
            elif action == 'SELL':
                if ticker in holdings:
                    del holdings[ticker]
                cash = row.get('cash_balance', 0)
    
    return holdings, cash


def get_latest_prices(tickers: list, db_path: str) -> dict:
    """Fetch latest prices for given tickers from the database."""
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
    """Get the most recent date in the database."""
    if not os.path.exists(db_path):
        return datetime.now().strftime('%Y-%m-%d')
    
    con = sqlite3.connect(db_path)
    try:
        result = pd.read_sql_query("SELECT MAX(date) as max_date FROM price_data", con)
        return result['max_date'].iloc[0] if not result.empty else datetime.now().strftime('%Y-%m-%d')
    finally:
        con.close()


def compute_portfolio_value(holdings: dict, prices: dict, cash: float) -> float:
    """Calculate total portfolio value."""
    total = cash
    for ticker, shares in holdings.items():
        if ticker in prices:
            total += shares * prices[ticker]
    return total


def main():
    print("=" * 50)
    print("COMPUTING PORTFOLIO SNAPSHOT")
    print("=" * 50)
    
    snapshot = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'price_date': get_latest_date(DB_PATH),
        'initial_capital': INITIAL_CAPITAL,
        'portfolios': {}
    }
    
    # Process each ledger
    ledgers = {
        'momentum': 'ledger_momentum.csv',
        'ml': 'ledger_ml.csv'
    }
    
    all_tickers = set()
    portfolio_data = {}
    
    # First pass: collect all tickers and holdings
    for name, path in ledgers.items():
        holdings, cash = get_current_holdings(path)
        portfolio_data[name] = {'holdings': holdings, 'cash': cash}
        all_tickers.update(holdings.keys())
        print(f"\n{name.upper()}:")
        print(f"  Holdings: {len(holdings)} positions")
        print(f"  Cash: ${cash:,.2f}")
    
    # Fetch prices for all tickers at once
    print(f"\nFetching prices for {len(all_tickers)} tickers...")
    prices = get_latest_prices(list(all_tickers), DB_PATH)
    print(f"  Got prices for {len(prices)} tickers")
    
    # Second pass: compute values
    for name, data in portfolio_data.items():
        holdings = data['holdings']
        cash = data['cash']
        value = compute_portfolio_value(holdings, prices, cash)
        return_pct = ((value / INITIAL_CAPITAL) - 1) * 100
        
        snapshot['portfolios'][name] = {
            'value': round(value, 2),
            'return_pct': round(return_pct, 2),
            'cash': round(cash, 2),
            'holdings': {k: int(v) for k, v in holdings.items()},
            'positions': len(holdings)
        }
        
        print(f"\n{name.upper()} RESULT:")
        print(f"  Value: ${value:,.2f}")
        print(f"  Return: {return_pct:+.2f}%")
    
    # Add SPY benchmark
    print("\n📈 Computing SPY benchmark...")
    try:
        # Get first ledger date
        first_date = None
        for path in ledgers.values():
            if os.path.exists(path):
                df = pd.read_csv(path)
                if not df.empty and 'date' in df.columns:
                    first_date = df['date'].min()
                    break
        
        if first_date and os.path.exists(DB_PATH):
            con = sqlite3.connect(DB_PATH)
            spy_df = pd.read_sql_query("""
                SELECT date, COALESCE(adj_close, close) AS price
                FROM price_data
                WHERE ticker = 'SPY'
                  AND date >= ?
                ORDER BY date ASC
            """, con, params=(first_date,))
            con.close()
            
            if not spy_df.empty:
                spy_start = spy_df['price'].iloc[0]
                spy_end = spy_df['price'].iloc[-1]
                spy_return = ((spy_end / spy_start) - 1) * 100
                spy_value = INITIAL_CAPITAL * (1 + spy_return / 100)
                
                snapshot['benchmark'] = {
                    'ticker': 'SPY',
                    'start_price': round(spy_start, 2),
                    'end_price': round(spy_end, 2),
                    'return_pct': round(spy_return, 2),
                    'value': round(spy_value, 2),
                    'start_date': first_date,
                    'end_date': spy_df['date'].iloc[-1]
                }
                print(f"  SPY: {spy_return:+.2f}% (${spy_value:,.2f})")
            else:
                print("  No SPY data found")
        else:
            print("  Skipped - no date range")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Save snapshot
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(snapshot, f, indent=2)
    
    print(f"\n✅ Saved snapshot to {OUTPUT_PATH}")
    print(f"   Price date: {snapshot['price_date']}")
    
    return snapshot


if __name__ == "__main__":
    main()
