"""
Compute current portfolio values from ledger holdings and latest prices.
Outputs a JSON snapshot for the dashboard to read.

IMPORTANT: Portfolio values come from the ledger's PORTFOLIO,VALUE rows
(the authoritative source), NOT from recalculating holdings × prices.
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


def get_ledger_portfolio_value(ledger_path: str) -> tuple[float, str]:
    """
    Get the latest PORTFOLIO,VALUE entry from the ledger.
    This is the authoritative source of truth for portfolio value.
    Returns: (portfolio_value, date_str)
    """
    if not os.path.exists(ledger_path):
        return 0, ""
    
    df = pd.read_csv(ledger_path)
    if df.empty:
        return 0, ""
    
    # Filter to PORTFOLIO,VALUE rows only
    value_rows = df[(df['ticker'] == 'PORTFOLIO') & (df['action'] == 'VALUE')]
    
    if value_rows.empty:
        return 0, ""
    
    # Get the latest entry
    value_rows = value_rows.sort_values('date')
    latest = value_rows.iloc[-1]
    
    # The value is stored in 'total_value' or 'cash_balance' column
    value = latest.get('total_value', latest.get('cash_balance', 0))
    date_str = str(latest['date'])[:10]
    
    return float(value), date_str


def get_current_holdings(ledger_path: str) -> tuple[dict, float]:
    """
    Parse ledger to get current holdings and cash balance.
    
    IMPORTANT: Cash is calculated from transaction amounts, NOT from the
    cash_balance column (which may be incorrect in some ledgers).
    
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
    
    for _, row in df.iterrows():
        action = row.get('action', '')
        ticker = row.get('ticker', '')
        amount = row.get('amount', 0)
        shares = row.get('shares', 0)
        
        if action == 'DEPOSIT':
            # Deposit adds to cash
            cash += amount if amount > 0 else row.get('cash_balance', 0)
        elif action == 'BUY':
            # BUY: subtract amount from cash, add shares to holdings
            cash -= amount
            holdings[ticker] = holdings.get(ticker, 0) + shares
        elif action == 'SELL':
            # SELL: add amount to cash, subtract shares from holdings
            cash += amount
            if ticker in holdings:
                holdings[ticker] -= shares
                if holdings[ticker] <= 0:
                    del holdings[ticker]
        # Skip PORTFOLIO,VALUE rows (they don't affect holdings/cash)
    
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
        'ml': 'ledger_ml.csv',
        'lstm': 'ledger_lstm.csv'
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
        
        spy_df = None
        
        # ALWAYS fetch SPY fresh from yfinance for accurate benchmark
        # This ensures SPY chart stays current regardless of cache state
        if first_date:
            try:
                import yfinance as yf
                ticker = yf.Ticker('SPY')
                hist = ticker.history(start=first_date, end=None)
                if not hist.empty:
                    hist = hist.reset_index()
                    spy_df = pd.DataFrame({
                        'date': hist['Date'].astype(str).str[:10],
                        'price': hist['Close']
                    })
                    print(f"  Fetched {len(spy_df)} days of SPY from yfinance (fresh)")
            except Exception as e:
                print(f"  yfinance fetch failed: {e}, trying database cache...")
        
        # Fallback to database cache if yfinance fails
        if spy_df is None and first_date and os.path.exists(DB_PATH):
            try:
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
                    print(f"  Using cached SPY data ({len(spy_df)} days)")
                else:
                    spy_df = None
            except Exception as db_err:
                print(f"  Database fallback failed: {db_err}")
        
        if spy_df is not None and not spy_df.empty:
            spy_start = spy_df['price'].iloc[0]
            spy_end = spy_df['price'].iloc[-1]
            spy_return = ((spy_end / spy_start) - 1) * 100
            spy_value = INITIAL_CAPITAL * (1 + spy_return / 100)
            
            snapshot['benchmark'] = {
                'ticker': 'SPY',
                'start_price': round(float(spy_start), 2),
                'end_price': round(float(spy_end), 2),
                'return_pct': round(spy_return, 2),
                'value': round(spy_value, 2),
                'start_date': first_date,
                'end_date': str(spy_df['date'].iloc[-1])
            }
            print(f"  SPY: {spy_return:+.2f}% (${spy_value:,.2f})")
        else:
            print("  No SPY data found")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Save snapshot
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(snapshot, f, indent=2)
    
    # Also save SPY benchmark history for the chart
    if spy_df is not None and not spy_df.empty:
        spy_history = {
            'ticker': 'SPY',
            'start_date': first_date,
            'end_date': str(spy_df['date'].iloc[-1]),
            'initial_value': INITIAL_CAPITAL,
            'final_value': round(INITIAL_CAPITAL * (spy_df['price'].iloc[-1] / spy_df['price'].iloc[0]), 2),
            'total_return': round(((spy_df['price'].iloc[-1] / spy_df['price'].iloc[0]) - 1) * 100, 2),
            'portfolio_history': [
                [str(row['date'])[:10], round(INITIAL_CAPITAL * (row['price'] / spy_df['price'].iloc[0]), 2)]
                for _, row in spy_df.iterrows()
            ]
        }
        with open('data/spy_benchmark.json', 'w') as f:
            json.dump(spy_history, f, indent=2)
        print(f"   Saved SPY history ({len(spy_history['portfolio_history'])} days) to data/spy_benchmark.json")
    
    print(f"\n✅ Saved snapshot to {OUTPUT_PATH}")
    print(f"   Price date: {snapshot['price_date']}")
    
    return snapshot


if __name__ == "__main__":
    main()
