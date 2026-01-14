"""
Compute current portfolio values from ledger holdings and latest prices.
Outputs a JSON snapshot for the dashboard to read.

IMPORTANT: Portfolio values come from the ledger's PORTFOLIO,VALUE rows
(the authoritative source), NOT from recalculating holdings × prices.

Usage:
  python compute_portfolio_snapshot.py                     # All strategies (legacy, conflicts)
  python compute_portfolio_snapshot.py --strategy ml       # Only ML strategy
  python compute_portfolio_snapshot.py --strategy lstm     # Only LSTM strategy
  python compute_portfolio_snapshot.py --strategy momentum # Only Momentum strategy
"""

import pandas as pd
import sqlite3
import json
import os
import argparse
from datetime import datetime

# Paths
DB_PATH = 'data/market.db'
OUTPUT_PATH = 'data/portfolio_snapshot.json'
SNAPSHOTS_DIR = 'data/snapshots'
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


def append_portfolio_value_to_ledger(ledger_path: str, value: float, date_str: str, strategy: str) -> bool:
    """
    Append a PORTFOLIO,VALUE row to the ledger for the given date.
    
    This ensures the ledger has a daily record of portfolio value,
    which is used by the dashboard chart. Only appends if no entry
    exists for this date yet.
    
    Returns: True if appended, False if already exists or error
    """
    if not os.path.exists(ledger_path):
        return False
    
    try:
        df = pd.read_csv(ledger_path)
        
        # Check if we already have an entry for this date
        existing = df[(df['ticker'] == 'PORTFOLIO') & 
                      (df['action'] == 'VALUE') & 
                      (df['date'].astype(str).str[:10] == date_str)]
        
        if not existing.empty:
            print(f"  {strategy}: Already has entry for {date_str}")
            return False
        
        # Create new row matching ledger format
        new_row = pd.DataFrame([{
            'date': date_str,
            'ticker': 'PORTFOLIO',
            'action': 'VALUE',
            'price': 1.0,
            'shares': 0,
            'amount': 0.0,
            'cash_balance': 0.0,
            'total_value': round(value, 2),
            'strategy': strategy,
            'momentum_score' if 'momentum' in strategy else 'confidence': ''
        }])
        
        # Append to ledger
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(ledger_path, index=False)
        print(f"  {strategy}: Appended value ${value:,.2f} for {date_str}")
        return True
        
    except Exception as e:
        print(f"  {strategy}: Error appending value - {e}")
        return False


def process_single_strategy(strategy: str) -> dict:
    """
    Process a single strategy and write its snapshot to data/snapshots/{strategy}.json.
    This is the conflict-free approach - each workflow writes to its own file.
    """
    ledgers = {
        'momentum': 'ledger_momentum.csv',
        'ml': 'ledger_ml.csv',
        'lstm': 'ledger_lstm.csv'
    }
    
    if strategy not in ledgers:
        print(f"Unknown strategy: {strategy}")
        return {}
    
    ledger_path = ledgers[strategy]
    print(f"\n{'='*50}")
    print(f"COMPUTING {strategy.upper()} SNAPSHOT")
    print(f"{'='*50}")
    
    # Get holdings and compute value
    holdings, cash = get_current_holdings(ledger_path)
    print(f"\n{strategy.upper()}:")
    print(f"  Holdings: {len(holdings)} positions")
    print(f"  Cash: ${cash:,.2f}")
    
    # Fetch prices
    prices = get_latest_prices(list(holdings.keys()), DB_PATH)
    print(f"  Got prices for {len(prices)} tickers")
    
    # Compute value
    value = compute_portfolio_value(holdings, prices, cash)
    return_pct = ((value / INITIAL_CAPITAL) - 1) * 100
    
    print(f"\n{strategy.upper()} RESULT:")
    print(f"  Value: ${value:,.2f}")
    print(f"  Return: {return_pct:+.2f}%")
    
    # Append daily value to ledger
    price_date = get_latest_date(DB_PATH)
    print(f"\n📊 Updating ledger with daily value for {price_date}...")
    append_portfolio_value_to_ledger(ledger_path, value, price_date, strategy)
    
    # Create snapshot
    snapshot = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'price_date': price_date,
        'initial_capital': INITIAL_CAPITAL,
        'strategy': strategy,
        'value': round(value, 2),
        'return_pct': round(return_pct, 2),
        'cash': round(cash, 2),
        'holdings': {k: int(v) for k, v in holdings.items()},
        'positions': len(holdings)
    }
    
    # Save to strategy-specific file
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    snapshot_path = os.path.join(SNAPSHOTS_DIR, f'{strategy}.json')
    with open(snapshot_path, 'w') as f:
        json.dump(snapshot, f, indent=2)
    
    print(f"\n✅ Saved snapshot to {snapshot_path}")
    
    return snapshot


def consolidate_snapshots() -> dict:
    """
    Read all per-strategy snapshots and merge into portfolio_snapshot.json.
    Called after each workflow to update the combined snapshot.
    """
    print(f"\n{'='*50}")
    print("CONSOLIDATING SNAPSHOTS")
    print(f"{'='*50}")
    
    combined = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'price_date': get_latest_date(DB_PATH),
        'initial_capital': INITIAL_CAPITAL,
        'portfolios': {}
    }
    
    # Read each strategy snapshot
    strategies = ['momentum', 'ml', 'lstm']
    for strategy in strategies:
        snapshot_path = os.path.join(SNAPSHOTS_DIR, f'{strategy}.json')
        if os.path.exists(snapshot_path):
            with open(snapshot_path, 'r') as f:
                data = json.load(f)
            combined['portfolios'][strategy] = {
                'value': data.get('value', 0),
                'return_pct': data.get('return_pct', 0),
                'cash': data.get('cash', 0),
                'holdings': data.get('holdings', {}),
                'positions': data.get('positions', 0)
            }
            print(f"  ✓ Loaded {strategy}: ${data.get('value', 0):,.2f}")
        else:
            print(f"  ⚠ No snapshot for {strategy}")
    
    # Add SPY benchmark
    add_spy_benchmark(combined)
    
    # Save combined snapshot
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(combined, f, indent=2)
    
    print(f"\n✅ Saved combined snapshot to {OUTPUT_PATH}")
    return combined


def add_spy_benchmark(snapshot: dict):
    """Add SPY benchmark to snapshot."""
    print("\n📈 Computing SPY benchmark...")
    try:
        # Get first ledger date
        first_date = None
        ledgers = ['ledger_momentum.csv', 'ledger_ml.csv', 'ledger_lstm.csv']
        for path in ledgers:
            if os.path.exists(path):
                df = pd.read_csv(path)
                if not df.empty and 'date' in df.columns:
                    first_date = df['date'].min()
                    break
        
        spy_df = None
        
        # Fetch SPY from yfinance
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
                    print(f"  Fetched {len(spy_df)} days of SPY from yfinance")
            except Exception as e:
                print(f"  yfinance failed: {e}, trying database...")
        
        # Fallback to database
        if spy_df is None and first_date and os.path.exists(DB_PATH):
            try:
                con = sqlite3.connect(DB_PATH)
                spy_df = pd.read_sql_query("""
                    SELECT date, COALESCE(adj_close, close) AS price
                    FROM price_data
                    WHERE ticker = 'SPY' AND date >= ?
                    ORDER BY date ASC
                """, con, params=(first_date,))
                con.close()
                if not spy_df.empty:
                    print(f"  Using cached SPY data ({len(spy_df)} days)")
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
            
            # Save SPY history for chart
            spy_history = {
                'ticker': 'SPY',
                'start_date': first_date,
                'end_date': str(spy_df['date'].iloc[-1]),
                'initial_value': INITIAL_CAPITAL,
                'final_value': round(spy_value, 2),
                'total_return': round(spy_return, 2),
                'portfolio_history': [
                    [str(row['date'])[:10], round(INITIAL_CAPITAL * (row['price'] / spy_start), 2)]
                    for _, row in spy_df.iterrows()
                ]
            }
            with open('data/spy_benchmark.json', 'w') as f:
                json.dump(spy_history, f, indent=2)
    except Exception as e:
        print(f"  Error: {e}")


def update_strategy_value(strategy: str) -> bool:
    """
    Update a strategy's JSON snapshot with the latest value from its ledger.
    
    This is used to keep monthly strategy snapshots current without
    recalculating holdings. Only updates value and return_pct, preserving
    the holdings/cash from the last time the strategy actually ran.
    
    Usage: Called by cache_refresh workflow to keep all strategy values fresh.
    
    Returns: True if updated, False if no update needed or error
    """
    ledger_paths = {
        'momentum': 'ledger_momentum.csv',
        'ml': 'ledger_ml.csv',
        'lstm': 'ledger_lstm.csv'
    }
    
    if strategy not in ledger_paths:
        print(f"Unknown strategy: {strategy}")
        return False
    
    print(f"\n📊 Updating {strategy.upper()} value from ledger...")
    
    # Get current value from ledger (authoritative source)
    ledger_path = ledger_paths[strategy]
    ledger_value, value_date = get_ledger_portfolio_value(ledger_path)
    
    if ledger_value <= 0:
        print(f"  ⚠ No value found in ledger for {strategy}")
        return False
    
    # Read existing snapshot (to preserve holdings/cash)
    snapshot_path = os.path.join(SNAPSHOTS_DIR, f'{strategy}.json')
    if not os.path.exists(snapshot_path):
        print(f"  ⚠ No existing snapshot for {strategy}")
        return False
    
    with open(snapshot_path, 'r') as f:
        data = json.load(f)
    
    old_value = data.get('value', 0)
    old_date = data.get('price_date', 'unknown')
    
    # Check if update is needed
    if abs(ledger_value - old_value) < 0.01:
        print(f"  ✓ {strategy}: Value unchanged (${ledger_value:,.2f})")
        return False
    
    # Update value and return_pct only (preserve holdings/cash)
    return_pct = ((ledger_value / INITIAL_CAPITAL) - 1) * 100
    data['value'] = round(ledger_value, 2)
    data['return_pct'] = round(return_pct, 2)
    data['price_date'] = value_date
    data['timestamp'] = datetime.utcnow().isoformat() + 'Z'
    
    # Save updated snapshot
    with open(snapshot_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  ✓ {strategy}: ${old_value:,.2f} ({old_date}) → ${ledger_value:,.2f} ({value_date})")
    return True


def update_all_strategy_values() -> int:
    """
    Update all strategy snapshots with current values from ledgers.
    Used by cache_refresh workflow to keep monthly strategy values fresh.
    
    Returns: Number of strategies updated
    """
    print(f"\n{'='*50}")
    print("UPDATING ALL STRATEGY VALUES FROM LEDGERS")
    print(f"{'='*50}")
    
    updated = 0
    for strategy in ['momentum', 'ml', 'lstm']:
        if update_strategy_value(strategy):
            updated += 1
    
    # Re-consolidate with updated values
    consolidate_snapshots()
    
    print(f"\n✅ Updated {updated} strategy value(s)")
    return updated


def main():
    parser = argparse.ArgumentParser(description='Compute portfolio snapshot')
    parser.add_argument('--strategy', type=str, choices=['ml', 'lstm', 'momentum'],
                       help='Process only this strategy (writes to data/snapshots/{strategy}.json)')
    parser.add_argument('--consolidate', action='store_true',
                       help='Consolidate all strategy snapshots into portfolio_snapshot.json')
    parser.add_argument('--update-value-only', action='store_true',
                       help='Update all strategy snapshot values from ledgers (preserves holdings)')
    args = parser.parse_args()
    
    if args.update_value_only:
        # Update values from ledgers (used by cache_refresh for monthly strategies)
        update_all_strategy_values()
    elif args.strategy:
        # Single strategy mode (used by workflows)
        process_single_strategy(args.strategy)
        # Also consolidate after each strategy update
        consolidate_snapshots()
    elif args.consolidate:
        # Just consolidate existing snapshots
        consolidate_snapshots()
    else:
        # Legacy mode: process all strategies at once (can cause conflicts)
        print("WARNING: Running in legacy mode. Use --strategy for conflict-free operation.")
        for strategy in ['momentum', 'ml', 'lstm']:
            process_single_strategy(strategy)
        consolidate_snapshots()


if __name__ == "__main__":
    main()

