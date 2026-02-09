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

import json
import os
import argparse
from datetime import datetime
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.trading.ledger_utils import (
    get_ledger_portfolio_value,
    get_current_holdings_from_ledger,
    append_portfolio_value_to_ledger
)
from src.data.price_utils import (
    get_market_date,
    get_latest_prices,
    get_latest_date,
    compute_portfolio_value
)


# Paths
DB_PATH = 'data/market.db'
OUTPUT_PATH = 'data/portfolio_snapshot.json'
SNAPSHOTS_DIR = 'data/snapshots'
INITIAL_CAPITAL = 10000
TRADING_START_DATE = "2025-10-01"  # Known first trading day - fallback if data corrupted


def is_github_actions() -> bool:
    """Check if running in GitHub Actions environment."""
    return os.environ.get('GITHUB_ACTIONS', '').lower() == 'true'


def get_db_freshness_days() -> int:
    """Get how many days old the database is."""
    import sqlite3
    import pandas as pd
    
    if not os.path.exists(DB_PATH):
        return 999
    
    con = sqlite3.connect(DB_PATH)
    try:
        result = pd.read_sql_query("SELECT MAX(date) as max_date FROM price_data", con)
        if result.empty or pd.isna(result['max_date'].iloc[0]):
            return 999
        import pandas as pd
        db_date = pd.to_datetime(result['max_date'].iloc[0])
        today = pd.to_datetime(get_market_date())
        return (today - db_date).days
    except:
        return 999
    finally:
        con.close()


def check_data_safety(force: bool = False) -> bool:
    """
    Check if it's safe to modify data files.
    
    Returns True if safe to proceed, False otherwise.
    Warns if running locally with stale database.
    """
    if is_github_actions():
        return True  # Always safe in CI
    
    stale_days = get_db_freshness_days()
    
    if stale_days > 1:
        print(f"\n{'='*60}")
        print("WARNING: LOCAL EXECUTION DETECTED")
        print(f"{'='*60}")
        print(f"Database is {stale_days} days old (last: {get_latest_date(DB_PATH)})")
        print("Running this script locally with stale data can corrupt")
        print("the remote repository with incorrect portfolio values.")
        print()
        
        if force:
            print("--force flag provided. Proceeding anyway...")
            return True
        else:
            print("Use --force to override this safety check.")
            print("Or run this via GitHub Actions for accurate data.")
            print(f"{'='*60}\n")
            return False
    
    return True


def process_single_strategy(strategy: str) -> dict:
    """
    Process a single strategy and write its snapshot to data/snapshots/{strategy}.json.
    This is the conflict-free approach - each workflow writes to its own file.
    """
    ledgers = {
        'momentum': 'data/ledgers/ledger_momentum.csv',
        'ml': 'data/ledgers/ledger_ml.csv',
        'lstm': 'data/ledgers/ledger_lstm.csv'
    }
    
    if strategy not in ledgers:
        print(f"Unknown strategy: {strategy}")
        return {}
    
    ledger_path = ledgers[strategy]
    print(f"\n{'='*50}")
    print(f"COMPUTING {strategy.upper()} SNAPSHOT")
    print(f"{'='*50}")
    
    # Get holdings and compute value
    holdings, cash = get_current_holdings_from_ledger(ledger_path)
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
    
    # Use database max date since that's when the prices are from
    price_date = get_latest_date(DB_PATH)
    print(f"\n[Ledger] Updating with daily value for {price_date}...")
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
    
    print(f"\nSaved snapshot to {snapshot_path}")
    
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
            print(f"  Loaded {strategy}: ${data.get('value', 0):,.2f}")
        else:
            print(f"  No snapshot for {strategy}")
    
    # Add SPY benchmark
    add_spy_benchmark(combined)
    
    # Save combined snapshot
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(combined, f, indent=2)
    
    print(f"\nSaved combined snapshot to {OUTPUT_PATH}")
    return combined


def add_spy_benchmark(snapshot: dict):
    """Add SPY benchmark to snapshot."""
    import pandas as pd
    import sqlite3
    
    print("\nComputing SPY benchmark...")
    try:
        # PRIORITY 1: Preserve existing start_date from spy_benchmark.json
        # This prevents data corruption when ledgers are temporarily corrupted
        existing_start = None
        if os.path.exists('data/spy_benchmark.json'):
            try:
                with open('data/spy_benchmark.json', 'r') as f:
                    existing = json.load(f)
                    existing_start = existing.get('start_date')
                    if existing_start:
                        print(f"  Preserving existing start_date: {existing_start}")
            except Exception as e:
                print(f"  Could not read existing spy_benchmark.json: {e}")
        
        # PRIORITY 2: Get first ledger date (only if no existing start)
        first_date = existing_start
        if not first_date:
            ledgers = ['data/ledgers/ledger_momentum.csv', 'data/ledgers/ledger_ml.csv', 'data/ledgers/ledger_lstm.csv']
            for path in ledgers:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    if not df.empty and 'date' in df.columns:
                        first_date = df['date'].min()
                        print(f"  Using ledger first_date: {first_date}")
                        break
        
        # PRIORITY 3: Fallback to known trading start date
        if not first_date:
            first_date = TRADING_START_DATE
            print(f"  Using fallback TRADING_START_DATE: {first_date}")
        
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
    Compute current portfolio value and append VALUE row to ledger.
    
    Used by cache_refresh to keep ALL strategies current daily.
    
    Returns: True if updated, False if no update needed or error
    """
    ledger_paths = {
        'momentum': 'data/ledgers/ledger_momentum.csv',
        'ml': 'data/ledgers/ledger_ml.csv',
        'lstm': 'data/ledgers/ledger_lstm.csv'
    }
    
    if strategy not in ledger_paths:
        print(f"Unknown strategy: {strategy}")
        return False
    
    ledger_path = ledger_paths[strategy]
    print(f"\n[Value Update] {strategy.upper()}...")
    
    # Read existing snapshot for holdings and cash
    snapshot_path = os.path.join(SNAPSHOTS_DIR, f'{strategy}.json')
    if not os.path.exists(snapshot_path):
        print(f"  No existing snapshot for {strategy}")
        return False
    
    with open(snapshot_path, 'r') as f:
        data = json.load(f)
    
    holdings = data.get('holdings', {})
    cash = data.get('cash', 0)
    old_value = data.get('value', 0)
    
    if not holdings:
        print(f"  No holdings in snapshot for {strategy}")
        return False
    
    # Fetch latest prices from database
    prices = get_latest_prices(list(holdings.keys()), DB_PATH)
    if not prices:
        print(f"  Could not fetch prices for {strategy}")
        return False
    
    # Compute portfolio value
    value = compute_portfolio_value(holdings, prices, cash)
    return_pct = ((value / INITIAL_CAPITAL) - 1) * 100
    
    # Use database max date since that's when the prices are from
    today = get_latest_date(DB_PATH)
    appended = append_portfolio_value_to_ledger(ledger_path, value, today, strategy)
    
    # Update JSON snapshot
    data['value'] = round(value, 2)
    data['return_pct'] = round(return_pct, 2)
    data['price_date'] = today
    data['timestamp'] = datetime.utcnow().isoformat() + 'Z'
    
    with open(snapshot_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    if appended:
        print(f"  {strategy}: ${old_value:,.2f} -> ${value:,.2f} (appended to ledger)")
    else:
        print(f"  {strategy}: ${value:,.2f} (already in ledger for {today})")
    
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
    
    print(f"\nUpdated {updated} strategy value(s)")
    return updated


def main():
    parser = argparse.ArgumentParser(description='Compute portfolio snapshot')
    parser.add_argument('--strategy', type=str, choices=['ml', 'lstm', 'momentum'],
                       help='Process only this strategy (writes to data/snapshots/{strategy}.json)')
    parser.add_argument('--consolidate', action='store_true',
                       help='Consolidate all strategy snapshots into portfolio_snapshot.json')
    parser.add_argument('--update-value-only', action='store_true',
                       help='Update all strategy snapshot values from ledgers (preserves holdings)')
    parser.add_argument('--force', action='store_true',
                       help='Force execution even with stale local database (use with caution)')
    args = parser.parse_args()
    
    # Safety check for data-modifying operations
    if args.update_value_only or args.strategy or (not args.consolidate):
        if not check_data_safety(force=args.force):
            print("Aborted. No changes made.")
            return
    
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
