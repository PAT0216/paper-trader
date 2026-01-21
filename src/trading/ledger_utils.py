"""
Ledger Utilities - Functions for parsing and updating trade ledgers.

Extracted from compute_portfolio_snapshot.py for reusability.
"""

import os
import pandas as pd
from typing import Dict, Tuple
from datetime import datetime


def get_ledger_portfolio_value(ledger_path: str) -> Tuple[float, str]:
    """
    Get the latest PORTFOLIO,VALUE entry from the ledger.
    This is the authoritative source of truth for portfolio value.
    
    Args:
        ledger_path: Path to the ledger CSV file
        
    Returns:
        Tuple of (portfolio_value, date_str)
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


def get_current_holdings_from_ledger(ledger_path: str) -> Tuple[Dict[str, int], float]:
    """
    Parse ledger to get current holdings and cash balance.
    
    IMPORTANT: Cash is calculated from transaction amounts, NOT from the
    cash_balance column (which may be incorrect in some ledgers).
    
    Args:
        ledger_path: Path to the ledger CSV file
        
    Returns:
        Tuple of (holdings_dict, cash_balance)
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
            holdings[ticker] = holdings.get(ticker, 0) + int(shares)
        elif action == 'SELL':
            # SELL: add amount to cash, subtract shares from holdings
            cash += amount
            if ticker in holdings:
                holdings[ticker] -= int(shares)
                if holdings[ticker] <= 0:
                    del holdings[ticker]
        # Skip PORTFOLIO,VALUE rows (they don't affect holdings/cash)
    
    return holdings, cash


def append_portfolio_value_to_ledger(
    ledger_path: str, 
    value: float, 
    date_str: str, 
    strategy: str
) -> bool:
    """
    Append a PORTFOLIO,VALUE row to the ledger for the given date.
    
    This ensures the ledger has a daily record of portfolio value,
    which is used by the dashboard chart. Only appends if no entry
    exists for this date yet.
    
    Args:
        ledger_path: Path to the ledger CSV file
        value: Portfolio value to record
        date_str: Date string (YYYY-MM-DD)
        strategy: Strategy name for logging
        
    Returns:
        True if appended, False if already exists or error
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
            'strategy': strategy
        }])
        
        # Append to ledger
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(ledger_path, index=False)
        print(f"  {strategy}: Appended value ${value:,.2f} for {date_str}")
        return True
        
    except Exception as e:
        print(f"  {strategy}: Error appending value - {e}")
        return False
