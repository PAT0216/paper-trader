#!/usr/bin/env python3
"""
Backfill total_value in ledger.csv

Recalculates portfolio total_value for all existing trades.
This fixes historical data where total_value was incorrectly set to 0.
"""

import pandas as pd
import os

def backfill_ledger_total_value():
    """Recalculate and update total_value for all ledger entries."""
    
    ledger_file = "ledger.csv"
    
    if not os.path.exists(ledger_file):
        print(f"❌ {ledger_file} not found")
        return
    
    print("📊 Loading ledger...")
    ledger = pd.read_csv(ledger_file)
    print(f"   Found {len(ledger)} entries")
    
    # Track positions
    positions = {}  # {ticker: {'shares': X, 'avg_price': Y}}
    
    print("\n🔄 Recalculating total_value for each entry...")
    
    for idx, row in ledger.iterrows():
        ticker = row['ticker']
        action = row['action']
        price = row['price']
        shares = row['shares']
        cash = row['cash_balance']
        
        # Update positions
        if action == 'BUY':
            if ticker not in positions:
                positions[ticker] = {'shares': 0, 'avg_price': 0}
            
            # Update shares
            old_shares = positions[ticker]['shares']
            new_shares = old_shares + shares
            
            # Weighted average price
            if old_shares > 0:
                old_value = old_shares * positions[ticker]['avg_price']
                new_value = shares * price
                positions[ticker]['avg_price'] = (old_value + new_value) / new_shares
            else:
                positions[ticker]['avg_price'] = price
            
            positions[ticker]['shares'] = new_shares
            
        elif action == 'SELL':
            if ticker in positions:
                positions[ticker]['shares'] -= shares
                if positions[ticker]['shares'] <= 0:
                    del positions[ticker]
        
        # Calculate portfolio value (cash + all positions)
        position_value = sum(
            pos['shares'] * pos['avg_price'] 
            for pos in positions.values()
        )
        
        total_value = cash + position_value
        
        # Update the ledger
        ledger.at[idx, 'total_value'] = total_value
        
        if idx % 5 == 0:
            print(f"   [{idx+1}/{len(ledger)}] {ticker:8s} {action:7s} | Cash: ${cash:>10.2f} | Positions: ${position_value:>10.2f} | Total: ${total_value:>10.2f}")
    
    # Save updated ledger
    print(f"\n💾 Saving updated ledger to {ledger_file}...")
    ledger.to_csv(ledger_file, index=False)
    
    print("\n✅ Backfill complete!")
    print(f"\nSummary:")
    print(f"  Total entries: {len(ledger)}")
    print(f"  Final cash: ${ledger.iloc[-1]['cash_balance']:.2f}")
    print(f"  Final total value: ${ledger.iloc[-1]['total_value']:.2f}")
    print(f"  Current positions: {len(positions)}")
    
    return ledger


if __name__ == "__main__":
    print("=" * 70)
    print("Ledger total_value Backfill Script")
    print("=" * 70)
    print()
    
    backfill_ledger_total_value()
    
    print("\n" + "=" * 70)
    print("Dashboard will now show correct portfolio values!")
    print("=" * 70)
