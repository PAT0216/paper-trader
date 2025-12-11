import pandas as pd
import os

LEDGER_FILE = "ledger.csv"

def repair_ledger():
    if not os.path.exists(LEDGER_FILE):
        print(f"❌ {LEDGER_FILE} not found.")
        return

    df = pd.read_csv(LEDGER_FILE)
    print(f"Loaded {len(df)} rows from {LEDGER_FILE}")

    # Reconstruct state
    cash = 10000.0  # Initial assumption, will be overridden by transactions
    # holdings: {ticker: shares}
    holdings = {}
    # entry_costs: {ticker: total_cost} -> avg_price = total_cost / shares
    entry_costs = {}

    # We need to preserve the original columns but update total_value
    # We will also sanity check cash_balance
    
    recalculated_values = []
    
    for index, row in df.iterrows():
        date = row['date']
        ticker = row['ticker']
        action = row['action']
        price = float(row['price'])
        shares = float(row['shares'])
        amount = float(row['amount'])
        
        # logic from Portfolio
        if action == "DEPOSIT":
             cash = amount # specific for the initial deposit line
             # no change to holdings
        elif action == "BUY":
            cash -= amount
            
            # Update holdings
            current_shares = holdings.get(ticker, 0.0)
            holdings[ticker] = current_shares + shares
            
            # Update entry cost (weighted average logic)
            # current_cost = entry_costs.get(ticker, 0.0)
            # entry_costs[ticker] = current_cost + amount
            # Actually Portfolio.py calculates avg price on the fly from the ledger
            # But here we are iterating forward.
            # Simplified: cost basis increases by amount paid.
            current_total_cost = entry_costs.get(ticker, 0.0)
            entry_costs[ticker] = current_total_cost + amount

        elif action == "SELL":
            cash += amount
            
            # Update holdings
            current_shares = holdings.get(ticker, 0.0)
            if current_shares < shares:
                print(f"⚠️ Warning at index {index}: Selling {shares} but only have {current_shares} of {ticker}")
                current_shares = shares # force to 0? or just allow negative validation?
            
            # When selling, we reduce the Cost Basis proportionally
            # avg_price = total_cost / current_shares
            # cost_of_sold_shares = avg_price * shares
            # new_total_cost = total_cost - cost_of_sold_shares
            
            total_cost = entry_costs.get(ticker, 0.0)
            if current_shares > 0:
                avg_price = total_cost / current_shares
                cost_of_sold = avg_price * shares
                entry_costs[ticker] = total_cost - cost_of_sold
            else:
                 entry_costs[ticker] = 0.0

            holdings[ticker] = current_shares - shares
            if holdings[ticker] <= 1e-9: # float tolerance
                del holdings[ticker]
                if ticker in entry_costs: del entry_costs[ticker]

        # Calculate Total Book Value
        # Cash + Sum of (current_shares * avg_entry_price)
        # But wait, (current_shares * avg_entry_price) IS exactly entry_costs[ticker] 
        # (minus floating point drift)
        
        holdings_value = sum(entry_costs.values())
        total_value = cash + holdings_value
        
        recalculated_values.append(total_value)
        
        # Optional: update row's cash_balance to match our recalc to be safe?
        # df.at[index, 'cash_balance'] = cash 

    # Update dataframe
    df['total_value'] = recalculated_values
    
    # Validation
    last_val = df.iloc[-1]['total_value']
    print(f"✅ Repaired. Final Portfolio Value: ${last_val:,.2f}")
    
    df.to_csv(LEDGER_FILE, index=False)
    print(f"Saved to {LEDGER_FILE}")

if __name__ == "__main__":
    repair_ledger()
