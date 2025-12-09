import pandas as pd
import os
from datetime import datetime

LEDGER_FILE = "ledger.csv"

class Portfolio:
    def __init__(self, start_cash=10000.0):
        self.ledger_file = LEDGER_FILE
        self.columns = ["date", "ticker", "action", "price", "shares", "amount", "cash_balance", "total_value"]
        
        if os.path.exists(self.ledger_file):
            self.ledger = pd.read_csv(self.ledger_file)
        else:
            self.ledger = pd.DataFrame(columns=self.columns)
            # Initialize with starting cash
            self._add_entry(datetime.now().strftime("%Y-%m-%d"), "CASH", "DEPOSIT", 1.0, start_cash, start_cash, start_cash, start_cash)

    def _add_entry(self, date, ticker, action, price, shares, amount, cash_balance, total_value):
        new_entry = pd.DataFrame([{
            "date": date,
            "ticker": ticker,
            "action": action,
            "price": price,
            "shares": shares,
            "amount": amount,
            "cash_balance": cash_balance,
            "total_value": total_value
        }])
        self.ledger = pd.concat([self.ledger, new_entry], ignore_index=True)
        self.ledger.to_csv(self.ledger_file, index=False)

    def get_last_balance(self):
        if self.ledger.empty:
            return 10000.0 # Default fallback
        # Get the very last cash_balance recorded
        return self.ledger.iloc[-1]["cash_balance"]

    def get_holdings(self):
        """
        Returns a dictionary of current holdings {ticker: shares}
        """
        holdings = {}
        # Get unique tickers
        tickers = self.ledger[self.ledger["ticker"] != "CASH"]["ticker"].unique()
        
        for t in tickers:
            ticker_ledger = self.ledger[self.ledger["ticker"] == t]
            buy_shares = ticker_ledger[ticker_ledger["action"] == "BUY"]["shares"].sum()
            sell_shares = ticker_ledger[ticker_ledger["action"] == "SELL"]["shares"].sum()
            current_shares = buy_shares - sell_shares
            if current_shares > 0:
                holdings[t] = current_shares
                
        return holdings
    
    def get_portfolio_value(self, current_prices):
        """
        Calculate total portfolio value (Cash + Holdings Value).
        current_prices: dict {ticker: price}
        """
        cash = self.get_last_balance()
        holdings_value = 0.0
        
        holdings = self.get_holdings()
        for t, shares in holdings.items():
            price = current_prices.get(t, 0.0)
            holdings_value += shares * price
                
        return cash + holdings_value

    def record_trade(self, ticker, action, price, shares):
        """
        Records a trade and updates the ledger.
        """
        date = datetime.now().strftime("%Y-%m-%d")
        current_cash = self.get_last_balance()
        amount = price * shares
        
        if action == "BUY":
            if current_cash < amount:
                print(f"Insufficient funds to buy {ticker}")
                return False
            new_cash = current_cash - amount
        elif action == "SELL":
            # Verification already done in logic usually, but double check
            new_cash = current_cash + amount
        else:
            return False
            
        self._add_entry(date, ticker, action, price, shares, amount, new_cash, 0.0) 
        return True
