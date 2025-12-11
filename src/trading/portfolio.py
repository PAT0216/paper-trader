import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Tuple

LEDGER_FILE = "ledger.csv"
DEFAULT_STOP_LOSS_PCT = 0.08  # 8% stop-loss from entry price

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
    
    def get_entry_prices(self) -> Dict[str, float]:
        """
        Calculate average entry price for each current holding.
        Uses FIFO-like weighted average across all buys.
        
        Returns:
            Dict of {ticker: avg_entry_price}
        """
        entry_prices = {}
        holdings = self.get_holdings()
        
        for ticker in holdings.keys():
            ticker_ledger = self.ledger[self.ledger["ticker"] == ticker]
            buys = ticker_ledger[ticker_ledger["action"] == "BUY"]
            
            if buys.empty:
                continue
            
            # Calculate weighted average entry price
            total_shares = buys["shares"].sum()
            total_cost = (buys["shares"] * buys["price"]).sum()
            
            if total_shares > 0:
                entry_prices[ticker] = total_cost / total_shares
        
        return entry_prices
    
    def check_stop_losses(
        self, 
        current_prices: Dict[str, float], 
        stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT
    ) -> List[Tuple[str, float, float, float]]:
        """
        Check which positions have breached stop-loss threshold.
        
        Args:
            current_prices: Dict of {ticker: current_price}
            stop_loss_pct: Stop-loss threshold (default 8%)
            
        Returns:
            List of (ticker, current_price, entry_price, loss_pct) for triggered stops
        """
        triggered = []
        holdings = self.get_holdings()
        entry_prices = self.get_entry_prices()
        
        for ticker, shares in holdings.items():
            if ticker not in current_prices or ticker not in entry_prices:
                continue
            
            current = current_prices[ticker]
            entry = entry_prices[ticker]
            
            if entry <= 0:
                continue
            
            loss_pct = (entry - current) / entry
            
            if loss_pct >= stop_loss_pct:
                triggered.append((ticker, current, entry, loss_pct))
        
        return triggered
    
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

        return cash + holdings_value

    def get_positions(self) -> Dict[str, Dict]:
        """
        Returns a dictionary of current positions with detailed info.
        Format: {ticker: {'ticker': t, 'shares': s, 'avg_price': p}}
        """
        positions = {}
        holdings = self.get_holdings()
        entry_prices = self.get_entry_prices()
        
        for ticker, shares in holdings.items():
            positions[ticker] = {
                'ticker': ticker,
                'shares': shares,
                'avg_price': entry_prices.get(ticker, 0.0)
            }
        return positions

    def has_traded_today(self, ticker, date=None):
        """
        Checks if a trade for the given ticker has already occurred today.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
            
        if self.ledger.empty:
            return False
            
        # Ensure date column is string for comparison
        # (It should be, but safety first if pandas inferred objects differently)
        today_trades = self.ledger[
            (self.ledger["date"].astype(str) == date) & 
            (self.ledger["ticker"] == ticker)
        ]
        return not today_trades.empty

    def record_trade(self, ticker, action, price, shares):
        """
        Records a trade and updates the ledger.
        """
        date = datetime.now().strftime("%Y-%m-%d")
        
        # 🛡️ Safety: Idempotency Check
        if self.has_traded_today(ticker, date):
            print(f"⚠️  Skipping {ticker}: Already traded today ({date}).")
            return False

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
        
        # Calculate total portfolio value (cash + position values)
        # Note: Using entry prices for simplicity - ideally would use current market prices
        positions = self.get_positions()
        position_value = sum(pos['shares'] * price if pos['ticker'] == ticker else pos['shares'] * pos['avg_price'] 
                            for pos in positions.values())
        
        # For BUY: add new position value, for SELL: position_value already adjusted
        if action == "BUY":
            total_value = new_cash + position_value + (shares * price)
        else:
            total_value = new_cash + position_value
            
        self._add_entry(date, ticker, action, price, shares, amount, new_cash, total_value) 
        return True

