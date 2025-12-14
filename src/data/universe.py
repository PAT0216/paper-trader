"""
Universe Manager
Reconstructs point-in-time S&P 500 universes by walking backwards from the current constituent list
using historical changes (adds/removes).
"""

import pandas as pd
from datetime import datetime
import os

class UniverseManager:
    def __init__(self, current_tickers: list[str], changes_csv: str):
        self.current_tickers = set(current_tickers)
        self.changes = self._load_changes(changes_csv)
        
    def _load_changes(self, csv_path: str) -> pd.DataFrame:
        """Load and parse changes CSV."""
        if not os.path.exists(csv_path):
            return pd.DataFrame(columns=['date', 'add', 'remove'])
            
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date', ascending=False)
    
    def get_universe_at(self, target_date: str) -> list[str]:
        """
        Get the list of tickers that were in the universe on target_date.
        Method: Start with current universe and reverse changes until target_date.
        """
        target_dt = pd.to_datetime(target_date)
        current_dt = datetime.now()
        
        # Start with current universe
        universe = self.current_tickers.copy()
        
        # Iterate through changes from NOW backwards to TARGET
        # Filter changes that happened AFTER target_date
        relevant_changes = self.changes[self.changes['date'] > target_dt]
        
        # Sort descending (newest to oldest) to reverse them in order
        relevant_changes = relevant_changes.sort_values('date', ascending=False)
        
        for _, row in relevant_changes.iterrows():
            # Reverse ADDs: If it was added, it wasn't there before -> REMOVE it
            if pd.notna(row['add']):
                added_tickers = [t.strip() for t in str(row['add']).split(',') if t.strip()]
                for ticker in added_tickers:
                    if ticker in universe:
                        universe.remove(ticker)
            
            # Reverse REMOVEs: If it was removed, it WAS there before -> ADD it
            if pd.notna(row['remove']):
                removed_tickers = [t.strip() for t in str(row['remove']).split(',') if t.strip()]
                for ticker in removed_tickers:
                    universe.add(ticker)
                    
        return list(universe)


def fetch_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 tickers from Wikipedia."""
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = tables[0]
        tickers = df['Symbol'].tolist()
        # Clean tickers: BRK.B -> BRK-B (Yahoo Finance format)
        return [t.replace('.', '-') for t in tickers]
    except Exception as e:
        print(f"Error fetching S&P 500 from Wikipedia: {e}")
        return []


def get_mega_caps() -> list[str]:
    """Fallback: hardcoded mega-cap stocks if Wikipedia fetch fails."""
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 
        'LLY', 'V', 'UNH', 'JPM', 'XOM', 'JNJ', 'MA', 'PG', 'HD', 'CVX',
        'MRK', 'ABBV', 'COST', 'PEP', 'KO', 'AVGO', 'WMT', 'MCD', 'CSCO',
        'CRM', 'ACN', 'LIN', 'TMO', 'ABT', 'ADBE', 'AMD', 'ORCL', 'NKE'
    ]
