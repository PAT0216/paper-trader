"""
Portfolio Tracker - Daily Monitoring

Tracks portfolio value daily without executing trades.
Updates ledger with mark-to-market values.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime, date
from dataclasses import dataclass
import csv
import os


@dataclass
class PortfolioSnapshot:
    """Daily portfolio snapshot."""
    date: date
    total_value: float
    cash: float
    positions_value: float
    daily_return: float
    cumulative_return: float


class PortfolioTracker:
    """
    Tracks portfolio performance daily.
    
    Updates ledger with current values without executing trades.
    Designed for daily cron job monitoring.
    """
    
    def __init__(self, ledger_path: str = 'ledger.csv'):
        """
        Initialize portfolio tracker.
        
        Args:
            ledger_path: Path to ledger CSV file
        """
        self.ledger_path = ledger_path
        self._last_value: Optional[float] = None
        self._initial_value: Optional[float] = None
    
    def get_current_holdings(self) -> Dict[str, int]:
        """
        Get current holdings from ledger.
        
        Returns:
            {ticker: shares}
        """
        if not os.path.exists(self.ledger_path):
            return {}
        
        holdings = {}
        
        with open(self.ledger_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row.get('ticker', row.get('symbol', ''))
                action = row.get('action', row.get('type', ''))
                shares = int(float(row.get('shares', row.get('quantity', 0))))
                
                if not ticker:
                    continue
                
                if action.upper() == 'BUY':
                    holdings[ticker] = holdings.get(ticker, 0) + shares
                elif action.upper() == 'SELL':
                    holdings[ticker] = holdings.get(ticker, 0) - shares
        
        # Remove zero positions
        return {t: s for t, s in holdings.items() if s > 0}
    
    def calculate_portfolio_value(
        self,
        holdings: Dict[str, int],
        prices: Dict[str, float],
        cash: float = 0
    ) -> Tuple[float, float]:
        """
        Calculate current portfolio value.
        
        Args:
            holdings: {ticker: shares}
            prices: {ticker: current_price}
            cash: Cash balance
            
        Returns:
            (total_value, positions_value)
        """
        positions_value = sum(
            holdings.get(t, 0) * prices.get(t, 0)
            for t in holdings
        )
        
        total_value = positions_value + cash
        
        return total_value, positions_value
    
    def create_snapshot(
        self,
        holdings: Dict[str, int],
        prices: Dict[str, float],
        cash: float = 0,
        snapshot_date: Optional[date] = None
    ) -> PortfolioSnapshot:
        """
        Create a daily portfolio snapshot.
        
        Args:
            holdings: Current positions
            prices: Current prices
            cash: Cash balance
            snapshot_date: Date for snapshot (default: today)
            
        Returns:
            PortfolioSnapshot object
        """
        if snapshot_date is None:
            snapshot_date = date.today()
        
        total_value, positions_value = self.calculate_portfolio_value(
            holdings, prices, cash
        )
        
        # Calculate returns
        if self._initial_value is None:
            self._initial_value = total_value
        
        daily_return = 0.0
        if self._last_value and self._last_value > 0:
            daily_return = (total_value / self._last_value) - 1
        
        cumulative_return = 0.0
        if self._initial_value and self._initial_value > 0:
            cumulative_return = (total_value / self._initial_value) - 1
        
        self._last_value = total_value
        
        return PortfolioSnapshot(
            date=snapshot_date,
            total_value=total_value,
            cash=cash,
            positions_value=positions_value,
            daily_return=daily_return,
            cumulative_return=cumulative_return
        )
    
    def check_data_staleness(
        self,
        prices: Dict[str, float],
        holdings: Dict[str, int]
    ) -> List[str]:
        """
        Check for missing or stale price data.
        
        Returns:
            List of tickers with missing/stale data
        """
        stale_tickers = []
        
        for ticker in holdings:
            if ticker not in prices or prices[ticker] <= 0:
                stale_tickers.append(ticker)
        
        return stale_tickers
    
    def check_drawdown(
        self,
        current_value: float,
        peak_value: float,
        warning_threshold: float = 0.15,
        halt_threshold: float = 0.25
    ) -> Tuple[bool, bool, float]:
        """
        Check drawdown levels.
        
        Returns:
            (warning_triggered, halt_triggered, drawdown_pct)
        """
        if peak_value <= 0:
            return False, False, 0.0
        
        drawdown = (peak_value - current_value) / peak_value
        
        warning = drawdown >= warning_threshold
        halt = drawdown >= halt_threshold
        
        return warning, halt, drawdown
    
    def log_daily_value(
        self,
        snapshot: PortfolioSnapshot,
        log_path: str = 'logs/portfolio_daily.csv'
    ):
        """
        Log daily portfolio value to CSV.
        
        Args:
            snapshot: Portfolio snapshot to log
            log_path: Path to daily log file
        """
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        file_exists = os.path.exists(log_path)
        
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            if not file_exists:
                writer.writerow([
                    'date', 'total_value', 'cash', 'positions_value',
                    'daily_return', 'cumulative_return'
                ])
            
            writer.writerow([
                snapshot.date.isoformat(),
                f'{snapshot.total_value:.2f}',
                f'{snapshot.cash:.2f}',
                f'{snapshot.positions_value:.2f}',
                f'{snapshot.daily_return:.4f}',
                f'{snapshot.cumulative_return:.4f}'
            ])


def run_daily_monitor():
    """
    Daily monitoring routine.
    
    Called by daily cron job. Does NOT execute trades.
    """
    from src.data.cache import DataCache
    from src.trading.portfolio import Portfolio
    
    print(f"=== Daily Portfolio Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")
    
    # Initialize
    tracker = PortfolioTracker()
    cache = DataCache()
    portfolio = Portfolio()
    
    # Get current holdings
    holdings = tracker.get_current_holdings()
    print(f"Current positions: {len(holdings)}")
    
    if not holdings:
        print("No positions to monitor")
        return
    
    # Fetch latest prices
    prices = {}
    for ticker in holdings:
        df = cache.get_price_data(ticker, period='5d')
        if df is not None and len(df) > 0:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            prices[ticker] = df['Close'].iloc[-1]
    
    # Check for stale data
    stale = tracker.check_data_staleness(prices, holdings)
    if stale:
        print(f"⚠️  Stale data for: {stale}")
    
    # Calculate portfolio value
    cash = portfolio.get_last_balance() if hasattr(portfolio, 'get_last_balance') else 0
    snapshot = tracker.create_snapshot(holdings, prices, cash)
    
    # Log daily value
    tracker.log_daily_value(snapshot)
    
    # Print summary
    print(f"\n📊 Portfolio Value: ${snapshot.total_value:,.2f}")
    print(f"   Daily Return: {snapshot.daily_return:+.2%}")
    print(f"   Cumulative: {snapshot.cumulative_return:+.2%}")
    
    # Check drawdown
    # In production, load peak from persistent storage
    peak = snapshot.total_value  # Placeholder
    warning, halt, dd = tracker.check_drawdown(snapshot.total_value, peak)
    
    if warning:
        print(f"⚠️  DRAWDOWN WARNING: {dd:.1%}")
    if halt:
        print(f"🛑 DRAWDOWN HALT: {dd:.1%}")
    
    print("\n✅ Daily monitoring complete (no trades executed)")


if __name__ == "__main__":
    run_daily_monitor()
