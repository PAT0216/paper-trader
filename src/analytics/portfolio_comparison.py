"""
Portfolio Comparison Analytics

Compares multiple portfolios and calculates performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import os


@dataclass
class PortfolioMetrics:
    """Performance metrics for a portfolio."""
    portfolio_id: str
    total_return: float
    cagr: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    current_value: float
    num_trades: int


class PortfolioComparison:
    """
    Compare multiple portfolios and calculate relative performance.
    """
    
    def __init__(self):
        self.daily_log_dir = "logs"
    
    def load_portfolio(self, portfolio_id: str) -> Optional[pd.DataFrame]:
        """Load ledger for a portfolio."""
        if portfolio_id == "default":
            path = "data/ledgers/ledger.csv"
        else:
            path = f"data/ledgers/ledger_{portfolio_id}.csv"
        
        if not os.path.exists(path):
            return None
        
        return pd.read_csv(path)
    
    def load_daily_values(self, portfolio_id: str) -> Optional[pd.DataFrame]:
        """Load daily portfolio values."""
        path = os.path.join(self.daily_log_dir, f"portfolio_{portfolio_id}_daily.csv")
        
        if not os.path.exists(path):
            return None
        
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def calculate_metrics(self, portfolio_id: str) -> Optional[PortfolioMetrics]:
        """Calculate performance metrics for a portfolio."""
        ledger = self.load_portfolio(portfolio_id)
        daily = self.load_daily_values(portfolio_id)
        
        if ledger is None or ledger.empty:
            return None
        
        # Get trades (exclude CASH deposits)
        trades = ledger[ledger['ticker'] != 'CASH']
        num_trades = len(trades)
        
        # Current value
        current_value = ledger.iloc[-1]['total_value'] if not ledger.empty else 0
        
        # Starting value
        start_value = ledger[ledger['action'] == 'DEPOSIT']['amount'].sum()
        if start_value == 0:
            start_value = 100000  # Default
        
        # Total return
        total_return = (current_value / start_value) - 1 if start_value > 0 else 0
        
        # Calculate from daily values if available
        if daily is not None and len(daily) > 1:
            returns = daily['total_value'].pct_change().dropna()
            
            # CAGR
            years = (daily['date'].iloc[-1] - daily['date'].iloc[0]).days / 365.25
            if years > 0:
                cagr = (daily['total_value'].iloc[-1] / daily['total_value'].iloc[0]) ** (1/years) - 1
            else:
                cagr = 0
            
            # Sharpe (annualized)
            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Max drawdown
            cummax = daily['total_value'].cummax()
            drawdown = (daily['total_value'] - cummax) / cummax
            max_drawdown = drawdown.min()
        else:
            cagr = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Win rate from trades
        if num_trades > 0:
            # Simplified: count buys vs sells with profit
            win_rate = 0.5  # Placeholder - would need entry/exit matching
        else:
            win_rate = 0
        
        return PortfolioMetrics(
            portfolio_id=portfolio_id,
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            current_value=current_value,
            num_trades=num_trades
        )
    
    def compare(self, portfolio_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple portfolios.
        
        Returns DataFrame with metrics for each portfolio.
        """
        metrics = []
        
        for pid in portfolio_ids:
            m = self.calculate_metrics(pid)
            if m:
                metrics.append({
                    'Portfolio': pid,
                    'Value': f"${m.current_value:,.0f}",
                    'Return': f"{m.total_return*100:+.1f}%",
                    'CAGR': f"{m.cagr*100:.1f}%",
                    'Sharpe': f"{m.sharpe_ratio:.2f}",
                    'Max DD': f"{m.max_drawdown*100:.1f}%",
                    'Trades': m.num_trades
                })
        
        return pd.DataFrame(metrics)
    
    def get_combined_daily_returns(self, portfolio_ids: List[str]) -> pd.DataFrame:
        """Get daily returns for all portfolios in one DataFrame."""
        dfs = []
        
        for pid in portfolio_ids:
            daily = self.load_daily_values(pid)
            if daily is not None:
                daily = daily[['date', 'total_value']].rename(
                    columns={'total_value': pid}
                )
                dfs.append(daily)
        
        if not dfs:
            return pd.DataFrame()
        
        # Merge all on date
        result = dfs[0]
        for df in dfs[1:]:
            result = result.merge(df, on='date', how='outer')
        
        return result.sort_values('date')


def get_portfolio_summary() -> Dict:
    """Quick summary of all portfolios."""
    comparison = PortfolioComparison()
    
    # Find all ledger files
    portfolios = []
    ledger_dir = 'data/ledgers'
    if os.path.exists(ledger_dir):
        for f in os.listdir(ledger_dir):
            if f.startswith('ledger') and f.endswith('.csv'):
                if f == 'ledger.csv':
                    portfolios.append('default')
                else:
                    pid = f.replace('ledger_', '').replace('.csv', '')
                    portfolios.append(pid)
    
    return {
        'portfolios': portfolios,
        'comparison': comparison.compare(portfolios)
    }
