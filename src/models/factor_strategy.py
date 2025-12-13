"""
Factor Investing Strategy

Long/short strategy based on Fama-French factors:
- Value: Buy cheap stocks (high B/P)
- Quality: Buy profitable, low-debt stocks
- Momentum: Buy recent winners (12-1 month)

Academic basis: Fama-French (1993, 2015), extensive peer-reviewed evidence.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from src.features.factor_features import (
    calculate_factor_scores,
    select_stocks,
    FACTOR_NAMES
)


class FactorStrategy:
    """
    Multi-factor investing strategy.
    
    Combines Value + Quality + Momentum to rank stocks,
    then goes long top N and short bottom N.
    """
    
    def __init__(self, 
                 n_long: int = 10, 
                 n_short: int = 10,
                 factor_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the factor strategy.
        
        Args:
            n_long: Number of stocks to long
            n_short: Number of stocks to short (0 for long-only)
            factor_weights: Custom weights for factors (default: equal weight)
        """
        self.n_long = n_long
        self.n_short = n_short
        
        # Factor weights (default: equal)
        if factor_weights is None:
            self.factor_weights = {
                'value': 1.0,
                'quality': 1.0,
                'momentum': 1.0
            }
        else:
            self.factor_weights = factor_weights
        
        # Normalize weights
        total_weight = sum(self.factor_weights.values())
        self.factor_weights = {k: v/total_weight for k, v in self.factor_weights.items()}
        
        self.last_factor_scores = None
    
    def calculate_composite_score(self, factor_df: pd.DataFrame) -> pd.Series:
        """
        Calculate weighted composite score from individual factors.
        """
        scores = pd.Series(0.0, index=factor_df.index)
        
        for factor, weight in self.factor_weights.items():
            if factor in factor_df.columns:
                scores += factor_df[factor].fillna(0.5) * weight
        
        return scores
    
    def generate_signals(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """
        Generate trading signals for all stocks.
        
        Args:
            data_dict: {ticker: OHLCV DataFrame}
            
        Returns:
            {ticker: 'BUY'|'SELL'|'HOLD'}
        """
        # Calculate factor scores
        factor_df = calculate_factor_scores(data_dict)
        
        if len(factor_df) == 0:
            return {ticker: 'HOLD' for ticker in data_dict}
        
        # Calculate weighted composite
        factor_df['composite'] = self.calculate_composite_score(factor_df)
        
        # Store for analysis
        self.last_factor_scores = factor_df
        
        # Select stocks
        long_tickers, short_tickers = select_stocks(
            factor_df, 
            n_long=self.n_long, 
            n_short=self.n_short
        )
        
        # Generate signals
        signals = {}
        for ticker in data_dict:
            if ticker in long_tickers:
                signals[ticker] = 'BUY'
            elif ticker in short_tickers:
                signals[ticker] = 'SELL'
            else:
                signals[ticker] = 'HOLD'
        
        return signals
    
    def get_portfolio_weights(self, signals: Dict[str, str]) -> Dict[str, float]:
        """
        Get portfolio weights based on signals.
        
        Returns equal weight within long and short portfolios.
        """
        long_tickers = [t for t, s in signals.items() if s == 'BUY']
        short_tickers = [t for t, s in signals.items() if s == 'SELL']
        
        weights = {}
        
        # Equal weight for long positions
        if long_tickers:
            long_weight = 0.5 / len(long_tickers)  # 50% to long
            for t in long_tickers:
                weights[t] = long_weight
        
        # Equal weight for short positions (negative)
        if short_tickers:
            short_weight = -0.5 / len(short_tickers)  # 50% to short
            for t in short_tickers:
                weights[t] = short_weight
        
        return weights
    
    def get_factor_analysis(self) -> Optional[pd.DataFrame]:
        """
        Get the last calculated factor scores for analysis.
        """
        return self.last_factor_scores


def backtest_factor_strategy(data_dict: Dict[str, pd.DataFrame],
                              strategy: FactorStrategy,
                              start_date: str,
                              end_date: str,
                              rebalance_freq: str = 'monthly',
                              transaction_cost: float = 0.001) -> Dict:
    """
    Backtest the factor strategy.
    
    Args:
        data_dict: {ticker: full OHLCV DataFrame}
        strategy: FactorStrategy instance
        start_date: Start of backtest period
        end_date: End of backtest period
        rebalance_freq: 'daily', 'weekly', or 'monthly'
        transaction_cost: One-way transaction cost as decimal
        
    Returns:
        Dict with performance metrics
    """
    # Get trading dates
    sample_df = list(data_dict.values())[0]
    if isinstance(sample_df.index, pd.DatetimeIndex):
        all_dates = sample_df.index
    else:
        all_dates = pd.to_datetime(sample_df.index)
    
    all_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]
    
    # Determine rebalance dates
    if rebalance_freq == 'monthly':
        # First trading day of each month
        rebalance_dates = all_dates.to_series().groupby(
            [all_dates.year, all_dates.month]
        ).first().values
    elif rebalance_freq == 'weekly':
        rebalance_dates = all_dates[::5]  # Every 5 trading days
    else:
        rebalance_dates = all_dates  # Daily
    
    # Track portfolio
    portfolio_values = [1.0]  # Start with $1
    current_weights = {}
    daily_returns = []
    
    for i, date in enumerate(all_dates):
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
        
        # Check if rebalance day
        if date in rebalance_dates:
            # Get data up to this date for each stock
            current_data = {}
            for ticker, df in data_dict.items():
                df_to_date = df[df.index <= date]
                if len(df_to_date) >= 252:  # Need 1 year for momentum
                    current_data[ticker] = df_to_date
            
            if current_data:
                # Generate new signals
                signals = strategy.generate_signals(current_data)
                new_weights = strategy.get_portfolio_weights(signals)
                
                # Apply transaction costs for turnover
                turnover = 0
                for ticker in set(list(current_weights.keys()) + list(new_weights.keys())):
                    old_w = current_weights.get(ticker, 0)
                    new_w = new_weights.get(ticker, 0)
                    turnover += abs(new_w - old_w)
                
                tx_cost_impact = turnover * transaction_cost
                portfolio_values[-1] *= (1 - tx_cost_impact)
                
                current_weights = new_weights
        
        # Calculate daily return
        if i > 0 and current_weights:
            prev_date = all_dates[i-1]
            daily_pnl = 0
            
            for ticker, weight in current_weights.items():
                if ticker in data_dict:
                    df = data_dict[ticker]
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    
                    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                    
                    if date in df.index and prev_date in df.index:
                        curr_price = df.loc[date, price_col]
                        prev_price = df.loc[prev_date, price_col]
                        
                        if prev_price > 0:
                            stock_return = (curr_price / prev_price) - 1
                            daily_pnl += weight * stock_return
            
            daily_returns.append(daily_pnl)
            portfolio_values.append(portfolio_values[-1] * (1 + daily_pnl))
    
    # Calculate metrics
    portfolio_values = np.array(portfolio_values)
    daily_returns = np.array(daily_returns)
    
    cumulative_return = portfolio_values[-1] / portfolio_values[0] - 1
    annualized_return = (1 + cumulative_return) ** (252 / len(daily_returns)) - 1 if daily_returns.size > 0 else 0
    volatility = np.std(daily_returns) * np.sqrt(252) if daily_returns.size > 0 else 0
    sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
    
    # Max drawdown
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = portfolio_values / running_max - 1
    max_drawdown = np.min(drawdowns)
    
    return {
        'cumulative_return': cumulative_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'n_days': len(daily_returns),
        'portfolio_values': portfolio_values,
        'daily_returns': daily_returns
    }


if __name__ == "__main__":
    print("Factor Investing Strategy")
    print("=" * 40)
    print("Factors: Value + Quality + Momentum")
    print("Rebalancing: Monthly")
    print("Based on: Fama-French (1993, 2015)")
