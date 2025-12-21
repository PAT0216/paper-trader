"""
Base Strategy - Abstract Base Class for Trading Strategies

All strategies must inherit from BaseStrategy and implement:
- get_name(): Strategy identifier
- rank_universe(): Score all tickers
- generate_signals(): Generate BUY/SELL/HOLD signals
- needs_training(): Whether strategy requires model training
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    All trading strategies (Momentum, ML, Mean Reversion, etc.) must inherit
    from this class and implement the required methods.
    
    This enables:
    - Consistent API across all strategies
    - Dynamic strategy loading via registry
    - Easy addition of new strategies without modifying main.py
    """
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Return unique strategy identifier.
        
        Used for:
        - Ledger filenames (ledger_{name}.csv)
        - CLI argument (--strategy {name})
        - Logging and display
        
        Returns:
            Strategy name (e.g., 'momentum', 'ml', 'mean_reversion')
        """
        pass
    
    @abstractmethod
    def rank_universe(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Score all tickers in the universe.
        
        Higher scores indicate better buy candidates.
        
        Args:
            data_dict: Dictionary of {ticker: OHLCV DataFrame}
            
        Returns:
            Dictionary of {ticker: score} where higher = better
        """
        pass
    
    @abstractmethod
    def generate_signals(
        self, 
        data_dict: Dict[str, pd.DataFrame],
        top_pct: float = 0.10,
        bottom_pct: float = 0.10
    ) -> Dict[str, str]:
        """
        Generate trading signals for all tickers.
        
        Default implementation uses cross-sectional ranking:
        - Top X% by score → BUY
        - Bottom X% by score → SELL
        - Middle → HOLD
        
        Args:
            data_dict: Dictionary of {ticker: OHLCV DataFrame}
            top_pct: Percentile for BUY signals (default 10%)
            bottom_pct: Percentile for SELL signals (default 10%)
            
        Returns:
            Dictionary of {ticker: 'BUY' | 'SELL' | 'HOLD'}
        """
        pass
    
    @abstractmethod
    def needs_training(self) -> bool:
        """
        Return True if strategy requires model training.
        
        Used to determine if training step should run:
        - Momentum: False (no ML model)
        - ML/XGBoost: True (requires trained model)
        - Mean Reversion: False (no ML model)
        
        Returns:
            True if training required, False otherwise
        """
        pass
    
    def get_ledger_filename(self) -> str:
        """
        Return ledger filename for this strategy.
        
        Default: ledger_{strategy_name}.csv
        
        Returns:
            Filename for trade ledger
        """
        return f"ledger_{self.get_name()}.csv"
    
    def get_display_name(self) -> str:
        """
        Return human-readable strategy name for display.
        
        Override for custom display names.
        
        Returns:
            Display name (e.g., 'Momentum 12-1', 'ML Ensemble')
        """
        return self.get_name().upper()
    
    def validate_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Validate and filter data dictionary.
        
        Default: require at least 200 days of data.
        Override for strategy-specific requirements.
        
        Args:
            data_dict: Raw data dictionary
            
        Returns:
            Filtered data dictionary with only valid tickers
        """
        min_days = 200
        return {
            ticker: df 
            for ticker, df in data_dict.items() 
            if len(df) >= min_days
        }
    
    def _cross_sectional_rank(
        self, 
        scores: Dict[str, float],
        top_pct: float = 0.10,
        bottom_pct: float = 0.10
    ) -> Dict[str, str]:
        """
        Helper: Convert scores to signals using cross-sectional ranking.
        
        Args:
            scores: Dictionary of {ticker: score}
            top_pct: Top percentile for BUY
            bottom_pct: Bottom percentile for SELL
            
        Returns:
            Dictionary of {ticker: signal}
        """
        if not scores:
            return {}
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        n = len(sorted_scores)
        n_buy = max(1, int(n * top_pct))
        n_sell = max(1, int(n * bottom_pct))
        
        signals = {}
        for i, (ticker, _) in enumerate(sorted_scores):
            if i < n_buy:
                signals[ticker] = 'BUY'
            elif i >= n - n_sell:
                signals[ticker] = 'SELL'
            else:
                signals[ticker] = 'HOLD'
        
        return signals
