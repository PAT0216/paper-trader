"""
Strategies Module - Trading Strategy Infrastructure

Provides:
- BaseStrategy: Abstract base class for all strategies
- MomentumStrategy: 12-1 month momentum factor strategy
- MLStrategy: XGBoost ensemble prediction strategy
- get_strategy(): Factory function for dynamic loading
- list_strategies(): List available strategy names
"""

from .base import BaseStrategy
from .momentum_strategy import MomentumStrategy, RebalanceOrder, load_tickers_from_file
from .ml_strategy import MLStrategy
from .registry import get_strategy, list_strategies, register_strategy, get_strategy_choices

__all__ = [
    # Base
    'BaseStrategy',
    # Strategies
    'MomentumStrategy', 
    'MLStrategy',
    # Registry
    'get_strategy', 
    'list_strategies', 
    'register_strategy',
    'get_strategy_choices',
    # Utilities
    'RebalanceOrder', 
    'load_tickers_from_file',
]
