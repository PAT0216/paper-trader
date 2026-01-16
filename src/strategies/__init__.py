"""
Strategies Module - Trading Strategy Infrastructure

Provides:
- BaseStrategy: Abstract base class for all strategies
- MomentumStrategy: 12-1 month momentum factor strategy
- MLStrategy: XGBoost ensemble prediction strategy
- LSTMStrategy: LSTM V4 threshold classification strategy
- get_strategy(): Factory function for dynamic loading
- list_strategies(): List available strategy names
"""

from .base import BaseStrategy
from .momentum_strategy import MomentumStrategy
from .ml_strategy import MLStrategy
from .lstm_strategy import LSTMStrategy
from .registry import get_strategy, list_strategies, register_strategy, get_strategy_choices

__all__ = [
    # Base
    'BaseStrategy',
    # Strategies
    'MomentumStrategy', 
    'MLStrategy',
    'LSTMStrategy',
    # Registry
    'get_strategy', 
    'list_strategies', 
    'register_strategy',
    'get_strategy_choices',
]

