"""
Strategy Registry - Factory Pattern for Dynamic Strategy Loading

Provides:
- STRATEGIES: Dictionary of registered strategy classes
- get_strategy(name): Factory function to instantiate by name
- list_strategies(): List all available strategy names
- register_strategy(name, cls): Register a new strategy at runtime

Usage:
    from src.strategies import get_strategy, list_strategies
    
    # Get strategy by name
    strategy = get_strategy("momentum")
    
    # List available strategies
    print(list_strategies())  # ['momentum', 'ml']
    
    # Use strategy
    signals = strategy.generate_signals(data_dict)
"""

from typing import Dict, Type, List

from src.strategies.base import BaseStrategy


# ==================== Strategy Registry ====================

# Import strategies (lazy to avoid circular imports)
def _get_strategies() -> Dict[str, Type[BaseStrategy]]:
    """Lazy load strategy classes."""
    from src.strategies.momentum_strategy import MomentumStrategy
    from src.strategies.ml_strategy import MLStrategy
    from src.strategies.lstm_strategy import LSTMStrategy
    
    return {
        "momentum": MomentumStrategy,
        "ml": MLStrategy,
        "lstm": LSTMStrategy,
    }


def get_strategy(name: str, **kwargs) -> BaseStrategy:
    """
    Factory function to get strategy instance by name.
    
    Args:
        name: Strategy identifier (e.g., 'momentum', 'ml')
        **kwargs: Additional arguments passed to strategy constructor
        
    Returns:
        Instantiated strategy
        
    Raises:
        ValueError: If strategy name is not registered
    """
    strategies = _get_strategies()
    
    if name not in strategies:
        available = ", ".join(strategies.keys())
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    
    return strategies[name](**kwargs)


def list_strategies() -> List[str]:
    """
    List all registered strategy names.
    
    Returns:
        List of strategy identifiers
    """
    return list(_get_strategies().keys())


def register_strategy(name: str, cls: Type[BaseStrategy]) -> None:
    """
    Register a new strategy at runtime.
    
    Note: For permanent registration, add to _get_strategies().
    
    Args:
        name: Strategy identifier
        cls: Strategy class (must inherit from BaseStrategy)
        
    Raises:
        TypeError: If cls doesn't inherit from BaseStrategy
    """
    if not issubclass(cls, BaseStrategy):
        raise TypeError(f"{cls.__name__} must inherit from BaseStrategy")
    
    # Add to registry (this modifies the cached dict)
    strategies = _get_strategies()
    strategies[name] = cls


# Convenience export for argparse
def get_strategy_choices() -> List[str]:
    """Get strategy names for argparse choices."""
    return list_strategies()
