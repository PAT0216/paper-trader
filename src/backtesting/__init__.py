"""
Backtesting module for Paper Trader.

Provides event-driven backtesting with realistic transaction costs,
risk management integration, and comprehensive performance metrics.
"""

from src.backtesting.backtester import (
    Backtester,
    BacktestConfig,
    BacktestPortfolio,
    BacktestTrade,
    BacktestPosition,
    create_ml_signal_generator,
    create_simple_signal_generator
)
from src.backtesting.performance import (
    PerformanceCalculator,
    PerformanceMetrics,
    generate_performance_summary
)
from src.backtesting.costs import (
    TransactionCostModel,
    CostTracker,
    CostConfig
)

__all__ = [
    # Main backtester
    'Backtester',
    'BacktestConfig',
    'BacktestPortfolio',
    'BacktestTrade',
    'BacktestPosition',
    'create_ml_signal_generator',
    'create_simple_signal_generator',
    
    # Performance
    'PerformanceCalculator',
    'PerformanceMetrics',
    'generate_performance_summary',
    
    # Costs
    'TransactionCostModel',
    'CostTracker',
    'CostConfig',
]
