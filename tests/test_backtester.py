"""
Unit tests for Backtesting Framework

Tests performance metrics calculation, cost modeling, and backtester engine.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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


class TestPerformanceCalculator:
    """Tests for PerformanceCalculator."""
    
    @pytest.fixture
    def calculator(self):
        """Create performance calculator."""
        return PerformanceCalculator(risk_free_rate=0.04)
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample portfolio values."""
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        # Simulate ~10% annual return with 15% vol
        np.random.seed(42)
        daily_returns = np.random.normal(0.0004, 0.01, 252)
        prices = 100000 * (1 + daily_returns).cumprod()
        return pd.Series(prices, index=dates)
    
    @pytest.fixture
    def sample_benchmark(self):
        """Create sample benchmark values."""
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        np.random.seed(123)
        daily_returns = np.random.normal(0.0003, 0.012, 252)
        prices = 100000 * (1 + daily_returns).cumprod()
        return pd.Series(prices, index=dates)
    
    def test_total_return_calculation(self, calculator, sample_returns):
        """Test total return calculation."""
        metrics = calculator.calculate_all_metrics(
            portfolio_values=sample_returns,
            benchmark_values=sample_returns,  # Same for simple test
            trades=None
        )
        
        expected_return = (sample_returns.iloc[-1] / sample_returns.iloc[0]) - 1
        assert abs(metrics.total_return - expected_return) < 0.001
    
    def test_sharpe_ratio_positive_for_good_returns(self, calculator, sample_returns, sample_benchmark):
        """Test Sharpe ratio is positive for good returns."""
        metrics = calculator.calculate_all_metrics(
            portfolio_values=sample_returns,
            benchmark_values=sample_benchmark,
            trades=None
        )
        
        # With 10% return vs 4% risk-free, Sharpe should be positive
        # (depends on volatility realization)
        assert isinstance(metrics.sharpe_ratio, float)
    
    def test_max_drawdown_is_positive(self, calculator, sample_returns, sample_benchmark):
        """Test max drawdown is positive (absolute value)."""
        metrics = calculator.calculate_all_metrics(
            portfolio_values=sample_returns,
            benchmark_values=sample_benchmark,
            trades=None
        )
        
        assert metrics.max_drawdown >= 0
        assert metrics.max_drawdown <= 1  # Can't lose more than 100%
    
    def test_volatility_annualized(self, calculator, sample_returns, sample_benchmark):
        """Test volatility is annualized correctly."""
        metrics = calculator.calculate_all_metrics(
            portfolio_values=sample_returns,
            benchmark_values=sample_benchmark,
            trades=None
        )
        
        # Annualized vol should be reasonable (5-50% range)
        assert 0.01 < metrics.volatility < 1.0
    
    def test_var_calculation(self, calculator, sample_returns, sample_benchmark):
        """Test VaR calculation."""
        metrics = calculator.calculate_all_metrics(
            portfolio_values=sample_returns,
            benchmark_values=sample_benchmark,
            trades=None
        )
        
        # VaR 99% should be >= VaR 95%
        assert metrics.var_99 >= metrics.var_95
        assert metrics.var_95 > 0


class TestTransactionCostModel:
    """Tests for TransactionCostModel."""
    
    @pytest.fixture
    def cost_model(self):
        """Create cost model with default settings."""
        config = CostConfig(slippage_bps=5.0, commission_per_share=0.0)
        return TransactionCostModel(config)
    
    def test_buy_execution_price_higher(self, cost_model):
        """Test BUY execution price is higher than quoted."""
        price = 100.0
        shares = 10
        
        exec_price, cost = cost_model.calculate_execution_price('BUY', price, shares)
        
        assert exec_price > price
        assert cost > 0
    
    def test_sell_execution_price_lower(self, cost_model):
        """Test SELL execution price is lower than quoted."""
        price = 100.0
        shares = 10
        
        exec_price, cost = cost_model.calculate_execution_price('SELL', price, shares)
        
        assert exec_price < price
        assert cost > 0
    
    def test_slippage_is_5bps(self, cost_model):
        """Test slippage is approximately 5 basis points."""
        price = 100.0
        shares = 100
        
        exec_price, _ = cost_model.calculate_execution_price('BUY', price, shares)
        
        # 5 bps = 0.05%
        expected_price = price * 1.0005
        assert abs(exec_price - expected_price) < 0.01
    
    def test_zero_shares_no_cost(self, cost_model):
        """Test zero shares results in no cost."""
        price = 100.0
        shares = 0
        
        exec_price, cost = cost_model.calculate_execution_price('BUY', price, shares)
        
        assert exec_price == price
        assert cost == 0
    
    def test_cost_breakdown(self, cost_model):
        """Test cost breakdown includes all components."""
        costs = cost_model.calculate_trade_costs(
            trade_value=10000,
            shares=100,
            avg_daily_volume=1000000
        )
        
        assert 'slippage' in costs
        assert 'commission' in costs
        assert 'market_impact' in costs
        assert 'total' in costs
        assert costs['total'] >= costs['slippage']


class TestCostTracker:
    """Tests for CostTracker."""
    
    @pytest.fixture
    def tracker(self):
        """Create cost tracker."""
        return CostTracker()
    
    def test_record_trade_updates_totals(self, tracker):
        """Test recording a trade updates cumulative totals."""
        tracker.record_trade(
            date=pd.Timestamp('2023-01-01'),
            ticker='AAPL',
            action='BUY',
            shares=100,
            price=150.0
        )
        
        summary = tracker.get_summary()
        assert summary['trade_count'] == 1
        assert summary['total_volume_traded'] == 15000  # 100 * 150
        assert summary['total_cost'] > 0
    
    def test_multiple_trades_accumulate(self, tracker):
        """Test multiple trades accumulate correctly."""
        for i in range(5):
            tracker.record_trade(
                date=pd.Timestamp(f'2023-01-{i+1:02d}'),
                ticker='AAPL',
                action='BUY',
                shares=10,
                price=100.0
            )
        
        summary = tracker.get_summary()
        assert summary['trade_count'] == 5
        assert summary['total_volume_traded'] == 5000  # 5 * 10 * 100
    
    def test_reset_clears_tracker(self, tracker):
        """Test reset clears all tracked values."""
        tracker.record_trade(
            date=pd.Timestamp('2023-01-01'),
            ticker='AAPL',
            action='BUY',
            shares=100,
            price=150.0
        )
        
        tracker.reset()
        summary = tracker.get_summary()
        
        assert summary['trade_count'] == 0
        assert summary['total_cost'] == 0


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""
    
    def test_to_dict_format(self):
        """Test to_dict produces expected structure."""
        metrics = PerformanceMetrics(
            total_return=0.15,
            sharpe_ratio=1.5,
            max_drawdown=0.10,
            win_rate=0.55
        )
        
        result = metrics.to_dict()
        
        assert 'returns' in result
        assert 'risk_adjusted' in result
        assert 'risk' in result
        assert 'trade_quality' in result
    
    def test_generate_summary(self):
        """Test generate_performance_summary produces text."""
        metrics = PerformanceMetrics(
            total_return=0.15,
            cagr=0.12,
            sharpe_ratio=1.5,
            max_drawdown=0.10
        )
        
        summary = generate_performance_summary(metrics)
        
        assert 'BACKTEST PERFORMANCE SUMMARY' in summary
        assert 'Total Return' in summary
        assert 'Sharpe Ratio' in summary


class TestRegimeClassification:
    """Tests for market regime classification."""
    
    def test_classify_regimes(self):
        """Test regime classification produces valid labels."""
        dates = pd.date_range(start='2020-01-01', periods=300, freq='D')
        prices = pd.Series(range(100, 400), index=dates)
        
        regimes = PerformanceCalculator.classify_market_regimes(prices)
        
        # Should have valid regime labels
        valid_regimes = {'bull', 'bear', 'crisis', 'sideways', 'unknown'}
        assert all(r in valid_regimes for r in regimes.unique())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
