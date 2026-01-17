"""
Tests for dual portfolio system.

Tests portfolio isolation and comparison functionality.
"""

import pytest
import pandas as pd
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDualPortfolio:
    """Test dual portfolio isolation."""
    
    def setup_method(self):
        """Create temp directory for test ledgers."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir)
        os.makedirs("data/ledgers", exist_ok=True)
    
    def teardown_method(self):
        """Clean up temp files."""
        os.chdir(self.original_dir)
        shutil.rmtree(self.temp_dir)
    
    def test_portfolio_id_creates_separate_ledger(self):
        """Each portfolio should have its own ledger file."""
        from src.trading.portfolio import Portfolio
        
        # Create two portfolios
        mom_pf = Portfolio(portfolio_id="momentum", start_cash=100000)
        ml_pf = Portfolio(portfolio_id="ml", start_cash=100000)
        
        # Verify separate files
        assert mom_pf.ledger_file == "data/ledgers/ledger_momentum.csv"
        assert ml_pf.ledger_file == "data/ledgers/ledger_ml.csv"
        assert os.path.exists("data/ledgers/ledger_momentum.csv")
        assert os.path.exists("data/ledgers/ledger_ml.csv")
    
    def test_portfolios_are_isolated(self):
        """Trades in one portfolio should not affect another."""
        from src.trading.portfolio import Portfolio
        
        mom_pf = Portfolio(portfolio_id="momentum", start_cash=100000)
        ml_pf = Portfolio(portfolio_id="ml", start_cash=100000)
        
        # Trade in momentum portfolio
        mom_pf.record_trade("AAPL", "BUY", 150.0, 10, strategy="momentum")
        
        # ML portfolio should be unaffected
        assert len(ml_pf.get_holdings()) == 0
        assert "AAPL" not in ml_pf.get_holdings()
        
        # Momentum portfolio should have the trade
        assert "AAPL" in mom_pf.get_holdings()
        assert mom_pf.get_holdings()["AAPL"] == 10
    
    def test_default_portfolio_backwards_compatible(self):
        """Default portfolio should use ledger.csv."""
        from src.trading.portfolio import Portfolio
        
        pf = Portfolio(portfolio_id="default", start_cash=100000)
        assert pf.ledger_file == "data/ledgers/ledger.csv"
    
    def test_start_cash_configurable(self):
        """Each portfolio should respect its start_cash."""
        from src.trading.portfolio import Portfolio
        
        pf1 = Portfolio(portfolio_id="small", start_cash=10000)
        pf2 = Portfolio(portfolio_id="large", start_cash=1000000)
        
        # Check initial deposit
        assert pf1.ledger.iloc[0]['amount'] == 10000
        assert pf2.ledger.iloc[0]['amount'] == 1000000


class TestPortfolioComparison:
    """Test portfolio comparison analytics."""
    
    def setup_method(self):
        """Create temp directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir)
        os.makedirs("data/ledgers", exist_ok=True)
    
    def teardown_method(self):
        """Clean up."""
        os.chdir(self.original_dir)
        shutil.rmtree(self.temp_dir)
    
    def test_load_portfolio_returns_dataframe(self):
        """Should return DataFrame for existing portfolio."""
        from src.trading.portfolio import Portfolio
        from src.analytics.portfolio_comparison import PortfolioComparison
        
        # Create a portfolio
        Portfolio(portfolio_id="test", start_cash=100000)
        
        # Load it
        comparison = PortfolioComparison()
        df = comparison.load_portfolio("test")
        
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_load_nonexistent_portfolio_returns_none(self):
        """Should return None for missing portfolio."""
        from src.analytics.portfolio_comparison import PortfolioComparison
        
        comparison = PortfolioComparison()
        df = comparison.load_portfolio("nonexistent")
        
        assert df is None
    
    def test_compare_returns_dataframe(self):
        """Compare should return DataFrame with metrics."""
        from src.trading.portfolio import Portfolio
        from src.analytics.portfolio_comparison import PortfolioComparison
        
        # Create portfolios
        Portfolio(portfolio_id="a", start_cash=100000)
        Portfolio(portfolio_id="b", start_cash=100000)
        
        comparison = PortfolioComparison()
        result = comparison.compare(["a", "b"])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
