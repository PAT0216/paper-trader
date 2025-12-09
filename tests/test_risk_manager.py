"""
Unit tests for Risk Manager

Tests position sizing, constraint validation, VaR calculation, and trade validation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.trading.risk_manager import RiskManager, RiskLimits


@pytest.fixture
def sample_historical_data():
    """Create sample OHLC data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    np.random.seed(42)
    
    closes = 100 * (1 + np.random.randn(100).cumsum() * 0.01)
    
    df = pd.DataFrame({
        'Open': closes * (1 + np.random.randn(100) * 0.005),
        'High': closes * (1 + abs(np.random.randn(100)) * 0.01),
        'Low': closes * (1 - abs(np.random.randn(100)) * 0.01),
        'Close': closes,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    return df


@pytest.fixture
def risk_manager():
    """Create RiskManager with default limits."""
    return RiskManager()


def test_risk_limits_defaults():
    """Test that RiskLimits has sensible defaults."""
    limits = RiskLimits()
    assert limits.max_position_pct == 0.15
    assert limits.max_sector_pct == 0.40
    assert limits.min_cash_buffer == 100.0
    assert limits.max_daily_var_pct == 0.025


def test_calculate_volatility(risk_manager, sample_historical_data):
    """Test volatility calculation."""
    vol = risk_manager._calculate_volatility(sample_historical_data)
    
    assert vol is not None
    assert vol > 0
    assert 0.01 < vol < 2.0  # Reasonable volatility range (1% to 200%)


def test_position_sizing_basic(risk_manager, sample_historical_data):
    """Test basic position sizing without constraints."""
    ticker = "AAPL"
    current_price = 150.0
    available_cash = 10000.0
    portfolio_value = 10000.0
    current_holdings = {}
    current_prices = {}
    
    shares, reason = risk_manager.calculate_position_size(
        ticker, current_price, available_cash, portfolio_value,
        sample_historical_data, current_holdings, current_prices
    )
    
    assert shares >= 0
    assert isinstance(reason, str)
    assert shares * current_price <= available_cash
    
    # Should respect max position limit (15% of portfolio = $1500)
    max_allowed_value = portfolio_value * 0.15
    assert shares * current_price <= max_allowed_value * 1.01  # Small tolerance


def test_position_sizing_insufficient_cash(risk_manager, sample_historical_data):
    """Test position sizing when price exceeds cash."""
    ticker = "AAPL"
    current_price = 150.0
    available_cash = 100.0  # Less than one share
    portfolio_value = 10000.0
    current_holdings = {}
    current_prices = {}
    
    shares, reason = risk_manager.calculate_position_size(
        ticker, current_price, available_cash, portfolio_value,
        sample_historical_data, current_holdings, current_prices
    )
    
    assert shares == 0
    assert "Insufficient cash" in reason


def test_sector_exposure_calculation(risk_manager):
    """Test sector exposure calculation."""
    holdings = {
        'AAPL': 10,  # Technology
        'MSFT': 5,   # Technology
        'JPM': 20    # Financials
    }
    prices = {
        'AAPL': 150.0,
        'MSFT': 300.0,
        'JPM': 100.0
    }
    
    tech_exposure = risk_manager._calculate_sector_exposure(holdings, prices, 'Technology')
    fin_exposure = risk_manager._calculate_sector_exposure(holdings, prices, 'Financials')
    
    assert tech_exposure == 10 * 150 + 5 * 300  # = 3000
    assert fin_exposure == 20 * 100  # = 2000


def test_sector_concentration_limit(risk_manager, sample_historical_data):
    """Test that sector concentration limits are enforced."""
    ticker = "NVDA"  # Technology sector
    current_price = 500.0
    available_cash = 10000.0
    portfolio_value = 10000.0
    
    # Already have large tech exposure
    current_holdings = {
        'AAPL': 10,    # $1500
        'MSFT': 5      # $1500  
    }
    current_prices = {
        'AAPL': 150.0,
        'MSFT': 300.0
    }
    # Tech sector = $3000 / $10000 = 30%
    # Limit is 40%, so can add max $1000 more to tech
    
    shares, reason = risk_manager.calculate_position_size(
        ticker, current_price, available_cash, portfolio_value,
        sample_historical_data, current_holdings, current_prices
    )
    
    # Should be limited by sector constraint
    assert shares * current_price <= (0.40 * portfolio_value - 3000) * 1.01


def test_validate_trade_buy_valid(risk_manager):
    """Test trade validation for a valid BUY."""
    holdings = {'AAPL': 10}
    prices = {'AAPL': 150.0}
    cash = 5000.0
    portfolio_value = 10000.0
    
    is_valid, reason = risk_manager.validate_trade(
        ticker='MSFT',
        action='BUY',
        shares=5,
        price=300.0,  # $1500 total = 15% of portfolio (at limit)
        current_holdings=holdings,
        current_prices=prices,
        cash_balance=cash,
        portfolio_value=portfolio_value
    )
    
    assert is_valid
    assert "validated" in reason.lower()


def test_validate_trade_buy_insufficient_cash(risk_manager):
    """Test trade validation rejects buy with insufficient cash."""
    is_valid, reason = risk_manager.validate_trade(
        ticker='AAPL',
        action='BUY',
        shares=100,
        price=150.0,  # $15,000
        current_holdings={},
        current_prices={},
        cash_balance=1000.0,  # Not enough
        portfolio_value=10000.0
    )
    
    assert not is_valid
    assert "Insufficient cash" in reason


def test_validate_trade_buy_position_too_large(risk_manager):
    """Test trade validation rejects oversized position."""
    is_valid, reason = risk_manager.validate_trade(
        ticker='AAPL',
        action='BUY',
        shares=100,
        price=150.0,  # $15,000 = 150% of portfolio
        current_holdings={},
        current_prices={},
        cash_balance=20000.0,
        portfolio_value=10000.0
    )
    
    assert not is_valid
    assert "Position too large" in reason


def test_validate_trade_sell_valid(risk_manager):
    """Test trade validation for valid SELL."""
    holdings = {'AAPL': 10}
    prices = {'AAPL': 150.0}
    
    is_valid, reason = risk_manager.validate_trade(
        ticker='AAPL',
        action='SELL',
        shares=5,  # Have 10, selling 5
        price=150.0,
        current_holdings=holdings,
        current_prices=prices,
        cash_balance=5000.0,
        portfolio_value=10000.0
    )
    
    assert is_valid


def test_validate_trade_sell_insufficient_shares(risk_manager):
    """Test trade validation rejects selling more than owned."""
    holdings = {'AAPL': 10}
    prices = {'AAPL': 150.0}
    
    is_valid, reason = risk_manager.validate_trade(
        ticker='AAPL',
        action='SELL',
        shares=20,  # Only have 10
        price=150.0,
        current_holdings=holdings,
        current_prices=prices,
        cash_balance=5000.0,
        portfolio_value=10000.0
    )
    
    assert not is_valid
    assert "only own" in reason


def test_sector_exposure_summary(risk_manager):
    """Test sector exposure summary calculation."""
    holdings = {
        'AAPL': 10,   # Tech
        'MSFT': 5,    # Tech
        'JPM': 20,    # Financials
        'UNH': 10     # Healthcare
    }
    prices = {
        'AAPL': 150.0,
        'MSFT': 300.0,
        'JPM': 100.0,
        'UNH': 400.0
    }
    portfolio_value = 10000.0
    
    exposure = risk_manager.get_sector_exposure_summary(holdings, prices, portfolio_value)
    
    assert 'Technology' in exposure
    assert 'Financials' in exposure
    assert 'Healthcare' in exposure
    
    # Technology = 1500 + 1500 = 3000 / 10000 = 30%
    assert abs(exposure['Technology'] - 0.30) < 0.01
    # Financials = 2000 / 10000 = 20%
    assert abs(exposure['Financials'] - 0.20) < 0.01
    # Healthcare = 4000 / 10000 = 40%
    assert abs(exposure['Healthcare'] - 0.40) < 0.01


def test_calculate_portfolio_var(risk_manager, sample_historical_data):
    """Test VaR calculation."""
    holdings = {'AAPL': 10}
    prices = {'AAPL': 100.0}
    historical_data = {'AAPL': sample_historical_data}
    
    var = risk_manager.calculate_portfolio_var(holdings, prices, historical_data)
    
    assert var is not None
    assert var >= 0
    # VaR should be reasonable (e.g., 1-10% of portfolio for 95% confidence)
    portfolio_value = 1000.0
    assert var < portfolio_value * 0.2  # Less than 20% is reasonable


def test_correlation_penalty(risk_manager, sample_historical_data):
    """Test correlation penalty for same-sector holdings."""
    ticker = "AAPL"  # Technology
    current_holdings = {
        'MSFT': 20,  # Already tech-heavy (3000 / 10000 = 30%)
        'NVDA': 10
    }
    current_prices = {
        'MSFT': 100.0,
        'NVDA': 300.0
    }
    portfolio_value = 10000.0
    
    penalty = risk_manager._calculate_correlation_penalty(
        ticker, sample_historical_data, current_holdings, current_prices, portfolio_value
    )
    
    # Should have some penalty since sector is already 30% (above 25% threshold)
    assert penalty >= 0
    assert penalty <= 0.5  # Max penalty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
