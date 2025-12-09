"""
Unit tests for Data Validator

Tests data quality checks including missing values, outliers, OHLC validity, and freshness.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.validator import DataValidator, ValidationResult


@pytest.fixture
def sample_valid_data():
    """Create sample valid OHLC data."""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    np.random.seed(42)
    
    closes = 100 * (1 + np.random.randn(100).cumsum() * 0.01)
    
    df = pd.DataFrame({
        'Open': closes * 0.99,
        'High': closes * 1.02,
        'Low': closes * 0.98,
        'Close': closes,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    return df


@pytest.fixture
def validator():
    """Create DataValidator with default settings."""
    return DataValidator()


def test_validation_result_creation():
    """Test ValidationResult dataclass."""
    result = ValidationResult(
        is_valid=True,
        errors=[],
        warnings=["Minor warning"]
    )
    
    assert result.is_valid
    assert len(result.errors) == 0
    assert len(result.warnings) == 1


def test_validate_valid_dataframe(validator, sample_valid_data):
    """Test validation of clean, valid data."""
    result = validator.validate_dataframe(sample_valid_data, "AAPL")
    
    assert result.is_valid
    # May have warnings about data freshness, but should be valid
    assert len(result.errors) == 0


def test_validate_empty_dataframe(validator):
    """Test validation rejects empty DataFrame."""
    df = pd.DataFrame()
    result = validator.validate_dataframe(df, "AAPL")
    
    assert not result.is_valid
    assert any("empty" in error.lower() for error in result.errors)


def test_validate_missing_columns(validator, sample_valid_data):
    """Test validation detects missing required columns."""
    df = sample_valid_data.drop(columns=['Volume'])
    result = validator.validate_dataframe(df, "AAPL")
    
    assert not result.is_valid
    assert any("missing required columns" in error.lower() for error in result.errors)


def test_check_price_validity_negative_prices(validator):
    """Test detection of negative prices."""
    dates = pd.date_range(end=datetime.now(), periods=10, freq='D')
    df = pd.DataFrame({
        'Open': [100, 101, -5, 102, 103, 104, 105, 106, 107, 108],  # Negative price
        'High': [105] * 10,
        'Low': [95] * 10,
        'Close': [100] * 10,
        'Volume': [1000000] * 10
    }, index=dates)
    
    errors = validator._check_price_validity(df, "TEST")
    
    assert len(errors) > 0
    assert any("non-positive" in error.lower() for error in errors)


def test_check_ohlc_relationships_invalid(validator):
    """Test detection of invalid OHLC relationships."""
    dates = pd.date_range(end=datetime.now(), periods=5, freq='D')
    df = pd.DataFrame({
        'Open': [100, 100, 100, 100, 100],
        'High': [105, 95, 105, 105, 105],  # High < Low on second row
        'Low': [95, 100, 95, 95, 95],
        'Close': [100, 100, 100, 100, 100],
        'Volume': [1000000] * 5
    }, index=dates)
    
    errors = validator._check_ohlc_relationships(df, "TEST")
    
    assert len(errors) > 0
    assert any("high < low" in error.lower() for error in errors)


def test_check_missing_values(validator):
    """Test detection of missing values."""
    dates = pd.date_range(end=datetime.now(), periods=10, freq='D')
    df = pd.DataFrame({
        'Open': [100] * 10,
        'High': [105] * 10,
        'Low': [95] * 10,
        'Close': [100, np.nan, 100, 100, 100, 100, 100, 100, 100, 100],  # One missing
        'Volume': [1000000] * 10
    }, index=dates)
    
    warning = validator._check_missing_values(df, "TEST")
    
    assert warning is not None
    assert "close" in warning.lower()


def test_check_outliers_extreme_returns(validator):
    """Test detection of outlier returns."""
    dates = pd.date_range(end=datetime.now(), periods=10, freq='D')
    # Create price series with one extreme jump (100 to 500 = 400% return)
    closes = [100, 100, 100, 500, 100, 100, 100, 100, 100, 100]
    df = pd.DataFrame({
        'Open': closes,
        'High': [c * 1.01 for c in closes],
        'Low': [c * 0.99 for c in closes],
        'Close': closes,
        'Volume': [1000000] * 10
    }, index=dates)
    
    warnings = validator._check_outliers(df, "TEST")
    
    assert len(warnings) > 0
    # Should detect extreme return or statistical outlier


def test_check_volume_zero_volume(validator):
    """Test detection of zero volume days."""
    dates = pd.date_range(end=datetime.now(), periods=10, freq='D')
    df = pd.DataFrame({
        'Open': [100] * 10,
        'High': [105] * 10,
        'Low': [95] * 10,
        'Close': [100] * 10,
        'Volume': [1000000, 0, 1000000, 0, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000]
    }, index=dates)
    
    warnings = validator._check_volume(df, "TEST")
    
    assert len(warnings) > 0
    assert any("zero volume" in warning.lower() for warning in warnings)


def test_check_duplicates(validator):
    """Test detection of duplicate dates."""
    dates = pd.date_range(end=datetime.now(), periods=9, freq='D')
    # Add duplicate date
    dates = dates.append(pd.DatetimeIndex([dates[0]]))
    
    df = pd.DataFrame({
        'Open': [100] * 10,
        'High': [105] * 10,
        'Low': [95] * 10,
        'Close': [100] * 10,
        'Volume': [1000000] * 10
    }, index=dates)
    
    error = validator._check_duplicates(df, "TEST")
    
    assert error is not None
    assert "duplicate" in error.lower()


def test_validate_data_dict(validator, sample_valid_data):
    """Test validation of multiple tickers."""
    # Create second invalid dataframe
    invalid_df = pd.DataFrame({
        'Open': [100, -5],  # Negative price
        'High': [105, 105],
        'Low': [95, 95],
        'Close': [100, 100],
        'Volume': [1000000, 1000000]
    }, index=pd.date_range(end=datetime.now(), periods=2, freq='D'))
    
    data_dict = {
        'AAPL': sample_valid_data,
        'BAD': invalid_df
    }
    
    results = validator.validate_data_dict(data_dict)
    
    assert len(results) == 2
    assert 'AAPL' in results
    assert 'BAD' in results
    assert results['AAPL'].is_valid or len(results['AAPL'].errors) == 0  # May have warnings
    assert not results['BAD'].is_valid  # Should fail due to negative price


def test_data_freshness_old_data(validator):
    """Test detection of stale data."""
    # Create data from 1 week ago
    old_date = datetime.now() - timedelta(days=7)
    dates = pd.date_range(end=old_date, periods=100, freq='D')
    
    df = pd.DataFrame({
        'Open': [100] * 100,
        'High': [105] * 100,
        'Low': [95] * 100,
        'Close': [100] * 100,
        'Volume': [1000000] * 100
    }, index=dates)
    
    warning = validator._check_data_freshness(df, "TEST")
    
    assert warning is not None
    assert "stale" in warning.lower()


def test_validation_edge_case_single_row(validator):
    """Test validation with minimal data (single row)."""
    df = pd.DataFrame({
        'Open': [100],
        'High': [105],
        'Low': [95],
        'Close': [100],
        'Volume': [1000000]
    }, index=pd.date_range(end=datetime.now(), periods=1, freq='D'))
    
    result = validator.validate_dataframe(df, "TEST")
    
    # Single row should be valid structure-wise
    # (though may have warnings about insufficient data for some checks)
    assert isinstance(result, ValidationResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
