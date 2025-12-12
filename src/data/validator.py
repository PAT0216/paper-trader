"""
Data Validation Module for Paper Trader

Ensures data quality and integrity by detecting missing values, outliers,
stale data, and potential data errors before they impact trading decisions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of data validation checks."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def __str__(self):
        if self.is_valid:
            msg = "✅ Data validation passed"
            if self.warnings:
                msg += f" ({len(self.warnings)} warnings)"
            return msg
        else:
            return f"❌ Data validation failed: {len(self.errors)} errors, {len(self.warnings)} warnings"


class DataValidator:
    """
    Validates market data quality to prevent garbage-in-garbage-out scenarios.
    """
    
    def __init__(
        self,
        max_missing_pct: float = 0.05,  # Max 5% missing data allowed
        outlier_std_threshold: float = 10.0,  # Flag returns > 10 std deviations
        max_data_age_hours: int = 48,  # Data should be within 48 hours
        min_price: float = 0.01,  # Minimum valid stock price
        max_daily_return: float = 0.50,  # Max 50% single-day move (likely error if higher)
        ohlc_error_tolerance: float = 0.005,  # Allow 0.5% of bars to have OHLC issues
        backtest_mode: bool = False  # Relax checks for backtesting
    ):
        """
        Initialize validator with configurable thresholds.
        
        Args:
            max_missing_pct: Maximum allowed percentage of missing values
            outlier_std_threshold: Standard deviations for outlier detection
            max_data_age_hours: Maximum age of data before considered stale
            min_price: Minimum valid price (detect data errors)
            max_daily_return: Maximum realistic single-day return
            ohlc_error_tolerance: Max % of bars with OHLC issues before rejection
            backtest_mode: If True, skip data freshness checks
        """
        self.max_missing_pct = max_missing_pct
        self.outlier_std_threshold = outlier_std_threshold
        self.max_data_age_hours = max_data_age_hours
        self.min_price = min_price
        self.max_daily_return = max_daily_return
        self.backtest_mode = backtest_mode
        
        # Filter to 2010+ data (matches model training filter)
        # Older data has quality issues we don't care about since model ignores it
        self.min_date = pd.Timestamp('2010-01-01')
        
        # Relax thresholds significantly for backtesting (old data has issues)
        if backtest_mode:
            self.ohlc_error_tolerance = 0.025  # 2.5% for old historical data
            self.min_price = 0.0001  # Some stocks traded <$1 before splits
        else:
            self.ohlc_error_tolerance = ohlc_error_tolerance

    
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        ticker: str,
        required_columns: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Comprehensive validation of a single ticker's OHLC DataFrame.
        
        Args:
            df: DataFrame with OHLC data
            ticker: Stock ticker symbol (for error messages)
            required_columns: List of required column names
            
        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []
        
        if required_columns is None:
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check 1: DataFrame is not empty
        if df.empty:
            errors.append(f"{ticker}: DataFrame is empty")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # Filter to 2010+ data (matches model training filter)
        # Pre-2010 data has quality issues we don't care about
        if isinstance(df.index, pd.DatetimeIndex):
            df = df[df.index >= self.min_date]
            if df.empty:
                warnings.append(f"{ticker}: No data after {self.min_date.strftime('%Y-%m-%d')}")
                return ValidationResult(is_valid=True, errors=errors, warnings=warnings)
        
        # Check 2: Required columns exist
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"{ticker}: Missing required columns: {missing_cols}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # Check 3: Index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            errors.append(f"{ticker}: Index is not DatetimeIndex")
        
        # Check 4: Data freshness (skip in backtest mode)
        if not self.backtest_mode:
            freshness_check = self._check_data_freshness(df, ticker)
            if freshness_check:
                warnings.append(freshness_check)
        
        # Check 5: Missing values
        missing_check = self._check_missing_values(df, ticker)
        if missing_check:
            if "critical" in missing_check.lower():
                errors.append(missing_check)
            else:
                warnings.append(missing_check)
        
        # Check 6: Price validity (positive, reasonable values)
        price_check = self._check_price_validity(df, ticker)
        if price_check:
            errors.extend(price_check)
        
        # Check 7: OHLC relationship validity (with tolerance)
        ohlc_errors, ohlc_warnings = self._check_ohlc_relationships(df, ticker)
        errors.extend(ohlc_errors)
        warnings.extend(ohlc_warnings)
        
        # Check 8: Outlier detection in returns
        outlier_check = self._check_outliers(df, ticker)
        if outlier_check:
            warnings.extend(outlier_check)
        
        # Check 9: Volume validity
        volume_check = self._check_volume(df, ticker)
        if volume_check:
            warnings.extend(volume_check)
        
        # Check 10: Duplicate dates
        duplicate_check = self._check_duplicates(df, ticker)
        if duplicate_check:
            errors.append(duplicate_check)
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
    
    def validate_data_dict(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, ValidationResult]:
        """
        Validate all tickers in a data dictionary.
        
        Args:
            data_dict: Dict of {ticker: DataFrame}
            
        Returns:
            Dict of {ticker: ValidationResult}
        """
        results = {}
        
        for ticker, df in data_dict.items():
            results[ticker] = self.validate_dataframe(df, ticker)
        
        return results
    
    def _check_data_freshness(self, df: pd.DataFrame, ticker: str) -> Optional[str]:
        """Check if data is recent enough for trading decisions."""
        if df.empty:
            return None
        
        last_date = df.index[-1]
        now = pd.Timestamp.now(tz=last_date.tz if last_date.tz else None)
        
        # Remove timezone for comparison if needed
        if last_date.tz is not None:
            last_date = last_date.tz_localize(None)
        if now.tz is not None:
            now = now.tz_localize(None)
        
        age_hours = (now - last_date).total_seconds() / 3600
        
        if age_hours > self.max_data_age_hours:
            return f"{ticker}: Data is stale ({age_hours:.1f} hours old, last: {last_date.strftime('%Y-%m-%d %H:%M')})"
        
        return None
    
    def _check_missing_values(self, df: pd.DataFrame, ticker: str) -> Optional[str]:
        """Detect missing values in critical columns."""
        critical_columns = ['Close', 'Volume']
        
        for col in critical_columns:
            if col not in df.columns:
                continue
            
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_pct = missing_count / len(df)
                
                if missing_pct > self.max_missing_pct:
                    return f"{ticker}: {col} has {missing_pct:.1%} missing values (CRITICAL)"
                else:
                    return f"{ticker}: {col} has {missing_count} missing values ({missing_pct:.1%})"
        
        return None
    
    def _check_price_validity(self, df: pd.DataFrame, ticker: str) -> List[str]:
        """Check that prices are positive and reasonable."""
        errors = []
        price_columns = ['Open', 'High', 'Low', 'Close']
        
        for col in price_columns:
            if col not in df.columns:
                continue
            
            # Check for non-positive prices
            non_positive = (df[col] <= 0).sum()
            if non_positive > 0:
                errors.append(f"{ticker}: {col} has {non_positive} non-positive values")
            
            # Check for extremely low prices (likely data error)
            too_low = (df[col] < self.min_price).sum()
            if too_low > 0:
                errors.append(f"{ticker}: {col} has {too_low} values below ${self.min_price}")
        
        return errors
    
    def _check_ohlc_relationships(self, df: pd.DataFrame, ticker: str) -> Tuple[List[str], List[str]]:
        """
        Validate OHLC internal consistency with tolerance.
        
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        n_bars = len(df)
        
        if n_bars == 0:
            return errors, warnings
        
        required = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required):
            return errors, warnings
        
        # Count issues
        issues = {
            'High < Low': (df['High'] < df['Low']).sum(),
            'High < Open': (df['High'] < df['Open']).sum(),
            'High < Close': (df['High'] < df['Close']).sum(),
            'Low > Open': (df['Low'] > df['Open']).sum(),
            'Low > Close': (df['Low'] > df['Close']).sum()
        }
        
        total_issues = sum(issues.values())
        issue_pct = total_issues / n_bars
        
        # Only error if exceeds tolerance (default 0.5%)
        if issue_pct > self.ohlc_error_tolerance:
            errors.append(
                f"{ticker}: {issue_pct:.2%} OHLC errors ({total_issues}/{n_bars} bars) exceeds {self.ohlc_error_tolerance:.2%} threshold"
            )
        elif total_issues > 0:
            # Warn but don't reject for minor issues
            warnings.append(
                f"{ticker}: {total_issues} minor OHLC issues ({issue_pct:.3%}) - within tolerance"
            )
        
        return errors, warnings
    
    def _check_outliers(self, df: pd.DataFrame, ticker: str) -> List[str]:
        """Detect outliers in daily returns."""
        warnings = []
        
        if 'Close' not in df.columns or len(df) < 2:
            return warnings
        
        returns = df['Close'].pct_change().dropna()
        
        if len(returns) == 0:
            return warnings
        
        # Statistical outlier detection
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0 or np.isnan(std_return):
            return warnings
        
        z_scores = np.abs((returns - mean_return) / std_return)
        outliers = returns[z_scores > self.outlier_std_threshold]
        
        if len(outliers) > 0:
            outlier_dates = [date.strftime('%Y-%m-%d') for date in outliers.index]
            warnings.append(
                f"{ticker}: {len(outliers)} statistical outliers detected "
                f"(>{self.outlier_std_threshold}σ): {outlier_dates[:3]}..."
            )
        
        # Realistic return check (e.g., >50% in one day is suspicious)
        extreme_returns = returns[np.abs(returns) > self.max_daily_return]
        if len(extreme_returns) > 0:
            extreme_dates = [(date.strftime('%Y-%m-%d'), ret) for date, ret in extreme_returns.items()]
            warnings.append(
                f"{ticker}: {len(extreme_returns)} extreme returns (>{self.max_daily_return:.0%}): "
                f"{extreme_dates[:3]}"
            )
        
        return warnings
    
    def _check_volume(self, df: pd.DataFrame, ticker: str) -> List[str]:
        """Validate volume data."""
        warnings = []
        
        if 'Volume' not in df.columns:
            return warnings
        
        # Check for zero volume (illiquid or data error)
        zero_volume = (df['Volume'] == 0).sum()
        if zero_volume > 0:
            zero_pct = zero_volume / len(df)
            if zero_pct > 0.01:  # >1% of days have zero volume
                warnings.append(
                    f"{ticker}: {zero_volume} days with zero volume ({zero_pct:.1%})"
                )
        
        # Check for negative volume (data error)
        negative_volume = (df['Volume'] < 0).sum()
        if negative_volume > 0:
            warnings.append(f"{ticker}: {negative_volume} days with negative volume (data error)")
        
        return warnings
    
    def _check_duplicates(self, df: pd.DataFrame, ticker: str) -> Optional[str]:
        """Check for duplicate date entries."""
        if df.index.duplicated().sum() > 0:
            duplicate_count = df.index.duplicated().sum()
            return f"{ticker}: {duplicate_count} duplicate date entries found"
        
        return None
    
    def print_validation_summary(self, results: Dict[str, ValidationResult]) -> None:
        """
        Print a summary of validation results for all tickers.
        
        Args:
            results: Dict of {ticker: ValidationResult}
        """
        total_tickers = len(results)
        valid_tickers = sum(1 for r in results.values() if r.is_valid)
        invalid_tickers = total_tickers - valid_tickers
        
        print("\n" + "="*60)
        print("DATA VALIDATION SUMMARY")
        print("="*60)
        print(f"Total Tickers: {total_tickers}")
        print(f"✅ Valid: {valid_tickers}")
        print(f"❌ Invalid: {invalid_tickers}")
        print("="*60)
        
        # Print detailed results for invalid tickers
        if invalid_tickers > 0:
            print("\n⚠️  INVALID TICKERS:")
            for ticker, result in results.items():
                if not result.is_valid:
                    print(f"\n{ticker}:")
                    for error in result.errors:
                        print(f"  ❌ {error}")
                    for warning in result.warnings:
                        print(f"  ⚠️  {warning}")
        
        # Print warnings for valid tickers
        tickers_with_warnings = [
            (ticker, result) for ticker, result in results.items()
            if result.is_valid and result.warnings
        ]
        
        if tickers_with_warnings:
            print(f"\n⚠️  {len(tickers_with_warnings)} tickers have warnings:")
            for ticker, result in tickers_with_warnings:
                print(f"\n{ticker}:")
                for warning in result.warnings:
                    print(f"  ⚠️  {warning}")
