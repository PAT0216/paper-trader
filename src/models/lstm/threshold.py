"""
LSTM Threshold Target Construction

Implements the V4 threshold classification target:
Target = 1 if Return > (mean + k * sigma), else 0

This fixes the "mean trap" problem from regression approaches.
"""

import pandas as pd
import numpy as np


def compute_threshold_target(
    returns: pd.Series,
    sigma_multiplier: float = 2.0,
    lookback: int = 60
) -> pd.Series:
    """
    Compute threshold classification target.
    
    A return exceeds threshold if it's > (rolling_mean + k * rolling_std).
    This identifies statistically significant positive returns.
    
    Args:
        returns: Series of log returns
        sigma_multiplier: Number of standard deviations above mean (default: 2.0)
        lookback: Rolling window for mean/std calculation
    
    Returns:
        Binary series (1 if exceeds threshold, 0 otherwise)
    """
    rolling_mean = returns.rolling(lookback).mean()
    rolling_std = returns.rolling(lookback).std()
    threshold = rolling_mean + sigma_multiplier * rolling_std
    
    return (returns > threshold).astype(int)


def compute_cagr_normalized_return(
    price_start: float,
    price_end: float,
    n_days: int
) -> float:
    """
    Compute CAGR-normalized daily return.
    
    Fixes the linear normalization error:
    WRONG:  return / n_days
    RIGHT:  (1 + return)^(1/n_days) - 1
    
    Args:
        price_start: Starting price
        price_end: Ending price
        n_days: Number of trading days
    
    Returns:
        Daily CAGR-equivalent return
    """
    if n_days <= 0 or price_start <= 0:
        return 0.0
    
    total_return = price_end / price_start - 1
    daily_return = (1 + total_return) ** (1 / n_days) - 1
    return daily_return


def compute_forward_returns(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """
    Compute forward returns (point-in-time safe).
    
    Uses SHIFT(-horizon) to look forward, which is only valid during
    training with known future data. Never use for inference.
    
    Args:
        df: DataFrame with 'Close' column
        horizon: Forward looking period in days
    
    Returns:
        Series of forward log returns
    """
    close = df['Close'] if 'Close' in df.columns else df['close']
    forward_close = close.shift(-horizon)
    forward_return = np.log(forward_close / close)
    return forward_return


def create_pit_target(
    df: pd.DataFrame,
    horizon: int = 1,
    sigma_multiplier: float = 2.0,
    lookback: int = 60
) -> pd.Series:
    """
    Create point-in-time safe target for training.
    
    IMPORTANT: This function looks forward and should ONLY be used
    during training, never during live inference.
    
    Args:
        df: DataFrame with OHLCV data
        horizon: Forward return horizon (days)
        sigma_multiplier: Threshold multiplier
        lookback: Rolling window for stats
    
    Returns:
        Binary target series
    """
    forward_returns = compute_forward_returns(df, horizon)
    target = compute_threshold_target(forward_returns, sigma_multiplier, lookback)
    return target
