"""
Momentum-Aligned ML Features (Steve Cohen + Karpathy Edition)

Key insight: The old features (RSI, MACD, BB) are mean-reversion signals.
Using them with "buy top predictions" = buying oversold losers.

New approach:
- MOMENTUM features: 12m/6m/3m returns (what actually predicts)
- QUALITY filters: Avoid distressed momentum
- LONGER HORIZON: 20-day target (1-day is noise)

Academic reference: Jegadeesh & Titman (1993) - momentum persists 3-12 months
"""

import pandas as pd
import numpy as np


def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate momentum-aligned features for ML prediction.
    
    These features align with momentum strategy logic:
    - Winners tend to keep winning (momentum)
    - But avoid overstretched momentum (quality)
    
    Args:
        df: DataFrame with OHLCV columns
        
    Returns:
        DataFrame with momentum features
    """
    df = df.copy()
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # ==========================================================================
    # CORE MOMENTUM (what actually works in quant)
    # ==========================================================================
    
    # 1. Multi-timeframe momentum (the core alpha source)
    df['Mom_12m'] = close.pct_change(252)  # 12-month momentum
    df['Mom_6m'] = close.pct_change(126)   # 6-month momentum
    df['Mom_3m'] = close.pct_change(63)    # 3-month momentum
    df['Mom_1m'] = close.pct_change(21)    # 1-month momentum (skip month)
    
    # 2. Momentum acceleration (is momentum strengthening?)
    df['Mom_Accel'] = df['Mom_3m'] - df['Mom_6m'].shift(63)
    
    # 3. 52-week high proximity (strong momentum signal)
    rolling_high = high.rolling(252).max()
    df['High_52w_Dist'] = close / rolling_high  # 1.0 = at high, 0.8 = 20% below
    
    # 4. Relative momentum (skip last month to avoid reversal)
    df['Mom_12_1'] = (close.shift(21) / close.shift(252)) - 1  # Classic 12-1 factor
    
    # ==========================================================================
    # TREND QUALITY (avoid distressed momentum)
    # ==========================================================================
    
    # 5. Trend consistency (how smooth is the trend?)
    monthly_returns = close.pct_change(21)
    df['Trend_Consistency'] = monthly_returns.rolling(12).apply(
        lambda x: (x > 0).sum() / len(x), raw=True
    )
    
    # 6. Drawdown from peak (momentum with low DD is better)
    df['Drawdown'] = (close - rolling_high) / rolling_high
    
    # 7. Volatility-adjusted momentum (Sharpe-like)
    returns = close.pct_change()
    df['Vol_Adj_Mom'] = df['Mom_6m'] / (returns.rolling(126).std() * np.sqrt(252) + 0.01)
    
    # ==========================================================================
    # VOLUME CONFIRMATION
    # ==========================================================================
    
    # 8. Volume trend (is volume confirming the move?)
    df['Vol_Trend'] = volume.rolling(20).mean() / volume.rolling(60).mean()
    
    # 9. Price-volume divergence (volume should rise with price)
    price_trend = close.pct_change(20)
    vol_trend = volume.pct_change(20)
    df['PV_Divergence'] = price_trend - vol_trend.clip(-1, 1)
    
    # ==========================================================================
    # MEAN-REVERSION FILTERS (to AVOID, not buy)
    # ==========================================================================
    
    # 10. RSI - but inverted interpretation (high RSI = strong momentum, not overbought)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 11. Distance from 200-day MA (trend following, not mean reversion)
    sma_200 = close.rolling(200).mean()
    df['Trend_Dist'] = (close / sma_200) - 1
    
    return df.dropna()


# Feature columns for momentum ML
MOMENTUM_FEATURE_COLUMNS = [
    # Core momentum (5)
    'Mom_12m', 'Mom_6m', 'Mom_3m', 'Mom_1m', 'Mom_12_1',
    # Momentum quality (3)
    'Mom_Accel', 'High_52w_Dist', 'Vol_Adj_Mom',
    # Trend quality (2)
    'Trend_Consistency', 'Drawdown',
    # Volume (2)
    'Vol_Trend', 'PV_Divergence',
    # Trend filters (2)
    'RSI', 'Trend_Dist'
]


def create_momentum_target(df: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """
    Create forward-looking target for ML training.
    
    Uses 20-day horizon (not 1-day) because:
    - 1-day returns are noise (R² ~ 0)
    - 20-day returns have momentum signal
    - Aligns with monthly rebalance frequency
    
    Args:
        df: DataFrame with 'Close' column
        horizon: Days ahead to predict (default: 20)
        
    Returns:
        DataFrame with 'Target' column
    """
    df = df.copy()
    df['Target'] = df['Close'].pct_change(horizon).shift(-horizon)
    return df
