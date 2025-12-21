"""
LSTM Feature Engineering

Generates normalized features for LSTM input.
All features are Z-score normalized and clipped to [-3, 3].
"""

import pandas as pd
import numpy as np
from typing import List


# Core features ranked by importance (from permutation analysis)
LSTM_FEATURES = [
    'close_normalized',    # close / SMA(20) - price relative to trend
    'open_log_return',     # log(open / prev_close) - overnight gap
    'momentum_20',         # 20-day momentum
    'sma_10_ratio',        # close / SMA(10)
    'sma_20_ratio',        # close / SMA(20)
    'volatility_20',       # Annualized 20-day volatility
    'volume_ratio',        # volume / SMA(volume, 20)
    'rsi_14',              # RSI normalized
    'macd_normalized',     # MACD / price
    'macd_hist',           # MACD histogram
    'atr_ratio',           # ATR / price
    'bb_position',         # Bollinger Band position [-1, 1]
]


def generate_lstm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate features for LSTM input.
    
    Args:
        df: DataFrame with OHLCV columns (must have Open, High, Low, Close, Volume)
    
    Returns:
        DataFrame with normalized features
    """
    result = pd.DataFrame(index=df.index)
    
    # Ensure column names are consistent
    close = df['Close'] if 'Close' in df.columns else df['close']
    open_price = df['Open'] if 'Open' in df.columns else df['open']
    high = df['High'] if 'High' in df.columns else df['high']
    low = df['Low'] if 'Low' in df.columns else df['low']
    volume = df['Volume'] if 'Volume' in df.columns else df['volume']
    
    # Moving averages
    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    
    # 1. Close normalized to 20-day SMA
    result['close_normalized'] = close / sma_20 - 1
    
    # 2. Open log return (overnight gap)
    result['open_log_return'] = np.log(open_price / close.shift(1))
    
    # 3. 20-day momentum
    result['momentum_20'] = close / close.shift(20) - 1
    
    # 4. SMA ratios
    result['sma_10_ratio'] = close / sma_10 - 1
    result['sma_20_ratio'] = close / sma_20 - 1
    
    # 5. Annualized volatility
    returns = np.log(close / close.shift(1))
    result['volatility_20'] = returns.rolling(20).std() * np.sqrt(252)
    
    # 6. Volume ratio
    vol_sma = volume.rolling(20).mean()
    result['volume_ratio'] = volume / vol_sma - 1
    
    # 7. RSI (14-day)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    result['rsi_14'] = (100 - (100 / (1 + rs))) / 50 - 1  # Normalize to [-1, 1]
    
    # 8. MACD normalized
    ema_12 = close.ewm(span=12).mean()
    ema_26 = close.ewm(span=26).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9).mean()
    result['macd_normalized'] = macd / close
    result['macd_hist'] = (macd - signal) / close
    
    # 9. ATR ratio
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    result['atr_ratio'] = atr / close
    
    # 10. Bollinger Band position
    bb_mid = sma_20
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    result['bb_position'] = (close - bb_mid) / (bb_upper - bb_lower + 1e-10)
    
    # Clip all features to [-3, 3] to handle outliers
    for col in LSTM_FEATURES:
        if col in result.columns:
            result[col] = result[col].clip(-3, 3)
    
    return result[LSTM_FEATURES].dropna()


def create_sequences(
    features_df: pd.DataFrame,
    target: pd.Series,
    sequence_length: int = 60,
    step_size: int = 1
) -> tuple:
    """
    Create sequences for LSTM training.
    
    Args:
        features_df: DataFrame with feature columns
        target: Series with target values
        sequence_length: Length of each sequence
        step_size: Step between sequences (1=overlapping, sequence_length=non-overlapping)
    
    Returns:
        (X, y) numpy arrays
    """
    X, y = [], []
    
    # Align target with features
    aligned_target = target.loc[features_df.index]
    
    for i in range(0, len(features_df) - sequence_length - 1, step_size):
        seq = features_df.iloc[i:i+sequence_length].values
        label = aligned_target.iloc[i + sequence_length]
        
        if not np.isnan(label) and not np.isnan(seq).any():
            X.append(seq)
            y.append(label)
    
    return np.array(X), np.array(y)
