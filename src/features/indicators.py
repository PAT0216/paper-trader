import pandas as pd
import numpy as np

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_bollinger_bands(series, window=20, num_std=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper = rolling_mean + (rolling_std * num_std)
    lower = rolling_mean - (rolling_std * num_std)
    return upper, lower

def generate_features(df, include_target=False):
    """
    Generates technical indicator features from OHLCV data.
    
    IMPORTANT: This function ONLY uses data available at time t.
    No look-ahead bias - all features are calculated using past data only.
    
    Args:
        df: DataFrame with OHLCV columns
        include_target: If True, also creates target column (only for training)
        
    Returns:
        DataFrame with feature columns added
    """
    df = df.copy()
    close = df['Close']
    
    # 1. Momentum Indicators (use only past data)
    df['RSI'] = compute_rsi(close)
    df['MACD'], df['MACD_signal'] = compute_macd(close)
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # 2. Volatility (past data only)
    df['BB_Upper'], df['BB_Lower'] = compute_bollinger_bands(close)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / close
    
    # 3. Trend (SMAs use only past data)
    df['SMA_50'] = close.rolling(window=50).mean()
    df['SMA_200'] = close.rolling(window=200).mean()
    df['Dist_SMA50'] = close / df['SMA_50'] - 1
    df['Dist_SMA200'] = close / df['SMA_200'] - 1
    
    # 4. Returns (use only past data - pct_change looks backward)
    df['Return_1d'] = close.pct_change()
    df['Return_5d'] = close.pct_change(5)
    
    # 5. Target (ONLY if requested - keeps feature generation separate)
    # This should be called separately during training to avoid leakage
    if include_target:
        df = create_target(df, target_type='regression')
    
    return df.dropna()


def create_target(df, target_type='regression', horizon=1):
    """
    Creates target variable for ML training.
    
    IMPORTANT: Call this function ONLY in the training pipeline,
    AFTER splitting data into train/test to prevent look-ahead bias.
    
    Args:
        df: DataFrame with 'Close' column
        target_type: 'regression' (predict return) or 'classification' (predict direction)
        horizon: Number of days ahead to predict (default: 1 day)
        
    Returns:
        DataFrame with 'Target' column added
    """
    df = df.copy()
    close = df['Close']
    
    if target_type == 'regression':
        # Predict next-day return (percentage)
        # shift(-horizon) looks into the future - ONLY use in training data
        df['Target'] = close.pct_change().shift(-horizon)
    else:
        # Binary classification: 1 if next close > current close
        df['Target'] = (close.shift(-horizon) > close).astype(int)
    
    return df


# Feature column list - use this for consistency
FEATURE_COLUMNS = [
    'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
    'BB_Width', 'Dist_SMA50', 'Dist_SMA200',
    'Return_1d', 'Return_5d'
]
