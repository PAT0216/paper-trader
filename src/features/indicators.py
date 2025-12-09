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

def generate_features(df):
    """
    Adds technical indicators and target variable to the dataframe.
    """
    df = df.copy()
    close = df['Close']
    
    # 1. Momentum
    df['RSI'] = compute_rsi(close)
    df['MACD'], df['MACD_signal'] = compute_macd(close)
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # 2. Volatility
    df['BB_Upper'], df['BB_Lower'] = compute_bollinger_bands(close)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / close
    
    # 3. Trend
    df['SMA_50'] = close.rolling(window=50).mean()
    df['SMA_200'] = close.rolling(window=200).mean()
    df['Dist_SMA50'] = close / df['SMA_50'] - 1
    df['Dist_SMA200'] = close / df['SMA_200'] - 1
    
    # 4. Returns
    df['Return_1d'] = close.pct_change()
    df['Return_5d'] = close.pct_change(5)
    
    # 5. Target: 1 if Next Close > Current Close
    df['Target'] = (close.shift(-1) > close).astype(int)
    
    return df.dropna()
