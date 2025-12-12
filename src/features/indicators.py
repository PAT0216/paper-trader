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


# ==================== PHASE 3.5: ENHANCED FEATURES ====================

def compute_obv(close, volume):
    """
    On-Balance Volume (OBV): Cumulative volume in direction of price movement.
    Rising OBV confirms price trend; divergence signals reversal.
    """
    direction = np.sign(close.diff())
    direction.iloc[0] = 0  # First value has no direction
    obv = (direction * volume).cumsum()
    return obv


def compute_obv_momentum(close, volume, period=10):
    """
    OBV Momentum: Rate of change of OBV over period.
    Measures volume conviction behind price moves.
    """
    obv = compute_obv(close, volume)
    return obv.pct_change(period)


def compute_volume_ratio(volume, short_period=5, long_period=20):
    """
    Volume Ratio: Short-term volume vs long-term average.
    >1 = unusual volume, <1 = quiet trading.
    """
    return volume.rolling(short_period).mean() / volume.rolling(long_period).mean()


def compute_vwap_deviation(close, volume, period=20):
    """
    VWAP Deviation: Price distance from Volume-Weighted Average Price.
    Positive = trading above fair value, negative = below.
    """
    typical_price = close  # Simplified; could use (H+L+C)/3
    vwap = (typical_price * volume).rolling(period).sum() / volume.rolling(period).sum()
    return (close / vwap) - 1


def compute_atr(high, low, close, period=14):
    """
    Average True Range (ATR): Volatility measure.
    Higher ATR = more volatile, useful for position sizing.
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def compute_bollinger_pctb(close, window=20, num_std=2):
    """
    Bollinger %B: Position within Bollinger Bands (0-1 scale).
    <0 = below lower band (oversold), >1 = above upper band (overbought).
    """
    upper, lower = compute_bollinger_bands(close, window, num_std)
    pct_b = (close - lower) / (upper - lower)
    return pct_b


def compute_volatility_ratio(close, short_period=10, long_period=60):
    """
    Volatility Ratio: Short-term vol vs long-term vol.
    >1 = volatility expansion, <1 = volatility contraction.
    """
    returns = close.pct_change()
    short_vol = returns.rolling(short_period).std()
    long_vol = returns.rolling(long_period).std()
    return short_vol / long_vol

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
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
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
    
    # ==================== PHASE 3.5: NEW FEATURES ====================
    
    # 5. Volume Features (past data only)
    df['OBV_Momentum'] = compute_obv_momentum(close, volume, period=10)
    df['Volume_Ratio'] = compute_volume_ratio(volume, short_period=5, long_period=20)
    df['VWAP_Dev'] = compute_vwap_deviation(close, volume, period=20)
    
    # 6. Enhanced Volatility Features (past data only)
    df['ATR_14'] = compute_atr(high, low, close, period=14)
    df['ATR_Pct'] = df['ATR_14'] / close  # ATR as % of price
    df['BB_PctB'] = compute_bollinger_pctb(close, window=20, num_std=2)
    df['Vol_Ratio'] = compute_volatility_ratio(close, short_period=10, long_period=60)
    
    # ==================== END PHASE 3.5 ====================
    
    # Target (ONLY if requested - keeps feature generation separate)
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
# Phase 4: Expanded to 22 features with Fama-French fundamentals
FEATURE_COLUMNS = [
    # Momentum (4)
    'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
    # Trend (4)
    'BB_Width', 'Dist_SMA50', 'Dist_SMA200',
    'Return_1d', 'Return_5d',
    # Volume (3) - Phase 3.5
    'OBV_Momentum', 'Volume_Ratio', 'VWAP_Dev',
    # Volatility (4) - Phase 3.5
    'ATR_Pct', 'BB_PctB', 'Vol_Ratio',
    # ===== FUNDAMENTAL FACTORS (7) - Phase 4 =====
    # Fama-French 5-Factor Model + extras
    'size_score',              # SMB: Small Minus Big (market cap)
    'pe_score',                # HML: Value (P/E ratio)
    'pb_score',                # HML: Value (P/B ratio)
    'roe_score',               # RMW: Profitability (ROE)
    'margin_score',            # RMW: Profitability (margins)
    'earnings_growth_score',   # Growth (earnings)
    'composite_score',         # Combined quality score
]

# Technical-only features (for backwards compatibility)
TECHNICAL_COLUMNS = [
    'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
    'BB_Width', 'Dist_SMA50', 'Dist_SMA200',
    'Return_1d', 'Return_5d',
    'OBV_Momentum', 'Volume_Ratio', 'VWAP_Dev',
    'ATR_Pct', 'BB_PctB', 'Vol_Ratio',
]

# Fundamental factor columns (from fundamentals.py)
FUNDAMENTAL_COLUMNS = [
    'size_score', 'pe_score', 'pb_score', 
    'roe_score', 'margin_score', 'earnings_growth_score',
    'composite_score',
]
