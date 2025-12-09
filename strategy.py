import pandas as pd

def calculate_indicators(df):
    """
    Adds SMA_50, SMA_200, and RSI_14 to the dataframe.
    """
    df = df.copy()
    
    # Calculate SMAs
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def generate_signal(df, current_holdings):
    """
    Decides whether to BUY, SELL, or HOLD based on the latest data.
    """
    last_row = df.iloc[-1]
    
    sma_50 = last_row['SMA_50']
    sma_200 = last_row['SMA_200']
    rsi = last_row['RSI']
    
    # Check if we have enough data (SMAs might be NaN at start)
    if pd.isna(sma_50) or pd.isna(sma_200) or pd.isna(rsi):
        return 'HOLD'

    # Strategy Logic:
    # BUY if SMA_50 > SMA_200 (Golden Cross) AND RSI < 70 (Not overbought)
    # SELL if SMA_50 < SMA_200 (Death Cross) OR RSI > 30 (logic tweak: usually sell on cross down, or stop loss)
    
    # Simplified Trend Following:
    if sma_50 > sma_200:
        if current_holdings == 0 and rsi < 70:
            return 'BUY'
    elif sma_50 < sma_200:
        if current_holdings > 0:
            return 'SELL'
            
    return 'HOLD'
