"""
Behavioral Alpha Factors (40 Factors)

Based on research paper: "Dual-Task MLP on Behavioral Alpha Factors"

Groups:
1. Momentum & Herding (12 factors): Capture trend-following behavior
2. Volume-Price Divergence (9 factors): Detect conviction and reversals
3. Oversold Reversals (12 factors): Bottom detection signals
4. Advanced Combinations (7 factors): Multi-indicator products
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute base technical indicators needed for alpha factors.
    Input: DataFrame with OHLCV columns
    """
    # Make a complete copy first
    df = df.copy()
    
    # Handle MultiIndex columns from yfinance FIRST before any operations
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Ensure we have required columns
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Moving Averages
    df['ma5'] = df['Close'].rolling(5).mean()
    df['ma20'] = df['Close'].rolling(20).mean()
    df['ma200'] = df['Close'].rolling(200).mean()
    df['ma5_volume'] = df['Volume'].rolling(5).mean()
    
    # VWAP (20-day rolling)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['vwap'] = (typical_price * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
    
    # MACD (12, 26, 9)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['macd_diff'] = macd_line - signal_line  # Histogram
    
    # RSI (14-period)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20, 2)
    df['boll_mid'] = df['Close'].rolling(20).mean()
    boll_std = df['Close'].rolling(20).std()
    df['boll_upper'] = df['boll_mid'] + 2 * boll_std
    df['boll_lower'] = df['boll_mid'] - 2 * boll_std
    
    # Volatility
    df['volatility_10d'] = df['Close'].rolling(10).std()
    
    return df


def compute_alpha_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 40 behavioral alpha factors.
    
    Args:
        df: DataFrame with OHLCV columns
        
    Returns:
        DataFrame with 40 alpha factor columns
    """
    # First compute base indicators
    df = compute_technical_indicators(df)
    
    factors = pd.DataFrame(index=df.index)
    
    # Helper for price range
    price_range = df['High'] - df['Low'] + 0.001
    
    # ========== Group 1: Momentum & Herding (12 factors) ==========
    
    # 1. Open down signal
    factors['alpha_signal_open_down'] = (df['Open'] < df['Open'].shift(1)).astype(int)
    
    # 2. Kline body strength
    factors['alpha_kline_body_strength'] = (df['Close'] - df['Open']) / price_range
    
    # 3. MACD × RSI product
    factors['alpha_macd_rsi_product'] = df['macd_diff'] * df['rsi_14']
    
    # 4. RSI × VWAP deviation rank
    vwap_dev = (df['Close'] - df['vwap']).abs()
    factors['alpha_rsi_times_vwapdev_rank'] = df['rsi_14'] * vwap_dev.rank(pct=True)
    
    # 5. Volume spike × body strength
    vol_spike = df['Volume'] / (df['ma5_volume'] + 1)
    factors['alpha_volspike_times_body'] = vol_spike * factors['alpha_kline_body_strength']
    
    # 6. MA200 diff × RSI
    ma200_diff = df['ma200'] - df['Close']
    factors['alpha_ma200diff_times_rsi'] = ma200_diff.rank(pct=True) * df['rsi_14']
    
    # 7. Short diff × MACD
    short_diff_5 = (df['Close'] - df['ma5']).rank(pct=True)
    short_diff_20 = (df['Close'] - df['ma20']).rank(pct=True)
    factors['alpha_shortdiff_times_macd'] = short_diff_5 * short_diff_20 * df['macd_diff']
    
    # 8. Median deviation × RSI
    median_price = (df['High'] + df['Low']) / 2
    median_dev = (df['Close'] - median_price) / price_range
    factors['alpha_median_dev_times_rsi'] = median_dev * df['rsi_14']
    
    # 9. MACD × low deviation
    low_dev = (df['Close'] - df['Low']) / price_range
    factors['alpha_macd_times_lowdev'] = df['macd_diff'] * low_dev
    
    # 10. MACD × volatility
    factors['alpha_macd_times_volatility'] = df['macd_diff'] * df['volatility_10d']
    
    # 11. RSI bounce strength
    rsi_rising = df['rsi_14'] > df['rsi_14'].shift(1)
    rsi_oversold = df['rsi_14'] < 30
    factors['alpha_rsi_bounce_strength'] = (rsi_rising & rsi_oversold).astype(int) * (df['rsi_14'] - 50)
    
    # 12. Body × RSI
    factors['alpha_body_times_rsi'] = factors['alpha_kline_body_strength'] * df['rsi_14']
    
    # ========== Group 2: Volume-Price Divergence (9 factors) ==========
    
    # 13. Volume spike ratio
    factors['alpha_volume_spike_ratio'] = df['Volume'] / (df['ma5_volume'] + 1)
    
    # 14. Close-VWAP diff rank
    factors['alpha_close_vwap_diff_rank'] = (df['Close'] - df['vwap']).rank(pct=True)
    
    # 15. Close near low
    factors['alpha_close_near_low'] = (df['High'] - df['Close']) / price_range
    
    # 16. Short MA diff rank product
    factors['alpha_short_ma_diff_rank_mul'] = short_diff_5 * short_diff_20
    
    # 17. Volume × MACD
    factors['alpha_volume_times_macd'] = df['Volume'] * df['macd_diff']
    
    # 18. MACD per volume
    factors['alpha_macd_per_volume'] = df['macd_diff'] / (df['Volume'] + 1e-10)
    
    # 19. Close range high rank
    close_range = (df['Close'] - df['Low']) / price_range
    factors['alpha_close_range_high_rank'] = close_range.rank(pct=True)
    
    # 20. Volatility × RSI
    factors['alpha_volatility_times_rsi'] = df['volatility_10d'] * df['rsi_14']
    
    # 21. VWAP dev over MACD
    factors['alpha_vwapdev_over_macd'] = (df['Close'] - df['vwap']).rank(pct=True) / (df['macd_diff'].abs() + 1e-6)
    
    # ========== Group 3: Oversold Reversals (12 factors) ==========
    
    # 22. Close delta (negative = reversal signal)
    factors['alpha_close_delta_1d'] = -1 * df['Close'].diff()
    
    # 23. Momentum 5d negative rank
    momentum_5d = df['Close'] - df['Close'].shift(5)
    factors['alpha_momentum_5d_min_rank'] = -1 * momentum_5d.rank(pct=True)
    
    # 24. MA200 - Close rank
    factors['alpha_ma200_close_diff_rank'] = (df['ma200'] - df['Close']).rank(pct=True)
    
    # 25. Volatility negative rank
    factors['alpha_volatility_10d_rank_neg'] = -1 * df['volatility_10d'].rank(pct=True)
    
    # 26. Reverse RSI × low dev
    factors['alpha_rev_rsi_times_lowdev'] = (100 - df['rsi_14']) * low_dev
    
    # 27. Vol rank × reverse RSI
    vol_rank = df['volatility_10d'].rank(pct=True)
    factors['alpha_volrank_times_revrsi'] = -1 * vol_rank * (100 - df['rsi_14'])
    
    # 28. RSI vs 50
    factors['alpha_rsi_vs_50'] = df['rsi_14'] - 50
    
    # 29. RSI bounce rank
    bounce_signal = (rsi_rising & rsi_oversold).astype(int) * (df['rsi_14'] - 50)
    factors['alpha_rsi_bounce_rank'] = bounce_signal.rank(pct=True)
    
    # 30. MACD cross
    macd_positive = df['macd_diff'] > 0
    macd_was_negative = df['macd_diff'].shift(1) <= 0
    factors['alpha_macd_cross'] = (macd_positive & macd_was_negative).astype(float)
    
    # 31. MACD cross strength
    factors['alpha_macd_cross_strength'] = factors['alpha_macd_cross'] * df['macd_diff']
    
    # 32. Price vs Bollinger lower
    factors['alpha_price_vs_boll'] = (df['Close'] < df['boll_lower']).astype(int) * df['Close']
    
    # 33. Bollinger rebound
    below_lower_yesterday = df['Close'].shift(1) < df['boll_lower']
    above_lower_today = df['Close'] > df['boll_lower']
    factors['alpha_boll_rebound'] = (below_lower_yesterday & above_lower_today).astype(int)
    
    # ========== Group 4: Advanced Combinations (7 factors) ==========
    
    # 34. Close median deviation ratio
    factors['alpha_close_median_dev_ratio'] = median_dev
    
    # 35. MACD rank × RSI
    factors['alpha_macd_rank_times_rsi'] = df['macd_diff'].rank(pct=True) * df['rsi_14']
    
    # 36. MACD rank × price
    factors['alpha_macd_rank_times_price'] = df['macd_diff'].rank(pct=True) * df['Close']
    
    # 37. RSI rank × close
    factors['alpha_rsi_rank_times_close'] = df['rsi_14'].rank(pct=True) * df['Close']
    
    # 38. RSI rise velocity
    factors['alpha_rsi_rise_velocity'] = df['rsi_14'] - df['rsi_14'].shift(3)
    
    # 39. Bollinger bandwidth
    factors['alpha_boll_bandwidth'] = df['boll_upper'] - df['boll_lower']
    
    # 40. Close-Bollinger mid ratio
    boll_width = df['boll_upper'] - df['boll_lower'] + 1e-6
    factors['alpha_close_boll_mid_ratio'] = (df['Close'] - df['boll_mid']) / boll_width
    
    return factors


# List of all factor names for reference
ALPHA_FACTOR_NAMES = [
    'alpha_signal_open_down', 'alpha_kline_body_strength', 'alpha_macd_rsi_product',
    'alpha_rsi_times_vwapdev_rank', 'alpha_volspike_times_body', 'alpha_ma200diff_times_rsi',
    'alpha_shortdiff_times_macd', 'alpha_median_dev_times_rsi', 'alpha_macd_times_lowdev',
    'alpha_macd_times_volatility', 'alpha_rsi_bounce_strength', 'alpha_body_times_rsi',
    'alpha_volume_spike_ratio', 'alpha_close_vwap_diff_rank', 'alpha_close_near_low',
    'alpha_short_ma_diff_rank_mul', 'alpha_volume_times_macd', 'alpha_macd_per_volume',
    'alpha_close_range_high_rank', 'alpha_volatility_times_rsi', 'alpha_vwapdev_over_macd',
    'alpha_close_delta_1d', 'alpha_momentum_5d_min_rank', 'alpha_ma200_close_diff_rank',
    'alpha_volatility_10d_rank_neg', 'alpha_rev_rsi_times_lowdev', 'alpha_volrank_times_revrsi',
    'alpha_rsi_vs_50', 'alpha_rsi_bounce_rank', 'alpha_macd_cross',
    'alpha_macd_cross_strength', 'alpha_price_vs_boll', 'alpha_boll_rebound',
    'alpha_close_median_dev_ratio', 'alpha_macd_rank_times_rsi', 'alpha_macd_rank_times_price',
    'alpha_rsi_rank_times_close', 'alpha_rsi_rise_velocity', 'alpha_boll_bandwidth',
    'alpha_close_boll_mid_ratio'
]


def prepare_training_data(data_dict: Dict[str, pd.DataFrame], 
                          forward_days: int = 5,
                          min_data_points: int = 250) -> tuple:
    """
    Prepare training data from multiple stocks.
    
    Args:
        data_dict: {ticker: OHLCV DataFrame}
        forward_days: Days ahead for target return
        min_data_points: Minimum data points required per stock
        
    Returns:
        (X, y_reg, y_clf, dates, tickers)
    """
    all_X = []
    all_y_reg = []
    all_y_clf = []
    all_dates = []
    all_tickers = []
    
    for ticker, df in data_dict.items():
        if len(df) < min_data_points + forward_days:
            continue
        
        try:
            # Compute factors
            factors = compute_alpha_factors(df)
            
            # Handle MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Compute forward returns
            forward_ret = df['Close'].shift(-forward_days) / df['Close'] - 1
            
            # Create classification label (1 = up, 0 = down)
            clf_label = (forward_ret > 0).astype(int)
            
            # Merge and drop NaN
            combined = factors.copy()
            combined['y_reg'] = forward_ret
            combined['y_clf'] = clf_label
            combined = combined.dropna()
            
            if len(combined) < 50:
                continue
            
            X = combined[ALPHA_FACTOR_NAMES].values
            y_reg = combined['y_reg'].values
            y_clf = combined['y_clf'].values
            
            # Clip extreme returns
            y_reg = np.clip(y_reg, -0.30, 0.30)
            
            all_X.append(X)
            all_y_reg.append(y_reg)
            all_y_clf.append(y_clf)
            all_dates.extend(combined.index.tolist())
            all_tickers.extend([ticker] * len(combined))
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    
    if not all_X:
        return None, None, None, None, None
    
    X = np.vstack(all_X)
    y_reg = np.concatenate(all_y_reg)
    y_clf = np.concatenate(all_y_clf)
    
    return X, y_reg, y_clf, all_dates, all_tickers


def normalize_features_zscore(X: np.ndarray) -> np.ndarray:
    """Z-score normalize features (per-column)."""
    mean = np.nanmean(X, axis=0, keepdims=True)
    std = np.nanstd(X, axis=0, keepdims=True) + 1e-8
    return (X - mean) / std


if __name__ == "__main__":
    print(f"Behavioral Alpha Factors Module")
    print(f"Total factors: {len(ALPHA_FACTOR_NAMES)}")
    print(f"Groups: Momentum(12), Volume-Price(9), Reversals(12), Advanced(7)")
