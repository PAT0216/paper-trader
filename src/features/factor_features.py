"""
Factor Investing Features

Based on Fama-French academic research (1993, 2015).

Factors implemented:
1. Value: Book-to-Market ratio (inverse of P/B)
2. Quality: Return on Equity (ROE) + low debt
3. Momentum: 12-month return excluding last month (12-1)

Combined into composite score for stock ranking.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta


def get_fundamental_data(ticker: str) -> Optional[Dict]:
    """
    Fetch fundamental data for a stock.
    
    Returns:
        Dict with P/B, ROE, debt ratio, or None if failed
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Value factors
        pb_ratio = info.get('priceToBook', None)
        
        # Quality factors
        roe = info.get('returnOnEquity', None)  # Already as decimal
        debt_to_equity = info.get('debtToEquity', None)
        profit_margin = info.get('profitMargins', None)
        
        # Return as dict
        return {
            'pb_ratio': pb_ratio,
            'roe': roe,
            'debt_to_equity': debt_to_equity,
            'profit_margin': profit_margin
        }
    except Exception as e:
        return None


def calculate_momentum(df: pd.DataFrame, skip_recent: int = 21) -> float:
    """
    Calculate 12-1 momentum (12-month return excluding most recent month).
    
    This is the standard Fama-French momentum definition.
    Skip recent month to avoid short-term reversal effects.
    
    Args:
        df: OHLCV DataFrame with at least 252 trading days
        skip_recent: Days to skip (default 21 = 1 month)
        
    Returns:
        Momentum as decimal (0.10 = 10% return)
    """
    if len(df) < 252:
        return np.nan
    
    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    
    # Price 12 months ago
    price_12m_ago = df[price_col].iloc[-252]
    
    # Price 1 month ago (excluding recent month)
    price_1m_ago = df[price_col].iloc[-skip_recent]
    
    # 12-1 momentum
    momentum = (price_1m_ago / price_12m_ago) - 1
    
    return momentum


def calculate_value_score(pb_ratio: float) -> float:
    """
    Calculate value score.
    
    Lower P/B = higher value = better score.
    Returns percentile-ready value (higher = more value).
    """
    if pb_ratio is None or pb_ratio <= 0:
        return np.nan
    
    # Invert P/B to get B/P (book-to-price)
    # Higher B/P = more value
    return 1.0 / pb_ratio


def calculate_quality_score(roe: float, debt_to_equity: float, 
                            profit_margin: float) -> float:
    """
    Calculate quality score.
    
    Quality = High ROE + Low Debt + High Margins
    """
    scores = []
    
    # ROE (higher is better)
    if roe is not None and not np.isnan(roe):
        # Clip to reasonable range
        roe_score = min(max(roe, 0), 0.5)  # 0-50%
        scores.append(roe_score)
    
    # Debt (lower is better)
    if debt_to_equity is not None and not np.isnan(debt_to_equity) and debt_to_equity >= 0:
        # Invert: low debt = high score
        debt_score = 1.0 / (1 + debt_to_equity / 100)
        scores.append(debt_score)
    
    # Profit margin (higher is better)
    if profit_margin is not None and not np.isnan(profit_margin):
        margin_score = min(max(profit_margin, 0), 0.5)
        scores.append(margin_score)
    
    if not scores:
        return np.nan
    
    return np.mean(scores)


def calculate_factor_scores(data_dict: Dict[str, pd.DataFrame],
                            use_cache: bool = True) -> pd.DataFrame:
    """
    Calculate all factor scores for a universe of stocks.
    
    Args:
        data_dict: {ticker: OHLCV DataFrame}
        use_cache: Whether to cache fundamental data
        
    Returns:
        DataFrame with columns: [ticker, value, quality, momentum, composite]
    """
    results = []
    
    for ticker, df in data_dict.items():
        # Get fundamental data
        fundamentals = get_fundamental_data(ticker)
        
        if fundamentals is None:
            continue
        
        # Calculate individual factor scores
        value_score = calculate_value_score(fundamentals['pb_ratio'])
        
        quality_score = calculate_quality_score(
            fundamentals['roe'],
            fundamentals['debt_to_equity'],
            fundamentals['profit_margin']
        )
        
        momentum_score = calculate_momentum(df)
        
        results.append({
            'ticker': ticker,
            'value_raw': value_score,
            'quality_raw': quality_score,
            'momentum_raw': momentum_score,
            'pb_ratio': fundamentals['pb_ratio'],
            'roe': fundamentals['roe']
        })
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Convert to percentile ranks (0-1) for each factor
    for factor in ['value', 'quality', 'momentum']:
        raw_col = f'{factor}_raw'
        if raw_col in df.columns:
            # Rank (higher raw score = higher percentile)
            df[factor] = df[raw_col].rank(pct=True, na_option='keep')
    
    # Composite score: equal weight of all factors
    factor_cols = ['value', 'quality', 'momentum']
    available_factors = [f for f in factor_cols if f in df.columns]
    
    if available_factors:
        df['composite'] = df[available_factors].mean(axis=1, skipna=True)
    
    return df


def select_stocks(factor_df: pd.DataFrame, 
                  n_long: int = 10, 
                  n_short: int = 10) -> Tuple[list, list]:
    """
    Select stocks based on composite factor score.
    
    Args:
        factor_df: DataFrame from calculate_factor_scores
        n_long: Number of stocks for long portfolio
        n_short: Number of stocks for short portfolio
        
    Returns:
        (long_tickers, short_tickers)
    """
    # Remove stocks with missing composite score
    valid_df = factor_df.dropna(subset=['composite'])
    
    if len(valid_df) < n_long + n_short:
        n_long = len(valid_df) // 2
        n_short = len(valid_df) // 2
    
    # Sort by composite score
    sorted_df = valid_df.sort_values('composite', ascending=False)
    
    # Top N for long, Bottom N for short
    long_tickers = sorted_df.head(n_long)['ticker'].tolist()
    short_tickers = sorted_df.tail(n_short)['ticker'].tolist()
    
    return long_tickers, short_tickers


# Factor names for reference
FACTOR_NAMES = ['value', 'quality', 'momentum']


if __name__ == "__main__":
    print("Factor Investing Features")
    print("=" * 40)
    print("Factors: Value (B/P), Quality (ROE+), Momentum (12-1)")
    print("Based on: Fama-French (1993, 2015)")
