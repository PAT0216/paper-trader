#!/usr/bin/env python
"""
Dual-Task MLP Strategy Backtest

Extensive testing of the Dual-Task MLP strategy with:
- Multiple time periods (Zhang paper timeframe + walk-forward)
- Various portfolio sizes (5, 10, 20)
- Transaction cost sensitivity
- IC/IR calculation
- Comparison with buy-and-hold
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

from src.data.cache import DataCache
from src.data.universe import fetch_sp500_tickers
from src.features.behavioral_factors import (
    compute_alpha_factors, 
    ALPHA_FACTOR_NAMES,
    normalize_features_zscore
)
from src.models.dual_task_mlp import DualTaskMLP, DualTaskTrainer, compute_ic

# ============================================================================
# Configuration
# ============================================================================

# Time periods (matching Zhang paper)
TRAIN_START = '2015-01-01'
TRAIN_END = '2017-12-31'
VAL_START = '2018-01-01'
VAL_END = '2018-12-31'
TEST_START = '2019-01-01'
TEST_END = '2024-12-31'

# Strategy parameters
FORWARD_DAYS = 5  # 5-day forward return prediction
PORTFOLIO_SIZES = [5, 10, 20]  # Number of stocks to long/short
TRANSACTION_COST = 0.0005  # 0.05% per half-turn

# Model parameters
HIDDEN1 = 64
HIDDEN2 = 32
DROPOUT = 0.1
LEARNING_RATE = 5e-4
EPOCHS = 100
BATCH_SIZE = 2048


def load_data(tickers: list, start_date: str, end_date: str, 
              cache: DataCache) -> dict:
    """Load OHLCV data for all tickers."""
    data_dict = {}
    for ticker in tickers:
        df = cache.get_price_data(ticker, start_date, end_date)
        if df is not None and len(df) > 250:
            data_dict[ticker] = df
    return data_dict


def prepare_factors_for_date(data_dict: dict, target_date: str, 
                             lookback_days: int = 220) -> tuple:
    """
    Prepare factors for all stocks on a specific date.
    
    Returns:
        (X, tickers, next_returns) where X is (n_stocks, 40)
    """
    X_list = []
    tickers_list = []
    next_returns = {}
    
    for ticker, df in data_dict.items():
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Get data up to target date
        df_to_date = df[df.index <= target_date]
        if len(df_to_date) < lookback_days:
            continue
        
        try:
            # Compute factors
            factors = compute_alpha_factors(df_to_date)
            factors = factors.dropna()
            
            if len(factors) == 0:
                continue
            
            # Get latest factors
            latest_factors = factors[ALPHA_FACTOR_NAMES].iloc[-1].values
            
            # Handle any remaining NaN
            if np.any(np.isnan(latest_factors)):
                continue
            
            X_list.append(latest_factors)
            tickers_list.append(ticker)
            
            # Get next-day return for signal evaluation
            df_after = df[df.index > target_date]
            if len(df_after) > 0:
                price_col = 'Close'
                curr = df_to_date[price_col].iloc[-1]
                next_val = df_after[price_col].iloc[0]
                next_returns[ticker] = (next_val - curr) / curr
        
        except Exception as e:
            continue
    
    if not X_list:
        return None, None, None
    
    X = np.array(X_list)
    return X, tickers_list, next_returns


def prepare_training_set(data_dict: dict, start_date: str, end_date: str) -> tuple:
    """
    Prepare full training dataset.
    """
    all_X = []
    all_y_reg = []
    all_y_clf = []
    
    for ticker, df in data_dict.items():
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        try:
            # Compute factors
            factors = compute_alpha_factors(df)
            
            # Compute forward returns
            forward_ret = df['Close'].shift(-FORWARD_DAYS) / df['Close'] - 1
            clf_label = (forward_ret > 0).astype(int)
            
            # Merge and filter by date
            combined = factors.copy()
            combined['y_reg'] = forward_ret
            combined['y_clf'] = clf_label
            combined = combined.dropna()
            combined = combined[(combined.index >= start_date) & (combined.index <= end_date)]
            
            if len(combined) < 50:
                continue
            
            X = combined[ALPHA_FACTOR_NAMES].values
            y_reg = np.clip(combined['y_reg'].values, -0.30, 0.30)
            y_clf = combined['y_clf'].values
            
            all_X.append(X)
            all_y_reg.append(y_reg)
            all_y_clf.append(y_clf)
        
        except Exception as e:
            continue
    
    if not all_X:
        return None, None, None
    
    X = np.vstack(all_X)
    y_reg = np.concatenate(all_y_reg)
    y_clf = np.concatenate(all_y_clf)
    
    return X, y_reg, y_clf


def run_backtest(trainer: DualTaskTrainer, data_dict: dict, 
                 start_date: str, end_date: str, 
                 n_top: int = 10, tx_cost: float = 0.0005) -> dict:
    """
    Run daily backtest.
    
    Returns:
        Dictionary with performance metrics
    """
    trading_dates = pd.bdate_range(start_date, end_date)
    
    daily_returns = []
    daily_ics = []
    
    for trade_date in trading_dates:
        date_str = trade_date.strftime('%Y-%m-%d')
        
        # Prepare factors for this date
        X, tickers, next_returns = prepare_factors_for_date(data_dict, date_str)
        
        if X is None or len(X) < 2 * n_top:
            continue
        
        # Normalize
        X = normalize_features_zscore(X)
        
        # Handle any NaN after normalization
        valid_mask = ~np.any(np.isnan(X), axis=1)
        if valid_mask.sum() < 2 * n_top:
            continue
        
        X = X[valid_mask]
        tickers = [t for t, v in zip(tickers, valid_mask) if v]
        
        # Get predictions
        reg_pred, clf_pred, deep_alpha = trainer.predict(X)
        
        # Sort by deep_alpha
        sorted_idx = np.argsort(deep_alpha)[::-1]  # Descending
        
        # Long top n, short bottom n
        long_tickers = [tickers[i] for i in sorted_idx[:n_top]]
        short_tickers = [tickers[i] for i in sorted_idx[-n_top:]]
        
        # Calculate returns (only for tickers with next return data)
        long_rets = [next_returns[t] for t in long_tickers if t in next_returns]
        short_rets = [next_returns[t] for t in short_tickers if t in next_returns]
        
        if long_rets and short_rets:
            long_ret = np.mean(long_rets)
            short_ret = np.mean(short_rets)
            
            # Long-short return (50/50 allocation)
            ls_ret = (long_ret - short_ret) / 2
            
            # Apply transaction costs (assume full turnover daily)
            ls_ret_after_cost = ls_ret - (2 * tx_cost)  # Buy + sell
            
            daily_returns.append(ls_ret_after_cost)
            
            # Calculate IC for this day
            actual_rets = np.array([next_returns.get(t, np.nan) for t in tickers])
            valid_ic = ~np.isnan(actual_rets)
            if valid_ic.sum() > 10:
                ic = spearmanr(deep_alpha[valid_ic], actual_rets[valid_ic])[0]
                if not np.isnan(ic):
                    daily_ics.append(ic)
    
    if not daily_returns:
        return None
    
    daily_returns = np.array(daily_returns)
    daily_ics = np.array(daily_ics) if daily_ics else np.array([0])
    
    # Calculate metrics
    cumulative_return = np.prod(1 + daily_returns) - 1
    annualized_return = (1 + cumulative_return) ** (252 / len(daily_returns)) - 1
    volatility = np.std(daily_returns) * np.sqrt(252)
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
    
    # Max drawdown
    equity = np.cumprod(1 + daily_returns)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_drawdown = np.min(drawdowns)
    
    # IC metrics
    mean_ic = np.mean(daily_ics)
    ic_std = np.std(daily_ics)
    ir = mean_ic / ic_std if ic_std > 0 else 0
    
    return {
        'n_days': len(daily_returns),
        'cumulative_return': cumulative_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'mean_ic': mean_ic,
        'ic_std': ic_std,
        'ir': ir,
        'daily_returns': daily_returns
    }


def main():
    print("=" * 70)
    print("DUAL-TASK MLP STRATEGY - EXTENSIVE BACKTEST")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Training: {TRAIN_START} to {TRAIN_END}")
    print(f"  Validation: {VAL_START} to {VAL_END}")
    print(f"  Test: {TEST_START} to {TEST_END}")
    print(f"  Forward Days: {FORWARD_DAYS}")
    print(f"  Portfolio Sizes: {PORTFOLIO_SIZES}")
    print(f"  Transaction Cost: {TRANSACTION_COST*100:.2f}% per half-turn")
    print()
    
    # Load data
    print("📊 Loading S&P 500 data...")
    cache = DataCache()
    tickers = fetch_sp500_tickers()
    
    data_dict = {}
    for t in tickers:
        df = cache.get_price_data(t, '2014-01-01', '2024-12-31')
        if df is not None and len(df) > 500:
            data_dict[t] = df
    
    print(f"   Loaded {len(data_dict)} tickers with sufficient data")
    
    # Prepare training data
    print("\n📈 Preparing training data...")
    X_train, y_reg_train, y_clf_train = prepare_training_set(
        data_dict, TRAIN_START, TRAIN_END
    )
    print(f"   Training samples: {len(X_train):,}")
    
    # Prepare validation data
    X_val, y_reg_val, y_clf_val = prepare_training_set(
        data_dict, VAL_START, VAL_END
    )
    print(f"   Validation samples: {len(X_val):,}")
    
    # Normalize
    X_train = normalize_features_zscore(X_train)
    X_val = normalize_features_zscore(X_val)
    
    # Handle NaN
    train_valid = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_reg_train) | np.isnan(y_clf_train))
    X_train = X_train[train_valid]
    y_reg_train = y_reg_train[train_valid]
    y_clf_train = y_clf_train[train_valid]
    
    val_valid = ~(np.isnan(X_val).any(axis=1) | np.isnan(y_reg_val) | np.isnan(y_clf_val))
    X_val = X_val[val_valid]
    y_reg_val = y_reg_val[val_valid]
    y_clf_val = y_clf_val[val_valid]
    
    print(f"   After cleaning: train={len(X_train):,}, val={len(X_val):,}")
    
    # Train model
    print("\n🚀 Training Dual-Task MLP...")
    model = DualTaskMLP(input_dim=40, hidden1=HIDDEN1, hidden2=HIDDEN2, dropout=DROPOUT)
    trainer = DualTaskTrainer(model, learning_rate=LEARNING_RATE)
    
    history = trainer.fit(
        X_train, y_reg_train, y_clf_train,
        X_val, y_reg_val, y_clf_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=True
    )
    
    print(f"   Final val accuracy: {history['val_acc'][-1]:.3f}")
    
    # ========================================================================
    # Run Backtests
    # ========================================================================
    
    results = {}
    
    # Test on validation period (2018) - same as Zhang paper
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS: VALIDATION PERIOD (2018)")
    print("=" * 70)
    
    for n_top in PORTFOLIO_SIZES:
        result = run_backtest(
            trainer, data_dict, VAL_START, VAL_END, 
            n_top=n_top, tx_cost=TRANSACTION_COST
        )
        if result:
            results[f'val_n{n_top}'] = result
            print(f"\nPortfolio Size: Long {n_top} / Short {n_top}")
            print(f"  Days traded: {result['n_days']}")
            print(f"  Cumulative Return: {result['cumulative_return']*100:.2f}%")
            print(f"  Annualized Return: {result['annualized_return']*100:.2f}%")
            print(f"  Volatility: {result['volatility']*100:.2f}%")
            print(f"  Sharpe Ratio: {result['sharpe']:.3f}")
            print(f"  Max Drawdown: {result['max_drawdown']*100:.2f}%")
            print(f"  Mean IC: {result['mean_ic']:.4f}")
            print(f"  IR: {result['ir']:.4f}")
    
    # Test on out-of-sample period (2019-2024)
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS: OUT-OF-SAMPLE (2019-2024)")
    print("=" * 70)
    
    for n_top in PORTFOLIO_SIZES:
        result = run_backtest(
            trainer, data_dict, TEST_START, TEST_END, 
            n_top=n_top, tx_cost=TRANSACTION_COST
        )
        if result:
            results[f'test_n{n_top}'] = result
            print(f"\nPortfolio Size: Long {n_top} / Short {n_top}")
            print(f"  Days traded: {result['n_days']}")
            print(f"  Cumulative Return: {result['cumulative_return']*100:.2f}%")
            print(f"  Annualized Return: {result['annualized_return']*100:.2f}%")
            print(f"  Volatility: {result['volatility']*100:.2f}%")
            print(f"  Sharpe Ratio: {result['sharpe']:.3f}")
            print(f"  Max Drawdown: {result['max_drawdown']*100:.2f}%")
            print(f"  Mean IC: {result['mean_ic']:.4f}")
            print(f"  IR: {result['ir']:.4f}")
    
    # ========================================================================
    # Transaction Cost Sensitivity
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("TRANSACTION COST SENSITIVITY (2018, n=10)")
    print("=" * 70)
    
    tx_costs = [0.0, 0.0001, 0.0005, 0.001, 0.002]
    for tx in tx_costs:
        result = run_backtest(
            trainer, data_dict, VAL_START, VAL_END, 
            n_top=10, tx_cost=tx
        )
        if result:
            print(f"  {tx*100:.2f}% cost: Sharpe={result['sharpe']:.3f}, Return={result['cumulative_return']*100:.2f}%")
    
    # ========================================================================
    # Summary Comparison
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("SUMMARY: DUAL-TASK MLP vs ZHANG 2019")
    print("=" * 70)
    
    print("\n| Metric | Zhang Paper (2018) | Dual-Task MLP (2018) | Dual-Task MLP (2019-24) |")
    print("|--------|-------------------|---------------------|------------------------|")
    
    val_n10 = results.get('val_n10', {})
    test_n10 = results.get('test_n10', {})
    
    print(f"| Return | 91.32% | {val_n10.get('cumulative_return', 0)*100:.2f}% | {test_n10.get('cumulative_return', 0)*100:.2f}% |")
    print(f"| Sharpe | 2.20 | {val_n10.get('sharpe', 0):.3f} | {test_n10.get('sharpe', 0):.3f} |")
    print(f"| Max DD | ? | {val_n10.get('max_drawdown', 0)*100:.2f}% | {test_n10.get('max_drawdown', 0)*100:.2f}% |")
    print(f"| Mean IC | ? | {val_n10.get('mean_ic', 0):.4f} | {test_n10.get('mean_ic', 0):.4f} |")
    
    print("\n✅ Backtest complete!")
    
    return results


if __name__ == "__main__":
    results = main()
