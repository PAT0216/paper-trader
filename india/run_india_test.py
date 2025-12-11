#!/usr/bin/env python3
"""
INDIAN MARKET WALKTHROUGH TEST

Tests if the Paper Trader model generalizes to Indian markets:
1. Train model on NIFTY 50 stocks (2018-2022)
2. Test on 2023-2024
3. Compare performance to NIFTY 50 index

This is a TRUE cross-market generalization test.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from india.fetch_india_data import get_india_data, get_all_india_tickers, INDIA_CACHE_DB
from india.nifty50_universe import NIFTY_INDEX

# Import model components from main project
from src.features.indicators import generate_features
from src.models.trainer import create_target, FEATURE_COLUMNS
import xgboost as xgb


def train_model_india(data_dict, train_end):
    """Train XGBoost model on Indian stock data."""
    all_features = []
    
    for ticker, df in data_dict.items():
        if ticker.startswith('^'):  # Skip indices
            continue
        
        df_train = df[df.index < train_end].copy()
        if len(df_train) < 252:  # Need at least 1 year
            continue
        
        try:
            processed = generate_features(df_train, include_target=False)
            processed = create_target(processed, target_type='regression')
            processed = processed.dropna(subset=['Target'])
            
            if len(processed) > 50:
                all_features.append(processed)
        except Exception:
            continue
    
    if not all_features:
        return None
    
    full_df = pd.concat(all_features).sort_index()
    
    # Clean data
    X = full_df[FEATURE_COLUMNS].values
    y = full_df['Target'].values
    X = np.where(np.isinf(X), np.nan, X)
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[valid], y[valid]
    
    if len(X) < 500:
        return None
    
    print(f"   Training on {len(X)} samples from {len(all_features)} stocks")
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(X, y)
    
    return model


def run_india_walkthrough():
    """Run walk-through test on Indian market."""
    print("=" * 70)
    print("INDIAN MARKET WALKTHROUGH TEST")
    print("Testing if US-designed features work on NSE/BSE")
    print("=" * 70)
    
    # Check data availability
    if not os.path.exists(INDIA_CACHE_DB):
        print("\n❌ No Indian data cache found!")
        print("   Please run: python india/fetch_india_data.py")
        print("   OR download from GitHub Actions cache")
        return
    
    # Load all tickers
    tickers = get_all_india_tickers()
    print(f"\n📊 Found {len(tickers)} tickers in cache")
    
    # Load data
    data_dict = {}
    for ticker in tickers:
        df = get_india_data(ticker)
        if df is not None and len(df) > 500:
            data_dict[ticker] = df
    
    print(f"   Valid tickers: {len(data_dict)}")
    
    if len(data_dict) < 20:
        print("❌ Insufficient data")
        return
    
    # Configuration
    TRAIN_END = "2022-12-31"
    TEST_START = "2023-01-01"
    TEST_END = "2024-12-01"
    INITIAL_CASH = 1000000  # ₹10 lakh
    STOP_LOSS_PCT = 0.15
    
    print(f"\n📋 Configuration:")
    print(f"   Training: Up to {TRAIN_END}")
    print(f"   Testing: {TEST_START} to {TEST_END}")
    print(f"   Initial Capital: ₹{INITIAL_CASH:,}")
    print(f"   Stop-Loss: {STOP_LOSS_PCT*100}%")
    
    # Load NIFTY benchmark
    nifty = get_india_data(NIFTY_INDEX)
    if nifty is None:
        print("❌ NIFTY 50 index data not found")
        return
    
    # Train model
    print("\n" + "=" * 70)
    print("TRAINING MODEL ON INDIAN DATA (up to 2022)")
    print("=" * 70)
    
    model = train_model_india(data_dict, TRAIN_END)
    if model is None:
        print("❌ Training failed")
        return
    
    print("✅ Model trained successfully")
    
    # Test
    print("\n" + "=" * 70)
    print("TESTING ON 2023-2024")
    print("=" * 70)
    
    # Get test period dates
    test_dates = pd.date_range(TEST_START, TEST_END, freq='B')
    
    # Storage
    results_sl = {'trades': [], 'values': []}
    results_no_sl = {'trades': [], 'values': []}
    
    positions_sl = {}
    positions_no_sl = {}
    cash_sl = INITIAL_CASH
    cash_no_sl = INITIAL_CASH
    
    for date in test_dates:
        # Get prices for this date
        prices = {}
        for ticker, df in data_dict.items():
            if ticker.startswith('^'):
                continue
            if date in df.index:
                prices[ticker] = df.loc[date, 'Close']
        
        if len(prices) < 10:
            continue
        
        # Stop-loss check
        for ticker in list(positions_sl.keys()):
            if ticker not in prices:
                continue
            entry = positions_sl[ticker]['entry']
            current = prices[ticker]
            loss = (current - entry) / entry
            
            if loss < -STOP_LOSS_PCT:
                shares = positions_sl[ticker]['shares']
                cash_sl += shares * current * 0.999
                results_sl['trades'].append({
                    'date': date, 'ticker': ticker,
                    'pnl': loss * 100, 'type': 'STOP_LOSS'
                })
                del positions_sl[ticker]
        
        # Generate predictions
        predictions = {}
        for ticker, df in data_dict.items():
            if ticker.startswith('^'):
                continue
            if date not in df.index:
                continue
            
            hist = df[df.index <= date]
            if len(hist) < 60:
                continue
            
            try:
                features = generate_features(hist, include_target=False)
                if len(features) == 0:
                    continue
                X = features[FEATURE_COLUMNS].iloc[-1:].values
                X = np.where(np.isinf(X), 0, X)
                X = np.nan_to_num(X, 0)
                pred = model.predict(X)[0]
                predictions[ticker] = pred
            except:
                continue
        
        if not predictions:
            continue
        
        # Signals
        buy_thresh = 0.005
        sell_thresh = -0.005
        
        # Sells
        for ticker in list(positions_sl.keys()):
            if ticker in predictions and predictions[ticker] < sell_thresh:
                if ticker in prices:
                    entry = positions_sl[ticker]['entry']
                    pnl = (prices[ticker] - entry) / entry * 100
                    cash_sl += positions_sl[ticker]['shares'] * prices[ticker] * 0.999
                    results_sl['trades'].append({
                        'date': date, 'ticker': ticker, 'pnl': pnl, 'type': 'SIGNAL'
                    })
                    del positions_sl[ticker]
        
        for ticker in list(positions_no_sl.keys()):
            if ticker in predictions and predictions[ticker] < sell_thresh:
                if ticker in prices:
                    entry = positions_no_sl[ticker]['entry']
                    pnl = (prices[ticker] - entry) / entry * 100
                    cash_no_sl += positions_no_sl[ticker]['shares'] * prices[ticker] * 0.999
                    results_no_sl['trades'].append({
                        'date': date, 'ticker': ticker, 'pnl': pnl, 'type': 'SIGNAL'
                    })
                    del positions_no_sl[ticker]
        
        # Buys
        buy_candidates = [(t, p) for t, p in predictions.items() if p > buy_thresh]
        buy_candidates.sort(key=lambda x: x[1], reverse=True)
        
        max_positions = 10
        position_size = INITIAL_CASH * 0.10  # 10% per position
        
        for ticker, pred in buy_candidates[:5]:
            if ticker not in prices or prices[ticker] <= 0:
                continue
            
            price = prices[ticker]
            
            if ticker not in positions_sl and len(positions_sl) < max_positions:
                if cash_sl >= position_size:
                    shares = int(position_size / price)
                    if shares > 0:
                        cash_sl -= shares * price * 1.001
                        positions_sl[ticker] = {'entry': price, 'shares': shares}
            
            if ticker not in positions_no_sl and len(positions_no_sl) < max_positions:
                if cash_no_sl >= position_size:
                    shares = int(position_size / price)
                    if shares > 0:
                        cash_no_sl -= shares * price * 1.001
                        positions_no_sl[ticker] = {'entry': price, 'shares': shares}
    
    # Calculate final values
    final_prices = {t: df['Close'].iloc[-1] for t, df in data_dict.items() if len(df) > 0 and not t.startswith('^')}
    
    final_sl = cash_sl + sum(p['shares'] * final_prices.get(t, p['entry']) for t, p in positions_sl.items())
    final_no_sl = cash_no_sl + sum(p['shares'] * final_prices.get(t, p['entry']) for t, p in positions_no_sl.items())
    
    # NIFTY benchmark
    nifty_test = nifty[(nifty.index >= TEST_START) & (nifty.index <= TEST_END)]
    if len(nifty_test) > 0:
        nifty_return = (nifty_test['Close'].iloc[-1] - nifty_test['Close'].iloc[0]) / nifty_test['Close'].iloc[0] * 100
    else:
        nifty_return = 0
    
    # Results
    sl_return = (final_sl - INITIAL_CASH) / INITIAL_CASH * 100
    no_sl_return = (final_no_sl - INITIAL_CASH) / INITIAL_CASH * 100
    
    sl_trades = pd.DataFrame(results_sl['trades']) if results_sl['trades'] else pd.DataFrame()
    no_sl_trades = pd.DataFrame(results_no_sl['trades']) if results_no_sl['trades'] else pd.DataFrame()
    
    print("\n" + "=" * 70)
    print("INDIAN MARKET RESULTS (2023-2024)")
    print("=" * 70)
    
    print("\n┌────────────────────┬──────────────────┬──────────────────┬──────────────────┐")
    print("│ Metric             │ WITH STOP-LOSS   │ NO STOP-LOSS     │ NIFTY 50 INDEX   │")
    print("├────────────────────┼──────────────────┼──────────────────┼──────────────────┤")
    print(f"│ Total Return       │ {sl_return:>15.1f}% │ {no_sl_return:>15.1f}% │ {nifty_return:>15.1f}% │")
    print(f"│ Trades             │ {len(sl_trades):>16} │ {len(no_sl_trades):>16} │              N/A │")
    if len(sl_trades) > 0:
        sl_win = (sl_trades['pnl'] > 0).mean() * 100
        no_sl_win = (no_sl_trades['pnl'] > 0).mean() * 100 if len(no_sl_trades) > 0 else 0
        print(f"│ Win Rate           │ {sl_win:>15.1f}% │ {no_sl_win:>15.1f}% │              N/A │")
    print("└────────────────────┴──────────────────┴──────────────────┴──────────────────┘")
    
    # Winner
    if sl_return > nifty_return:
        print("\n🏆 STRATEGY BEATS NIFTY 50!")
    else:
        print("\n📊 NIFTY 50 outperforms strategy")
    
    if sl_return > no_sl_return:
        print("🛡️ Stop-loss still helps!")
    
    print("\n" + "=" * 70)
    print("CROSS-MARKET GENERALIZATION TEST COMPLETE")
    print("If returns are positive, the model learned universal patterns!")
    print("=" * 70)


if __name__ == "__main__":
    run_india_walkthrough()
