#!/usr/bin/env python3
"""
DOUBLE HOLDOUT VALIDATION: The Gold Standard for ML Trading

This is the most rigorous test possible:
1. TIME HOLDOUT: Train on 2017-2022, Test on 2023-2024
2. TICKER HOLDOUT: Train on Set A (80 tickers), Test on Set B (20 DIFFERENT tickers)

If the model works on tickers it has NEVER seen, it has learned
real market patterns, not just ticker-specific quirks.

Professional quant methodology:
- Simulates deploying model to new stocks
- Eliminates overfitting to specific tickers
- Most realistic performance estimate

Estimated runtime: 10-15 minutes
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, '.')

from src.data import loader
from src.data.validator import DataValidator
from src.data.universe import fetch_sp500_tickers
from src.models.trainer import train_model
from src.models.predictor import Predictor


def run_double_holdout():
    print("=" * 70)
    print("DOUBLE HOLDOUT VALIDATION")
    print("The Gold Standard for ML Trading Systems")
    print("=" * 70)
    
    # ==================== CONFIGURATION ====================
    TRAIN_START = "2017-01-01"
    TRAIN_END = "2022-12-31"
    TEST_START = "2023-01-01"
    TEST_END = "2024-12-01"
    
    INITIAL_CASH = 100000.0
    STOP_LOSS_PCT = 0.15
    
    # Split: 80 tickers for training, 20 DIFFERENT for testing
    N_TRAIN_TICKERS = 80
    N_TEST_TICKERS = 20
    
    print(f"\n📋 Configuration:")
    print(f"   Training Period: {TRAIN_START} to {TRAIN_END}")
    print(f"   Testing Period:  {TEST_START} to {TEST_END}")
    print(f"   Train Tickers:   {N_TRAIN_TICKERS} (model sees these)")
    print(f"   Test Tickers:    {N_TEST_TICKERS} (model NEVER sees these)")
    
    # ==================== LOAD S&P 500 ====================
    print("\n📊 Loading S&P 500 universe...")
    try:
        all_tickers = fetch_sp500_tickers()
        print(f"   Found {len(all_tickers)} tickers")
    except Exception as e:
        print(f"   Error: {e}")
        return
    
    # Random split (fixed seed for reproducibility)
    np.random.seed(42)
    shuffled = np.random.permutation(all_tickers)
    
    train_tickers = list(shuffled[:N_TRAIN_TICKERS])
    test_tickers = list(shuffled[N_TRAIN_TICKERS:N_TRAIN_TICKERS + N_TEST_TICKERS])
    
    # Ensure no overlap
    assert len(set(train_tickers) & set(test_tickers)) == 0, "Train/test overlap!"
    
    print(f"\n   Train tickers (first 10): {train_tickers[:10]}")
    print(f"   Test tickers: {test_tickers}")
    
    # ==================== FETCH DATA ====================
    print("\n📥 Fetching data...")
    all_needed = train_tickers + test_tickers
    data_dict = loader.fetch_data(all_needed, period="10y")
    print(f"   Loaded {len(data_dict)} tickers")
    
    # Validate
    print("\n🔍 Validating data...")
    validator = DataValidator(backtest_mode=True)
    for ticker in list(data_dict.keys()):
        df = data_dict[ticker]
        data_dict[ticker] = df[df.index >= '2015-01-01']
    
    results = validator.validate_data_dict(data_dict)
    valid_tickers = [t for t, r in results.items() if r.is_valid]
    data_dict = {t: data_dict[t] for t in valid_tickers}
    
    train_data = {t: data_dict[t] for t in train_tickers if t in data_dict}
    test_data = {t: data_dict[t] for t in test_tickers if t in data_dict}
    
    print(f"   Valid train tickers: {len(train_data)}")
    print(f"   Valid test tickers: {len(test_data)}")
    
    if len(train_data) < 20 or len(test_data) < 5:
        print("❌ Insufficient data")
        return
    
    # ==================== TRAIN MODEL (on train tickers only) ====================
    print("\n" + "=" * 70)
    print("TRAINING MODEL (2017-2022 on Train Tickers ONLY)")
    print("=" * 70)
    
    # Filter training data to training period
    train_dict_filtered = {}
    for ticker, df in train_data.items():
        df_train = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]
        if len(df_train) >= 252:
            train_dict_filtered[ticker] = df_train
    
    print(f"   Training on {len(train_dict_filtered)} tickers")
    
    model = train_model(train_dict_filtered, n_splits=3, save_model=False)
    
    if model is None:
        print("❌ Training failed")
        return
    
    # Create predictor
    predictor = Predictor()
    predictor.model = model
    
    print("\n✅ Model trained successfully")
    print("   Model has NEVER seen the test tickers!")
    
    # ==================== TEST ON UNSEEN TICKERS ====================
    print("\n" + "=" * 70)
    print("TESTING ON UNSEEN TICKERS (2023-2024)")
    print("=" * 70)
    
    # Results storage
    results_with_sl = {'trades': [], 'portfolio_values': []}
    results_no_sl = {'trades': [], 'portfolio_values': []}
    
    # Get test dates
    test_dates = pd.date_range(TEST_START, TEST_END, freq='B')
    test_dates = [d for d in test_dates if any(d in test_data[t].index for t in test_data)]
    
    print(f"   Test dates: {len(test_dates)}")
    
    # Track positions
    positions_sl = {}   # {ticker: {'entry_price': x, 'shares': n}}
    positions_no_sl = {}
    
    cash_sl = INITIAL_CASH
    cash_no_sl = INITIAL_CASH
    
    for i, date in enumerate(test_dates):
        # Get prices
        prices = {}
        for ticker, df in test_data.items():
            if date in df.index:
                prices[ticker] = df.loc[date, 'Close']
        
        if len(prices) < 3:
            continue
        
        # ===== STOP-LOSS CHECK =====
        for ticker in list(positions_sl.keys()):
            if ticker not in prices:
                continue
            entry = positions_sl[ticker]['entry_price']
            current = prices[ticker]
            loss_pct = (current - entry) / entry
            
            if loss_pct < -STOP_LOSS_PCT:
                shares = positions_sl[ticker]['shares']
                cash_sl += shares * current * 0.999
                pnl = loss_pct * 100
                results_with_sl['trades'].append({
                    'date': date, 'ticker': ticker, 'pnl_pct': pnl, 'type': 'STOP_LOSS'
                })
                del positions_sl[ticker]
        
        # ===== GENERATE SIGNALS =====
        predictions = {}
        for ticker, df in test_data.items():
            if date not in df.index:
                continue
            df_to_date = df[df.index <= date]
            if len(df_to_date) < 60:
                continue
            try:
                pred = predictor.predict(df_to_date)
                predictions[ticker] = pred
            except:
                continue
        
        if not predictions:
            continue
        
        # Thresholds
        buy_thresh = 0.005
        sell_thresh = -0.005
        
        # ===== SELLS =====
        for ticker in list(positions_sl.keys()):
            if ticker in predictions and predictions[ticker] < sell_thresh:
                shares = positions_sl[ticker]['shares']
                price = prices.get(ticker, 0)
                if price > 0:
                    pnl = (price - positions_sl[ticker]['entry_price']) / positions_sl[ticker]['entry_price'] * 100
                    cash_sl += shares * price * 0.999
                    results_with_sl['trades'].append({
                        'date': date, 'ticker': ticker, 'pnl_pct': pnl, 'type': 'SIGNAL'
                    })
                    del positions_sl[ticker]
        
        for ticker in list(positions_no_sl.keys()):
            if ticker in predictions and predictions[ticker] < sell_thresh:
                shares = positions_no_sl[ticker]['shares']
                price = prices.get(ticker, 0)
                if price > 0:
                    pnl = (price - positions_no_sl[ticker]['entry_price']) / positions_no_sl[ticker]['entry_price'] * 100
                    cash_no_sl += shares * price * 0.999
                    results_no_sl['trades'].append({
                        'date': date, 'ticker': ticker, 'pnl_pct': pnl, 'type': 'SIGNAL'
                    })
                    del positions_no_sl[ticker]
        
        # ===== BUYS =====
        buy_candidates = [(t, p) for t, p in predictions.items() if p > buy_thresh]
        buy_candidates.sort(key=lambda x: x[1], reverse=True)
        
        max_positions = 5
        position_size = min(cash_sl / max(max_positions - len(positions_sl), 1), INITIAL_CASH * 0.15)
        
        for ticker, pred in buy_candidates[:3]:
            price = prices.get(ticker, 0)
            if price <= 0:
                continue
            
            if ticker not in positions_sl and len(positions_sl) < max_positions:
                if cash_sl > position_size:
                    shares = int(position_size / price)
                    if shares > 0:
                        cash_sl -= shares * price * 1.001
                        positions_sl[ticker] = {'entry_price': price, 'shares': shares}
            
            if ticker not in positions_no_sl and len(positions_no_sl) < max_positions:
                if cash_no_sl > position_size:
                    shares = int(position_size / price)
                    if shares > 0:
                        cash_no_sl -= shares * price * 1.001
                        positions_no_sl[ticker] = {'entry_price': price, 'shares': shares}
        
        # Track portfolio value weekly
        if i % 5 == 0:
            sl_value = cash_sl + sum(p['shares'] * prices.get(t, p['entry_price']) for t, p in positions_sl.items())
            no_sl_value = cash_no_sl + sum(p['shares'] * prices.get(t, p['entry_price']) for t, p in positions_no_sl.items())
            results_with_sl['portfolio_values'].append((date, sl_value))
            results_no_sl['portfolio_values'].append((date, no_sl_value))
    
    # Final values
    final_prices = {t: df['Close'].iloc[-1] for t, df in test_data.items() if len(df) > 0}
    
    final_sl = cash_sl + sum(p['shares'] * final_prices.get(t, p['entry_price']) for t, p in positions_sl.items())
    final_no_sl = cash_no_sl + sum(p['shares'] * final_prices.get(t, p['entry_price']) for t, p in positions_no_sl.items())
    
    # ==================== RESULTS ====================
    print("\n" + "=" * 70)
    print("DOUBLE HOLDOUT RESULTS")
    print("Model tested on tickers it has NEVER seen")
    print("=" * 70)
    
    sl_return = (final_sl - INITIAL_CASH) / INITIAL_CASH * 100
    no_sl_return = (final_no_sl - INITIAL_CASH) / INITIAL_CASH * 100
    
    sl_trades = pd.DataFrame(results_with_sl['trades']) if results_with_sl['trades'] else pd.DataFrame()
    no_sl_trades = pd.DataFrame(results_no_sl['trades']) if results_no_sl['trades'] else pd.DataFrame()
    
    sl_win_rate = (sl_trades['pnl_pct'] > 0).mean() * 100 if len(sl_trades) > 0 else 0
    no_sl_win_rate = (no_sl_trades['pnl_pct'] > 0).mean() * 100 if len(no_sl_trades) > 0 else 0
    
    print("\n┌────────────────────┬──────────────────┬──────────────────┐")
    print("│ Metric             │ WITH STOP-LOSS   │ NO STOP-LOSS     │")
    print("├────────────────────┼──────────────────┼──────────────────┤")
    print(f"│ Final Value        │ ${final_sl:>14,.2f} │ ${final_no_sl:>14,.2f} │")
    print(f"│ Total Return       │ {sl_return:>15.1f}% │ {no_sl_return:>15.1f}% │")
    print(f"│ Trades             │ {len(sl_trades):>16} │ {len(no_sl_trades):>16} │")
    print(f"│ Win Rate           │ {sl_win_rate:>15.1f}% │ {no_sl_win_rate:>15.1f}% │")
    print("└────────────────────┴──────────────────┴──────────────────┘")
    
    # Winner
    if sl_return > no_sl_return:
        print("\n🏆 WINNER: 15% STOP-LOSS")
    else:
        print("\n🏆 WINNER: NO STOP-LOSS")
    
    print("\n" + "=" * 70)
    print("TEST PARAMETERS")
    print(f"   Train tickers: {len(train_dict_filtered)} (model trained on these)")
    print(f"   Test tickers: {len(test_data)} (model NEVER saw these)")
    print(f"   Train period: {TRAIN_START} to {TRAIN_END}")
    print(f"   Test period: {TEST_START} to {TEST_END}")
    print("=" * 70)
    
    # Reality check
    print("\n📊 REALITY CHECK")
    print("   These results are on truly unseen tickers.")
    print("   If returns are much lower than biased backtests,")
    print("   it means the model was overfitting to specific stocks.")


if __name__ == "__main__":
    run_double_holdout()
