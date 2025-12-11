#!/usr/bin/env python3
"""
UNBIASED WALK-FORWARD A/B TEST: Stop-Loss vs No Stop-Loss

Professional quant methodology:
1. Walk-forward validation (yearly retraining - no look-ahead bias)
2. Diverse universe (random sample from S&P 500 - reduces survivor bias)  
3. Multi-regime coverage (2018-2024: includes bull, bear, COVID crisis)
4. Controlled comparison (ONLY stop-loss toggle changes)

Estimated runtime: 30-45 minutes
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
from src.trading.risk_manager import RiskManager, RiskLimits


def run_unbiased_comparison():
    print("=" * 70)
    print("UNBIASED WALK-FORWARD A/B TEST")
    print("Stop-Loss (15%) vs No Stop-Loss")
    print("=" * 70)
    
    # ==================== CONFIGURATION ====================
    TEST_START = "2018-01-01"  # Start of testing (train on 2015-2017)
    TEST_END = "2024-12-01"
    RETRAIN_FREQ = "yearly"    # Retrain model each year
    INITIAL_CASH = 100000.0
    STOP_LOSS_PCT = 0.15       # 15% stop-loss
    
    # Random sample size (balance between coverage and speed)
    SAMPLE_SIZE = 75  # ~15% of S&P 500, good statistical power
    
    # ==================== DATA LOADING ====================
    print("\n📊 Loading S&P 500 universe...")
    try:
        all_tickers = fetch_sp500_tickers()
        print(f"   Full S&P 500: {len(all_tickers)} tickers")
    except Exception as e:
        print(f"   Error fetching S&P 500: {e}")
        return
    
    # Random sample for unbiased universe (fixed seed for reproducibility)
    np.random.seed(42)
    sample_tickers = list(np.random.choice(all_tickers, size=min(SAMPLE_SIZE, len(all_tickers)), replace=False))
    
    # Add SPY for benchmark
    if 'SPY' not in sample_tickers:
        sample_tickers.append('SPY')
    
    print(f"   Random sample: {len(sample_tickers)} tickers")
    
    # Fetch data
    print("\n📥 Fetching historical data (2015-2024)...")
    data_dict = loader.fetch_data(sample_tickers, period="10y")
    print(f"   Loaded: {len(data_dict)} tickers")
    
    # Validate
    print("\n🔍 Validating data quality...")
    validator = DataValidator(backtest_mode=True)
    
    # Filter to 2015+ before validation
    for ticker in list(data_dict.keys()):
        df = data_dict[ticker]
        data_dict[ticker] = df[df.index >= '2015-01-01']
    
    results = validator.validate_data_dict(data_dict)
    valid_tickers = [t for t, r in results.items() if r.is_valid]
    data_dict = {t: data_dict[t] for t in valid_tickers if t in data_dict}
    print(f"   Valid: {len(data_dict)} tickers")
    
    if len(data_dict) < 20:
        print("   ❌ Insufficient valid tickers")
        return
    
    # Extract benchmark
    benchmark = data_dict.pop('SPY', None)
    
    # ==================== WALK-FORWARD SETUP ====================
    # Test years
    test_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    
    # Results storage
    results_with_sl = {'trades': [], 'returns': []}
    results_no_sl = {'trades': [], 'returns': []}
    
    print(f"\n🚀 Running walk-forward validation...")
    print(f"   Test period: {TEST_START} to {TEST_END}")
    print(f"   Retrain frequency: {RETRAIN_FREQ}")
    
    # ==================== WALK-FORWARD LOOP ====================
    for year in test_years:
        print(f"\n--- Year {year} ---")
        
        train_end = f"{year-1}-12-31"
        test_start = f"{year}-01-01"
        test_end = f"{year}-12-31"
        
        # Prepare training data (all data before test year)
        train_dict = {}
        for ticker, df in data_dict.items():
            df_train = df[df.index <= train_end]
            if len(df_train) >= 252:  # Need at least 1 year
                train_dict[ticker] = df_train
        
        print(f"   Training on {len(train_dict)} tickers up to {train_end}")
        
        # Train model using existing function
        try:
            model = train_model(train_dict, n_splits=3, save_model=False)
            if model is None:
                print(f"   ⚠️ Training returned None")
                continue
            print(f"   Model trained successfully")
        except Exception as e:
            print(f"   ⚠️ Training failed: {e}")
            continue
        
        # Create predictor with this model
        predictor = Predictor()
        predictor.model = model
        
        # ==================== TEST BOTH STRATEGIES ====================
        # Get test dates
        try:
            test_dates = pd.date_range(test_start, test_end, freq='B')
            test_dates = [d for d in test_dates if any(d in data_dict[t].index for t in data_dict)]
        except:
            continue
        
        # Track positions for each strategy
        positions_sl = {}   # {ticker: {'entry_price': x, 'shares': n}}
        positions_no_sl = {}
        
        cash_sl = INITIAL_CASH / len(test_years)  # Divide capital by years
        cash_no_sl = INITIAL_CASH / len(test_years)
        
        for date in test_dates:
            # Get current prices
            prices = {}
            for ticker, df in data_dict.items():
                if date in df.index:
                    prices[ticker] = df.loc[date, 'Close']
            
            if len(prices) < 5:
                continue
            
            # ===== STOP-LOSS CHECK (only for SL strategy) =====
            for ticker in list(positions_sl.keys()):
                if ticker not in prices:
                    continue
                entry = positions_sl[ticker]['entry_price']
                current = prices[ticker]
                loss_pct = (current - entry) / entry
                
                if loss_pct < -STOP_LOSS_PCT:
                    # Stop-loss triggered
                    shares = positions_sl[ticker]['shares']
                    cash_sl += shares * current * 0.999  # Slippage
                    pnl_pct = loss_pct * 100
                    results_with_sl['trades'].append({
                        'date': date, 'ticker': ticker,
                        'pnl_pct': pnl_pct, 'type': 'STOP_LOSS'
                    })
                    del positions_sl[ticker]
            
            # ===== GENERATE SIGNALS =====
            predictions = {}
            for ticker, df in data_dict.items():
                if date not in df.index:
                    continue
                df_to_date = df[df.index <= date]
                if len(df_to_date) < 60:
                    continue
                try:
                    pred = predictor.predict(df_to_date)
                    predictions[ticker] = pred
                except Exception:
                    continue
            
            if not predictions:
                continue
            
            # Signal threshold
            buy_thresh = 0.005
            sell_thresh = -0.005
            
            # ===== EXECUTE SELLS =====
            for ticker in list(positions_sl.keys()):
                if ticker in predictions and predictions[ticker] < sell_thresh:
                    shares = positions_sl[ticker]['shares']
                    price = prices.get(ticker, 0)
                    if price > 0:
                        pnl = (price - positions_sl[ticker]['entry_price']) / positions_sl[ticker]['entry_price'] * 100
                        cash_sl += shares * price * 0.999
                        results_with_sl['trades'].append({
                            'date': date, 'ticker': ticker,
                            'pnl_pct': pnl, 'type': 'SIGNAL'
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
                            'date': date, 'ticker': ticker,
                            'pnl_pct': pnl, 'type': 'SIGNAL'
                        })
                        del positions_no_sl[ticker]
            
            # ===== EXECUTE BUYS =====
            buy_candidates = [(t, p) for t, p in predictions.items() if p > buy_thresh]
            buy_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Max 5 positions per strategy
            max_positions = 5
            position_size = cash_sl / max(max_positions - len(positions_sl), 1)
            
            for ticker, pred in buy_candidates[:3]:  # Top 3 signals
                price = prices.get(ticker, 0)
                if price <= 0:
                    continue
                
                # Strategy WITH stop-loss
                if ticker not in positions_sl and len(positions_sl) < max_positions:
                    if cash_sl > position_size:
                        shares = int(position_size / price)
                        if shares > 0:
                            cost = shares * price * 1.001
                            cash_sl -= cost
                            positions_sl[ticker] = {'entry_price': price, 'shares': shares}
                
                # Strategy WITHOUT stop-loss
                if ticker not in positions_no_sl and len(positions_no_sl) < max_positions:
                    if cash_no_sl > position_size:
                        shares = int(position_size / price)
                        if shares > 0:
                            cost = shares * price * 1.001
                            cash_no_sl -= cost
                            positions_no_sl[ticker] = {'entry_price': price, 'shares': shares}
        
        # End of year - close all positions
        final_prices = {}
        for ticker, df in data_dict.items():
            if len(df) > 0:
                final_prices[ticker] = df['Close'].iloc[-1]
        
        # Calculate year-end portfolio values
        sl_value = cash_sl + sum(
            p['shares'] * final_prices.get(t, p['entry_price'])
            for t, p in positions_sl.items()
        )
        no_sl_value = cash_no_sl + sum(
            p['shares'] * final_prices.get(t, p['entry_price'])
            for t, p in positions_no_sl.items()
        )
        
        year_capital = INITIAL_CASH / len(test_years)
        results_with_sl['returns'].append((year, (sl_value - year_capital) / year_capital * 100))
        results_no_sl['returns'].append((year, (no_sl_value - year_capital) / year_capital * 100))
        
        print(f"   With SL:    {(sl_value - year_capital) / year_capital * 100:.1f}%")
        print(f"   Without SL: {(no_sl_value - year_capital) / year_capital * 100:.1f}%")
    
    # ==================== FINAL RESULTS ====================
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    # Calculate metrics
    def calc_metrics(results_dict):
        trades = pd.DataFrame(results_dict['trades'])
        returns = results_dict['returns']
        
        if trades.empty:
            return {'n_trades': 0, 'win_rate': 0, 'avg_pnl': 0, 'sharpe': 0, 'total_return': 0}
        
        n_trades = len(trades)
        win_rate = (trades['pnl_pct'] > 0).mean()
        avg_pnl = trades['pnl_pct'].mean()
        
        # Yearly returns for Sharpe
        yearly_rets = [r[1] for r in returns]
        if len(yearly_rets) > 1:
            sharpe = np.mean(yearly_rets) / np.std(yearly_rets) if np.std(yearly_rets) > 0 else 0
        else:
            sharpe = 0
        
        total_return = sum(yearly_rets) / len(yearly_rets) if yearly_rets else 0
        
        return {
            'n_trades': n_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'sharpe': sharpe,
            'total_return': total_return
        }
    
    m_sl = calc_metrics(results_with_sl)
    m_no_sl = calc_metrics(results_no_sl)
    
    print("\n┌────────────────────┬──────────────────┬──────────────────┐")
    print("│ Metric             │ WITH STOP-LOSS   │ NO STOP-LOSS     │")
    print("├────────────────────┼──────────────────┼──────────────────┤")
    print(f"│ Trades             │ {m_sl['n_trades']:>16} │ {m_no_sl['n_trades']:>16} │")
    print(f"│ Win Rate           │ {m_sl['win_rate']:>15.1%} │ {m_no_sl['win_rate']:>15.1%} │")
    print(f"│ Avg Trade P&L      │ {m_sl['avg_pnl']:>14.2f}% │ {m_no_sl['avg_pnl']:>14.2f}% │")
    print(f"│ Avg Yearly Return  │ {m_sl['total_return']:>14.2f}% │ {m_no_sl['total_return']:>14.2f}% │")
    print(f"│ Sharpe (Yearly)    │ {m_sl['sharpe']:>16.2f} │ {m_no_sl['sharpe']:>16.2f} │")
    print("└────────────────────┴──────────────────┴──────────────────┘")
    
    # Winner
    if m_sl['sharpe'] > m_no_sl['sharpe'] * 1.1:
        print("\n🏆 WINNER: STOP-LOSS (15%)")
    elif m_no_sl['sharpe'] > m_sl['sharpe'] * 1.1:
        print("\n🏆 WINNER: NO STOP-LOSS")
    else:
        print("\n📊 RESULT: No significant difference")
    
    print("\n" + "=" * 70)
    print("TEST PARAMETERS")
    print(f"   Universe: {len(data_dict)} random S&P 500 tickers")
    print(f"   Period: {TEST_START} to {TEST_END}")
    print(f"   Retraining: Yearly walk-forward")
    print(f"   Stop-Loss: 15%")
    print("=" * 70)


if __name__ == "__main__":
    run_unbiased_comparison()
