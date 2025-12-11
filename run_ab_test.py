#!/usr/bin/env python3
"""
3-Way A/B Test: Fixed vs Z-Score vs Hybrid

Methods:
1. FIXED: Buy if prediction > 0.5% (original)
2. ZSCORE: Buy if z-score > 1.0 (too selective)
3. HYBRID: Buy if prediction > 0.5%, weight by z-score (best of both)
"""

import sys
import numpy as np
import pandas as pd
import os

sys.path.insert(0, '.')

def simulate_with_existing_data():
    from src.data.cache import get_cache
    from src.models.predictor import Predictor
    
    print("=" * 70)
    print("3-WAY A/B TEST: Fixed vs Z-Score vs Hybrid")
    print("=" * 70)
    
    # Load cached data
    print("\n📊 Loading cached market data...")
    cache = get_cache()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 
               'JPM', 'V', 'MA', 'UNH', 'JNJ', 'HD', 'PG', 'XOM', 'CVX',
               'BAC', 'WMT', 'KO', 'PEP', 'MRK', 'COST', 'DIS', 'NFLX', 'AMD']
    
    data_dict = {}
    for ticker in tickers:
        df = cache.get_price_data(ticker)
        if df is not None and len(df) > 100:
            data_dict[ticker] = df
    
    print(f"   Loaded {len(data_dict)} tickers from cache")
    
    if len(data_dict) < 10:
        print("   Insufficient cached data.")
        return
    
    predictor = Predictor()
    
    # Results storage
    results_fixed = []
    results_zscore = []
    results_hybrid = []
    
    sample_df = list(data_dict.values())[0]
    test_dates = sample_df.index[-252:-1]
    
    print(f"\n🧪 Running simulation over {len(test_dates)} trading days...\n")
    
    for i, date in enumerate(test_dates):
        if i % 50 == 0:
            print(f"   Day {i}/{len(test_dates)}...")
        
        predictions = {}
        next_returns = {}
        
        for ticker, df in data_dict.items():
            if date not in df.index:
                continue
            
            idx = df.index.get_loc(date)
            if idx < 60 or idx >= len(df) - 1:
                continue
            
            historical = df.iloc[:idx+1]
            
            try:
                pred = predictor.predict(historical)
                predictions[ticker] = pred
                
                today_close = df.iloc[idx]['Close']
                next_close = df.iloc[idx + 1]['Close']
                actual_ret = (next_close - today_close) / today_close
                next_returns[ticker] = actual_ret
            except:
                continue
        
        if len(predictions) < 5:
            continue
        
        # Calculate z-scores for all predictions
        preds_array = np.array(list(predictions.values()))
        mu, sigma = np.mean(preds_array), np.std(preds_array)
        
        buy_thresh = 0.005  # 0.5% return threshold
        
        for ticker, pred in predictions.items():
            if ticker not in next_returns:
                continue
            
            actual = next_returns[ticker]
            z = (pred - mu) / sigma if sigma > 0 else 0
            
            # Method 1: FIXED (any pred > 0.5%)
            if pred > buy_thresh:
                results_fixed.append({
                    'date': date, 'ticker': ticker,
                    'pred': pred, 'actual': actual,
                    'weight': 1.0  # Equal weight
                })
            
            # Method 2: ZSCORE (z > 1.0 only)
            if z > 1.0:
                results_zscore.append({
                    'date': date, 'ticker': ticker,
                    'pred': pred, 'actual': actual, 'z': z,
                    'weight': 1.0
                })
            
            # Method 3: HYBRID (pred > 0.5% for entry, z-score for weight)
            if pred > buy_thresh:
                # Weight: higher z-score = larger position
                # Min weight 0.5, max weight 1.5 based on z-score
                weight = np.clip(0.5 + z * 0.25, 0.5, 1.5)
                results_hybrid.append({
                    'date': date, 'ticker': ticker,
                    'pred': pred, 'actual': actual, 'z': z,
                    'weight': weight
                })
    
    df_fixed = pd.DataFrame(results_fixed)
    df_zscore = pd.DataFrame(results_zscore)
    df_hybrid = pd.DataFrame(results_hybrid)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    def calc_metrics(df, weighted=False):
        if df.empty:
            return {k: 0 for k in ['N Trades', 'Win Rate', 'Avg Return', 'Wtd Return', 'Sharpe', 'Sortino']}
        
        rets = df['actual']
        weights = df['weight'] if weighted and 'weight' in df.columns else pd.Series([1.0] * len(df))
        
        # Weighted returns
        wtd_rets = rets * weights
        
        n = len(rets)
        win_rate = (rets > 0).mean()
        avg_ret = rets.mean()
        wtd_avg = wtd_rets.mean()
        sharpe = wtd_avg / wtd_rets.std() * np.sqrt(252) if wtd_rets.std() > 0 else 0
        
        downside = wtd_rets[wtd_rets < 0]
        sortino = wtd_avg / downside.std() * np.sqrt(252) if len(downside) > 1 else 0
        
        return {
            'N Trades': n,
            'Win Rate': win_rate,
            'Avg Return': avg_ret,
            'Wtd Return': wtd_avg,
            'Sharpe': sharpe,
            'Sortino': sortino
        }
    
    m_fixed = calc_metrics(df_fixed, weighted=False)
    m_zscore = calc_metrics(df_zscore, weighted=False)
    m_hybrid = calc_metrics(df_hybrid, weighted=True)
    
    print("\n┌─────────────────┬─────────────┬─────────────┬─────────────┐")
    print("│ Metric          │   FIXED     │   Z-SCORE   │   HYBRID    │")
    print("├─────────────────┼─────────────┼─────────────┼─────────────┤")
    
    for key in ['N Trades', 'Win Rate', 'Avg Return', 'Wtd Return', 'Sharpe', 'Sortino']:
        f = m_fixed.get(key, 0)
        z = m_zscore.get(key, 0)
        h = m_hybrid.get(key, 0)
        
        if key == 'N Trades':
            print(f"│ {key:<15} │ {f:>11.0f} │ {z:>11.0f} │ {h:>11.0f} │")
        elif key == 'Win Rate':
            print(f"│ {key:<15} │ {f:>10.1%} │ {z:>10.1%} │ {h:>10.1%} │")
        elif key in ['Avg Return', 'Wtd Return']:
            print(f"│ {key:<15} │ {f*100:>9.2f}% │ {z*100:>9.2f}% │ {h*100:>9.2f}% │")
        else:
            print(f"│ {key:<15} │ {f:>11.2f} │ {z:>11.2f} │ {h:>11.2f} │")
    
    print("└─────────────────┴─────────────┴─────────────┴─────────────┘")
    
    # Find winner
    sharpes = {'FIXED': m_fixed['Sharpe'], 'Z-SCORE': m_zscore['Sharpe'], 'HYBRID': m_hybrid['Sharpe']}
    winner = max(sharpes, key=sharpes.get)
    
    print(f"\n🏆 WINNER (by Sharpe): {winner}")
    print(f"   FIXED Sharpe:  {m_fixed['Sharpe']:.2f}")
    print(f"   Z-SCORE Sharpe: {m_zscore['Sharpe']:.2f}")
    print(f"   HYBRID Sharpe: {m_hybrid['Sharpe']:.2f}")
    
    if winner == "HYBRID":
        improvement = (m_hybrid['Sharpe'] - m_fixed['Sharpe']) / max(abs(m_fixed['Sharpe']), 0.01) * 100
        print(f"\n✅ HYBRID improves on FIXED by {improvement:.1f}%")
    elif winner == "FIXED":
        print(f"\n📊 FIXED remains the best approach")
    
    return m_fixed, m_zscore, m_hybrid


if __name__ == "__main__":
    simulate_with_existing_data()
