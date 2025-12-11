#!/usr/bin/env python3
"""
Stop-Loss Threshold Sweep Test

Tests multiple stop-loss thresholds to find optimal setting.
Uses cached data for speed.
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '.')

from src.data.cache import get_cache
from src.models.predictor import Predictor


def run_threshold_sweep():
    # Load cached data
    print("📊 Loading data from cache...")
    cache = get_cache()
    
    # DIVERSE MIX: Include underperformers, cyclicals, value traps
    # Not just FAANG - real S&P 500 mix
    tickers = [
        # Tech (mix of good and bad)
        'AAPL', 'INTC', 'IBM', 'CSCO',  # INTC, IBM underperformed
        # Financials (volatile in crises)
        'BAC', 'C', 'WFC',  # Bank stocks - crashed in 2008, 2020
        # Energy (crashed in 2020)
        'XOM', 'CVX', 'OXY',  # Energy - massive drawdowns
        # Retail (mixed)
        'WMT', 'TGT', 'KSS',  # KSS is a value trap
        # Pharma
        'PFE', 'MRK', 'BMY',  # Slower growth, more volatile
        # Industrials
        'BA', 'GE', 'MMM',  # BA crashed, GE struggled
        # Consumer
        'KO', 'MCD', 'DIS',  # DIS crashed post-COVID
    ]
    
    print(f"   Testing with DIVERSE MIX: {len(tickers)} tickers")
    print(f"   Includes: Tech underperformers, Banks, Energy, Cyclicals")
    
    data_dict = {}
    for ticker in tickers:
        df = cache.get_price_data(ticker)
        if df is not None and len(df) > 100:
            data_dict[ticker] = df
    
    print(f"   Loaded {len(data_dict)} tickers\n")
    
    predictor = Predictor()
    
    # Test period: last 2 years
    sample_df = list(data_dict.values())[0]
    test_dates = sample_df.index[-504:-1]  # ~2 years
    
    # Stop-loss thresholds to test
    thresholds = [None, 0.05, 0.08, 0.10, 0.15, 0.20]
    
    print("=" * 70)
    print("STOP-LOSS THRESHOLD SWEEP")
    print("=" * 70)
    
    results = []
    
    for stop_loss_pct in thresholds:
        trades = []
        positions = {}  # {ticker: {'entry_price': x, 'shares': n, 'entry_date': d}}
        
        for i, date in enumerate(test_dates):
            if i % 100 == 0 and i > 0:
                pass  # Silent progress
            
            current_prices = {}
            predictions = {}
            
            for ticker, df in data_dict.items():
                if date not in df.index:
                    continue
                
                idx = df.index.get_loc(date)
                if idx < 60 or idx >= len(df) - 1:
                    continue
                
                current_prices[ticker] = df.iloc[idx]['Close']
                
                try:
                    pred = predictor.predict(df.iloc[:idx+1])
                    predictions[ticker] = pred
                except:
                    continue
            
            # Check stop-losses first (if enabled)
            if stop_loss_pct is not None:
                for ticker in list(positions.keys()):
                    if ticker not in current_prices:
                        continue
                    
                    entry_price = positions[ticker]['entry_price']
                    current_price = current_prices[ticker]
                    loss_pct = (current_price - entry_price) / entry_price
                    
                    if loss_pct < -stop_loss_pct:
                        # Stop-loss triggered
                        pnl = loss_pct * 100  # Convert to %
                        trades.append({
                            'exit_type': 'STOP_LOSS',
                            'pnl_pct': pnl,
                            'holding_days': (date - positions[ticker]['entry_date']).days
                        })
                        del positions[ticker]
            
            # Generate new signals
            buy_thresh = 0.005
            
            for ticker, pred in predictions.items():
                if pred > buy_thresh and ticker not in positions:
                    # Enter position
                    positions[ticker] = {
                        'entry_price': current_prices[ticker],
                        'entry_date': date
                    }
                elif pred < -buy_thresh and ticker in positions:
                    # Exit signal
                    entry_price = positions[ticker]['entry_price']
                    current_price = current_prices[ticker]
                    pnl = (current_price - entry_price) / entry_price * 100
                    trades.append({
                        'exit_type': 'SIGNAL',
                        'pnl_pct': pnl,
                        'holding_days': (date - positions[ticker]['entry_date']).days
                    })
                    del positions[ticker]
        
        # Calculate metrics
        if not trades:
            continue
        
        trades_df = pd.DataFrame(trades)
        n_trades = len(trades_df)
        win_rate = (trades_df['pnl_pct'] > 0).mean()
        avg_return = trades_df['pnl_pct'].mean()
        total_return = trades_df['pnl_pct'].sum()
        
        stop_loss_count = (trades_df['exit_type'] == 'STOP_LOSS').sum()
        
        std_return = trades_df['pnl_pct'].std()
        sharpe = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0
        
        results.append({
            'stop_loss': f"{stop_loss_pct*100:.0f}%" if stop_loss_pct else "None",
            'n_trades': n_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'total_return': total_return,
            'sharpe': sharpe,
            'stop_triggered': stop_loss_count
        })
    
    # Print results
    print("\n┌────────────┬────────┬──────────┬───────────┬────────────┬─────────┬───────────┐")
    print("│ Stop-Loss  │ Trades │ Win Rate │ Avg Ret   │ Total Ret  │ Sharpe  │ Stopped   │")
    print("├────────────┼────────┼──────────┼───────────┼────────────┼─────────┼───────────┤")
    
    for r in results:
        print(f"│ {r['stop_loss']:>10} │ {r['n_trades']:>6} │ {r['win_rate']:>7.1%} │ {r['avg_return']:>8.2f}% │ {r['total_return']:>9.1f}% │ {r['sharpe']:>7.2f} │ {r['stop_triggered']:>9} │")
    
    print("└────────────┴────────┴──────────┴───────────┴────────────┴─────────┴───────────┘")
    
    # Find best
    best = max(results, key=lambda x: x['sharpe'])
    print(f"\n🏆 BEST: {best['stop_loss']} stop-loss (Sharpe: {best['sharpe']:.2f})")
    
    return results


if __name__ == "__main__":
    run_threshold_sweep()
