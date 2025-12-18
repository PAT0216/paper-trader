#!/usr/bin/env python3
"""
Point-in-Time Backtest: Oct 1, 2024 to Today
============================================
Simulates what production would have done with daily retraining + rebalancing.
Updates ledger for production website.
"""

import matplotlib
matplotlib.use('Agg')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import warnings
import json
from datetime import datetime
from typing import Dict

warnings.filterwarnings('ignore')

from src.data.cache import DataCache
from src.features.indicators import generate_features, FEATURE_COLUMNS

# =============================================================================
# CONFIGURATION - PRODUCTION (DAILY RETRAIN + DAILY REBALANCE)
# =============================================================================

START_DATE = '2025-10-01'
END_DATE = '2025-12-17'  # Today
INITIAL_CASH = 10000
TOP_N = 10  # Top 10% of ~500 = ~50, but we cap at 10
HORIZON_WEIGHTS = {1: 0.5, 5: 0.3, 20: 0.2}


def load_data():
    """Load all price data from cache."""
    print("📦 Loading data from cache...")
    cache = DataCache()
    all_tickers = cache.get_cached_tickers()
    
    all_price_data = {}
    for ticker in all_tickers:
        df = cache.get_price_data(ticker)
        if df is not None and len(df) > 200:
            df = df[df.index >= '2010-01-01']
            if len(df) > 200:
                all_price_data[ticker] = df.copy()
    
    spy_data = all_price_data.pop('SPY', None)
    print(f"Loaded {len(all_price_data)} tickers")
    return all_price_data, spy_data


def train_ensemble(train_data: Dict[str, pd.DataFrame]):
    """Train ensemble using updated trainer.py (matching production)."""
    from src.models.trainer import train_ensemble as trainer_train
    
    try:
        ensemble = trainer_train(train_data, n_splits=3, save_model=False)
        if ensemble:
            ensemble['weights'] = HORIZON_WEIGHTS
        return ensemble
    except Exception as e:
        print(f"Training error: {e}")
        return None


def predict_ensemble(ensemble: Dict, df: pd.DataFrame) -> float:
    """Generate normalized prediction using ensemble."""
    if ensemble is None or not ensemble.get('models'):
        return 0.0
    
    features_df = generate_features(df.copy(), include_target=False)
    if features_df.empty:
        return 0.0
    features_df = features_df.tail(1)
    
    total_pred = 0
    total_weight = 0
    
    for horizon, model in ensemble['models'].items():
        selected = ensemble['selected_features'].get(horizon, FEATURE_COLUMNS)
        weight = HORIZON_WEIGHTS.get(horizon, 0.33)
        
        try:
            feature_list = [f for f in selected if f in features_df.columns]
            if not feature_list:
                continue
            X = features_df[feature_list].values
            X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
            raw_pred = model.predict(X)[0]
            
            # NORMALIZE to daily return
            daily_pred = raw_pred / horizon
            total_pred += weight * daily_pred
            total_weight += weight
        except:
            continue
    
    return total_pred / total_weight if total_weight > 0 else 0.0


def run_backtest():
    """Run daily backtest from Oct 1 to today."""
    print("=" * 70)
    print("POINT-IN-TIME BACKTEST: Oct 1, 2024 - Today")
    print("Configuration: DAILY retrain + DAILY rebalance (Production)")
    print("=" * 70)
    print()
    
    # Load data
    all_price_data, spy_data = load_data()
    
    # Get all trading days in range
    sample_df = list(all_price_data.values())[0]
    all_dates = sorted(sample_df.index)
    trading_days = [d for d in all_dates 
                    if d >= pd.Timestamp(START_DATE) 
                    and d <= pd.Timestamp(END_DATE)]
    
    print(f"Trading days: {len(trading_days)}")
    print(f"Date range: {trading_days[0].date()} to {trading_days[-1].date()}")
    print()
    
    # Initialize
    portfolio_value = INITIAL_CASH
    daily_returns = []
    portfolio_history = [(trading_days[0], portfolio_value)]
    model = None
    
    for day_idx in range(len(trading_days) - 1):
        today = trading_days[day_idx]
        next_day = trading_days[day_idx + 1]
        
        # DAILY retrain (matching production)
        train_data = {t: df[df.index < today] 
                     for t, df in all_price_data.items() 
                     if len(df[df.index < today]) > 100}
        
        if len(train_data) >= 50:
            model = train_ensemble(train_data)
        
        if model is None:
            continue
        
        # Generate predictions for all tickers
        predictions = {}
        for ticker, df in all_price_data.items():
            history = df[df.index <= today]
            if len(history) < 50:
                continue
            pred = predict_ensemble(model, history)
            predictions[ticker] = pred
        
        if len(predictions) < TOP_N:
            continue
        
        # Select top N
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        selected = [t for t, _ in sorted_preds[:TOP_N]]
        
        # Calculate daily returns
        day_returns = []
        for ticker in selected:
            df = all_price_data[ticker]
            if today not in df.index or next_day not in df.index:
                continue
            
            entry = df.loc[today, 'Close']
            exit_price = df.loc[next_day, 'Close']
            ret = (exit_price / entry) - 1
            day_returns.append(ret)
        
        if day_returns:
            avg_return = np.mean(day_returns)
            daily_returns.append(avg_return)
            portfolio_value *= (1 + avg_return)
            portfolio_history.append((next_day, portfolio_value))
            
            if day_idx % 10 == 0:
                print(f"  {today.date()}: ${portfolio_value:,.2f} ({avg_return*100:+.2f}%)")
    
    # Calculate metrics
    if not daily_returns:
        print("❌ No trades executed!")
        return None
    
    total_return = (portfolio_value / INITIAL_CASH) - 1
    days = len(daily_returns)
    annualized_return = ((1 + total_return) ** (252/days) - 1) if days > 0 else 0
    vol = np.std(daily_returns) * np.sqrt(252)
    sharpe = (np.mean(daily_returns) * 252) / vol if vol > 0 else 0
    
    # Max drawdown
    values = [v for _, v in portfolio_history]
    peak = values[0]
    max_dd = 0
    for v in values:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd
    
    win_rate = sum(1 for r in daily_returns if r > 0) / len(daily_returns)
    
    # SPY comparison
    if spy_data is not None:
        spy_period = spy_data[(spy_data.index >= trading_days[0]) & 
                              (spy_data.index <= trading_days[-1])]
        if len(spy_period) > 1:
            spy_return = (spy_period['Close'].iloc[-1] / spy_period['Close'].iloc[0]) - 1
        else:
            spy_return = 0
    else:
        spy_return = 0
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Trading Days: {days}")
    print()
    print(f"Initial Value:  ${INITIAL_CASH:,.2f}")
    print(f"Final Value:    ${portfolio_value:,.2f}")
    print(f"Total Return:   {total_return*100:+.2f}%")
    print(f"Annualized:     {annualized_return*100:+.2f}%")
    print(f"Sharpe Ratio:   {sharpe:.3f}")
    print(f"Max Drawdown:   {max_dd*100:.1f}%")
    print(f"Win Rate:       {win_rate*100:.1f}%")
    print()
    print(f"SPY Return:     {spy_return*100:+.2f}%")
    print(f"Excess Return:  {(total_return - spy_return)*100:+.2f}%")
    
    # Save results
    results = {
        'start_date': START_DATE,
        'end_date': END_DATE,
        'initial_value': INITIAL_CASH,
        'final_value': portfolio_value,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'spy_return': spy_return,
        'trading_days': days,
        'portfolio_history': [(str(d.date()), v) for d, v in portfolio_history]
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/pit_backtest_oct_dec_2024.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print(f"💾 Results saved to results/pit_backtest_oct_dec_2024.json")
    
    return results


if __name__ == "__main__":
    run_backtest()
