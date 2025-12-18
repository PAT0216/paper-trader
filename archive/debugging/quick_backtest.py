#!/usr/bin/env python3
"""
Quick test of walkforward_weekly_ensemble.py with just 2024
to compare fixed vs buggy results
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta
from src.data.cache import DataCache
from src.features.indicators import generate_features, create_target

ALL_FEATURES = [
    'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
    'BB_Width', 'Dist_SMA50', 'Dist_SMA200',
    'Return_1d', 'Return_5d',
    'OBV_Momentum', 'Volume_Ratio', 'VWAP_Dev',
    'ATR_Pct', 'BB_PctB', 'Vol_Ratio'
]

HORIZONS = [1, 5, 20]
HORIZON_WEIGHTS = {1: 0.5, 5: 0.3, 20: 0.2}
TOP_N = 10
INITIAL_CAPITAL = 100000

print("="*70)
print("QUICK TEST: walkforward_weekly_ensemble.py (2024 only)")
print("="*70)

# Load data
cache = DataCache()
tickers = cache.get_cached_tickers()[:100]  # Just 100 for speed

all_data = []
for ticker in tickers:
    df = cache.get_price_data(ticker)
    if df is not None and len(df) > 200:
        processed = generate_features(df, include_target=False)
        if len(processed) > 100:
            processed = processed.copy()
            processed['ticker'] = ticker
            all_data.append(processed)

full_df = pd.concat(all_data).sort_index()
full_df = full_df[full_df.index >= '2015-01-01']
full_df = full_df.dropna()

# Add noise features
np.random.seed(42)
for i in range(5):
    full_df[f'NOISE_{i}'] = np.random.randn(len(full_df))

print(f"Tickers: {full_df['ticker'].nunique()}")
print()

noise_features = [f'NOISE_{i}' for i in range(5)]
all_with_noise = ALL_FEATURES + noise_features

def select_features_better_than_noise(X, y, feature_names):
    model = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=4, 
                              random_state=42, verbosity=0)
    model.fit(X, y)
    importances = model.feature_importances_
    noise_indices = [i for i, f in enumerate(feature_names) if 'NOISE' in f]
    noise_baseline = np.mean([importances[i] for i in noise_indices])
    selected = [f for i, f in enumerate(feature_names) 
                if 'NOISE' not in f and importances[i] > noise_baseline]
    return selected if selected else feature_names[:8]

def train_ensemble(train_df, use_fixed_target=True):
    """Train ensemble. use_fixed_target controls which target calculation is used."""
    ensemble = {'models': {}, 'selected_features': {}, 'weights': HORIZON_WEIGHTS}
    
    for horizon in HORIZONS:
        df = train_df.copy()
        
        # Target calculation - FIXED vs BUGGY
        if use_fixed_target:
            # FIXED: N-day return
            df['Target'] = df.groupby('ticker')['Close'].transform(
                lambda x: x.pct_change(horizon).shift(-horizon)
            )
        else:
            # BUGGY: 1-day return shifted by N days
            df['Target'] = df.groupby('ticker')['Close'].transform(
                lambda x: x.pct_change().shift(-horizon)
            )
        
        df = df.dropna(subset=['Target'])
        
        if len(df) < 1000:
            continue
        
        X = df[all_with_noise].values
        y = df['Target'].values
        X = np.where(np.isinf(X), np.nan, X)
        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[valid], y[valid]
        
        if len(X) < 500:
            continue
        
        selected = select_features_better_than_noise(X, y, all_with_noise)
        selected_indices = [all_with_noise.index(f) for f in selected]
        X_selected = X[:, selected_indices]
        
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, 
                                  max_depth=5, random_state=42, verbosity=0)
        model.fit(X_selected, y)
        
        ensemble['models'][horizon] = model
        ensemble['selected_features'][horizon] = selected
    
    return ensemble

def predict_ensemble(ensemble, X_df, all_features):
    predictions = np.zeros(len(X_df))
    total_weight = 0
    
    for horizon, model in ensemble['models'].items():
        selected = ensemble['selected_features'].get(horizon, all_features[:8])
        weight = HORIZON_WEIGHTS.get(horizon, 0.33)
        
        try:
            X = X_df[selected].values
            X = np.where(np.isinf(X), 0, X)
            X = np.where(np.isnan(X), 0, X)
            raw_pred = model.predict(X)
            # NOTE: walkforward_weekly_ensemble does NOT normalize by horizon
            predictions += weight * raw_pred
            total_weight += weight
        except:
            continue
    
    return predictions / total_weight if total_weight > 0 else predictions

def run_backtest(full_df, use_fixed_target, label):
    """Run backtest for 2024 only."""
    all_dates = sorted(full_df.index.unique())
    mondays = [d for d in all_dates if d.weekday() == 0]
    
    # Filter to 2024
    mondays = [m for m in mondays if m >= pd.Timestamp('2024-01-01') 
               and m < pd.Timestamp('2025-01-01')]
    
    portfolio_value = INITIAL_CAPITAL
    weekly_returns = []
    ensemble = None
    
    for i, monday in enumerate(mondays[:-1]):
        next_monday = mondays[i + 1]
        
        # Retrain WEEKLY (like the original)
        train_df = full_df[full_df.index < monday]
        if len(train_df) < 5000:
            continue
        
        ensemble = train_ensemble(train_df, use_fixed_target)
        
        if ensemble is None or not ensemble['models']:
            continue
        
        # Get prediction data
        monday_data = full_df[full_df.index == monday]
        if monday_data.empty:
            continue
        monday_data = monday_data.set_index('ticker', append=True).reset_index(level=0, drop=True)
        
        if len(monday_data) < TOP_N:
            continue
        
        # Predict
        predictions = predict_ensemble(ensemble, monday_data, ALL_FEATURES)
        
        if len(predictions) == 0:
            continue
        
        # Select top N
        valid_tickers = monday_data.index.tolist()
        top_idx = np.argsort(predictions)[-TOP_N:]
        selected_tickers = [valid_tickers[i] for i in top_idx]
        
        # Calculate weekly return
        week_returns = []
        week_df = full_df[(full_df.index >= monday) & (full_df.index < next_monday)]
        
        for ticker in selected_tickers:
            ticker_week = week_df[week_df['ticker'] == ticker]
            if len(ticker_week) >= 2:
                ret = (ticker_week['Close'].iloc[-1] / ticker_week['Close'].iloc[0]) - 1
                week_returns.append(ret)
        
        if week_returns:
            avg_return = np.mean(week_returns)
            weekly_returns.append(avg_return)
            portfolio_value *= (1 + avg_return)
    
    # Results
    total_return = (portfolio_value / INITIAL_CAPITAL) - 1
    win_rate = sum(1 for r in weekly_returns if r > 0) / len(weekly_returns) if weekly_returns else 0
    
    print(f"\n{label}:")
    print(f"  Total Return: {total_return*100:+.1f}%")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Weeks traded: {len(weekly_returns)}")
    print(f"  Final Value: ${portfolio_value:,.2f}")
    
    return total_return

# Run both versions
print("\nRunning with BUGGY target (pct_change().shift(-horizon))...")
buggy_return = run_backtest(full_df, use_fixed_target=False, label="BUGGY TARGET")

print("\nRunning with FIXED target (pct_change(horizon).shift(-horizon))...")
fixed_return = run_backtest(full_df, use_fixed_target=True, label="FIXED TARGET")

print("\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"BUGGY: {buggy_return*100:+.1f}%")
print(f"FIXED: {fixed_return*100:+.1f}%")
print(f"Difference: {(fixed_return - buggy_return)*100:+.1f}%")
