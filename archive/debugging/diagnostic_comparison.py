#!/usr/bin/env python3
"""
DIAGNOSTIC: Side-by-side comparison of Fair Comparison vs Replication

This script:
1. Uses ONE training period (data up to 2020-01-01)
2. Trains BOTH implementations on the EXACT same data
3. Makes predictions for the SAME stocks on the SAME date
4. Compares predictions and stock rankings

Should run in ~2-3 minutes, not hours.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime

from src.data.cache import DataCache
from src.features.indicators import generate_features, create_target, FEATURE_COLUMNS

# Match fair_comparison.py exactly
ALL_FEATURES = [
    'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
    'BB_Width', 'Dist_SMA50', 'Dist_SMA200',
    'Return_1d', 'Return_5d',
    'OBV_Momentum', 'Volume_Ratio', 'VWAP_Dev',
    'ATR_Pct', 'BB_PctB', 'Vol_Ratio'
]

HORIZONS = [1, 5, 20]
HORIZON_WEIGHTS = {1: 0.5, 5: 0.3, 20: 0.2}

# Test configuration
TRAIN_END = pd.Timestamp('2020-01-01')  # Train on data before this
PREDICT_DATE = pd.Timestamp('2020-01-06')  # Predict for this Monday
TOP_N = 10

print("="*70)
print("DIAGNOSTIC: Implementation Comparison")
print("="*70)
print(f"Train data: before {TRAIN_END.date()}")
print(f"Predict date: {PREDICT_DATE.date()}")
print()

# Load data
cache = DataCache()
tickers = cache.get_cached_tickers()[:100]  # Use 100 tickers for speed

print(f"Loading {len(tickers)} tickers...")
all_price_data = {}
for ticker in tickers:
    df = cache.get_price_data(ticker)
    if df is not None and len(df) > 200:
        df = df[df.index >= '2010-01-01']
        if len(df) > 200:
            all_price_data[ticker] = df.copy()

print(f"Loaded {len(all_price_data)} valid tickers")
print()

# ============================================================================
# BUILD FULL FEATURE DATAFRAME (for Fair Comparison style)
# ============================================================================
print("Building full feature DataFrame...")
all_features_data = []
for ticker, df in all_price_data.items():
    processed = generate_features(df, include_target=False)
    if len(processed) > 100:
        processed = processed.copy()
        processed['ticker'] = ticker
        all_features_data.append(processed)

full_features_df = pd.concat(all_features_data).sort_index()
print(f"Full features shape: {full_features_df.shape}")
print(f"Date range: {full_features_df.index.min()} to {full_features_df.index.max()}")
print()

# ============================================================================
# IMPLEMENTATION A: Fair Comparison Style (inline training)
# ============================================================================
print("="*70)
print("IMPLEMENTATION A: Fair Comparison Style")
print("="*70)

def add_noise_features_A(df, n_noise=5):
    np.random.seed(42)  # FIXED seed like fair_comparison
    for i in range(n_noise):
        df[f'NOISE_{i}'] = np.random.randn(len(df))
    return df

def select_features_A(X, y, feature_names):
    """Fair comparison's feature selection."""
    model = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=4, 
                              random_state=42, verbosity=0)
    model.fit(X, y)
    importances = model.feature_importances_
    noise_indices = [i for i, f in enumerate(feature_names) if 'NOISE' in f]
    noise_baseline = np.mean([importances[i] for i in noise_indices])
    selected = [f for i, f in enumerate(feature_names) 
                if 'NOISE' not in f and importances[i] > noise_baseline]
    return selected if selected else feature_names[:8]

def train_A(full_df, train_end):
    """Train using Fair Comparison approach."""
    # Filter to training period only
    train_df = full_df[full_df.index < train_end].copy()
    train_df = add_noise_features_A(train_df)
    
    noise_features = [f'NOISE_{i}' for i in range(5)]
    all_with_noise = ALL_FEATURES + noise_features
    
    ensemble = {'models': {}, 'selected_features': {}, 'weights': HORIZON_WEIGHTS}
    
    for horizon in HORIZONS:
        df = train_df.copy()
        # Target calculation: groupby ticker
        df['Target'] = df.groupby('ticker')['Close'].transform(
            lambda x: x.pct_change(horizon).shift(-horizon)
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
        
        selected = select_features_A(X, y, all_with_noise)
        selected_indices = [all_with_noise.index(f) for f in selected]
        X_selected = X[:, selected_indices]
        
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, 
                                  max_depth=5, random_state=42, verbosity=0)
        model.fit(X_selected, y)
        
        ensemble['models'][horizon] = model
        ensemble['selected_features'][horizon] = selected
        print(f"  {horizon}-day: {len(selected)} features, {len(X):,} samples")
    
    return ensemble

ensemble_A = train_A(full_features_df, TRAIN_END)
print()

# ============================================================================
# IMPLEMENTATION B: Replication Style (using trainer.py logic)
# ============================================================================
print("="*70)
print("IMPLEMENTATION B: Replication Style (trainer.py)")
print("="*70)

def select_features_B(X, y, feature_names):
    """Trainer.py's feature selection - with RANDOM noise."""
    np.random.seed(None)  # RANDOM seed like trainer.py
    noise = np.random.randn(len(X), 5)
    X_with_noise = np.hstack([X, noise])
    
    model = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=4, 
                              random_state=42, verbosity=0)
    model.fit(X_with_noise, y)
    
    importances = model.feature_importances_
    noise_baseline = np.mean(importances[-5:])
    selected = [f for i, f in enumerate(feature_names) if importances[i] > noise_baseline]
    return selected if selected else feature_names[:8]

def train_B(all_price_data, train_end):
    """Train using trainer.py approach."""
    ensemble = {'models': {}, 'selected_features': {}, 'weights': HORIZON_WEIGHTS}
    
    for horizon in HORIZONS:
        # Process each ticker separately, then concatenate
        all_features = []
        for ticker, df in all_price_data.items():
            processed_df = generate_features(df, include_target=False)
            processed_df = create_target(processed_df, target_type='regression', horizon=horizon)
            train_data = processed_df[processed_df.index < train_end]
            train_data = train_data.dropna(subset=['Target'])
            if len(train_data) > 50:
                all_features.append(train_data)
        
        if not all_features:
            continue
        
        full_df = pd.concat(all_features).sort_index()
        
        X = full_df[FEATURE_COLUMNS].values
        y = full_df['Target'].values
        X = np.where(np.isinf(X), np.nan, X)
        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[valid], y[valid]
        
        if len(X) < 500:
            continue
        
        # Train final model on all features first
        final_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05,
                                        max_depth=5, random_state=42, verbosity=0)
        final_model.fit(X, y)
        
        # Feature selection AFTER training (trainer.py style)
        useful_features = select_features_B(X, y, FEATURE_COLUMNS)
        
        if len(useful_features) < len(FEATURE_COLUMNS):
            feature_indices = [i for i, f in enumerate(FEATURE_COLUMNS) if f in useful_features]
            X_selected = X[:, feature_indices]
            final_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05,
                                            max_depth=5, random_state=42, verbosity=0)
            final_model.fit(X_selected, y)
        
        ensemble['models'][horizon] = final_model
        ensemble['selected_features'][horizon] = useful_features
        print(f"  {horizon}-day: {len(useful_features)} features, {len(X):,} samples")
    
    return ensemble

ensemble_B = train_B(all_price_data, TRAIN_END)
print()

# ============================================================================
# COMPARE PREDICTIONS
# ============================================================================
print("="*70)
print("PREDICTION COMPARISON")
print("="*70)

# Get data for prediction date (from full features df)
monday_data = full_features_df[full_features_df.index == PREDICT_DATE]
if monday_data.empty:
    # Try next available date
    next_dates = full_features_df[full_features_df.index > PREDICT_DATE].index.unique()[:5]
    for d in next_dates:
        monday_data = full_features_df[full_features_df.index == d]
        if not monday_data.empty:
            print(f"Using date {d.date()} instead of {PREDICT_DATE.date()}")
            PREDICT_DATE = d
            break

print(f"Prediction data shape: {monday_data.shape}")
print(f"Tickers available: {monday_data['ticker'].nunique()}")
print()

def predict_A(ensemble, monday_data, all_features):
    """Fair comparison prediction."""
    if not ensemble.get('models'):
        return {}
    
    predictions = {}
    for ticker in monday_data['ticker'].unique():
        ticker_data = monday_data[monday_data['ticker'] == ticker].iloc[-1:]
        
        total_pred = 0
        total_weight = 0
        
        for horizon, model in ensemble['models'].items():
            selected = ensemble['selected_features'].get(horizon, all_features[:8])
            weight = HORIZON_WEIGHTS.get(horizon, 0.33)
            
            try:
                X = ticker_data[selected].values
                X = np.where(np.isinf(X), 0, X)
                X = np.where(np.isnan(X), 0, X)
                raw_pred = model.predict(X)[0]
                daily_pred = raw_pred / horizon
                total_pred += weight * daily_pred
                total_weight += weight
            except Exception as e:
                continue
        
        if total_weight > 0:
            predictions[ticker] = total_pred / total_weight
    
    return predictions

def predict_B(ensemble, all_price_data, predict_date):
    """Replication prediction."""
    if not ensemble.get('models'):
        return {}
    
    predictions = {}
    for ticker, df in all_price_data.items():
        history = df[df.index <= predict_date]
        if len(history) < 50:
            continue
        
        features_df = generate_features(history.copy(), include_target=False)
        if features_df.empty:
            continue
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
                daily_pred = raw_pred / horizon
                total_pred += weight * daily_pred
                total_weight += weight
            except:
                continue
        
        if total_weight > 0:
            predictions[ticker] = total_pred / total_weight
    
    return predictions

preds_A = predict_A(ensemble_A, monday_data, ALL_FEATURES)
preds_B = predict_B(ensemble_B, all_price_data, PREDICT_DATE)

print(f"Predictions A: {len(preds_A)} tickers")
print(f"Predictions B: {len(preds_B)} tickers")
print()

# Compare common tickers
common = set(preds_A.keys()) & set(preds_B.keys())
print(f"Common tickers: {len(common)}")
print()

if common:
    # Compare predictions
    print("Sample predictions (first 15 common tickers):")
    print(f"{'Ticker':<8} {'Pred A':>12} {'Pred B':>12} {'Diff':>12} {'%Diff':>10}")
    print("-"*60)
    
    diffs = []
    pct_diffs = []
    for ticker in sorted(common)[:15]:
        pred_a = preds_A[ticker]
        pred_b = preds_B[ticker]
        diff = pred_a - pred_b
        pct_diff = (diff / abs(pred_a) * 100) if pred_a != 0 else 0
        diffs.append(abs(diff))
        pct_diffs.append(abs(pct_diff))
        print(f"{ticker:<8} {pred_a:>12.6f} {pred_b:>12.6f} {diff:>12.6f} {pct_diff:>9.1f}%")
    
    print()
    print(f"Avg absolute difference: {np.mean(diffs):.6f}")
    print(f"Max absolute difference: {np.max(diffs):.6f}")
    print(f"Avg % difference: {np.mean(pct_diffs):.1f}%")
    print()
    
    # Correlation between predictions
    common_list = sorted(common)
    a_vals = [preds_A[t] for t in common_list]
    b_vals = [preds_B[t] for t in common_list]
    corr = np.corrcoef(a_vals, b_vals)[0, 1]
    print(f"Prediction correlation: {corr:.4f}")
    print()
    
    # Compare rankings
    sorted_A = sorted(preds_A.items(), key=lambda x: x[1], reverse=True)
    sorted_B = sorted(preds_B.items(), key=lambda x: x[1], reverse=True)
    
    top_A = [t for t, _ in sorted_A[:TOP_N]]
    top_B = [t for t, _ in sorted_B[:TOP_N]]
    
    overlap = len(set(top_A) & set(top_B))
    
    print(f"Top {TOP_N} stocks selected:")  
    print(f"  A: {top_A}")
    print(f"  B: {top_B}")
    print(f"  Overlap: {overlap}/{TOP_N}")
    print()
    
    # Compare selected features (this is key!)
    print("="*70)
    print("SELECTED FEATURES COMPARISON (KEY DIAGNOSTIC)")
    print("="*70)
    for horizon in HORIZONS:
        feats_A = set(ensemble_A['selected_features'].get(horizon, []))
        feats_B = set(ensemble_B['selected_features'].get(horizon, []))
        overlap_feat = len(feats_A & feats_B)
        print(f"\n  {horizon}-day horizon:")
        print(f"    A selected: {len(feats_A)} features: {sorted(feats_A)}")
        print(f"    B selected: {len(feats_B)} features: {sorted(feats_B)}")
        print(f"    Overlap: {overlap_feat}")
        if feats_A != feats_B:
            print(f"    Only in A: {sorted(feats_A - feats_B)}")
            print(f"    Only in B: {sorted(feats_B - feats_A)}")
else:
    print("NO COMMON TICKERS - something is wrong!")

print()
print("="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
