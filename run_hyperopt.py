#!/usr/bin/env python3
"""
Hyperparameter Optimization with Overfitting Safeguards

Quick Grid Search (Method 1):
- Test 3-4 key XGBoost params with 3 values each
- Use TimeSeriesSplit for proper walk-forward validation
- Early stopping to prevent overfitting

Usage: python run_hyperopt.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import fetch_data
from src.data.cache import DataCache
from src.features.indicators import generate_features
from src.models.trainer import create_target, FEATURE_COLUMNS
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit


def prepare_training_data(data_dict, end_date='2023-01-01'):
    """Prepare combined training data from multiple tickers."""
    all_features = []
    
    for ticker, df in data_dict.items():
        train_df = df[df.index < end_date].copy()
        if len(train_df) < 100:
            continue
            
        processed = generate_features(train_df, include_target=False)
        processed = create_target(processed, target_type='regression')
        processed = processed.dropna(subset=['Target'])
        
        if len(processed) > 50:
            all_features.append(processed)
    
    if not all_features:
        return None, None
    
    full_df = pd.concat(all_features).sort_index()
    
    X = full_df[FEATURE_COLUMNS].values
    y = full_df['Target'].values
    X = np.where(np.isinf(X), np.nan, X)
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[valid], y[valid]
    
    return X, y


def evaluate_params(X, y, params, n_splits=5):
    """
    Evaluate XGBoost parameters using TimeSeriesSplit.
    
    Returns:
        Dict with mean/std of metrics across folds
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    scores = {
        'train_rmse': [],
        'val_rmse': [],
        'train_dir_acc': [],
        'val_dir_acc': []
    }
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train with early stopping
        model = xgb.XGBRegressor(
            **params,
            objective='reg:squarederror',
            random_state=42,
            early_stopping_rounds=10
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Get predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # RMSE
        train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
        val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
        
        # Directional accuracy (most important for trading)
        train_dir_acc = np.mean(np.sign(train_pred) == np.sign(y_train))
        val_dir_acc = np.mean(np.sign(val_pred) == np.sign(y_val))
        
        scores['train_rmse'].append(train_rmse)
        scores['val_rmse'].append(val_rmse)
        scores['train_dir_acc'].append(train_dir_acc)
        scores['val_dir_acc'].append(val_dir_acc)
    
    return {
        'train_rmse_mean': np.mean(scores['train_rmse']),
        'val_rmse_mean': np.mean(scores['val_rmse']),
        'train_dir_acc_mean': np.mean(scores['train_dir_acc']),
        'val_dir_acc_mean': np.mean(scores['val_dir_acc']),
        'overfit_gap': np.mean(scores['train_dir_acc']) - np.mean(scores['val_dir_acc']),
    }


def run_grid_search():
    """
    Run quick grid search with 3-4 key parameters.
    
    Quant-approved approach:
    1. Focus on regularization to prevent overfitting
    2. Use TimeSeriesSplit (not random split)
    3. Early stopping on validation set
    4. Check train/val gap for overfitting
    """
    print("=" * 70)
    print("HYPERPARAMETER OPTIMIZATION (Quick Grid Search)")
    print("=" * 70)
    print("Quant Principles:")
    print("  1. TimeSeriesSplit for proper temporal validation")
    print("  2. Early stopping to prevent overfitting")
    print("  3. Check train/val gap (should be small)")
    print("  4. Prefer regularized params when performance is similar")
    print("=" * 70)
    
    # Load data from cache
    print("\n📊 Loading data from cache...")
    cache = DataCache()
    stats = cache.get_cache_stats()
    tickers = stats['ticker'].tolist()[:100]  # Use subset for speed
    
    data_dict = fetch_data(tickers, period='max')
    
    # Filter to 2010+
    MIN_DATE = '2010-01-01'
    for ticker in list(data_dict.keys()):
        df = data_dict[ticker]
        data_dict[ticker] = df[df.index >= MIN_DATE]
        if len(data_dict[ticker]) < 500:
            del data_dict[ticker]
    
    print(f"   {len(data_dict)} tickers loaded")
    
    # Prepare training data (up to 2023 for tuning)
    print("\n🔧 Preparing training data (2010-2022)...")
    X, y = prepare_training_data(data_dict, '2023-01-01')
    print(f"   {len(X):,} samples prepared")
    
    # Define parameter grid (focused, not exhaustive)
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_weight': [1, 5],
        'subsample': [0.8, 1.0],
        'reg_lambda': [1, 5],
    }
    
    # Calculate total combinations
    n_combos = 1
    for values in param_grid.values():
        n_combos *= len(values)
    
    print(f"\n🔍 Testing {n_combos} parameter combinations...")
    print("   (This will take ~15-30 minutes)")
    
    results = []
    best_score = 0
    best_params = None
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for i, combo in enumerate(product(*param_values)):
        params = dict(zip(param_names, combo))
        
        if (i + 1) % 20 == 0:
            print(f"   Progress: {i+1}/{n_combos} ({(i+1)/n_combos*100:.1f}%)")
        
        try:
            scores = evaluate_params(X, y, params, n_splits=5)
            
            result = {**params, **scores}
            results.append(result)
            
            # Track best (by validation directional accuracy)
            if scores['val_dir_acc_mean'] > best_score:
                # Only accept if not overfitting too much (gap < 5%)
                if scores['overfit_gap'] < 0.05:
                    best_score = scores['val_dir_acc_mean']
                    best_params = params.copy()
        except Exception as e:
            pass
    
    # Convert to DataFrame and sort
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('val_dir_acc_mean', ascending=False)
    
    # Save results
    results_df.to_csv('results/hyperopt_results.csv', index=False)
    
    # Print results
    print("\n" + "=" * 70)
    print("TOP 10 PARAMETER COMBINATIONS")
    print("=" * 70)
    print("\n(Sorted by Validation Directional Accuracy)")
    print("-" * 70)
    
    display_cols = ['max_depth', 'n_estimators', 'learning_rate', 'reg_lambda', 
                    'val_dir_acc_mean', 'train_dir_acc_mean', 'overfit_gap']
    
    print(results_df[display_cols].head(10).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("BEST PARAMETERS (with overfitting check)")
    print("=" * 70)
    
    if best_params:
        print(f"\nBest Validation Directional Accuracy: {best_score:.4f}")
        print("\nOptimal Parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        
        # Save best params
        best_params_str = str(best_params)
        with open('results/best_hyperparams.txt', 'w') as f:
            f.write(f"Best Hyperparameters (found {datetime.now().isoformat()})\n")
            f.write("=" * 50 + "\n\n")
            for k, v in best_params.items():
                f.write(f"{k}: {v}\n")
            f.write(f"\nValidation Directional Accuracy: {best_score:.4f}\n")
        
        print("\n✅ Best parameters saved to results/best_hyperparams.txt")
    else:
        print("\n⚠️ No parameters found with acceptable overfit gap")
    
    # Comparison with current defaults
    print("\n" + "=" * 70)
    print("COMPARISON: CURRENT vs OPTIMIZED")
    print("=" * 70)
    
    current_params = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.05,
        'min_child_weight': 1,
        'subsample': 1.0,
        'reg_lambda': 1
    }
    
    current_scores = evaluate_params(X, y, current_params)
    
    print(f"\n{'Metric':<30} {'Current':>15} {'Optimized':>15}")
    print("-" * 60)
    print(f"{'Validation Dir. Accuracy':<30} {current_scores['val_dir_acc_mean']:>15.4f} {best_score:>15.4f}")
    print(f"{'Train-Val Gap (Overfit)':<30} {current_scores['overfit_gap']:>15.4f} {results_df.iloc[0]['overfit_gap']:>15.4f}")
    
    improvement = (best_score - current_scores['val_dir_acc_mean']) / current_scores['val_dir_acc_mean'] * 100
    print(f"\n{'Improvement':<30} {improvement:>+15.2f}%")
    
    return best_params, results_df


if __name__ == "__main__":
    best_params, results = run_grid_search()
