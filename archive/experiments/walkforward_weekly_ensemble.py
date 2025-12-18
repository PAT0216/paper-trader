"""
Walk-Forward Backtest with Weekly Retraining

Features:
1. Weekly model retraining (every Monday)
2. Multi-horizon ensemble (1/5/20 day returns)
3. Dynamic noise-based feature selection at each training
4. Production-like portfolio construction
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta
from src.data.cache import DataCache
from src.features.indicators import generate_features, create_target

# All 15 features (will dynamically select better-than-noise at each training)
ALL_FEATURES = [
    'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
    'BB_Width', 'Dist_SMA50', 'Dist_SMA200',
    'Return_1d', 'Return_5d',
    'OBV_Momentum', 'Volume_Ratio', 'VWAP_Dev',
    'ATR_Pct', 'BB_PctB', 'Vol_Ratio'
]

HORIZONS = [1, 5, 20]
HORIZON_WEIGHTS = {1: 0.5, 5: 0.3, 20: 0.2}


def add_noise_features(df, n_noise=5):
    """Add random noise features for baseline comparison."""
    np.random.seed(42)
    for i in range(n_noise):
        df[f'NOISE_{i}'] = np.random.randn(len(df))
    return df


def select_features_better_than_noise(X, y, feature_names, n_noise=5):
    """
    Train a quick model and return only features that beat random noise.
    """
    # Train model with all features + noise
    model = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=4, 
                              random_state=42, verbosity=0)
    model.fit(X, y)
    
    # Get importances
    importances = model.feature_importances_
    
    # Calculate noise baseline (average importance of noise features)
    noise_indices = [i for i, f in enumerate(feature_names) if 'NOISE' in f]
    noise_baseline = np.mean([importances[i] for i in noise_indices])
    
    # Select features better than noise (excluding noise itself)
    selected = []
    for i, feat in enumerate(feature_names):
        if 'NOISE' not in feat and importances[i] > noise_baseline:
            selected.append(feat)
    
    return selected if selected else feature_names[:8]  # Fallback to first 8


def train_ensemble_with_noise_selection(train_df, all_features):
    """
    Train multi-horizon ensemble with dynamic noise-based feature selection.
    """
    ensemble = {
        'models': {},
        'selected_features': {},
        'weights': HORIZON_WEIGHTS,
        'horizons': HORIZONS
    }
    
    # Add noise features
    noise_features = [f'NOISE_{i}' for i in range(5)]
    all_with_noise = all_features + noise_features
    
    for horizon in HORIZONS:
        # Create target for this horizon
        df = train_df.copy()
        df['Target'] = df.groupby('ticker')['Close'].transform(
            lambda x: x.pct_change(horizon).shift(-horizon)  # FIXED: was pct_change().shift(-horizon)
        )
        df = df.dropna(subset=['Target'])
        
        if len(df) < 1000:
            continue
        
        # Prepare data
        X = df[all_with_noise].values
        y = df['Target'].values
        X = np.where(np.isinf(X), np.nan, X)
        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid]
        y = y[valid]
        
        if len(X) < 500:
            continue
        
        # Select features better than noise
        selected = select_features_better_than_noise(X, y, all_with_noise)
        
        # Get indices of selected features (from all_with_noise)
        selected_indices = [all_with_noise.index(f) for f in selected]
        X_selected = X[:, selected_indices]
        
        # Train final model
        model = xgb.XGBRegressor(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=5,
            random_state=42, 
            verbosity=0
        )
        model.fit(X_selected, y)
        
        ensemble['models'][horizon] = model
        ensemble['selected_features'][horizon] = selected
    
    return ensemble


def predict_ensemble(ensemble, X_df, all_features):
    """Make weighted ensemble prediction."""
    predictions = np.zeros(len(X_df))
    total_weight = 0
    
    for horizon, model in ensemble['models'].items():
        selected = ensemble['selected_features'].get(horizon, all_features[:8])
        weight = ensemble['weights'].get(horizon, 0.33)
        
        # Get feature values
        try:
            X = X_df[selected].values
            X = np.where(np.isinf(X), 0, X)
            X = np.where(np.isnan(X), 0, X)
            
            pred = model.predict(X)
            predictions += weight * pred
            total_weight += weight
        except Exception:
            continue
    
    return predictions / total_weight if total_weight > 0 else predictions


def run_walkforward_backtest(full_df, all_features, top_n=10, initial_capital=10000):
    """
    Walk-forward backtest with weekly retraining.
    """
    print('='*70)
    print('WALK-FORWARD BACKTEST: Weekly Retrain + Ensemble + Noise Selection')
    print('='*70)
    
    # Add noise features to dataframe
    full_df = add_noise_features(full_df.copy())
    
    all_dates = sorted(full_df.index.unique())
    
    # Find all Mondays for retraining
    mondays = [d for d in all_dates if d.weekday() == 0]
    
    portfolio_value = initial_capital
    results = []
    ensemble = None
    week_count = 0
    
    # Start from 2017 (2 years training buffer)
    start_date = pd.Timestamp('2017-01-01')
    mondays = [m for m in mondays if m >= start_date]
    
    print(f'Total weeks: {len(mondays)}')
    print(f'Date range: {mondays[0].date()} to {mondays[-1].date()}')
    print()
    
    for i, monday in enumerate(mondays[:-1]):  # Skip last week
        next_monday = mondays[i + 1]
        
        # Retrain every week
        train_df = full_df[full_df.index < monday]
        if len(train_df) < 5000:
            continue
        
        ensemble = train_ensemble_with_noise_selection(train_df, all_features)
        
        if not ensemble['models']:
            continue
        
        # Get Friday before next Monday for prediction
        week_df = full_df[(full_df.index >= monday) & (full_df.index < next_monday)]
        if week_df.empty:
            continue
        
        # Get latest data for each ticker on Monday
        monday_data = full_df[full_df.index == monday]
        if monday_data.empty:
            # Try next available day
            monday_data = week_df.groupby('ticker').first()
        else:
            monday_data = monday_data.set_index('ticker', append=True).reset_index(level=0, drop=True)
        
        if len(monday_data) < top_n:
            continue
        
        # Predict using ensemble
        predictions = predict_ensemble(ensemble, monday_data, all_features)
        
        # Select top N
        valid_tickers = monday_data.index.tolist()
        top_idx = np.argsort(predictions)[-top_n:]
        selected_tickers = [valid_tickers[i] for i in top_idx]
        
        # Calculate weekly return for selected stocks
        week_returns = []
        for ticker in selected_tickers:
            ticker_week = week_df[week_df['ticker'] == ticker]
            if len(ticker_week) >= 2:
                ret = (ticker_week['Close'].iloc[-1] / ticker_week['Close'].iloc[0]) - 1
                week_returns.append(ret)
        
        if week_returns:
            avg_return = np.mean(week_returns)
            portfolio_value *= (1 + avg_return)
            
            results.append({
                'date': monday,
                'year': monday.year,
                'return': avg_return,
                'value': portfolio_value,
                'n_features': len(ensemble['selected_features'].get(1, []))
            })
            
            week_count += 1
            if week_count % 52 == 0:  # Print yearly progress
                print(f'  Year {monday.year}: Value = ${portfolio_value:,.0f}')
    
    return pd.DataFrame(results)


def main():
    print('Loading data...')
    cache = DataCache()
    tickers = cache.get_cached_tickers()[:100]
    
    all_data = []
    for ticker in tickers:
        df = cache.get_price_data(ticker)
        if df is not None and len(df) > 200:
            processed = generate_features(df, include_target=False)
            processed['ticker'] = ticker
            all_data.append(processed)
    
    full_df = pd.concat(all_data).sort_index()
    full_df = full_df[full_df.index >= '2015-01-01']
    full_df = full_df.dropna()
    
    print(f'Total samples: {len(full_df):,}')
    print(f'Tickers: {full_df["ticker"].nunique()}')
    print()
    
    # Run backtest
    results_df = run_walkforward_backtest(full_df, ALL_FEATURES)
    
    if results_df.empty:
        print('No results!')
        return
    
    # Calculate overall metrics
    print()
    print('='*70)
    print('OVERALL RESULTS')
    print('='*70)
    
    total_ret = (results_df['value'].iloc[-1] / results_df['value'].iloc[0]) - 1
    years = len(results_df) / 52
    cagr = ((1 + total_ret) ** (1/years) - 1) if years > 0 else 0
    vol = results_df['return'].std() * np.sqrt(52)
    sharpe = (results_df['return'].mean() * 52) / vol if vol > 0 else 0
    max_dd = (results_df['value'] / results_df['value'].cummax() - 1).min()
    win_rate = (results_df['return'] > 0).mean()
    
    print(f'Total Return: {total_ret*100:.1f}%')
    print(f'CAGR: {cagr*100:.2f}%')
    print(f'Sharpe: {sharpe:.3f}')
    print(f'Max Drawdown: {max_dd*100:.1f}%')
    print(f'Win Rate: {win_rate*100:.1f}%')
    print(f'Final Value: ${results_df["value"].iloc[-1]:,.2f}')
    print()
    
    # Year-by-year
    print('='*70)
    print('YEAR-BY-YEAR BREAKDOWN')
    print('='*70)
    
    yearly = results_df.groupby('year').agg({
        'return': ['mean', 'std', 'sum', 'count'],
        'n_features': 'mean'
    })
    yearly.columns = ['mean_ret', 'std_ret', 'total_ret', 'weeks', 'avg_features']
    yearly['annual_ret'] = (1 + yearly['total_ret']).apply(lambda x: x - 1) * 100
    yearly['sharpe'] = (yearly['mean_ret'] * 52) / (yearly['std_ret'] * np.sqrt(52))
    
    print(f'{"Year":<6} {"Return":<12} {"Sharpe":<10} {"Weeks":<8} {"Avg Features"}')
    print('-'*55)
    for year, row in yearly.iterrows():
        print(f'{year:<6} {row["annual_ret"]:>+8.1f}%    {row["sharpe"]:>+.2f}      {int(row["weeks"]):<8} {row["avg_features"]:.1f}')
    
    # Save results
    results_df.to_csv('results/walkforward_weekly_ensemble.csv', index=False)
    print()
    print('💾 Saved to results/walkforward_weekly_ensemble.csv')


if __name__ == '__main__':
    main()
