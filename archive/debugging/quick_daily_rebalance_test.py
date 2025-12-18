"""
Quick Test: Weekly Retrain + Daily Rebalance
=============================================
Same model as walkforward_fair_comparison.py but with:
- 1 year test (2024)
- Weekly retraining (every week)
- Daily rebalancing (every trading day)

Purpose: Sanity check before production deployment
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from src.data.cache import DataCache
from src.features.indicators import generate_features, FEATURE_COLUMNS

# All 15 features
ALL_FEATURES = FEATURE_COLUMNS

HORIZONS = [1, 5, 20]
HORIZON_WEIGHTS = {1: 0.5, 5: 0.3, 20: 0.2}

# Configuration - MODIFIED FOR THIS TEST
RETRAIN_DAYS = 5      # Weekly retrain (every 5 trading days)
TOP_N = 10
INITIAL_CAPITAL = 100000
TEST_YEAR = 2024


def add_noise_features(df, n_noise=5):
    """Add random noise features for baseline comparison."""
    np.random.seed(42)
    for i in range(n_noise):
        df[f'NOISE_{i}'] = np.random.randn(len(df))
    return df


def select_features_better_than_noise(X, y, feature_names, n_noise=5):
    """Train a quick model and return only features that beat random noise."""
    model = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=4, 
                              random_state=42, verbosity=0)
    model.fit(X, y)
    
    importances = model.feature_importances_
    noise_indices = [i for i, f in enumerate(feature_names) if 'NOISE' in f]
    noise_baseline = np.mean([importances[i] for i in noise_indices])
    
    selected = []
    for i, feat in enumerate(feature_names):
        if 'NOISE' not in feat and importances[i] > noise_baseline:
            selected.append(feat)
    
    return selected if selected else feature_names[:8]


def train_ensemble_with_noise_selection(train_df, all_features):
    """Train multi-horizon ensemble with dynamic noise-based feature selection."""
    ensemble = {
        'models': {},
        'selected_features': {},
        'weights': HORIZON_WEIGHTS,
        'horizons': HORIZONS
    }
    
    noise_features = [f'NOISE_{i}' for i in range(5)]
    all_with_noise = all_features + noise_features
    
    for horizon in HORIZONS:
        df = train_df.copy()
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
        X = X[valid]
        y = y[valid]
        
        if len(X) < 500:
            continue
        
        selected = select_features_better_than_noise(X, y, all_with_noise)
        selected_indices = [all_with_noise.index(f) for f in selected]
        X_selected = X[:, selected_indices]
        
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


def predict_ensemble_normalized(ensemble, X_df, all_features):
    """Make weighted ensemble prediction with normalization."""
    if X_df.empty:
        return np.array([])
    
    predictions = np.zeros(len(X_df))
    total_weight = 0
    
    for horizon, model in ensemble['models'].items():
        selected = ensemble['selected_features'].get(horizon, all_features[:8])
        weight = ensemble['weights'].get(horizon, 0.33)
        
        try:
            X = X_df[selected].values
            X = np.where(np.isinf(X), 0, X)
            X = np.where(np.isnan(X), 0, X)
            
            raw_pred = model.predict(X)
            daily_pred = raw_pred / horizon  # NORMALIZE
            
            predictions += weight * daily_pred
            total_weight += weight
        except Exception:
            continue
    
    return predictions / total_weight if total_weight > 0 else predictions


def run_daily_rebalance_backtest(full_df, all_features, top_n=TOP_N, initial_capital=INITIAL_CAPITAL):
    """
    Walk-forward backtest with WEEKLY retraining, DAILY rebalancing.
    """
    print('='*70)
    print('QUICK TEST: Weekly Retrain + Daily Rebalance')
    print('='*70)
    print(f'Test Year: {TEST_YEAR}')
    print(f'Retrain: Weekly (every {RETRAIN_DAYS} trading days)')
    print(f'Rebalance: DAILY')
    print(f'Universe: {full_df["ticker"].nunique()} tickers')
    print(f'Capital: ${initial_capital:,}')
    print()
    
    # Add noise features
    full_df = add_noise_features(full_df.copy())
    
    # Get all trading days in test year
    all_dates = sorted(full_df.index.unique())
    test_dates = [d for d in all_dates 
                  if d >= pd.Timestamp(f'{TEST_YEAR}-01-01') 
                  and d < pd.Timestamp(f'{TEST_YEAR+1}-01-01')]
    
    print(f'Trading days in {TEST_YEAR}: {len(test_dates)}')
    print(f'Date range: {test_dates[0].date()} to {test_dates[-1].date()}')
    print()
    
    portfolio_value = initial_capital
    results = []
    ensemble = None
    retrain_count = 0
    
    for day_idx in range(len(test_dates) - 1):
        today = test_dates[day_idx]
        tomorrow = test_dates[day_idx + 1]
        
        # Retrain WEEKLY (every 5 trading days)
        if day_idx % RETRAIN_DAYS == 0:
            train_df = full_df[full_df.index < today]
            if len(train_df) < 5000:
                continue
            
            ensemble = train_ensemble_with_noise_selection(train_df, all_features)
            retrain_count += 1
            
            if retrain_count % 10 == 0:
                print(f'  Retrain #{retrain_count}: {today.date()}')
        
        if ensemble is None or not ensemble['models']:
            continue
        
        # Get today's data for prediction
        today_data = full_df[full_df.index == today]
        if today_data.empty:
            continue
        
        today_data = today_data.set_index('ticker', append=True).reset_index(level=0, drop=True)
        
        if len(today_data) < top_n:
            continue
        
        # Predict using ensemble
        predictions = predict_ensemble_normalized(ensemble, today_data, all_features)
        
        if len(predictions) == 0:
            continue
        
        # Select top N
        valid_tickers = today_data.index.tolist()
        top_idx = np.argsort(predictions)[-top_n:]
        selected_tickers = [valid_tickers[i] for i in top_idx]
        
        # Calculate DAILY return for selected stocks (Close to Close)
        day_returns = []
        
        for ticker in selected_tickers:
            # Get today's close and tomorrow's close
            today_row = full_df[(full_df.index == today) & (full_df['ticker'] == ticker)]
            tomorrow_row = full_df[(full_df.index == tomorrow) & (full_df['ticker'] == ticker)]
            
            if len(today_row) == 1 and len(tomorrow_row) == 1:
                entry = today_row['Close'].iloc[0]
                exit_price = tomorrow_row['Close'].iloc[0]
                ret = (exit_price / entry) - 1
                day_returns.append(ret)
        
        if day_returns:
            avg_return = np.mean(day_returns)
            portfolio_value *= (1 + avg_return)
            
            results.append({
                'date': today,
                'return': avg_return,
                'value': portfolio_value,
                'n_stocks': len(day_returns)
            })
    
    print(f'\nTotal retrains: {retrain_count}')
    print(f'Total trading days: {len(results)}')
    
    return pd.DataFrame(results)


def main():
    print('='*70)
    print('SANITY CHECK: Daily Rebalance Test')
    print('='*70)
    print()
    
    print('Loading data from cache...')
    cache = DataCache()
    tickers = cache.get_cached_tickers()
    print(f'Found {len(tickers)} tickers')
    
    all_data = []
    for ticker in tickers:
        df = cache.get_price_data(ticker)
        if df is not None and len(df) > 200:
            processed = generate_features(df, include_target=False)
            if len(processed) > 100:
                processed['ticker'] = ticker
                all_data.append(processed)
    
    full_df = pd.concat(all_data).sort_index()
    full_df = full_df[full_df.index >= '2015-01-01']
    full_df = full_df.dropna()
    
    print(f'Total samples: {len(full_df):,}')
    print(f'Tickers: {full_df["ticker"].nunique()}')
    print()
    
    # Run backtest
    results_df = run_daily_rebalance_backtest(full_df, ALL_FEATURES)
    
    if results_df.empty:
        print('No results!')
        return
    
    # Calculate metrics
    print()
    print('='*70)
    print('RESULTS')
    print('='*70)
    
    total_ret = (results_df['value'].iloc[-1] / results_df['value'].iloc[0]) - 1
    days = len(results_df)
    annual_ret = ((1 + total_ret) ** (252/days) - 1) if days > 0 else 0
    vol = results_df['return'].std() * np.sqrt(252)
    sharpe = (results_df['return'].mean() * 252) / vol if vol > 0 else 0
    max_dd = (results_df['value'] / results_df['value'].cummax() - 1).min()
    win_rate = (results_df['return'] > 0).mean()
    
    print(f'Test Period: {TEST_YEAR}')
    print(f'Trading Days: {days}')
    print(f'Total Return: {total_ret*100:+.1f}%')
    print(f'Annualized Return: {annual_ret*100:+.2f}%')
    print(f'Sharpe Ratio: {sharpe:.3f}')
    print(f'Max Drawdown: {max_dd*100:.1f}%')
    print(f'Win Rate: {win_rate*100:.1f}%')
    print(f'Final Value: ${results_df["value"].iloc[-1]:,.2f}')
    print()
    
    # Monthly breakdown
    print('='*70)
    print('MONTHLY BREAKDOWN')
    print('='*70)
    
    results_df['month'] = results_df['date'].dt.to_period('M')
    monthly = results_df.groupby('month').agg({
        'return': ['sum', 'count', 'mean', 'std']
    })
    monthly.columns = ['total_ret', 'days', 'avg_ret', 'std_ret']
    monthly['monthly_ret'] = (1 + monthly['total_ret']) - 1
    
    print(f'{"Month":<10} {"Return":<12} {"Days":<8} {"Win Rate"}')
    print('-'*45)
    for month, row in monthly.iterrows():
        wins = results_df[results_df['month'] == month]['return'].gt(0).mean()
        print(f'{str(month):<10} {row["monthly_ret"]*100:>+8.2f}%    {int(row["days"]):<8} {wins*100:.0f}%')
    
    # Save results
    results_df.to_csv(f'results/daily_rebalance_test_{TEST_YEAR}.csv', index=False)
    print()
    print(f'Saved to results/daily_rebalance_test_{TEST_YEAR}.csv')
    
    # Summary comparison
    print()
    print('='*70)
    print('COMPARISON WITH WEEKLY REBALANCE')
    print('='*70)
    print('This test: Weekly retrain, DAILY rebalance')
    print('Production: Monthly retrain, WEEKLY rebalance')
    print()
    print('If results are directionally similar (positive return, Sharpe > 0),')
    print('the model is robust and ready for production.')


if __name__ == '__main__':
    main()

