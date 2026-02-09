
"""
Hyperparameter Optimization & Robustness Test
---------------------------------------------
"Jim Simons + Karpathy" Mode:
Systematic grid search to find the most robust strategy configuration across different market regimes.

Variables:
1. Rebalancing Frequency: [Daily, Weekly]
2. Target Horizon: [5d, 10d, 20d]
3. Timelines (Regimes):
   - 2022 (Bear/Volatile)
   - 2023 (Recovery)
   - 2025 (Current)

Metric: Sharpe Ratio & Total Return
"""

import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from dataclasses import dataclass
from itertools import product
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.loader import fetch_data
from src.features.momentum_features import compute_momentum_features, create_momentum_target
from src.trading.risk_manager import RiskManager, RiskLimits

@dataclass
class SimConfig:
    name: str
    start_date: str
    end_date: str
    rebalance_freq: str
    target_horizon: int
    universe_size: int = 100 # Random sample for speed, or full?

def get_risk_manager(mode='relaxed'):
    # Defaulting to relaxed for pure strategy signal testing
    limits = RiskLimits(max_position_pct=0.15, max_sector_pct=1.0, volatility_lookback=30)
    return RiskManager(risk_limits=limits)

def feature_engineering(df, horizon):
    # Dynamic Horizon Target
    df = compute_momentum_features(df.copy())
    df = create_momentum_target(df, horizon=horizon)
    target_col = 'Target' # create_momentum_target makes 'Target' column
    return df, target_col

def train_model(df_train, features, target_col):
    valid = df_train.dropna(subset=[target_col] + features)
    if len(valid) < 100: return None
    
    X = valid[features].values
    y = valid[target_col].values
    
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0
    )
    model.fit(X, y)
    return model

def run_simulation(config: SimConfig, data_dict):
    print(f"  > Simulating: {config.name}")
    
    # 1. Feature Gen (Specific to Horizon)
    feature_dfs = {}
    from src.features.momentum_features import MOMENTUM_FEATURE_COLUMNS
    features = MOMENTUM_FEATURE_COLUMNS
    target_col = 'Target'
    
    for ticker, df in data_dict.items():
        if df.empty: continue
        if ticker == 'SPY': continue
        
        f_df, t_col = feature_engineering(df, config.target_horizon)
        feature_dfs[ticker] = f_df
        
    start_dt = pd.Timestamp(config.start_date)
    end_dt = pd.Timestamp(config.end_date)
    
    # Common Dates
    dates = sorted(list(set().union(*[df.index for df in feature_dfs.values()])))
    sim_dates = [d for d in dates if start_dt <= d <= end_dt]
    
    cash = 10000.0
    holdings = {}
    history = []
    
    # Pre-train initially? Or Daily Retrain?
    # For Optimization Speed: WEEKLY Retrain?
    # Daily Retrain is too slow for Grid Search.
    # Let's do Weekly Retrain for accurate approximation.
    
    model = None
    
    for i, date in enumerate(sim_dates):
        # Schedule
        is_rebalance_day = True
        if config.rebalance_freq == 'weekly':
            if i > 0 and date.dayofweek != 0: is_rebalance_day = False
            
        # Retrain (Weekly - Mondays)
        if date.dayofweek == 0:
            train_end = date - pd.Timedelta(days=1)
            training_data = []
            for t, df in feature_dfs.items():
                mask = (df.index <= train_end) & (df.index >= train_end - pd.Timedelta(days=365*2))
                subset = df[mask]
                if not subset.empty: training_data.append(subset)
            
            if training_data:
                full_train = pd.concat(training_data)
                model = train_model(full_train, features, target_col)
                
        # Predict & Trade
        if is_rebalance_day and model:
            scores = {}
            prices = {}
            for ticker, df in feature_dfs.items():
                if date in df.index:
                    row = df.loc[date]
                    prices[ticker] = row['Close']
                    # Predict
                    if not row[features].isnull().any():
                        X = row[features].values.reshape(1, -1)
                        scores[ticker] = model.predict(X)[0]

            # Rebalance
            port_val = cash + sum(holdings.get(t,0) * prices.get(t,0) for t in holdings)
            
            # Rank
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_picks = [t for t,s in ranked if s > 0][:10]
            
            # Sell
            for ticker in list(holdings.keys()):
                if ticker not in top_picks and ticker in prices:
                    price = prices[ticker]
                    shares = holdings[ticker]
                    cash += shares * price * 0.9995 # 5bps slip
                    del holdings[ticker]
            
            # Buy
            for ticker in top_picks:
                if ticker not in holdings and ticker in prices:
                    price = prices[ticker]
                    target_amt = port_val * 0.10
                    shares = int(target_amt / price)
                    if shares > 0 and (shares * price * 1.0005) <= cash:
                        cost = shares * price * 1.0005
                        cash -= cost
                        holdings[ticker] = shares

        # Record
        curr_val = cash + sum(holdings.get(t,0) * prices.get(t,0) for t in holdings)
        history.append({'date': date, 'value': curr_val})

    if not history: return 0, 0, 0

    df_h = pd.DataFrame(history).set_index('date')
    final_ret = (df_h['value'].iloc[-1] - 10000) / 10000
    
    df_h['ret'] = df_h['value'].pct_change()
    sharpe = (df_h['ret'].mean() / df_h['ret'].std()) * np.sqrt(252) if df_h['ret'].std() > 0 else 0
    
    return final_ret, sharpe, df_h['value'].iloc[-1]

def main():
    print("Fetching Data for Optimization (Top 100 Random)...")
    all_tickers = pd.read_csv("data/sp500_tickers.txt", header=None)[0].tolist()
    import random
    random.seed(42)
    random.shuffle(all_tickers) # Randomized selection to avoid alphabetical bias
    sample_tickers = all_tickers[:150] # Representative sample 
    
    data = fetch_data(sample_tickers, period="max", use_cache=True)
    
    # Config Grid
    horizons = [5, 10, 20]
    freqs = ['daily', 'weekly']
    timelines = [
        ('2022 Bear', '2022-01-01', '2022-12-31'),
        ('2023 Bull', '2023-01-01', '2023-12-31'),
        ('2025 Current', '2025-10-01', '2026-02-03')
    ]
    
    results = []
    
    print("\nStarting Grid Search...")
    print(f"{'Timeline':<15} | {'Horizon':<5} | {'Freq':<8} | {'Return':<8} | {'Sharpe':<6}")
    print("-" * 60)
    
    for timeline_name, start, end in timelines:
        for h in horizons:
            for f in freqs:
                cfg = SimConfig(
                    name=f"{timeline_name}_{h}d_{f}",
                    start_date=start,
                    end_date=end,
                    rebalance_freq=f,
                    target_horizon=h
                )
                
                ret, sharpe, final_val = run_simulation(cfg, data)
                results.append({
                    'timeline': timeline_name,
                    'horizon': h,
                    'freq': f,
                    'return': ret,
                    'sharpe': sharpe
                })
                print(f"{timeline_name:<15} | {h:<5}d | {f:<8} | {ret:>8.2%} | {sharpe:>6.2f}")

    print("\n--- Optimization Summary ---")
    df_res = pd.DataFrame(results)
    # Average Sharpe by Horizon
    print("\nAvg Sharpe by Horizon:")
    print(df_res.groupby('horizon')['sharpe'].mean())
    
    print("\nAvg Sharpe by Frequency:")
    print(df_res.groupby('freq')['sharpe'].mean())
    
    print("\nBest Configuration per Timeline:")
    for t in df_res['timeline'].unique():
        best = df_res[df_res['timeline'] == t].sort_values('sharpe', ascending=False).iloc[0]
        print(f"{t}: {best['horizon']}d / {best['freq']} -> Sharpe: {best['sharpe']:.2f}")

if __name__ == "__main__":
    main()
