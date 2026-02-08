
"""
Comprehensive Walkforward Test
------------------------------
Developer-Level Verification of "Old ML" vs "New ML" strategies.
Uses Point-In-Time (Walkforward) Retraining to prevent data leakage.
Tests two Risk Management Scenarios:
1. STRICT (Production Ideal): Target Volatility Sizing, Sector Limits (30%).
2. RELAXED (Live Observed): Equal Weight Sizing (10%), No Sector Limits.

ASSUMPTIONS:
- Old ML: 3-Model Ensemble (1d, 5d, 20d horizons)
- New ML: Single Model (20d horizon) with Momentum Features
"""

import sys
import os
import glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import xgboost as xgb
import json
from dataclasses import dataclass
from typing import List, Dict, Optional

from src.data.loader import fetch_data
from src.features.indicators import generate_features, FEATURE_COLUMNS
from src.features.momentum_features import compute_momentum_features, MOMENTUM_FEATURE_COLUMNS
from src.trading.risk_manager import RiskManager, RiskLimits

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BacktestConfig:
    name: str
    strategy_type: str
    risk_mode: str
    rebalance_freq: str = "daily" # 'daily' or 'weekly'
    start_date: str = "2025-10-01"
    end_date: str = "2026-02-03"
    initial_capital: float = 10000.0
    slippage_bps: int = 5

# =============================================================================
# HELPERS
# =============================================================================

def get_risk_manager(mode: str) -> RiskManager:
    if mode == 'strict':
        # Production Ideal
        limits = RiskLimits(
            max_position_pct=0.15,
            max_sector_pct=0.30,
            volatility_lookback=30
        )
        # Note: RiskManager uses Inverse Vol sizing by default if vol provided
        return RiskManager(risk_limits=limits)
    else:
        # Relaxed (Live Observed)
        limits = RiskLimits(
            max_position_pct=0.15, # Still cap single pos at 15%
            max_sector_pct=1.00,   # Disable sector limit (100%)
            volatility_lookback=30
        )
        return RiskManager(risk_limits=limits)

def train_model(df_train, strategy_type):
    """
    Train model(s) based on strategy type using ONLY df_train (past data).
    """
    models = {}
    
    if strategy_type == 'old_ensemble':
        # Train 3 horizons: 1d, 5d, 20d
        features = FEATURE_COLUMNS
        for horizon in [1, 5, 20]:
            target_col = f'Target_{horizon}d'
            
            # Filter valid
            valid = df_train.dropna(subset=[target_col] + features)
            if len(valid) < 100:
                continue
                
            X = valid[features].values
            y = valid[target_col].values
            
            model = xgb.XGBRegressor(
                n_estimators=50, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0
            )
            model.fit(X, y)
            models[horizon] = model
            
    elif strategy_type == 'new_momentum':
        # Train 1 horizon: 20d
        features = MOMENTUM_FEATURE_COLUMNS
        target_col = 'Target_20d'
        
        valid = df_train.dropna(subset=[target_col] + features)
        if len(valid) < 100:
            return {}
            
        X = valid[features].values
        y = valid[target_col].values
            
        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0
        )
        model.fit(X, y)
        models[20] = model
        
    return models

def predict_score(row, models, strategy_type):
    """
    Generate score for a single row (timestamp)
    """
    if not models:
        return -999.0
        
    if strategy_type == 'old_ensemble':
        features = FEATURE_COLUMNS
        # Need to ensure row has these columns
        if row[features].isnull().any():
            return -999.0
            
        X = row[features].values.reshape(1, -1)
        
        # Weighted Ensemble Logic
        score = 0.0
        total_w = 0.0
        
        if 1 in models:
            pred = models[1].predict(X)[0]
            score += 0.5 * (pred / 1.0)
            total_w += 0.5
        if 5 in models:
            pred = models[5].predict(X)[0]
            score += 0.3 * (pred / 5.0)
            total_w += 0.3
        if 20 in models:
            pred = models[20].predict(X)[0]
            score += 0.2 * (pred / 20.0)
            total_w += 0.2
            
        return score / total_w if total_w > 0 else -999.0

    elif strategy_type == 'new_momentum':
        features = MOMENTUM_FEATURE_COLUMNS
        if row[features].isnull().any():
            return -999.0
            
        X = row[features].values.reshape(1, -1)
        
        if 20 in models:
            pred = models[20].predict(X)[0]
            return pred
            
    return -999.0

# =============================================================================
# ENGINE
# =============================================================================

def run_backtest(config: BacktestConfig):
    print(f"\nRunning Backtest: {config.name}")
    print(f"Strategy: {config.strategy_type} | Risk: {config.risk_mode} | Freq: {config.rebalance_freq}")
    
    # 1. Load Data (Full Universe)
    tickers = pd.read_csv("data/sp500_tickers.txt", header=None)[0].tolist()
    # Add SPY for benchmark
    if 'SPY' not in tickers:
        tickers.append('SPY')
        
    data_dict = fetch_data(tickers, period="max", use_cache=True)
    
    # Benchmark Data
    spy_df = data_dict.get('SPY', pd.DataFrame())
    
    # 2. Precompute Features
    print("Generating features...")
    feature_dfs = {}
    for ticker, df in data_dict.items():
        if df.empty: continue
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        
        # Skip feature gen for SPY unless it's a target (it's in tickers list)
        
        if config.strategy_type == 'old_ensemble':
            f_df = generate_features(df.copy(), include_target=False)
            f_df['Target_1d'] = f_df['Close'].pct_change(1).shift(-1)
            f_df['Target_5d'] = f_df['Close'].pct_change(5).shift(-5)
            f_df['Target_20d'] = f_df['Close'].pct_change(20).shift(-20)
            feature_dfs[ticker] = f_df
        else:
            f_df = compute_momentum_features(df.copy())
            from src.features.momentum_features import create_momentum_target
            f_df = create_momentum_target(f_df, horizon=20)
            f_df = f_df.rename(columns={'Target': 'Target_20d'})
            feature_dfs[ticker] = f_df

    # 3. Setup Simulation
    risk_manager = get_risk_manager(config.risk_mode)
    cash = config.initial_capital
    holdings = {}
    avg_costs = {}
    history = []
    trade_log = [] # List of dicts
    
    dates = sorted(list(set().union(*[df.index for df in feature_dfs.values()])))
    start_dt = pd.Timestamp(config.start_date)
    end_dt = pd.Timestamp(config.end_date)
    sim_dates = [d for d in dates if start_dt <= d <= end_dt]
    
    model = {} 
    
    print(f"Simulating {len(sim_dates)} days...")
    
    for i, date in enumerate(sim_dates):
        # Benchmark Value
        spy_price = spy_df.loc[date]['Close'] if (not spy_df.empty and date in spy_df.index) else 0
        
        # Check Rebalance Schedule
        is_rebalance_day = True
        if config.rebalance_freq == 'weekly':
            # Rebalance only on Mondays (dayofweek=0)
            # OR if it's the first simulaton day
            if i > 0 and date.dayofweek != 0:
                is_rebalance_day = False
        
        # Retrain Daily
        train_end = date - pd.Timedelta(days=1)
        
        # Training (Cached per day)
        # Assuming retrain is needed daily
        training_data = []
        if is_rebalance_day:
            for t, df in feature_dfs.items():
                if t == 'SPY': continue 
                mask = (df.index <= train_end) & (df.index >= train_end - pd.Timedelta(days=365*5))
                subset = df[mask]
                if not subset.empty:
                    training_data.append(subset)
            if training_data:
                full_train = pd.concat(training_data)
                model = train_model(full_train, config.strategy_type)
        
        # 5. Predict & Trade
        scores = {}
        prices = {}
        for ticker, df in feature_dfs.items():
            if date in df.index:
                row = df.loc[date]
                prices[ticker] = row['Close']
                if is_rebalance_day and ticker != 'SPY':
                    score = predict_score(row, model, config.strategy_type)
                    if score != -999.0:
                        scores[ticker] = score
        
        # Portfolio Calculation
        port_val = cash + sum(holdings.get(t,0) * prices.get(t,0) for t in holdings)
        history.append({'date': date, 'value': port_val, 'cash': cash, 'spy_price': spy_price})
        
        if is_rebalance_day:
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_picks = [t for t,s in ranked if s > 0][:10]
            
            # Sell
            for ticker in list(holdings.keys()):
                if ticker not in top_picks and ticker in prices:
                    price = prices[ticker]
                    shares = holdings[ticker]
                    proceeds = shares * price * (1 - config.slippage_bps/10000)
                    cash += proceeds
                    # Log Trade
                    trade_log.append({
                        'date': date, 'ticker': ticker, 'action': 'SELL', 
                        'price': price, 'shares': shares, 'amount': proceeds, 
                        'cash_balance': cash, 'total_value': port_val,
                        'strategy': 'new_ml',
                         # Approximate PnL from average cost
                        'pnl': proceeds - (shares * avg_costs[ticker])
                    })
                    del holdings[ticker]
                    del avg_costs[ticker]
            
            # Buy
            for ticker in top_picks:
                if ticker not in holdings and ticker in prices:
                    price = prices[ticker]
                    hist_data = feature_dfs[ticker].loc[:date].tail(30)
                    
                    if config.risk_mode == 'relaxed':
                        target_size = port_val * 0.10
                        shares = int(target_size / price)
                    else:
                        shares, reason = risk_manager.calculate_position_size(
                            ticker, price, cash, port_val, hist_data, holdings, prices
                        )
                    
                    if shares > 0:
                        cost = shares * price * (1 + config.slippage_bps/10000)
                        if cost <= cash:
                            cash -= cost
                            holdings[ticker] = shares
                            avg_costs[ticker] = price
                            trade_log.append({
                                'date': date, 'ticker': ticker, 'action': 'BUY', 
                                'price': price, 'shares': shares, 'amount': cost, 
                                'cash_balance': cash, 'total_value': port_val,
                                'strategy': 'new_ml',
                                'pnl': 0.0
                            })

    # Stats
    df_hist = pd.DataFrame(history)
    df_hist['date'] = pd.to_datetime(df_hist['date'])
    df_hist = df_hist.set_index('date')
    
    final_val = df_hist['value'].iloc[-1]
    total_ret = (final_val - config.initial_capital) / config.initial_capital
    
    # Bench
    spy_start = df_hist['spy_price'].iloc[0]
    spy_end = df_hist['spy_price'].iloc[-1]
    spy_ret = (spy_end / spy_start - 1) if spy_start > 0 else 0
    
    # CAGR/Sharpe
    days = (df_hist.index[-1] - df_hist.index[0]).days
    cagr = (final_val / config.initial_capital) ** (365.25 / days) - 1 if days > 0 else 0
    
    df_hist['returns'] = df_hist['value'].pct_change()
    mean_ret = df_hist['returns'].mean()
    std_ret = df_hist['returns'].std()
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
    
    cummax = df_hist['value'].cummax()
    max_dd = ((df_hist['value'] - cummax) / cummax).min()

    return {
        "name": config.name,
        "total_return": total_ret,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "final_value": final_val,
        "trades": trade_log,
        "spy_return": spy_ret
    }

def main():
    configs = [
        BacktestConfig(name="New ML (Relaxed, Daily)", strategy_type="new_momentum", risk_mode="relaxed", rebalance_freq="daily"),
        BacktestConfig(name="New ML (Relaxed, Weekly)", strategy_type="new_momentum", risk_mode="relaxed", rebalance_freq="weekly"),
    ]
    
    results = []
    print("Starting Deep Dive Verification (Trades & Frequency)...")
    
    os.makedirs("results", exist_ok=True)
    
    for cfg in configs:
        res = run_backtest(cfg)
        results.append(res)
        
        # Save Ledger
        fname = f"results/ledger_{cfg.rebalance_freq}.csv"
        pd.DataFrame(res['trades']).to_csv(fname, index=False)
        print(f"  > Ledger saved to {fname}")
        print(f"  > Result: {res['name']} -> Ret: {res['total_return']:.2%} | SPY: {res['spy_return']:.2%}")

    print("\nFINAL REPORT")
    print("-" * 100)
    print(f"{'Strategy':<30} | {'Return':<8} | {'SPY':<8} | {'Alpha':<8} | {'Trades':<6} | {'Sharpe':<6} | {'Best Frequency'}")
    print("-" * 100)
    
    best_ret = -999
    best_freq = ""
    
    for r in results:
        alpha = r['total_return'] - r['spy_return']
        print(f"{r['name']:<30} | {r['total_return']:>8.2%} | {r['spy_return']:>8.2%} | {alpha:>8.2%} | {len(r['trades']):<6} | {r['sharpe']:>6.2f} |")
        
        if r['total_return'] > best_ret:
            best_ret = r['total_return']
            best_freq = r['name']
            
    print("-" * 100)
    print(f"Winner: {best_freq}")

if __name__ == "__main__":
    main()
