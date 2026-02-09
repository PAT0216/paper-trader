#!/usr/bin/env python3
"""
TRUE Head-to-Head Comparison: Daily Retraining

This script replicates the critical 'retrain_daily: true' setting from production.
It is computationally intensive but provides the most accurate simulation.

Process:
For each day in trading period:
1. Training: Retrain model on history up to previous day
2. Scoring: Predict using the fresh model
3. Trading: Execute trades
4. Repeat

Usage:
    python scripts/backtests/daily_retrain_comparison.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import json
import xgboost as xgb
from datetime import datetime, timedelta

from src.data.loader import fetch_data
from src.features.indicators import generate_features, FEATURE_COLUMNS
from src.features.momentum_features import (
    compute_momentum_features, 
    create_momentum_target,
    MOMENTUM_FEATURE_COLUMNS
)

# =============================================================================
# CONFIGURATION
# =============================================================================

TEST_START = "2025-10-01"
TEST_END = "2026-02-03"

INITIAL_CAPITAL = 10000.0
SLIPPAGE_BPS = 5
MAX_POSITIONS = 10  # Match Ledger
POSITION_SIZE_PCT = 0.10  # 10% per position

# Optimization: Limit training history to speed up daily retrain
MAX_TRAIN_HISTORY_DAYS = 365 * 5  # Train on last 5 years


# =============================================================================
# DATA LOADING
# =============================================================================

def load_sp500_tickers():
    cache_file = "data/sp500_tickers.txt"
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    return []

def load_market_data(tickers):
    print(f"Loading market data for {len(tickers)} tickers...")
    data_dict = fetch_data(tickers, period="max", use_cache=True)
    
    for ticker in list(data_dict.keys()):
        df = data_dict[ticker]
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        if len(df) < 500:
            del data_dict[ticker]
        else:
            data_dict[ticker] = df
    
    print(f"  Loaded {len(data_dict)} tickers")
    return data_dict


# =============================================================================
# TRAINING FUNCTIONS (FAST VERSION)
# =============================================================================

# =============================================================================
# TRAINING FUNCTIONS (ENSEMBLE SUPPORT)
# =============================================================================

def train_model_snapshot(all_features_df, feature_cols, target_col='Target'):
    """
    Train a model on the provided features dataframe.
    """
    if len(all_features_df) < 100:
        return None
        
    X = all_features_df[feature_cols].values
    y = all_features_df[target_col].values
    
    # Fast filtering
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isinf(X).any(axis=1))
    X, y = X[valid], y[valid]
    
    # Fast XGBoost params
    model = xgb.XGBRegressor(
        n_estimators=50,       
        learning_rate=0.1,     
        max_depth=4,           
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,             
        random_state=42,
        verbosity=0
    )
    model.fit(X, y)
    return model

# =============================================================================
# RISK MANAGEMENT (Replicating src/trading/risk_manager.py)
# =============================================================================

def calculate_volatility(price_series, lookback=30):
    if len(price_series) < lookback:
        return None
    returns = price_series.pct_change().dropna().tail(lookback)
    if len(returns) < 2:
        return None
    return returns.std() * np.sqrt(252)

def calculate_position_size_risk_managed(ticker, price, cash, portfolio_value, historical_prices, current_holdings):
    # Limits from RiskLimits / config
    MAX_POSITION_PCT = 0.15
    TARGET_VOL = 0.20
    
    # 1. Position Limit
    max_pos_value = portfolio_value * MAX_POSITION_PCT
    shares_limit = int(max_pos_value / price)
    
    # 2. Volatility Sizing
    vol = calculate_volatility(historical_prices)
    if vol is None or vol == 0:
        vol_shares = shares_limit # Default to limit if no vol data
    else:
        vol_scalar = min(TARGET_VOL / vol, 1.5)
        vol_shares = int(shares_limit * vol_scalar)
        
    final_shares = min(shares_limit, vol_shares)
    
    # Ensure strictly affordable
    max_cash_shares = int(cash / price)
    final_shares = min(final_shares, max_cash_shares)
    
    return final_shares

# =============================================================================
# PORTFOLIO
# =============================================================================

class Portfolio:
    def __init__(self, name, initial_cash, slippage_bps=5):
        self.name = name
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions = {}
        self.trades = []
        self.slippage_bps = slippage_bps
        self.value_history = []
    
    def get_value(self, prices):
        holdings = sum(
            pos['shares'] * prices.get(ticker, pos['avg_cost'])
            for ticker, pos in self.positions.items()
        )
        return self.cash + holdings
    
    def buy(self, ticker, shares, price, date):
        if shares <= 0: return
        exec_price = price * (1 + self.slippage_bps / 10000)
        cost = shares * exec_price
        
        # Double check affordabilty after slippage
        if cost > self.cash:
            shares = int(self.cash / exec_price)
            if shares <= 0: return
            cost = shares * exec_price
        
        self.cash -= cost
        if ticker in self.positions:
            pos = self.positions[ticker]
            total = pos['shares'] + shares
            avg = (pos['shares'] * pos['avg_cost'] + shares * exec_price) / total
            self.positions[ticker] = {'shares': total, 'avg_cost': avg}
        else:
            self.positions[ticker] = {'shares': shares, 'avg_cost': exec_price}
        
        self.trades.append({'date': date, 'action': 'BUY', 'ticker': ticker, 'shares': shares, 'price': exec_price})
    
    def sell(self, ticker, shares, price, date):
        if ticker not in self.positions: return
        pos = self.positions[ticker]
        shares = min(shares, pos['shares'])
        exec_price = price * (1 - self.slippage_bps / 10000)
        proceeds = shares * exec_price
        self.cash += proceeds
        pnl = (exec_price - pos['avg_cost']) * shares
        
        remaining = pos['shares'] - shares
        if remaining <= 0:
            del self.positions[ticker]
        else:
            self.positions[ticker]['shares'] = remaining
        
        self.trades.append({'date': date, 'action': 'SELL', 'ticker': ticker, 'shares': shares, 'price': exec_price, 'pnl': pnl})
    
    def record_value(self, date, prices):
        self.value_history.append({'date': date, 'value': self.get_value(prices)})


# =============================================================================
# DAILY RETRAINING SIMULATION
# =============================================================================

def run_daily_retrain_sim(data_dict, strategy_type, name):
    print(f"\n[{name}] Starting Daily Retraining Simulation (Risk Managed)...")
    
    start = pd.Timestamp(TEST_START)
    end = pd.Timestamp(TEST_END)
    
    # Pre-calculate features
    print("  Pre-calculating features...")
    feature_dfs = {}
    
    for ticker, df in data_dict.items():
        if strategy_type == 'old':
            feat_df = generate_features(df, include_target=False)
            feat_df['Target_1d'] = feat_df['Close'].pct_change(1).shift(-1)
            feat_df['Target_5d'] = feat_df['Close'].pct_change(5).shift(-5)
            feat_df['Target_20d'] = feat_df['Close'].pct_change(20).shift(-20)
            feature_cols = FEATURE_COLUMNS
        else:
            feat_df = compute_momentum_features(df)
            feat_df = create_momentum_target(feat_df, horizon=20)
            feature_cols = MOMENTUM_FEATURE_COLUMNS
            
        feat_df = feat_df.dropna(subset=feature_cols)
        feature_dfs[ticker] = feat_df
    
    
    # Get trading dates
    all_dates = set()
    for df in data_dict.values():
        mask = (df.index >= start) & (df.index <= end)
        all_dates.update(df.index[mask].tolist())
    trading_dates = sorted(all_dates)
    
    portfolio = Portfolio(name, INITIAL_CAPITAL, SLIPPAGE_BPS)
    
    for i, date in enumerate(trading_dates):
        # Progress indicator
        print(f"  Day {i+1}/{len(trading_dates)}: {date.date()}", end='\r')
        
        # 1. Update Prices
        current_prices = {}
        for ticker, df in data_dict.items():
            if date in df.index:
                current_prices[ticker] = df.loc[date, 'Close']
        portfolio.record_value(date, current_prices)
        
        # 2. Retrain Model (using data up to YESTERDAY)
        train_end_date = date - pd.Timedelta(days=1)
        
        # Prepare training data... (same as before)
        model = None
        ensemble_models = {}
        
        if strategy_type == 'old':
            for horizon, target_col in [(1, 'Target_1d'), (5, 'Target_5d'), (20, 'Target_20d')]:
                train_dfs = []
                for ticker, df in feature_dfs.items():
                    mask = (df.index <= train_end_date)
                    if MAX_TRAIN_HISTORY_DAYS:
                        start_hist = train_end_date - pd.Timedelta(days=MAX_TRAIN_HISTORY_DAYS)
                        mask = mask & (df.index >= start_hist)
                    hist = df[mask].dropna(subset=[target_col])
                    if len(hist) > 50:
                        train_dfs.append(hist)
                if train_dfs:
                    full_train = pd.concat(train_dfs)
                    ensemble_models[horizon] = train_model_snapshot(full_train, feature_cols, target_col)
        else:
            train_dfs = []
            for ticker, df in feature_dfs.items():
                mask = (df.index <= train_end_date)
                if MAX_TRAIN_HISTORY_DAYS:
                    start_hist = train_end_date - pd.Timedelta(days=MAX_TRAIN_HISTORY_DAYS)
                    mask = mask & (df.index >= start_hist)
                hist = df[mask].dropna(subset=['Target'])
                if len(hist) > 50:
                    train_dfs.append(hist)
            if train_dfs:
                full_train = pd.concat(train_dfs)
                model = train_model_snapshot(full_train, feature_cols, 'Target')

            
        # 3. Predict for TODAY
        scores = {}
        for ticker, df in feature_dfs.items():
            if ticker not in current_prices: continue
            if date not in df.index: continue
            
            row = df.loc[[date]]
            X_pred = row[feature_cols].values
            
            if np.isnan(X_pred).any() or np.isinf(X_pred).any():
                continue
                
            pred = 0.0
            if strategy_type == 'old':
                w_total = 0
                if 1 in ensemble_models and ensemble_models[1]:
                    pred += 0.5 * ensemble_models[1].predict(X_pred)[0]
                    w_total += 0.5
                if 5 in ensemble_models and ensemble_models[5]:
                    pred += 0.3 * ensemble_models[5].predict(X_pred)[0]
                    w_total += 0.3
                if 20 in ensemble_models and ensemble_models[20]:
                    pred += 0.2 * ensemble_models[20].predict(X_pred)[0]
                    w_total += 0.2
                if w_total > 0:
                    pred /= w_total
                else:
                    continue
            else:
                if model:
                    pred = model.predict(X_pred)[0]
                else:
                    continue
            
            scores[ticker] = pred
            
        # 4. Filter & Trade
        if not scores: continue
        
        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [t for t, _ in sorted_scores[:MAX_POSITIONS]]
        
        # DEBUG: Print Day 1 selections to compare with Ledger
        if i == 0:
            print(f"\n  [DEBUG Day 1] {name} Selected: {selected}")
        
        # Sell
        for ticker in list(portfolio.positions.keys()):
            if ticker not in selected and ticker in current_prices:
                portfolio.sell(ticker, portfolio.positions[ticker]['shares'], current_prices[ticker], date)
        
        # Buy (Risk Managed Sizing)
        current_holdings_dict = {t: p['shares'] for t, p in portfolio.positions.items()}
        
        for ticker in selected:
            if ticker not in portfolio.positions and ticker in current_prices:
                if len(portfolio.positions) >= MAX_POSITIONS: break
                
                price = current_prices[ticker]
                portfolio_val = portfolio.get_value(current_prices)
                cash = portfolio.cash
                
                # Get history for volatility calculation (slice up to today)
                # Note: This might be slow inside loop, but necessary for accuracy
                try:
                    hist_data = data_dict[ticker].loc[:date]['Close']
                except:
                    hist_data = pd.Series()
                
                shares = calculate_position_size_risk_managed(
                    ticker, price, cash, portfolio_val, 
                    hist_data, current_holdings_dict
                )
                
                if shares > 0:
                    portfolio.buy(ticker, shares, price, date)
                    current_holdings_dict[ticker] = shares
    
    print("\n  Simulation complete.")
    return portfolio


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(portfolio):
    if not portfolio.value_history: return {}
    values = pd.DataFrame(portfolio.value_history).set_index('date').sort_index()
    initial = values['value'].iloc[0]
    final = values['value'].iloc[-1]
    ret = (final / initial - 1) * 100
    
    returns = values['value'].pct_change().dropna()
    sharpe = (returns.mean() * 252 - 0.05) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    max_dd = ((values['value'] - values['value'].cummax()) / values['value'].cummax()).min() * 100
    
    return {
        'name': portfolio.name,
        'final': final, 
        'return_pct': ret, 
        'sharpe': sharpe, 
        'max_dd_pct': max_dd
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("DAILY RETRAINING COMPARISON (The 'Fair' Test)")
    print(f"Period: {TEST_START} to {TEST_END}")
    print("=" * 70)
    
    tickers = load_sp500_tickers()
    data_dict = load_market_data(tickers) # Use FULL universe
    
    if not data_dict:
        print("No data.")
        return
        
    # Run Old ML (Daily Retrain)
    pf_old = run_daily_retrain_sim(data_dict, 'old', 'Old ML (Daily Retrain)')
    m_old = calculate_metrics(pf_old)
    
    # Run New ML (Daily Retrain)
    pf_new = run_daily_retrain_sim(data_dict, 'new', 'New ML (Daily Retrain)')
    m_new = calculate_metrics(pf_new)
    
    print("\n" + "=" * 70)
    print("RESULTS (With Daily Retraining)")
    print("=" * 70)
    print(f"{'Metric':<15} {'Old ML':<15} {'New ML':<15} {'Delta':<10}")
    print("-" * 55)
    print(f"{'Return':<15} {m_old['return_pct']:+.1f}%{'':<9} {m_new['return_pct']:+.1f}%{'':<9} {m_new['return_pct']-m_old['return_pct']:+.1f}%")
    print(f"{'Sharpe':<15} {m_old['sharpe']:.2f}{'':<11} {m_new['sharpe']:.2f}{'':<11} {m_new['sharpe']-m_old['sharpe']:+.2f}")
    print(f"{'Max DD':<15} {m_old['max_dd_pct']:.1f}%{'':<9} {m_new['max_dd_pct']:.1f}%{'':<9} {m_new['max_dd_pct']-m_old['max_dd_pct']:+.1f}%")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/daily_retrain_comparison.json", "w") as f:
        json.dump({'old': m_old, 'new': m_new}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
