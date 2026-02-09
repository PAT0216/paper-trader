#!/usr/bin/env python3
"""
Head-to-Head ML Comparison: Old Features vs New Features

Both models:
- Trained ONCE on data before Oct 1, 2025 (point-in-time)
- Daily rebalancing during test period
- Same position sizing, slippage, constraints

This is a fair A/B test of the feature engineering change.

Usage:
    python scripts/backtests/daily_ml_comparison.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import json
from datetime import datetime
from scipy.stats import spearmanr
import xgboost as xgb

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
TEST_END = "2026-02-03"  # Most recent trading day
TRAIN_END = "2025-09-30"  # Train on data BEFORE test period

INITIAL_CAPITAL = 10000.0
SLIPPAGE_BPS = 5
MAX_POSITIONS = 5  # Match live ML strategy
POSITION_SIZE_PCT = 0.20  # 20% per position (5 positions = 100%)

# Target horizons
OLD_HORIZON = 1   # Old ML used 1-day target
NEW_HORIZON = 20  # New ML uses 20-day target


# =============================================================================
# DATA
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
        df = df[df.index >= '2010-01-01']
        if len(df) < 500:
            del data_dict[ticker]
        else:
            data_dict[ticker] = df
    
    print(f"  Loaded {len(data_dict)} tickers")
    return data_dict


# =============================================================================
# OLD ML MODEL (Mean-Reversion Features + 1-day Target)
# =============================================================================

def train_old_ml(data_dict, train_end_date):
    """Train using OLD feature set (RSI, MACD, Bollinger, etc.)"""
    print("\n" + "=" * 60)
    print("TRAINING OLD ML MODEL (Mean-Reversion Features)")
    print("=" * 60)
    
    all_data = []
    
    for ticker, df in data_dict.items():
        df_train = df[df.index <= train_end_date].copy()
        if len(df_train) < 300:
            continue
        
        # OLD feature generation
        df_feat = generate_features(df_train, include_target=False)
        
        # 1-day target (old approach)
        df_feat['Target'] = df_feat['Close'].pct_change(OLD_HORIZON).shift(-OLD_HORIZON)
        df_feat = df_feat.dropna(subset=['Target'])
        
        if len(df_feat) > 100:
            all_data.append(df_feat)
    
    if not all_data:
        return None, []
    
    full_df = pd.concat(all_data).sort_index()
    print(f"Training samples: {len(full_df):,}")
    
    X = full_df[FEATURE_COLUMNS].values
    y = full_df['Target'].values
    
    # Clean
    X = np.where(np.isinf(X), np.nan, X)
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[valid], y[valid]
    
    # Train/Val split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    
    # Validate
    y_pred = model.predict(X_val)
    dir_acc = ((y_pred > 0) == (y_val > 0)).mean()
    spearman = spearmanr(y_pred, y_val)[0]
    
    print(f"  Direction Accuracy: {dir_acc:.1%}")
    print(f"  Spearman: {spearman:.4f}")
    
    # Final model on all data
    final_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        verbosity=0
    )
    final_model.fit(X, y)
    
    return final_model, FEATURE_COLUMNS


# =============================================================================
# NEW ML MODEL (Momentum Features + 20-day Target)
# =============================================================================

def train_new_ml(data_dict, train_end_date):
    """Train using NEW feature set (momentum-aligned)"""
    print("\n" + "=" * 60)
    print("TRAINING NEW ML MODEL (Momentum-Aligned Features)")
    print("=" * 60)
    
    all_data = []
    
    for ticker, df in data_dict.items():
        df_train = df[df.index <= train_end_date].copy()
        if len(df_train) < 300:
            continue
        
        # NEW feature generation
        df_feat = compute_momentum_features(df_train)
        df_feat = create_momentum_target(df_feat, horizon=NEW_HORIZON)
        df_feat = df_feat.dropna(subset=['Target'])
        
        if len(df_feat) > 100:
            all_data.append(df_feat)
    
    if not all_data:
        return None, []
    
    full_df = pd.concat(all_data).sort_index()
    print(f"Training samples: {len(full_df):,}")
    
    X = full_df[MOMENTUM_FEATURE_COLUMNS].values
    y = full_df['Target'].values
    
    # Clean
    X = np.where(np.isinf(X), np.nan, X)
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[valid], y[valid]
    
    # Train/Val split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    
    # Validate
    y_pred = model.predict(X_val)
    dir_acc = ((y_pred > 0) == (y_val > 0)).mean()
    spearman = spearmanr(y_pred, y_val)[0]
    
    print(f"  Direction Accuracy: {dir_acc:.1%}")
    print(f"  Spearman: {spearman:.4f}")
    
    # Final model
    final_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    final_model.fit(X, y)
    
    return final_model, MOMENTUM_FEATURE_COLUMNS


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
        if shares <= 0:
            return
        exec_price = price * (1 + self.slippage_bps / 10000)
        cost = shares * exec_price
        if cost > self.cash:
            shares = int(self.cash / exec_price)
            if shares <= 0:
                return
            cost = shares * exec_price
        
        self.cash -= cost
        if ticker in self.positions:
            pos = self.positions[ticker]
            total = pos['shares'] + shares
            avg = (pos['shares'] * pos['avg_cost'] + shares * exec_price) / total
            self.positions[ticker] = {'shares': total, 'avg_cost': avg}
        else:
            self.positions[ticker] = {'shares': shares, 'avg_cost': exec_price}
        
        self.trades.append({'date': date, 'action': 'BUY', 'ticker': ticker})
    
    def sell(self, ticker, shares, price, date):
        if ticker not in self.positions:
            return
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
        
        self.trades.append({'date': date, 'action': 'SELL', 'ticker': ticker, 'pnl': pnl})
    
    def record_value(self, date, prices):
        self.value_history.append({'date': date, 'value': self.get_value(prices)})


# =============================================================================
# DAILY SIMULATION
# =============================================================================

def run_daily_simulation(data_dict, model, features, feature_type, name):
    """Run daily trading simulation."""
    print(f"\n[Simulating {name}]")
    
    start = pd.Timestamp(TEST_START)
    end = pd.Timestamp(TEST_END)
    
    # Get trading dates
    all_dates = set()
    for df in data_dict.values():
        mask = (df.index >= start) & (df.index <= end)
        all_dates.update(df.index[mask].tolist())
    trading_dates = sorted(all_dates)
    
    print(f"  Period: {trading_dates[0].date()} to {trading_dates[-1].date()}")
    print(f"  Trading days: {len(trading_dates)}")
    
    portfolio = Portfolio(name, INITIAL_CAPITAL, SLIPPAGE_BPS)
    
    for date in trading_dates:
        # Get prices
        prices = {}
        for ticker, df in data_dict.items():
            if date in df.index:
                prices[ticker] = df.loc[date, 'Close']
        
        portfolio.record_value(date, prices)
        
        # Score all tickers
        scores = {}
        for ticker, df in data_dict.items():
            if ticker not in prices:
                continue
            
            df_to_date = df.loc[:date]
            if len(df_to_date) < 260:
                continue
            
            try:
                if feature_type == 'old':
                    df_feat = generate_features(df_to_date, include_target=False)
                else:
                    df_feat = compute_momentum_features(df_to_date)
                
                if len(df_feat) < 1:
                    continue
                
                last_row = df_feat.iloc[[-1]][features]
                
                if last_row.isnull().any().any() or np.isinf(last_row.values).any():
                    continue
                
                score = model.predict(last_row)[0]
                scores[ticker] = score
            except Exception:
                continue
        
        if not scores:
            continue
        
        # Rank and select
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [t for t, _ in sorted_scores[:MAX_POSITIONS]]
        
        # Sell non-selected
        for ticker in list(portfolio.positions.keys()):
            if ticker not in selected and ticker in prices:
                portfolio.sell(ticker, portfolio.positions[ticker]['shares'], prices[ticker], date)
        
        # Buy selected
        for ticker in selected:
            if ticker not in portfolio.positions and ticker in prices:
                if len(portfolio.positions) >= MAX_POSITIONS:
                    break
                available = portfolio.cash * 0.95
                position_value = min(available, INITIAL_CAPITAL * POSITION_SIZE_PCT)
                shares = int(position_value / prices[ticker])
                if shares > 0:
                    portfolio.buy(ticker, shares, prices[ticker], date)
    
    return portfolio


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(portfolio):
    if not portfolio.value_history:
        return {}
    
    values = pd.DataFrame(portfolio.value_history)
    values['date'] = pd.to_datetime(values['date'])
    values = values.set_index('date').sort_index()
    
    initial = values['value'].iloc[0]
    final = values['value'].iloc[-1]
    total_return = (final / initial - 1) * 100
    
    returns = values['value'].pct_change().dropna()
    sharpe = (returns.mean() * 252 - 0.05) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    
    cummax = values['value'].cummax()
    drawdown = (values['value'] - cummax) / cummax
    max_dd = drawdown.min() * 100
    
    sells = [t for t in portfolio.trades if t['action'] == 'SELL' and 'pnl' in t]
    wins = [t for t in sells if t['pnl'] > 0]
    win_rate = len(wins) / len(sells) * 100 if sells else 0
    
    buys = [t for t in portfolio.trades if t['action'] == 'BUY']
    
    return {
        'name': portfolio.name,
        'initial': initial,
        'final': final,
        'return_pct': total_return,
        'sharpe': sharpe,
        'max_dd_pct': max_dd,
        'n_buys': len(buys),
        'n_sells': len(sells),
        'win_rate': win_rate,
        'values': values
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("OLD ML vs NEW ML - DAILY TRADING COMPARISON")
    print(f"Test Period: {TEST_START} to {TEST_END}")
    print(f"Training cutoff: {TRAIN_END} (point-in-time)")
    print("=" * 70)
    
    tickers = load_sp500_tickers()
    data_dict = load_market_data(tickers[:200])
    
    if not data_dict:
        print("No data!")
        return
    
    train_end = pd.Timestamp(TRAIN_END)
    
    # Train both models
    old_model, old_features = train_old_ml(data_dict, train_end)
    new_model, new_features = train_new_ml(data_dict, train_end)
    
    if old_model is None or new_model is None:
        print("Training failed!")
        return
    
    # Run simulations
    portfolio_old = run_daily_simulation(data_dict, old_model, old_features, 'old', 'Old ML (RSI/MACD)')
    portfolio_new = run_daily_simulation(data_dict, new_model, new_features, 'new', 'New ML (Momentum)')
    
    # Calculate metrics
    m_old = calculate_metrics(portfolio_old)
    m_new = calculate_metrics(portfolio_new)
    
    # Print results
    print("\n" + "=" * 70)
    print("DAILY TRADING RESULTS")
    print("=" * 70)
    
    print(f"\n{'Metric':<20} {'Old ML':<15} {'New ML':<15} {'Delta':<15}")
    print("-" * 65)
    print(f"{'Final Value':<20} ${m_old['final']:,.0f}{'':<5} ${m_new['final']:,.0f}{'':<5} ${m_new['final']-m_old['final']:+,.0f}")
    print(f"{'Return':<20} {m_old['return_pct']:+.1f}%{'':<10} {m_new['return_pct']:+.1f}%{'':<10} {m_new['return_pct']-m_old['return_pct']:+.1f}%")
    print(f"{'Sharpe':<20} {m_old['sharpe']:.2f}{'':<12} {m_new['sharpe']:.2f}{'':<12} {m_new['sharpe']-m_old['sharpe']:+.2f}")
    print(f"{'Max Drawdown':<20} {m_old['max_dd_pct']:.1f}%{'':<10} {m_new['max_dd_pct']:.1f}%{'':<10} {m_new['max_dd_pct']-m_old['max_dd_pct']:+.1f}%")
    print(f"{'Win Rate':<20} {m_old['win_rate']:.1f}%{'':<10} {m_new['win_rate']:.1f}%{'':<10} {m_new['win_rate']-m_old['win_rate']:+.1f}%")
    print(f"{'Total Trades':<20} {m_old['n_buys']+m_old['n_sells']:<15} {m_new['n_buys']+m_new['n_sells']:<15}")
    
    # Daily comparison
    print("\n" + "=" * 70)
    print("DAILY VALUE COMPARISON")
    print("=" * 70)
    
    # Merge values
    old_vals = m_old['values'].rename(columns={'value': 'Old ML'})
    new_vals = m_new['values'].rename(columns={'value': 'New ML'})
    comparison = old_vals.join(new_vals, how='outer')
    
    # Show weekly snapshots
    comparison['Week'] = comparison.index.isocalendar().week
    weekly = comparison.groupby('Week').last()
    
    print(f"\n{'Week':<8} {'Old ML':<15} {'New ML':<15} {'Diff':<10}")
    print("-" * 48)
    for week, row in weekly.iterrows():
        diff = row['New ML'] - row['Old ML']
        print(f"W{week:<7} ${row['Old ML']:,.0f}{'':<5} ${row['New ML']:,.0f}{'':<5} ${diff:+,.0f}")
    
    # Final values
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    if m_new['return_pct'] > m_old['return_pct']:
        print(f"\nNew ML wins by {m_new['return_pct']-m_old['return_pct']:.1f}%")
    else:
        print(f"\nOld ML wins by {m_old['return_pct']-m_new['return_pct']:.1f}%")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    results = {
        'test_period': f"{TEST_START} to {TEST_END}",
        'old_ml': {k: v for k, v in m_old.items() if k != 'values'},
        'new_ml': {k: v for k, v in m_new.items() if k != 'values'}
    }
    with open("results/daily_ml_comparison.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to results/daily_ml_comparison.json")


if __name__ == "__main__":
    main()
