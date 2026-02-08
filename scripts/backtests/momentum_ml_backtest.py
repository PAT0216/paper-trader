#!/usr/bin/env python3
"""
Momentum-Enhanced ML Strategy Backtest

Steve Cohen + Karpathy approach:
1. Train on momentum features (not mean-reversion)
2. 20-day prediction horizon (not 1-day noise)
3. ML enhances momentum selection, doesn't replace it
4. Compare vs pure momentum baseline

Usage:
    python scripts/backtests/momentum_ml_backtest.py
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
from src.features.momentum_features import (
    compute_momentum_features, 
    create_momentum_target,
    MOMENTUM_FEATURE_COLUMNS
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Backtest periods
SHORT_START = "2025-10-01"
SHORT_END = "2026-02-02"
LONG_START = "2015-01-01"
LONG_END = "2025-12-31"

# Strategy params
INITIAL_CAPITAL = 100000.0
SLIPPAGE_BPS = 5
MAX_POSITIONS = 10
POSITION_SIZE_PCT = 0.10
TARGET_HORIZON = 20  # 20-day forward returns


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
# ML TRAINING
# =============================================================================

def train_momentum_ml(data_dict, train_end_date):
    """
    Train XGBoost on momentum features with 20-day horizon.
    
    Key differences from old approach:
    1. Momentum features instead of mean-reversion
    2. 20-day target instead of 1-day
    3. TimeSeriesSplit CV
    """
    print("\n" + "=" * 60)
    print("TRAINING MOMENTUM-ENHANCED ML MODEL")
    print("=" * 60)
    
    # Prepare training data
    all_data = []
    
    for ticker, df in data_dict.items():
        # Filter to training period only (point-in-time)
        df_train = df[df.index < train_end_date].copy()
        
        if len(df_train) < 300:
            continue
        
        # Generate features
        df_feat = compute_momentum_features(df_train)
        df_feat = create_momentum_target(df_feat, horizon=TARGET_HORIZON)
        df_feat = df_feat.dropna(subset=['Target'])
        
        if len(df_feat) > 100:
            df_feat['ticker'] = ticker
            all_data.append(df_feat)
    
    if not all_data:
        print("No training data!")
        return None, []
    
    full_df = pd.concat(all_data).sort_index()
    print(f"Training samples: {len(full_df):,}")
    print(f"Training period: {full_df.index.min().date()} to {full_df.index.max().date()}")
    
    # Prepare X, y
    X = full_df[MOMENTUM_FEATURE_COLUMNS].values
    y = full_df['Target'].values
    
    # Clean inf/nan
    X = np.where(np.isinf(X), np.nan, X)
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[valid], y[valid]
    
    print(f"Clean samples: {len(X):,}")
    print(f"Features: {len(MOMENTUM_FEATURE_COLUMNS)}")
    
    # Train/Val split (last 20% for validation)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"\nTrain: {len(X_train):,}, Val: {len(X_val):,}")
    
    # Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,  # Shallow to avoid overfitting
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    
    # Validate
    y_pred_val = model.predict(X_val)
    
    # Metrics
    dir_acc = ((y_pred_val > 0) == (y_val > 0)).mean()
    spearman = spearmanr(y_pred_val, y_val)[0]
    
    # Top quintile analysis
    n_quintile = len(y_val) // 5
    pred_top_idx = np.argsort(y_pred_val)[-n_quintile:]
    actual_returns_top = y_val[pred_top_idx]
    avg_return_top = actual_returns_top.mean()
    
    pred_bottom_idx = np.argsort(y_pred_val)[:n_quintile]
    actual_returns_bottom = y_val[pred_bottom_idx]
    avg_return_bottom = actual_returns_bottom.mean()
    
    long_short_spread = avg_return_top - avg_return_bottom
    
    print(f"\n[Validation Metrics]")
    print(f"  Direction Accuracy: {dir_acc:.1%}")
    print(f"  Spearman Rank Corr: {spearman:.4f}")
    print(f"  Top Quintile Avg Return: {avg_return_top*100:.2f}%")
    print(f"  Bottom Quintile Avg Return: {avg_return_bottom*100:.2f}%")
    print(f"  Long-Short Spread: {long_short_spread*100:.2f}%")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': MOMENTUM_FEATURE_COLUMNS,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n[Top 5 Features]")
    for _, row in importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Train final model on all data
    print(f"\n[Training final model on all {len(X):,} samples]")
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
# PORTFOLIO SIMULATION
# =============================================================================

class Portfolio:
    def __init__(self, initial_cash, slippage_bps=5):
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
        
        self.trades.append({'date': date, 'action': 'BUY', 'ticker': ticker, 'shares': shares, 'price': exec_price})
    
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
        
        self.trades.append({'date': date, 'action': 'SELL', 'ticker': ticker, 'shares': shares, 'price': exec_price, 'pnl': pnl})
    
    def record_value(self, date, prices):
        self.value_history.append({'date': date, 'value': self.get_value(prices)})


def run_momentum_ml_simulation(data_dict, model, features, start_date, end_date, strategy_name):
    """Run simulation with momentum-enhanced ML."""
    print(f"\n[Simulating {strategy_name}] {start_date} to {end_date}")
    
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    # Get trading dates
    all_dates = set()
    for df in data_dict.values():
        mask = (df.index >= start) & (df.index <= end)
        all_dates.update(df.index[mask].tolist())
    trading_dates = sorted(all_dates)
    
    print(f"  Trading days: {len(trading_dates)}")
    
    # Monthly rebalance (like momentum baseline)
    rebalance_dates = set()
    current_month = None
    for d in trading_dates:
        if (d.year, d.month) != current_month:
            rebalance_dates.add(d)
            current_month = (d.year, d.month)
    
    print(f"  Rebalances: {len(rebalance_dates)}")
    
    portfolio = Portfolio(INITIAL_CAPITAL, SLIPPAGE_BPS)
    
    for date in trading_dates:
        prices = {t: df.loc[date, 'Close'] for t, df in data_dict.items() if date in df.index}
        portfolio.record_value(date, prices)
        
        if date not in rebalance_dates:
            continue
        
        # Score all tickers
        scores = {}
        for ticker, df in data_dict.items():
            if ticker not in prices:
                continue
            
            df_to_date = df.loc[:date]
            if len(df_to_date) < 260:
                continue
            
            try:
                df_feat = compute_momentum_features(df_to_date)
                if len(df_feat) < 1:
                    continue
                
                last_row = df_feat.iloc[[-1]][features]
                
                # Handle any inf/nan
                if last_row.isnull().any().any() or np.isinf(last_row.values).any():
                    continue
                
                score = model.predict(last_row)[0]
                scores[ticker] = score
            except Exception:
                continue
        
        if not scores:
            continue
        
        # Rank and select top N
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


def run_pure_momentum_simulation(data_dict, start_date, end_date):
    """Run pure 12-1 momentum baseline."""
    print(f"\n[Simulating Pure Momentum] {start_date} to {end_date}")
    
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    all_dates = set()
    for df in data_dict.values():
        mask = (df.index >= start) & (df.index <= end)
        all_dates.update(df.index[mask].tolist())
    trading_dates = sorted(all_dates)
    
    rebalance_dates = set()
    current_month = None
    for d in trading_dates:
        if (d.year, d.month) != current_month:
            rebalance_dates.add(d)
            current_month = (d.year, d.month)
    
    portfolio = Portfolio(INITIAL_CAPITAL, SLIPPAGE_BPS)
    
    for date in trading_dates:
        prices = {t: df.loc[date, 'Close'] for t, df in data_dict.items() if date in df.index}
        portfolio.record_value(date, prices)
        
        if date not in rebalance_dates:
            continue
        
        # Calculate 12-1 momentum
        scores = {}
        for ticker, df in data_dict.items():
            if ticker not in prices:
                continue
            df_to_date = df.loc[:date]
            if len(df_to_date) < 252:
                continue
            
            price_21d_ago = df_to_date['Close'].iloc[-21]
            price_252d_ago = df_to_date['Close'].iloc[-252]
            if price_252d_ago > 0:
                momentum = (price_21d_ago / price_252d_ago) - 1
                scores[ticker] = momentum
        
        if not scores:
            continue
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [t for t, _ in sorted_scores[:MAX_POSITIONS]]
        
        for ticker in list(portfolio.positions.keys()):
            if ticker not in selected and ticker in prices:
                portfolio.sell(ticker, portfolio.positions[ticker]['shares'], prices[ticker], date)
        
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

def calculate_metrics(portfolio, name):
    if not portfolio or not portfolio.value_history:
        return {'name': name, 'return_pct': 0, 'cagr': 0, 'sharpe': 0, 'max_dd_pct': 0, 'n_trades': 0}
    
    values = pd.DataFrame(portfolio.value_history)
    values['date'] = pd.to_datetime(values['date'])
    values = values.set_index('date').sort_index()
    
    initial = values['value'].iloc[0]
    final = values['value'].iloc[-1]
    total_return = (final / initial - 1) * 100
    
    years = (values.index[-1] - values.index[0]).days / 365.25
    cagr = ((final / initial) ** (1/years) - 1) * 100 if years > 0 else 0
    
    returns = values['value'].pct_change().dropna()
    sharpe = (returns.mean() * 252 - 0.05) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    
    cummax = values['value'].cummax()
    drawdown = (values['value'] - cummax) / cummax
    max_dd = drawdown.min() * 100
    
    sells = [t for t in portfolio.trades if t['action'] == 'SELL' and 'pnl' in t]
    win_rate = sum(1 for t in sells if t['pnl'] > 0) / len(sells) * 100 if sells else 0
    
    return {
        'name': name,
        'initial': initial,
        'final': final,
        'return_pct': total_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'max_dd_pct': max_dd,
        'n_trades': len(portfolio.trades),
        'win_rate': win_rate
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("MOMENTUM-ENHANCED ML STRATEGY BACKTEST")
    print("Steve Cohen + Karpathy Edition")
    print("=" * 70)
    
    tickers = load_sp500_tickers()
    if not tickers:
        print("No tickers!")
        return
    
    data_dict = load_market_data(tickers[:200])
    
    if not data_dict:
        print("No data!")
        return
    
    results = []
    
    # ==========================================================================
    # SHORT-TERM BACKTEST (Oct 2025 - Feb 2026)
    # ==========================================================================
    print("\n" + "=" * 70)
    print(f"SHORT-TERM BACKTEST: {SHORT_START} to {SHORT_END}")
    print("=" * 70)
    
    # Train model on data BEFORE test period (point-in-time)
    train_end = pd.Timestamp(SHORT_START)
    model, features = train_momentum_ml(data_dict, train_end)
    
    if model is None:
        print("Training failed!")
        return
    
    # Run simulations
    portfolio_ml = run_momentum_ml_simulation(data_dict, model, features, SHORT_START, SHORT_END, "Momentum ML")
    portfolio_pure = run_pure_momentum_simulation(data_dict, SHORT_START, SHORT_END)
    
    m_ml = calculate_metrics(portfolio_ml, "Momentum ML")
    m_pure = calculate_metrics(portfolio_pure, "Pure Momentum")
    
    print("\n[SHORT-TERM RESULTS]")
    print(f"{'Strategy':<20} {'Return':>10} {'Sharpe':>8} {'Max DD':>10} {'Trades':>8} {'Win%':>8}")
    print("-" * 64)
    print(f"{m_pure['name']:<20} {m_pure['return_pct']:>9.1f}% {m_pure['sharpe']:>8.2f} {m_pure['max_dd_pct']:>9.1f}% {m_pure['n_trades']:>8} {m_pure['win_rate']:>7.1f}%")
    print(f"{m_ml['name']:<20} {m_ml['return_pct']:>9.1f}% {m_ml['sharpe']:>8.2f} {m_ml['max_dd_pct']:>9.1f}% {m_ml['n_trades']:>8} {m_ml['win_rate']:>7.1f}%")
    
    results.append(('short', m_pure))
    results.append(('short', m_ml))
    
    # ==========================================================================
    # LONG-TERM BACKTEST (2015 - 2025)
    # ==========================================================================
    print("\n" + "=" * 70)
    print(f"LONG-TERM BACKTEST: {LONG_START} to {LONG_END}")
    print("=" * 70)
    
    # Train on first 50% of data for long-term test
    train_end_long = pd.Timestamp("2020-01-01")
    model_long, features_long = train_momentum_ml(data_dict, train_end_long)
    
    if model_long is None:
        print("Long-term training failed!")
    else:
        portfolio_ml_long = run_momentum_ml_simulation(data_dict, model_long, features_long, LONG_START, LONG_END, "Momentum ML")
        portfolio_pure_long = run_pure_momentum_simulation(data_dict, LONG_START, LONG_END)
        
        m_ml_long = calculate_metrics(portfolio_ml_long, "Momentum ML")
        m_pure_long = calculate_metrics(portfolio_pure_long, "Pure Momentum")
        
        print("\n[LONG-TERM RESULTS]")
        print(f"{'Strategy':<20} {'Return':>10} {'CAGR':>8} {'Sharpe':>8} {'Max DD':>10} {'Trades':>8}")
        print("-" * 74)
        print(f"{m_pure_long['name']:<20} {m_pure_long['return_pct']:>9.1f}% {m_pure_long['cagr']:>7.1f}% {m_pure_long['sharpe']:>8.2f} {m_pure_long['max_dd_pct']:>9.1f}% {m_pure_long['n_trades']:>8}")
        print(f"{m_ml_long['name']:<20} {m_ml_long['return_pct']:>9.1f}% {m_ml_long['cagr']:>7.1f}% {m_ml_long['sharpe']:>8.2f} {m_ml_long['max_dd_pct']:>9.1f}% {m_ml_long['n_trades']:>8}")
        
        results.append(('long', m_pure_long))
        results.append(('long', m_ml_long))
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/momentum_ml_backtest.json", 'w') as f:
        json.dump([r[1] for r in results], f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)


if __name__ == "__main__":
    main()
