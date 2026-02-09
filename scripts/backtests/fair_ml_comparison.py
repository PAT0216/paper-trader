#!/usr/bin/env python3
"""
Fair ML Comparison - Uses ACTUAL Pre-Trained Models

Old ML: Uses the LIVE pre-trained EnsemblePredictor (same as production)
New ML: Uses new momentum-aligned features (for comparison)

This properly replicates what the live system does.
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
    MOMENTUM_FEATURE_COLUMNS
)
# Use the ACTUAL live predictor
from src.models.predictor import EnsemblePredictor


# =============================================================================
# CONFIGURATION
# =============================================================================

TEST_START = "2025-10-01"
TEST_END = "2026-02-03"

INITIAL_CAPITAL = 10000.0
SLIPPAGE_BPS = 5
MAX_POSITIONS = 10  # Live system uses ~10 positions
POSITION_SIZE_PCT = 0.10


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
        if len(df) < 500:
            del data_dict[ticker]
        else:
            data_dict[ticker] = df
    
    print(f"  Loaded {len(data_dict)} tickers")
    return data_dict


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
# SIMULATION
# =============================================================================

def run_live_ml_simulation(data_dict, predictor, name):
    """
    Run simulation using the ACTUAL live EnsemblePredictor.
    
    This should match the live system's behavior.
    """
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
        
        # Score all tickers using the LIVE predictor
        scores = {}
        for ticker, df in data_dict.items():
            if ticker not in prices:
                continue
            
            df_to_date = df.loc[:date]
            if len(df_to_date) < 200:
                continue
            
            try:
                score = predictor.predict(df_to_date)
                if score is not None:
                    scores[ticker] = float(score)
            except Exception:
                continue
        
        if not scores:
            continue
        
        # Rank and select top 10%
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        n_select = max(1, int(len(sorted_scores) * 0.10))
        selected = [t for t, _ in sorted_scores[:n_select]][:MAX_POSITIONS]
        
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


def run_new_ml_simulation(data_dict, model, features, name):
    """Run simulation with NEW momentum-aligned ML."""
    print(f"\n[Simulating {name}]")
    
    start = pd.Timestamp(TEST_START)
    end = pd.Timestamp(TEST_END)
    
    all_dates = set()
    for df in data_dict.values():
        mask = (df.index >= start) & (df.index <= end)
        all_dates.update(df.index[mask].tolist())
    trading_dates = sorted(all_dates)
    
    print(f"  Period: {trading_dates[0].date()} to {trading_dates[-1].date()}")
    print(f"  Trading days: {len(trading_dates)}")
    
    portfolio = Portfolio(name, INITIAL_CAPITAL, SLIPPAGE_BPS)
    
    for date in trading_dates:
        prices = {}
        for ticker, df in data_dict.items():
            if date in df.index:
                prices[ticker] = df.loc[date, 'Close']
        
        portfolio.record_value(date, prices)
        
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
                if last_row.isnull().any().any() or np.isinf(last_row.values).any():
                    continue
                
                score = model.predict(last_row)[0]
                scores[ticker] = score
            except Exception:
                continue
        
        if not scores:
            continue
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        n_select = max(1, int(len(sorted_scores) * 0.10))
        selected = [t for t, _ in sorted_scores[:n_select]][:MAX_POSITIONS]
        
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


def train_new_ml(data_dict, train_end_date):
    """Train new momentum ML model."""
    print("\n[Training New ML Model]")
    
    all_data = []
    for ticker, df in data_dict.items():
        df_train = df[df.index <= train_end_date].copy()
        if len(df_train) < 300:
            continue
        
        from src.features.momentum_features import create_momentum_target
        df_feat = compute_momentum_features(df_train)
        df_feat = create_momentum_target(df_feat, horizon=20)
        df_feat = df_feat.dropna(subset=['Target'])
        
        if len(df_feat) > 100:
            all_data.append(df_feat)
    
    if not all_data:
        return None, []
    
    full_df = pd.concat(all_data).sort_index()
    print(f"  Training samples: {len(full_df):,}")
    
    X = full_df[MOMENTUM_FEATURE_COLUMNS].values
    y = full_df['Target'].values
    
    X = np.where(np.isinf(X), np.nan, X)
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[valid], y[valid]
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    model.fit(X, y)
    
    return model, MOMENTUM_FEATURE_COLUMNS


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
        'win_rate': win_rate
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("FAIR ML COMPARISON - USING ACTUAL LIVE MODEL")
    print(f"Test Period: {TEST_START} to {TEST_END}")
    print("=" * 70)
    
    tickers = load_sp500_tickers()
    data_dict = load_market_data(tickers[:200])
    
    if not data_dict:
        print("No data!")
        return
    
    # Load the ACTUAL live predictor
    print("\n[Loading LIVE EnsemblePredictor]")
    live_predictor = EnsemblePredictor()
    
    # Train new ML model
    train_end = pd.Timestamp("2025-09-30")
    new_model, new_features = train_new_ml(data_dict, train_end)
    
    if new_model is None:
        print("Training failed!")
        return
    
    # Run simulations
    portfolio_live = run_live_ml_simulation(data_dict, live_predictor, 'Live ML (Actual)')
    portfolio_new = run_new_ml_simulation(data_dict, new_model, new_features, 'New ML (Momentum)')
    
    # Calculate metrics
    m_live = calculate_metrics(portfolio_live)
    m_new = calculate_metrics(portfolio_new)
    
    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    print(f"\n{'Metric':<20} {'Live ML':<15} {'New ML':<15} {'Delta':<15}")
    print("-" * 65)
    print(f"{'Final Value':<20} ${m_live['final']:,.0f}{'':<5} ${m_new['final']:,.0f}{'':<5} ${m_new['final']-m_live['final']:+,.0f}")
    print(f"{'Return':<20} {m_live['return_pct']:+.1f}%{'':<10} {m_new['return_pct']:+.1f}%{'':<10} {m_new['return_pct']-m_live['return_pct']:+.1f}%")
    print(f"{'Sharpe':<20} {m_live['sharpe']:.2f}{'':<12} {m_new['sharpe']:.2f}{'':<12} {m_new['sharpe']-m_live['sharpe']:+.2f}")
    print(f"{'Max Drawdown':<20} {m_live['max_dd_pct']:.1f}%{'':<10} {m_new['max_dd_pct']:.1f}%{'':<10} {m_new['max_dd_pct']-m_live['max_dd_pct']:+.1f}%")
    print(f"{'Win Rate':<20} {m_live['win_rate']:.1f}%{'':<10} {m_new['win_rate']:.1f}%{'':<10} {m_new['win_rate']-m_live['win_rate']:+.1f}%")
    print(f"{'Total Trades':<20} {m_live['n_buys']+m_live['n_sells']:<15} {m_new['n_buys']+m_new['n_sells']:<15}")
    
    # Compare to actual ledger
    print("\n" + "=" * 70)
    print("COMPARISON TO ACTUAL LEDGER")
    print("=" * 70)
    
    try:
        with open('data/snapshots/ml.json', 'r') as f:
            live_snapshot = json.load(f)
        actual_return = live_snapshot.get('return_pct', 0)
        actual_value = live_snapshot.get('value', 0)
        
        print(f"\nActual Live ML (from ledger): ${actual_value:,.2f} ({actual_return:+.2f}%)")
        print(f"Simulated Live ML:            ${m_live['final']:,.0f} ({m_live['return_pct']:+.1f}%)")
        print(f"Difference:                   ${actual_value - m_live['final']:+,.0f}")
    except:
        pass
    
    # Save results
    os.makedirs("results", exist_ok=True)
    results = {
        'test_period': f"{TEST_START} to {TEST_END}",
        'live_ml': {k: v for k, v in m_live.items()},
        'new_ml': {k: v for k, v in m_new.items()}
    }
    with open("results/fair_ml_comparison.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to results/fair_ml_comparison.json")


if __name__ == "__main__":
    main()
