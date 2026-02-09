#!/usr/bin/env python3
"""
Strategy Improvement Backtest - Oct 2025 to Feb 2026

Point-in-time simulation using full cache data.
Compares baseline vs improved strategies.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from src.data.loader import fetch_data


# =============================================================================
# CONFIGURATION
# =============================================================================

BACKTEST_START = "2025-10-01"
BACKTEST_END = "2026-02-02"
INITIAL_CAPITAL = 10000.0
SLIPPAGE_BPS = 5
TOP_PCT = 0.10
MAX_POSITIONS = 10
POSITION_SIZE_PCT = 0.10


# =============================================================================
# DATA LOADING
# =============================================================================

def load_sp500_tickers():
    """Load S&P 500 tickers."""
    cache_file = "data/sp500_tickers.txt"
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    return []


def load_market_data(tickers):
    """Load full historical data for calculations."""
    print(f"Loading market data for {len(tickers)} tickers...")
    data_dict = fetch_data(tickers, period="max", use_cache=True)
    
    # Filter to 2010+ (matching training filter)
    for ticker in list(data_dict.keys()):
        df = data_dict[ticker]
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df[df.index >= '2010-01-01']
        if len(df) < 300:  # Need at least 300 days
            del data_dict[ticker]
        else:
            data_dict[ticker] = df
    
    print(f"  Loaded {len(data_dict)} tickers")
    return data_dict


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
            self.positions[ticker] = {'shares': total, 'avg_cost': avg, 'entry_date': pos['entry_date']}
        else:
            self.positions[ticker] = {'shares': shares, 'avg_cost': exec_price, 'entry_date': date}
        
        self.trades.append({'date': date, 'ticker': ticker, 'action': 'BUY', 'shares': shares, 'price': exec_price})
    
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
        
        self.trades.append({'date': date, 'ticker': ticker, 'action': 'SELL', 'shares': shares, 'price': exec_price, 'pnl': pnl})
    
    def record_value(self, date, prices):
        self.value_history.append({'date': date, 'value': self.get_value(prices)})


# =============================================================================
# STRATEGY FUNCTIONS
# =============================================================================

def calculate_momentum(df_to_date):
    """Calculate 12-1 month momentum."""
    if len(df_to_date) < 252:
        return None
    
    # Price 21 trading days ago (skip last month)
    price_recent = df_to_date['Close'].iloc[-21]
    # Price 252 trading days ago
    price_12m_ago = df_to_date['Close'].iloc[-252]
    
    if price_12m_ago <= 0:
        return None
    
    return (price_recent / price_12m_ago) - 1


def run_simulation(data_dict, strategy_name, rebalance_freq='daily', 
                   use_correlation_guard=False, corr_threshold=0.7):
    """Run strategy simulation."""
    
    # Get trading dates in backtest period
    start = pd.Timestamp(BACKTEST_START)
    end = pd.Timestamp(BACKTEST_END)
    
    all_dates = set()
    for df in data_dict.values():
        mask = (df.index >= start) & (df.index <= end)
        all_dates.update(df.index[mask].tolist())
    trading_dates = sorted(all_dates)
    
    if not trading_dates:
        print(f"  No trading dates found!")
        return None
    
    print(f"  Trading period: {trading_dates[0].date()} to {trading_dates[-1].date()}")
    print(f"  Trading days: {len(trading_dates)}")
    
    # Determine rebalance dates
    if rebalance_freq == 'weekly':
        rebalance_dates = set()
        current_week = None
        for d in trading_dates:
            week = (d.year, d.isocalendar()[1])
            if week != current_week:
                rebalance_dates.add(d)
                current_week = week
    elif rebalance_freq == 'monthly':
        rebalance_dates = set()
        current_month = None
        for d in trading_dates:
            ym = (d.year, d.month)
            if ym != current_month:
                rebalance_dates.add(d)
                current_month = ym
    else:
        rebalance_dates = set(trading_dates)
    
    print(f"  Rebalance frequency: {rebalance_freq} ({len(rebalance_dates)} rebalances)")
    
    # Initialize
    portfolio = Portfolio(INITIAL_CAPITAL, SLIPPAGE_BPS)
    
    # Load predictor if ML
    predictor = None
    if strategy_name.startswith('ML'):
        from src.models.predictor import EnsemblePredictor
        predictor = EnsemblePredictor()
    
    for date in trading_dates:
        # Get current prices
        prices = {}
        for ticker, df in data_dict.items():
            if date in df.index:
                prices[ticker] = df.loc[date, 'Close']
        
        portfolio.record_value(date, prices)
        
        if date not in rebalance_dates:
            continue
        
        # Calculate scores for all tickers
        scores = {}
        for ticker, df in data_dict.items():
            if ticker not in prices:
                continue
            
            df_to_date = df.loc[:date]
            
            if strategy_name.startswith('Momentum'):
                score = calculate_momentum(df_to_date)
            elif strategy_name.startswith('ML') and predictor:
                try:
                    score = predictor.predict(df_to_date)
                except:
                    score = None
            else:
                continue
            
            if score is not None:
                scores[ticker] = score
        
        if not scores:
            continue
        
        # Rank
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        n_buy = max(1, int(len(sorted_scores) * TOP_PCT))
        
        # Select top N (with optional correlation guard)
        selected = []
        
        if use_correlation_guard:
            for ticker, score in sorted_scores:
                if len(selected) >= MAX_POSITIONS:
                    break
                if ticker not in prices:
                    continue
                
                # Check correlation with already selected
                is_correlated = False
                if selected:
                    df_new = data_dict[ticker].loc[:date]['Close'].tail(60)
                    for existing in selected:
                        df_exist = data_dict[existing].loc[:date]['Close'].tail(60)
                        common = df_new.index.intersection(df_exist.index)
                        if len(common) >= 30:
                            corr = df_new.loc[common].corr(df_exist.loc[common])
                            if abs(corr) > corr_threshold:
                                is_correlated = True
                                break
                
                if not is_correlated:
                    selected.append(ticker)
        else:
            selected = [t for t, _ in sorted_scores[:n_buy]][:MAX_POSITIONS]
        
        # Sell positions not in selected
        for ticker in list(portfolio.positions.keys()):
            if ticker not in selected and ticker in prices:
                pos = portfolio.positions[ticker]
                portfolio.sell(ticker, pos['shares'], prices[ticker], date)
        
        # Buy new positions
        for ticker in selected:
            if ticker not in portfolio.positions and ticker in prices:
                if len(portfolio.positions) >= MAX_POSITIONS:
                    break
                available = portfolio.cash * 0.95  # Keep some buffer
                position_value = min(available, INITIAL_CAPITAL * POSITION_SIZE_PCT)
                shares = int(position_value / prices[ticker])
                if shares > 0:
                    portfolio.buy(ticker, shares, prices[ticker], date)
    
    return portfolio


def calculate_metrics(portfolio, name):
    """Calculate performance metrics."""
    if not portfolio or not portfolio.value_history:
        return {'name': name, 'return_pct': 0, 'sharpe': 0, 'max_dd_pct': 0, 'n_trades': 0, 'win_rate': 0}
    
    values = pd.DataFrame(portfolio.value_history)
    values['date'] = pd.to_datetime(values['date'])
    values = values.set_index('date').sort_index()
    
    initial = values['value'].iloc[0]
    final = values['value'].iloc[-1]
    total_return = (final / initial - 1) * 100
    
    returns = values['value'].pct_change().dropna()
    
    if len(returns) > 1 and returns.std() > 0:
        sharpe = (returns.mean() * 252 - 0.05) / (returns.std() * np.sqrt(252))
    else:
        sharpe = 0
    
    cummax = values['value'].cummax()
    drawdown = (values['value'] - cummax) / cummax
    max_dd = drawdown.min() * 100
    
    trades = portfolio.trades
    sells = [t for t in trades if t['action'] == 'SELL' and 'pnl' in t]
    win_rate = sum(1 for t in sells if t['pnl'] > 0) / len(sells) * 100 if sells else 0
    
    return {
        'name': name,
        'initial': initial,
        'final': final,
        'return_pct': total_return,
        'sharpe': sharpe,
        'max_dd_pct': max_dd,
        'n_trades': len(trades),
        'win_rate': win_rate
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("STRATEGY IMPROVEMENT BACKTEST")
    print(f"Period: {BACKTEST_START} to {BACKTEST_END}")
    print("=" * 70)
    
    tickers = load_sp500_tickers()
    if not tickers:
        print("No tickers found!")
        return
    
    # Use subset for faster testing
    data_dict = load_market_data(tickers[:200])
    
    if not data_dict:
        print("No data loaded!")
        return
    
    results = []
    
    # 1. Baseline Momentum (monthly)
    print("\n[1/4] Momentum Baseline (monthly)...")
    p1 = run_simulation(data_dict, 'Momentum', 'monthly', use_correlation_guard=False)
    m1 = calculate_metrics(p1, 'Momentum Baseline')
    results.append(m1)
    
    # 2. Momentum + Correlation Guard
    print("\n[2/4] Momentum + Correlation Guard...")
    p2 = run_simulation(data_dict, 'Momentum+Corr', 'monthly', use_correlation_guard=True, corr_threshold=0.7)
    m2 = calculate_metrics(p2, 'Momentum + Corr Guard')
    results.append(m2)
    
    # 3. ML Daily Baseline
    print("\n[3/4] ML Daily Baseline...")
    p3 = run_simulation(data_dict, 'ML Daily', 'daily', use_correlation_guard=False)
    m3 = calculate_metrics(p3, 'ML Daily')
    results.append(m3)
    
    # 4. ML Weekly (Improved)
    print("\n[4/4] ML Weekly (Improved)...")
    p4 = run_simulation(data_dict, 'ML Weekly', 'weekly', use_correlation_guard=False)
    m4 = calculate_metrics(p4, 'ML Weekly')
    results.append(m4)
    
    # Print results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    
    print(f"\n{'Strategy':<25} {'Return':>10} {'Sharpe':>8} {'Max DD':>10} {'Trades':>8} {'Win%':>8}")
    print("-" * 69)
    
    for r in results:
        print(f"{r['name']:<25} {r['return_pct']:>9.1f}% {r['sharpe']:>8.2f} {r['max_dd_pct']:>9.1f}% {r['n_trades']:>8} {r['win_rate']:>7.1f}%")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/improvement_backtest_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    # Momentum comparison
    print(f"\nMomentum:")
    print(f"  Baseline:   {m1['return_pct']:+.1f}%, Max DD: {m1['max_dd_pct']:.1f}%")
    print(f"  + Corr:     {m2['return_pct']:+.1f}%, Max DD: {m2['max_dd_pct']:.1f}%")
    if m1['max_dd_pct'] != 0:
        dd_improvement = m2['max_dd_pct'] - m1['max_dd_pct']
        print(f"  DD Change:  {dd_improvement:+.1f}%")
    
    # ML comparison
    print(f"\nML Strategy:")
    print(f"  Daily:  {m3['return_pct']:+.1f}%, {m3['n_trades']} trades")
    print(f"  Weekly: {m4['return_pct']:+.1f}%, {m4['n_trades']} trades")
    if m3['n_trades'] > 0:
        trade_reduction = (1 - m4['n_trades']/m3['n_trades']) * 100
        print(f"  Trade Reduction: {trade_reduction:.0f}%")


if __name__ == "__main__":
    main()
