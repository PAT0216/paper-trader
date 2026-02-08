#!/usr/bin/env python3
"""
Extended Walkforward Backtest - 2015 to 2025

Validates strategy improvements over longer period.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import json

from src.data.loader import fetch_data


# =============================================================================
# CONFIGURATION
# =============================================================================

BACKTEST_START = "2015-01-01"
BACKTEST_END = "2025-12-31"
INITIAL_CAPITAL = 100000.0
SLIPPAGE_BPS = 5
TOP_PCT = 0.10
MAX_POSITIONS = 10
POSITION_SIZE_PCT = 0.10


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
        df = df[df.index >= '2010-01-01']
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
# STRATEGY
# =============================================================================

def calculate_momentum(df_to_date):
    if len(df_to_date) < 252:
        return None
    price_recent = df_to_date['Close'].iloc[-21]
    price_12m_ago = df_to_date['Close'].iloc[-252]
    if price_12m_ago <= 0:
        return None
    return (price_recent / price_12m_ago) - 1


def run_momentum_simulation(data_dict, use_correlation_guard=False, corr_threshold=0.7):
    start = pd.Timestamp(BACKTEST_START)
    end = pd.Timestamp(BACKTEST_END)
    
    all_dates = set()
    for df in data_dict.values():
        mask = (df.index >= start) & (df.index <= end)
        all_dates.update(df.index[mask].tolist())
    trading_dates = sorted(all_dates)
    
    print(f"  Trading period: {trading_dates[0].date()} to {trading_dates[-1].date()}")
    print(f"  Trading days: {len(trading_dates)}")
    
    # Monthly rebalance
    rebalance_dates = set()
    current_month = None
    for d in trading_dates:
        ym = (d.year, d.month)
        if ym != current_month:
            rebalance_dates.add(d)
            current_month = ym
    
    print(f"  Rebalances: {len(rebalance_dates)}")
    
    portfolio = Portfolio(INITIAL_CAPITAL, SLIPPAGE_BPS)
    
    for date in trading_dates:
        prices = {}
        for ticker, df in data_dict.items():
            if date in df.index:
                prices[ticker] = df.loc[date, 'Close']
        
        portfolio.record_value(date, prices)
        
        if date not in rebalance_dates:
            continue
        
        scores = {}
        for ticker, df in data_dict.items():
            if ticker not in prices:
                continue
            df_to_date = df.loc[:date]
            score = calculate_momentum(df_to_date)
            if score is not None:
                scores[ticker] = score
        
        if not scores:
            continue
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select with optional correlation guard
        selected = []
        if use_correlation_guard:
            for ticker, score in sorted_scores:
                if len(selected) >= MAX_POSITIONS:
                    break
                if ticker not in prices:
                    continue
                
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
            n_buy = max(1, int(len(sorted_scores) * TOP_PCT))
            selected = [t for t, _ in sorted_scores[:n_buy]][:MAX_POSITIONS]
        
        # Sell non-selected
        for ticker in list(portfolio.positions.keys()):
            if ticker not in selected and ticker in prices:
                pos = portfolio.positions[ticker]
                portfolio.sell(ticker, pos['shares'], prices[ticker], date)
        
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


def calculate_metrics(portfolio, name):
    if not portfolio or not portfolio.value_history:
        return {'name': name, 'return_pct': 0, 'cagr': 0, 'sharpe': 0, 'max_dd_pct': 0, 'n_trades': 0, 'win_rate': 0}
    
    values = pd.DataFrame(portfolio.value_history)
    values['date'] = pd.to_datetime(values['date'])
    values = values.set_index('date').sort_index()
    
    initial = values['value'].iloc[0]
    final = values['value'].iloc[-1]
    total_return = (final / initial - 1) * 100
    
    # CAGR
    years = (values.index[-1] - values.index[0]).days / 365.25
    cagr = ((final / initial) ** (1/years) - 1) * 100 if years > 0 else 0
    
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
        'cagr': cagr,
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
    print("EXTENDED WALKFORWARD BACKTEST")
    print(f"Period: {BACKTEST_START} to {BACKTEST_END}")
    print("=" * 70)
    
    tickers = load_sp500_tickers()
    if not tickers:
        print("No tickers found!")
        return
    
    data_dict = load_market_data(tickers[:200])
    
    if not data_dict:
        print("No data loaded!")
        return
    
    results = []
    
    print("\n[1/2] Momentum Baseline...")
    p1 = run_momentum_simulation(data_dict, use_correlation_guard=False)
    m1 = calculate_metrics(p1, 'Momentum Baseline')
    results.append(m1)
    
    print("\n[2/2] Momentum + Correlation Guard...")
    p2 = run_momentum_simulation(data_dict, use_correlation_guard=True, corr_threshold=0.7)
    m2 = calculate_metrics(p2, 'Momentum + Corr Guard')
    results.append(m2)
    
    print("\n" + "=" * 70)
    print("EXTENDED BACKTEST RESULTS (2015-2025)")
    print("=" * 70)
    
    print(f"\n{'Strategy':<25} {'Return':>10} {'CAGR':>8} {'Sharpe':>8} {'Max DD':>10} {'Trades':>8} {'Win%':>8}")
    print("-" * 77)
    
    for r in results:
        print(f"{r['name']:<25} {r['return_pct']:>9.1f}% {r['cagr']:>7.1f}% {r['sharpe']:>8.2f} {r['max_dd_pct']:>9.1f}% {r['n_trades']:>8} {r['win_rate']:>7.1f}%")
    
    os.makedirs("results", exist_ok=True)
    with open("results/extended_backtest_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 70)
    
    print(f"\nMomentum Baseline:")
    print(f"  Total Return: {m1['return_pct']:+.1f}%")
    print(f"  CAGR: {m1['cagr']:.1f}%")
    print(f"  Sharpe: {m1['sharpe']:.2f}")
    print(f"  Max DD: {m1['max_dd_pct']:.1f}%")
    
    print(f"\nMomentum + Correlation Guard:")
    print(f"  Total Return: {m2['return_pct']:+.1f}%")
    print(f"  CAGR: {m2['cagr']:.1f}%")
    print(f"  Sharpe: {m2['sharpe']:.2f}")
    print(f"  Max DD: {m2['max_dd_pct']:.1f}%")
    
    print(f"\nImprovement:")
    print(f"  Return Delta: {m2['return_pct'] - m1['return_pct']:+.1f}%")
    print(f"  CAGR Delta: {m2['cagr'] - m1['cagr']:+.1f}%")
    print(f"  Max DD Change: {m2['max_dd_pct'] - m1['max_dd_pct']:+.1f}%")


if __name__ == "__main__":
    main()
