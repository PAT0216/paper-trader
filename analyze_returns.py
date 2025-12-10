#!/usr/bin/env python3
"""
Trade Return Distribution Analysis

Investigates the R² paradox: why does a model with R² = -0.01 
produce 630% returns in walk-forward validation?

This script analyzes:
1. P&L concentration (what % of returns come from top N trades)
2. Win rate and profit factor
3. Average win vs average loss
4. Holding period analysis

Usage: python analyze_returns.py
"""

import pandas as pd
import numpy as np
import os

def analyze_trade_returns():
    """Analyze trade return distribution from backtest results."""
    
    trades_file = "results/backtest_trades.csv"
    
    if not os.path.exists(trades_file):
        print(f"❌ Trade file not found: {trades_file}")
        print("   Run a backtest first: python run_backtest.py")
        return
    
    # Load trades
    trades = pd.read_csv(trades_file)
    print("=" * 60)
    print("TRADE RETURN DISTRIBUTION ANALYSIS")
    print("=" * 60)
    print(f"\nTotal trades: {len(trades)}")
    
    # Filter to completed trades (with P&L)
    completed = trades[trades['pnl'] != 0].copy()
    print(f"Completed trades (with P&L): {len(completed)}")
    
    if len(completed) == 0:
        print("No completed trades found.")
        return
    
    # Basic stats
    total_pnl = completed['pnl'].sum()
    winning = completed[completed['pnl'] > 0]
    losing = completed[completed['pnl'] < 0]
    
    print(f"\n--- SUMMARY STATISTICS ---")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"Win Rate: {len(winning)/len(completed)*100:.1f}% ({len(winning)}/{len(completed)})")
    
    if len(winning) > 0:
        avg_win = winning['pnl'].mean()
        print(f"Average Win: ${avg_win:,.2f}")
    else:
        avg_win = 0
    
    if len(losing) > 0:
        avg_loss = losing['pnl'].mean()
        print(f"Average Loss: ${avg_loss:,.2f}")
    else:
        avg_loss = 0
    
    if avg_loss != 0:
        profit_factor = abs(winning['pnl'].sum() / losing['pnl'].sum()) if len(losing) > 0 else float('inf')
        print(f"Profit Factor: {profit_factor:.2f}")
    
    # P&L CONCENTRATION ANALYSIS
    print(f"\n--- P&L CONCENTRATION ANALYSIS ---")
    print("(Checking if returns are driven by a few lucky trades)")
    
    completed_sorted = completed.sort_values('pnl', ascending=False)
    
    for n in [5, 10, 20, 50]:
        if len(completed_sorted) >= n:
            top_n_pnl = completed_sorted.head(n)['pnl'].sum()
            pct = (top_n_pnl / total_pnl * 100) if total_pnl != 0 else 0
            print(f"  Top {n} trades: ${top_n_pnl:,.2f} ({pct:.1f}% of total P&L)")
    
    # Top 10 trades
    print(f"\n--- TOP 10 WINNING TRADES ---")
    top_10 = completed_sorted.head(10)
    for _, row in top_10.iterrows():
        print(f"  {row['ticker']}: ${row['pnl']:,.2f} ({row['holding_days']} days)")
    
    # Bottom 10 trades
    print(f"\n--- TOP 10 LOSING TRADES ---")
    bottom_10 = completed_sorted.tail(10)
    for _, row in bottom_10.iterrows():
        print(f"  {row['ticker']}: ${row['pnl']:,.2f} ({row['holding_days']} days)")
    
    # Holding period analysis
    print(f"\n--- HOLDING PERIOD ANALYSIS ---")
    print(f"Average holding period: {completed['holding_days'].mean():.1f} days")
    print(f"Median holding period: {completed['holding_days'].median():.1f} days")
    
    # Check if winners held longer
    if len(winning) > 0 and len(losing) > 0:
        print(f"Avg holding (winners): {winning['holding_days'].mean():.1f} days")
        print(f"Avg holding (losers): {losing['holding_days'].mean():.1f} days")
    
    # RISK ASSESSMENT
    print(f"\n--- RISK ASSESSMENT ---")
    
    # Concentration check
    if len(completed_sorted) >= 10:
        top_10_pct = completed_sorted.head(10)['pnl'].sum() / total_pnl * 100 if total_pnl != 0 else 0
        if top_10_pct > 80:
            print(f"⚠️  HIGH CONCENTRATION: {top_10_pct:.1f}% of P&L from top 10 trades")
            print("   Returns are fragile - driven by a few outlier trades")
        elif top_10_pct > 50:
            print(f"⚠️  MODERATE CONCENTRATION: {top_10_pct:.1f}% of P&L from top 10 trades")
        else:
            print(f"✅ LOW CONCENTRATION: {top_10_pct:.1f}% of P&L from top 10 trades")
            print("   Returns are distributed across many trades")
    
    # Win rate check
    win_rate = len(winning) / len(completed) * 100
    if win_rate < 50:
        print(f"⚠️  Win rate below 50%: {win_rate:.1f}%")
    else:
        print(f"✅ Win rate: {win_rate:.1f}%")
    
    # Ticker concentration
    print(f"\n--- TICKER CONCENTRATION ---")
    ticker_pnl = completed.groupby('ticker')['pnl'].sum().sort_values(ascending=False)
    print("Top 5 tickers by total P&L:")
    for ticker, pnl in ticker_pnl.head(5).items():
        pct = (pnl / total_pnl * 100) if total_pnl != 0 else 0
        print(f"  {ticker}: ${pnl:,.2f} ({pct:.1f}%)")
    
    # Save analysis to file
    output_file = "results/trade_analysis.txt"
    with open(output_file, 'w') as f:
        f.write("Trade Return Distribution Analysis\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total Trades: {len(completed)}\n")
        f.write(f"Total P&L: ${total_pnl:,.2f}\n")
        f.write(f"Win Rate: {win_rate:.1f}%\n")
        f.write(f"Profit Factor: {profit_factor:.2f}\n\n")
        
        if len(completed_sorted) >= 10:
            top_10_pct = completed_sorted.head(10)['pnl'].sum() / total_pnl * 100 if total_pnl != 0 else 0
            f.write(f"P&L Concentration (Top 10): {top_10_pct:.1f}%\n")
        
        f.write(f"\nTop 10 Winners:\n")
        for _, row in top_10.iterrows():
            f.write(f"  {row['ticker']}: ${row['pnl']:,.2f}\n")
    
    print(f"\n✅ Analysis saved to {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    analyze_trade_returns()
