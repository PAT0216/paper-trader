#!/usr/bin/env python
"""
2025 Walk-Forward Trading Simulation

Simulates what would have happened if we ran the bot daily from Jan 1, 2025.
- Retrains model each day using data up to (date - 1)
- Generates predictions for date
- Simulates trades based on signals
- Tracks portfolio value

Output: results/simulation_2025.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import fetch_data
from src.data.universe import fetch_sp500_tickers
from src.features.indicators import generate_features, create_target, FEATURE_COLUMNS
from src.models.trainer import train_model
import xgboost as xgb

# Configuration
START_DATE = datetime(2025, 1, 2)  # First trading day of 2025
END_DATE = datetime(2025, 12, 11)  # Yesterday (we don't have today's close yet)
INITIAL_CASH = 10000.0
MAX_POSITION_PCT = 0.05  # 5% per position
BUY_THRESHOLD = 0.005  # 0.5% expected return
SELL_THRESHOLD = -0.005

print("=" * 60)
print("2025 WALK-FORWARD TRADING SIMULATION")
print("=" * 60)
print(f"Period: {START_DATE.date()} to {END_DATE.date()}")
print(f"Initial Capital: ${INITIAL_CASH:,.2f}")
print()

# Get S&P 500 tickers
print("📊 Fetching S&P 500 universe...")
try:
    tickers = fetch_sp500_tickers()
    print(f"   Loaded {len(tickers)} tickers")
except Exception as e:
    print(f"   Error: {e}, using mega-caps")
    from src.data.universe import get_mega_caps
    tickers = get_mega_caps()

# Fetch all data upfront (faster than fetching per-day)
print("\n📈 Fetching historical data (this may take a while)...")
all_data = fetch_data(tickers, period='max')  # Get all available data
print(f"   Loaded data for {len(all_data)} tickers")

# Filter to valid tickers with data before 2025
valid_tickers = []
for ticker, df in all_data.items():
    if len(df) > 200 and df.index.min().year < 2025:
        valid_tickers.append(ticker)
print(f"   {len(valid_tickers)} tickers have sufficient history")

# Simulation tracking
portfolio = {
    'cash': INITIAL_CASH,
    'positions': {},  # ticker -> {'shares': n, 'entry_price': p}
}
simulation_log = []
daily_values = []

def get_portfolio_value(portfolio, prices):
    """Calculate total portfolio value."""
    value = portfolio['cash']
    for ticker, pos in portfolio['positions'].items():
        if ticker in prices:
            value += pos['shares'] * prices[ticker]
    return value

def get_latest_prices(data_dict, date):
    """Get closing prices for a given date."""
    prices = {}
    for ticker, df in data_dict.items():
        try:
            # Get the most recent price on or before date
            mask = df.index <= pd.Timestamp(date)
            if mask.any():
                prices[ticker] = df.loc[mask, 'Close'].iloc[-1]
        except:
            pass
    return prices

# Main simulation loop
current_date = START_DATE
training_interval = 5  # Retrain every 5 days (faster simulation)
last_train_date = None
model = None
selected_features = FEATURE_COLUMNS

print(f"\n🚀 Starting simulation (retraining every {training_interval} days)...")
print("-" * 60)

day_count = 0
while current_date <= END_DATE:
    day_count += 1
    
    # Skip weekends
    if current_date.weekday() >= 5:
        current_date += timedelta(days=1)
        continue
    
    # Check if we need to retrain
    should_train = (
        model is None or 
        last_train_date is None or 
        (current_date - last_train_date).days >= training_interval
    )
    
    if should_train:
        print(f"\n📅 {current_date.date()} - Retraining model...")
        
        # Prepare training data (all data BEFORE current_date)
        train_data = {}
        for ticker in valid_tickers[:100]:  # Use top 100 for speed
            if ticker in all_data:
                df = all_data[ticker]
                train_df = df[df.index < pd.Timestamp(current_date)]
                if len(train_df) > 200:
                    train_data[ticker] = train_df
        
        if len(train_data) > 20:
            try:
                # Train model (silent mode for simulation)
                model = train_model(train_data, n_splits=2, save_model=False)
                selected_features = FEATURE_COLUMNS  # Use all features
                last_train_date = current_date
            except Exception as e:
                print(f"   ⚠️ Training failed: {e}")
    
    if model is None:
        current_date += timedelta(days=1)
        continue
    
    # Generate predictions for current_date
    predictions = {}
    for ticker in valid_tickers[:100]:
        if ticker not in all_data:
            continue
        
        df = all_data[ticker]
        # Data up to and including current_date for feature generation
        pred_df = df[df.index <= pd.Timestamp(current_date)]
        if len(pred_df) < 200:
            continue
        
        try:
            features_df = generate_features(pred_df, include_target=False)
            if len(features_df) == 0:
                continue
            
            last_row = features_df.iloc[[-1]][selected_features]
            pred = model.predict(last_row)[0]
            predictions[ticker] = pred
        except:
            pass
    
    # Generate signals
    signals = {}
    for ticker, pred in predictions.items():
        if pred > BUY_THRESHOLD:
            signals[ticker] = 'BUY'
        elif pred < SELL_THRESHOLD:
            signals[ticker] = 'SELL'
        else:
            signals[ticker] = 'HOLD'
    
    # Get current prices
    prices = get_latest_prices(all_data, current_date)
    
    # Execute trades
    trades_today = []
    
    # First, process SELLs
    for ticker, pos in list(portfolio['positions'].items()):
        if signals.get(ticker) == 'SELL' and ticker in prices:
            sell_amount = pos['shares'] * prices[ticker]
            portfolio['cash'] += sell_amount
            pnl = (prices[ticker] - pos['entry_price']) * pos['shares']
            trades_today.append({
                'date': current_date,
                'ticker': ticker,
                'action': 'SELL',
                'shares': pos['shares'],
                'price': prices[ticker],
                'pnl': pnl
            })
            del portfolio['positions'][ticker]
    
    # Then, process BUYs (limit to available cash)
    buy_signals = [t for t, s in signals.items() if s == 'BUY' and t not in portfolio['positions']]
    portfolio_value = get_portfolio_value(portfolio, prices)
    
    for ticker in buy_signals[:5]:  # Max 5 new positions per day
        if ticker not in prices:
            continue
        
        position_size = portfolio_value * MAX_POSITION_PCT
        shares = int(position_size / prices[ticker])
        cost = shares * prices[ticker]
        
        if shares > 0 and cost <= portfolio['cash'] * 0.98:  # Keep 2% cash buffer
            portfolio['cash'] -= cost
            portfolio['positions'][ticker] = {
                'shares': shares,
                'entry_price': prices[ticker]
            }
            trades_today.append({
                'date': current_date,
                'ticker': ticker,
                'action': 'BUY',
                'shares': shares,
                'price': prices[ticker],
                'pnl': 0
            })
    
    # Record daily value
    portfolio_value = get_portfolio_value(portfolio, prices)
    daily_values.append({
        'date': current_date,
        'portfolio_value': portfolio_value,
        'cash': portfolio['cash'],
        'positions': len(portfolio['positions']),
        'trades': len(trades_today)
    })
    
    # Log trades
    simulation_log.extend(trades_today)
    
    # Progress update every 20 days
    if day_count % 20 == 0:
        pct_return = (portfolio_value / INITIAL_CASH - 1) * 100
        print(f"   {current_date.date()}: ${portfolio_value:,.2f} ({pct_return:+.1f}%) | {len(portfolio['positions'])} positions")
    
    current_date += timedelta(days=1)

# Final summary
print("\n" + "=" * 60)
print("SIMULATION COMPLETE")
print("=" * 60)

final_value = daily_values[-1]['portfolio_value'] if daily_values else INITIAL_CASH
total_return = (final_value / INITIAL_CASH - 1) * 100

# Calculate metrics
df_daily = pd.DataFrame(daily_values)
df_daily['returns'] = df_daily['portfolio_value'].pct_change()
sharpe = df_daily['returns'].mean() / df_daily['returns'].std() * np.sqrt(252) if len(df_daily) > 1 else 0
max_dd = (df_daily['portfolio_value'] / df_daily['portfolio_value'].cummax() - 1).min() * 100

print(f"\n📈 Final Portfolio Value: ${final_value:,.2f}")
print(f"   Total Return: {total_return:+.2f}%")
print(f"   Sharpe Ratio: {sharpe:.2f}")
print(f"   Max Drawdown: {max_dd:.2f}%")
print(f"   Total Trades: {len(simulation_log)}")

# Save results
os.makedirs('results', exist_ok=True)
df_daily.to_csv('results/simulation_2025_daily.csv', index=False)
pd.DataFrame(simulation_log).to_csv('results/simulation_2025_trades.csv', index=False)

print(f"\n💾 Results saved to:")
print(f"   results/simulation_2025_daily.csv")
print(f"   results/simulation_2025_trades.csv")
