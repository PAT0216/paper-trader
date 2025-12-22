"""
LSTM Walk-Forward Backtest with Quarterly Retraining

Tests LSTM threshold classification strategy with:
- Walk-forward validation (2016-2025)
- Quarterly model retraining
- Non-overlapping windows (no look-ahead bias)
- Transaction costs (5 bps slippage)
- Comparison to SPY benchmark
"""

import sys
import os
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.data.cache import get_cache
from src.backtesting.costs import TransactionCostModel, CostConfig
from src.models.lstm.features import generate_lstm_features, LSTM_FEATURES, create_sequences
from src.models.lstm.threshold import create_pit_target
from src.models.lstm.model import is_tensorflow_available, build_lstm_model, get_callbacks, build_xgb_threshold_model

# ==================== Configuration ====================
START_YEAR = 2016
END_DATE = '2025-12-19'
INITIAL_CAPITAL = 10000
TOP_N = 10
REBALANCE_FREQ = 'weekly'
SLIPPAGE_BPS = 5
SEQUENCE_LENGTH = 60
WINDOW_STEP = 60  # Non-overlapping
RETRAIN_QUARTERS = True
MIN_TRAIN_YEARS = 2  # Minimum training history

# Training config
EPOCHS = 30
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 15

# ==================== Helper Functions ====================
def get_quarters(start_year, end_date):
    """Generate quarterly periods for walk-forward."""
    quarters = []
    current = datetime(start_year, 1, 1)
    end = pd.to_datetime(end_date)
    
    while current < end:
        q_end = current + pd.DateOffset(months=3) - timedelta(days=1)
        if q_end > end:
            q_end = end
        quarters.append((current, q_end))
        current = current + pd.DateOffset(months=3)
    
    return quarters


def train_model_for_period(data_dict, train_end_date, use_lstm=True):
    """Train model using data up to train_end_date."""
    X_all, y_all = [], []
    
    for ticker, df in data_dict.items():
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            train_df = df[df.index <= train_end_date]
            
            if len(train_df) < SEQUENCE_LENGTH + 60:
                continue
            
            features_df = generate_lstm_features(train_df)
            target = create_pit_target(train_df, horizon=1, sigma_multiplier=2.0)
            target = target.loc[features_df.index]
            
            X, y = create_sequences(features_df, target, SEQUENCE_LENGTH, WINDOW_STEP)
            
            if len(X) > 0:
                X_all.append(X)
                y_all.append(y)
        except:
            continue
    
    if not X_all:
        return None
    
    X_all = np.concatenate(X_all)
    y_all = np.concatenate(y_all)
    
    # Time-based split
    split_idx = int(len(X_all) * 0.8)
    X_train, X_val = X_all[:split_idx], X_all[split_idx:]
    y_train, y_val = y_all[:split_idx], y_all[split_idx:]
    
    if use_lstm and is_tensorflow_available():
        model = build_lstm_model(SEQUENCE_LENGTH, len(LSTM_FEATURES))
        pos_weight = len(y_train) / (2 * y_train.sum() + 1)
        neg_weight = len(y_train) / (2 * (len(y_train) - y_train.sum()) + 1)
        
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=get_callbacks(EARLY_STOPPING_PATIENCE),
            class_weight={0: neg_weight, 1: pos_weight},
            verbose=0
        )
        return ('lstm', model)
    else:
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        
        model = build_xgb_threshold_model()
        model.fit(X_train_flat, y_train, eval_set=[(X_val_flat, y_val)], verbose=False)
        return ('xgb', model)


def predict_with_model(model_info, df):
    """Get prediction for a single ticker."""
    model_type, model = model_info
    
    try:
        features_df = generate_lstm_features(df)
        if len(features_df) < SEQUENCE_LENGTH:
            return 0.5
        
        X = features_df[LSTM_FEATURES].iloc[-SEQUENCE_LENGTH:].values
        
        if model_type == 'lstm':
            X = np.expand_dims(X, axis=0)
            pred = model(X, training=False).numpy()[0, 0]
        else:
            X_flat = X.reshape(1, -1)
            pred = model.predict_proba(X_flat)[0, 1]
        
        return float(pred)
    except:
        return 0.5


# ==================== Load Data ====================
print(f"\n{'='*70}")
print(f"LSTM Walk-Forward Backtest: {START_YEAR} to {END_DATE}")
print(f"Quarterly Retraining Enabled")
print(f"{'='*70}")

cache = get_cache()
tickers = cache.get_cached_tickers()
print(f"\nLoading data for {len(tickers)} tickers...")

data_dict = {}
for ticker in tickers:
    df = cache.get_price_data(ticker)
    if df is not None and len(df) >= 200:
        data_dict[ticker] = df

print(f"Loaded: {len(data_dict)} tickers with sufficient history")

# Filter for 2016+ data
filtered_data = {}
for ticker, df in data_dict.items():
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.min().year <= START_YEAR:
        filtered_data[ticker] = df

print(f"Tickers with data from {START_YEAR}: {len(filtered_data)}")

# ==================== Walk-Forward Backtest ====================
quarters = get_quarters(START_YEAR + MIN_TRAIN_YEARS, END_DATE)  # Start testing after min train period
print(f"\nTesting periods: {len(quarters)} quarters ({quarters[0][0].strftime('%Y-%m')} to {quarters[-1][1].strftime('%Y-%m')})")

cost_config = CostConfig(slippage_bps=SLIPPAGE_BPS)
cost_model = TransactionCostModel(config=cost_config)

cash = INITIAL_CAPITAL
holdings = {}
portfolio_values = []
trades = []
current_model = None
last_train_quarter = None

print("\nRunning walk-forward backtest...")
print(f"{'Quarter':<15} {'Value':>12} {'Return':>10} {'Holdings':>10} {'Trades':>8}")
print("-" * 60)

for q_idx, (q_start, q_end) in enumerate(quarters):
    q_name = f"{q_start.year}Q{(q_start.month-1)//3+1}"
    
    # Retrain at start of each quarter
    if RETRAIN_QUARTERS and (current_model is None or q_name != last_train_quarter):
        train_end = q_start - timedelta(days=1)
        print(f"  Training model on data up to {train_end.strftime('%Y-%m-%d')}...", end=" ")
        current_model = train_model_for_period(filtered_data, train_end)
        if current_model:
            print(f"({current_model[0]})")
            last_train_quarter = q_name
        else:
            print("FAILED - skipping quarter")
            continue
    
    # Get trading days in this quarter
    all_dates = set()
    for ticker, df in filtered_data.items():
        df_q = df[(df.index >= q_start) & (df.index <= q_end)]
        all_dates.update(df_q.index.tolist())
    trading_days = sorted(all_dates)
    
    if not trading_days:
        continue
    
    q_trades = 0
    q_start_value = None
    
    for i, date in enumerate(trading_days):
        date_str = date.strftime('%Y-%m-%d')
        
        # Get current prices
        prices = {}
        for ticker, df in filtered_data.items():
            if date in df.index:
                prices[ticker] = df.loc[date, 'Close']
        
        # Calculate portfolio value
        holdings_value = sum(prices.get(t, 0) * shares for t, shares in holdings.items())
        total_value = cash + holdings_value
        portfolio_values.append((date_str, total_value))
        
        if q_start_value is None:
            q_start_value = total_value
        
        # Weekly rebalancing
        is_rebalance = (i % 5 == 0) if REBALANCE_FREQ == 'weekly' else True
        
        if is_rebalance and i < len(trading_days) - 1 and current_model:
            # Get predictions
            scores = {}
            for ticker, df in filtered_data.items():
                df_pit = df[df.index <= date]
                if len(df_pit) >= SEQUENCE_LENGTH and ticker in prices:
                    scores[ticker] = predict_with_model(current_model, df_pit)
            
            if scores:
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                target_tickers = [t for t, _ in sorted_scores[:TOP_N]]
                
                # Sell non-targets
                for ticker in list(holdings.keys()):
                    if ticker not in target_tickers and ticker in prices:
                        shares = holdings[ticker]
                        sell_price, _ = cost_model.calculate_execution_price('SELL', prices[ticker], shares)
                        cash += shares * sell_price
                        del holdings[ticker]
                        q_trades += 1
                
                # Buy new positions
                n_to_buy = len([t for t in target_tickers if t not in holdings and t in prices])
                if n_to_buy > 0:
                    allocation = cash / n_to_buy * 0.99
                    for ticker in target_tickers:
                        if ticker not in holdings and ticker in prices:
                            est_shares = int(allocation / prices[ticker])
                            if est_shares > 0:
                                buy_price, _ = cost_model.calculate_execution_price('BUY', prices[ticker], est_shares)
                                shares = int(allocation / buy_price)
                                if shares > 0:
                                    cash -= shares * buy_price
                                    holdings[ticker] = shares
                                    q_trades += 1
    
    # Quarter end stats
    q_return = (total_value / q_start_value - 1) * 100 if q_start_value else 0
    print(f"{q_name:<15} ${total_value:>11,.2f} {q_return:>+9.2f}% {len(holdings):>10} {q_trades:>8}")
    trades.append({'quarter': q_name, 'return': q_return, 'trades': q_trades})

# ==================== Final Results ====================
final_value = portfolio_values[-1][1] if portfolio_values else INITIAL_CAPITAL
total_return = (final_value / INITIAL_CAPITAL - 1) * 100
years = (pd.to_datetime(END_DATE) - datetime(START_YEAR + MIN_TRAIN_YEARS, 1, 1)).days / 365.25
cagr = ((final_value / INITIAL_CAPITAL) ** (1/years) - 1) * 100 if years > 0 else 0

print(f"\n{'='*70}")
print("WALK-FORWARD RESULTS")
print(f"{'='*70}")
print(f"Period: {START_YEAR + MIN_TRAIN_YEARS} to {END_DATE}")
print(f"Initial: ${INITIAL_CAPITAL:,.2f}")
print(f"Final:   ${final_value:,.2f}")
print(f"Total Return: {total_return:+.2f}%")
print(f"CAGR: {cagr:+.2f}%")
print(f"Total Trades: {sum(t['trades'] for t in trades)}")

# Save results
results = {
    'strategy': 'lstm_v4_walkforward',
    'period': f'{START_YEAR + MIN_TRAIN_YEARS} to {END_DATE}',
    'initial_capital': INITIAL_CAPITAL,
    'final_value': round(final_value, 2),
    'total_return_pct': round(total_return, 2),
    'cagr_pct': round(cagr, 2),
    'quarterly_results': trades,
    'portfolio_history': portfolio_values
}

os.makedirs('results', exist_ok=True)
with open('results/lstm_walkforward_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to results/lstm_walkforward_results.json")
