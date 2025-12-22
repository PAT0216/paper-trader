"""
LSTM Production Backtest: Oct 1 - Dec 19, 2025
- Quarterly retraining (model trained on data up to Sep 30, 2025)
- Daily rebalancing
- Full universe (~500 tickers)
- 5 bps slippage + transaction costs
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
START_DATE = '2025-10-01'
END_DATE = '2025-12-19'
TRAIN_END_DATE = '2025-09-30'  # Quarterly: train up to Q3 end
INITIAL_CAPITAL = 10000
TOP_N = 10
REBALANCE_FREQ = 'daily'  # Daily rebalancing
SLIPPAGE_BPS = 5
SEQUENCE_LENGTH = 60
WINDOW_STEP = 60

# Training config
EPOCHS = 30
BATCH_SIZE = 64

print(f"\n{'='*70}")
print(f"LSTM Production Backtest: {START_DATE} to {END_DATE}")
print(f"Model trained on data up to: {TRAIN_END_DATE} (quarterly)")
print(f"Rebalancing: {REBALANCE_FREQ.upper()}")
print(f"{'='*70}")

# ==================== Load Data ====================
cache = get_cache()
tickers = cache.get_cached_tickers()
print(f"\nLoading data for {len(tickers)} tickers...")

data_dict = {}
for ticker in tickers:
    df = cache.get_price_data(ticker)
    if df is not None and len(df) >= 200:
        df.index = pd.to_datetime(df.index)
        data_dict[ticker] = df

print(f"Loaded: {len(data_dict)} tickers")

# ==================== Train Model on Full Universe ====================
print(f"\nTraining LSTM on full universe up to {TRAIN_END_DATE}...")

X_all, y_all = [], []
for ticker, df in data_dict.items():
    try:
        train_df = df[df.index <= TRAIN_END_DATE]
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

X_all = np.concatenate(X_all)
y_all = np.concatenate(y_all)
print(f"Training samples: {len(X_all)}")

# Time-based split
split_idx = int(len(X_all) * 0.8)
X_train, X_val = X_all[:split_idx], X_all[split_idx:]
y_train, y_val = y_all[:split_idx], y_all[split_idx:]

if is_tensorflow_available():
    print("Training with TensorFlow + Metal GPU...")
    model = build_lstm_model(SEQUENCE_LENGTH, len(LSTM_FEATURES))
    pos_weight = len(y_train) / (2 * y_train.sum() + 1)
    neg_weight = len(y_train) / (2 * (len(y_train) - y_train.sum()) + 1)
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(15),
        class_weight={0: neg_weight, 1: pos_weight},
        verbose=1
    )
    model_info = ('lstm', model)
else:
    print("Training with XGBoost fallback...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    model = build_xgb_threshold_model()
    model.fit(X_train_flat, y_train, eval_set=[(X_val_flat, y_val)], verbose=False)
    model_info = ('xgb', model)

print(f"Model type: {model_info[0]}")

# ==================== Prediction Function ====================
def predict(model_info, df):
    model_type, model = model_info
    try:
        features_df = generate_lstm_features(df)
        if len(features_df) < SEQUENCE_LENGTH:
            return 0.5
        X = features_df[LSTM_FEATURES].iloc[-SEQUENCE_LENGTH:].values
        if model_type == 'lstm':
            X = np.expand_dims(X, axis=0)
            return float(model(X, training=False).numpy()[0, 0])
        else:
            return float(model.predict_proba(X.reshape(1, -1))[0, 1])
    except:
        return 0.5

# ==================== Backtest ====================
print(f"\nRunning daily-rebalancing backtest...")

# Filter to test period
test_data = {}
for ticker, df in data_dict.items():
    test_df = df[(df.index >= START_DATE) & (df.index <= END_DATE)]
    if len(test_df) >= 5:
        test_data[ticker] = df  # Keep full history for predictions

# Get trading days
sample_df = list(test_data.values())[0]
trading_days = sorted(sample_df[(sample_df.index >= START_DATE) & (sample_df.index <= END_DATE)].index)
print(f"Trading days: {len(trading_days)}")

cost_config = CostConfig(slippage_bps=SLIPPAGE_BPS)
cost_model = TransactionCostModel(config=cost_config)

cash = INITIAL_CAPITAL
holdings = {}
portfolio_values = []
all_trades = []

for i, date in enumerate(trading_days):
    date_str = date.strftime('%Y-%m-%d')
    
    # Get prices
    prices = {}
    for ticker, df in test_data.items():
        if date in df.index:
            prices[ticker] = df.loc[date, 'Close']
    
    # Portfolio value
    holdings_value = sum(prices.get(t, 0) * s for t, s in holdings.items())
    total_value = cash + holdings_value
    portfolio_values.append((date_str, total_value))
    
    # Daily rebalancing
    if i < len(trading_days) - 1:
        # Get predictions for all tickers
        scores = {}
        for ticker, df in test_data.items():
            pit_df = df[df.index <= date]
            if len(pit_df) >= SEQUENCE_LENGTH and ticker in prices:
                scores[ticker] = predict(model_info, pit_df)
        
        if scores:
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            target_tickers = [t for t, _ in sorted_scores[:TOP_N]]
            
            # Sell non-targets
            for ticker in list(holdings.keys()):
                if ticker not in target_tickers and ticker in prices:
                    shares = holdings[ticker]
                    sell_price, _ = cost_model.calculate_execution_price('SELL', prices[ticker], shares)
                    cash += shares * sell_price
                    all_trades.append({'date': date_str, 'ticker': ticker, 'action': 'SELL', 'shares': shares})
                    del holdings[ticker]
            
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
                                all_trades.append({'date': date_str, 'ticker': ticker, 'action': 'BUY', 'shares': shares})
    
    if i % 10 == 0:
        print(f"  {date_str}: ${total_value:,.2f}, Holdings: {len(holdings)}")

# Final value
final_value = portfolio_values[-1][1]

# ==================== Results ====================
print(f"\n{'='*70}")
print("LSTM PRODUCTION BACKTEST RESULTS")
print(f"{'='*70}")
print(f"Period: {START_DATE} to {END_DATE}")
print(f"Model trained on: Full universe up to {TRAIN_END_DATE}")
print(f"Rebalancing: DAILY")
print(f"Initial: ${INITIAL_CAPITAL:,.2f}")
print(f"Final:   ${final_value:,.2f}")
print(f"Return:  {(final_value/INITIAL_CAPITAL - 1)*100:+.2f}%")
print(f"Total Trades: {len(all_trades)}")
print(f"Slippage: {SLIPPAGE_BPS} bps")

# Save results
results = {
    'strategy': 'lstm_production',
    'period': f'{START_DATE} to {END_DATE}',
    'train_end': TRAIN_END_DATE,
    'rebalancing': 'daily',
    'initial_capital': INITIAL_CAPITAL,
    'final_value': round(final_value, 2),
    'return_pct': round((final_value/INITIAL_CAPITAL - 1)*100, 2),
    'total_trades': len(all_trades),
    'slippage_bps': SLIPPAGE_BPS,
    'portfolio_history': portfolio_values
}

with open('results/lstm_production_backtest.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to results/lstm_production_backtest.json")
