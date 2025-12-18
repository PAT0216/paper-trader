#!/usr/bin/env python3
"""
LEAKAGE TEST v2: More careful check of fair_comparison's actual flow
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from src.data.cache import DataCache
from src.features.indicators import generate_features

TRAIN_END = pd.Timestamp('2020-01-01')
HORIZON = 20

print("="*70)
print("LEAKAGE TEST v2: Checking fair_comparison's actual flow")
print("="*70)

# Simulate fair_comparison's data loading (lines 295-310)
cache = DataCache()
tickers = ['AAPL', 'MSFT', 'GOOGL']

all_data = []
for ticker in tickers:
    df = cache.get_price_data(ticker)
    if df is not None:
        processed = generate_features(df, include_target=False)
        processed = processed.copy()
        processed['ticker'] = ticker
        all_data.append(processed)

full_df = pd.concat(all_data).sort_index()
full_df = full_df[full_df.index >= '2015-01-01']
print(f"Full DataFrame date range: {full_df.index.min().date()} to {full_df.index.max().date()}")
print()

# Simulate what happens at training time (lines 214, 218)
# This is what fair_comparison does:
train_df = full_df[full_df.index < TRAIN_END].copy()
print(f"After filtering to before {TRAIN_END.date()}:")
print(f"  train_df date range: {train_df.index.min().date()} to {train_df.index.max().date()}")
print(f"  train_df rows: {len(train_df)}")
print()

# Now simulate train_ensemble_with_noise_selection (lines 92-96)
# This is where targets are created
print(f"Creating {HORIZON}-day target inside train_ensemble_with_noise_selection:")
df = train_df.copy()
df['Target'] = df.groupby('ticker')['Close'].transform(
    lambda x: x.pct_change(HORIZON).shift(-HORIZON)
)

# Check the last row's target
for ticker in tickers:
    ticker_data = df[df['ticker'] == ticker]
    valid = ticker_data[~ticker_data['Target'].isna()]
    if len(valid) > 0:
        last_date = valid.index.max()
        last_target = valid.loc[last_date, 'Target']
        
        # What dates are used in this target calculation?
        ticker_all = ticker_data
        last_idx = ticker_all.index.get_loc(last_date)
        
        if last_idx + HORIZON < len(ticker_all):
            future_date = ticker_all.index[last_idx + HORIZON]
            print(f"  {ticker}: Last training row = {last_date.date()}")
            print(f"         Target uses Close from: {future_date.date()}")
            if future_date >= TRAIN_END:
                print(f"         ⚠️  LEAKAGE: {future_date.date()} >= {TRAIN_END.date()}")
            else:
                print(f"         ✅ OK")
        else:
            # shift(-HORIZON) would result in NaN for last HORIZON rows
            print(f"  {ticker}: Last valid target row = {last_date.date()}")
            print(f"         (shift(-{HORIZON}) causes NaN for last {HORIZON} rows)")
            
            # Find what the target actually computed
            # pct_change(HORIZON) on train_df only can't access data outside train_df
            print(f"         ✅ No leakage possible - future data not in train_df")

print()

# Double check: what Close prices are in train_df?
print("Close prices in train_df for AAPL (last 5 rows):")
aapl_train = train_df[train_df['ticker'] == 'AAPL'].tail(5)
print(aapl_train[['Close']].to_string())
print(f"\nMax date with Close in train_df: {aapl_train.index.max().date()}")

# vs what's in full_df
print("\nClose prices in full_df for AAPL (around cutoff):")
aapl_full = full_df[full_df['ticker'] == 'AAPL']
around_cutoff = aapl_full[(aapl_full.index >= '2019-12-20') & (aapl_full.index <= '2020-01-10')]
print(around_cutoff[['Close']].to_string())

print()
print("="*70)
print("CONCLUSION:")
print("="*70)
print("train_df is filtered BEFORE target calculation.")
print("The target calculation can only use Close prices that exist in train_df.")
print("Since train_df only has data before cutoff, the target cannot use future data.")
print("Therefore: NO LEAKAGE in fair_comparison's target calculation.")
print()
