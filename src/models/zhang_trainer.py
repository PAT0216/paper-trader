"""
Zhang 2019 Strategy Trainer

Implements the complete training pipeline from:
"Investment Strategy Based on Machine Learning Models" by Jiayue Zhang (2019)

Pipeline:
1. Generate 20 PCA-optimized lagged return features
2. Create classification labels (beat cross-sectional median)
3. Train XGBoost classifier
4. Apply Vertical Ensemble (3-day weighted probabilities)
5. Long top 10, short bottom 10

XGBoost Parameters from paper:
- n_estimators (MXGB): 100
- max_depth (JXGB): 3
- learning_rate (λXGB): 0.1
- subset of features per split (mXGB): 16
"""

import xgboost as xgb
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit

from src.features.zhang_features import (
    generate_zhang_features,
    ZHANG_FEATURE_COLUMNS,
    ZHANG_PERIODS,
    apply_vertical_ensemble
)


class ZhangStrategy:
    """
    Complete Zhang 2019 investment strategy implementation.
    """
    
    def __init__(self, n_top: int = 10, use_mean_reversion: bool = False):
        """
        Initialize the Zhang strategy.
        
        Args:
            n_top: Number of stocks to long (and short)
            use_mean_reversion: If True, flip predictions (for S&P 500 which shows
                               mean reversion rather than momentum)
        """
        self.n_top = n_top
        self.use_mean_reversion = use_mean_reversion
        self.model = None
        self.probability_history = {}  # {ticker: [P[t-2], P[t-1], P[t]]}
        
    def train(self, data_dict: Dict[str, pd.DataFrame], 
              min_date: str = '2014-01-01',
              n_splits: int = 5) -> 'ZhangStrategy':
        """
        Train XGBoost classifier on historical data.
        
        Args:
            data_dict: {ticker: OHLCV DataFrame}
            min_date: Minimum date for training data
            n_splits: Number of cross-validation folds
            
        Returns:
            self (trained model)
        """
        print("=" * 60)
        print("ZHANG 2019 STRATEGY TRAINING")
        print("=" * 60)
        
        print(f"\n📊 Preparing training data...")
        print(f"   Features: {len(ZHANG_FEATURE_COLUMNS)} lagged returns")
        print(f"   Periods: {ZHANG_PERIODS}")
        
        # Prepare training data
        all_records = []
        
        for ticker, df in data_dict.items():
            # Generate Zhang features
            processed = generate_zhang_features(df)
            
            # Add next-day return as target
            price_col = 'Adj Close' if 'Adj Close' in processed.columns else 'Close'
            processed['NextReturn'] = processed[price_col].pct_change().shift(-1)
            
            # Drop rows with NaN
            processed = processed.dropna(subset=ZHANG_FEATURE_COLUMNS + ['NextReturn'])
            
            # Filter by date
            processed = processed[processed.index >= min_date]
            
            # Add ticker
            processed['Ticker'] = ticker
            
            if len(processed) > 0:
                all_records.append(processed)
        
        if not all_records:
            print("❌ No valid training data")
            return self
        
        full_df = pd.concat(all_records)
        full_df = full_df.sort_index()
        
        print(f"   Total samples: {len(full_df):,}")
        print(f"   Date range: {full_df.index.min()} to {full_df.index.max()}")
        
        # Create classification labels by date
        print(f"\n🏷️  Creating classification labels...")
        labels = []
        for date in full_df.index.unique():
            day_data = full_df.loc[[date]].copy()
            median = day_data['NextReturn'].median()
            day_data['Label'] = (day_data['NextReturn'] > median).astype(int)
            labels.append(day_data[['Label']])
        
        labels_df = pd.concat(labels)
        full_df['Label'] = labels_df['Label']
        
        # Prepare X and y
        X = full_df[ZHANG_FEATURE_COLUMNS].values
        y = full_df['Label'].values
        
        # Clean data
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"   Valid samples: {len(X):,}")
        print(f"   Class balance: {y.mean():.2%} positive")
        
        # Train XGBoost with Zhang's exact parameters
        print(f"\n🚀 Training XGBoost classifier...")
        print(f"   n_estimators: 100")
        print(f"   max_depth: 3")
        print(f"   learning_rate: 0.1")
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                colsample_bytree=16/len(ZHANG_FEATURE_COLUMNS),  # mXGB = 16
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42,
                verbosity=0
            )
            
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            cv_scores.append(score)
            print(f"   Fold {fold+1}: Accuracy = {score:.4f}")
        
        print(f"   Mean CV Accuracy: {np.mean(cv_scores):.4f}")
        
        # Train final model on all data
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            colsample_bytree=16/len(ZHANG_FEATURE_COLUMNS),
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            verbosity=0
        )
        self.model.fit(X, y)
        
        print(f"\n✅ Training complete!")
        
        return self
    
    def predict_probability(self, ticker: str, df: pd.DataFrame) -> float:
        """
        Predict probability of a stock beating the median.
        
        Uses Vertical Ensemble: weighted average of last 3 days.
        
        Args:
            ticker: Stock ticker
            df: OHLCV DataFrame
            
        Returns:
            Ensemble probability (0.0 to 1.0)
        """
        if self.model is None:
            return 0.5
        
        # Generate features
        processed = generate_zhang_features(df)
        processed = processed.dropna(subset=ZHANG_FEATURE_COLUMNS)
        
        if len(processed) == 0:
            return 0.5
        
        # Get latest features
        X = processed[ZHANG_FEATURE_COLUMNS].iloc[-1:].values
        
        # Predict probability
        prob = self.model.predict_proba(X)[0, 1]
        
        # Update probability history for vertical ensemble
        if ticker not in self.probability_history:
            self.probability_history[ticker] = []
        
        self.probability_history[ticker].append(prob)
        
        # Keep only last 3
        if len(self.probability_history[ticker]) > 3:
            self.probability_history[ticker] = self.probability_history[ticker][-3:]
        
        # Apply vertical ensemble
        ensemble_prob = apply_vertical_ensemble(self.probability_history[ticker])
        
        return ensemble_prob
    
    def generate_signals(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """
        Generate trading signals for all stocks.
        
        Long top n, short bottom n by ensemble probability.
        If use_mean_reversion=True, flip the logic (long LOW prob, short HIGH prob).
        
        Args:
            data_dict: {ticker: OHLCV DataFrame}
            
        Returns:
            {ticker: 'BUY'|'SELL'|'HOLD'}
        """
        # Get probabilities for all stocks
        probs = {}
        for ticker, df in data_dict.items():
            prob = self.predict_probability(ticker, df)
            probs[ticker] = prob
        
        # Sort by probability
        sorted_tickers = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        if self.use_mean_reversion:
            # Flip: long LOW probability (recent losers), short HIGH probability (recent winners)
            buy_set = set([t for t, _ in sorted_tickers[-self.n_top:]])  # Bottom n
            sell_set = set([t for t, _ in sorted_tickers[:self.n_top]])   # Top n
        else:
            # Original: long HIGH probability, short LOW probability
            buy_set = set([t for t, _ in sorted_tickers[:self.n_top]])   # Top n
            sell_set = set([t for t, _ in sorted_tickers[-self.n_top:]]) # Bottom n
        
        signals = {}
        for ticker in probs:
            if ticker in buy_set:
                signals[ticker] = 'BUY'
            elif ticker in sell_set:
                signals[ticker] = 'SELL'
            else:
                signals[ticker] = 'HOLD'
        
        return signals
    
    def save(self, path: str = 'models/zhang_model.json'):
        """Save the trained model."""
        if self.model is not None:
            self.model.save_model(path)
            print(f"📁 Model saved to {path}")
    
    def load(self, path: str = 'models/zhang_model.json'):
        """Load a trained model."""
        if os.path.exists(path):
            self.model = xgb.XGBClassifier()
            self.model.load_model(path)
            print(f"📁 Model loaded from {path}")
        return self


def train_zhang_model(data_dict: Dict[str, pd.DataFrame],
                      min_date: str = '2014-01-01',
                      save_path: str = 'models/zhang_model.json') -> ZhangStrategy:
    """
    Convenience function to train Zhang 2019 model.
    
    Args:
        data_dict: {ticker: OHLCV DataFrame}
        min_date: Minimum date for training
        save_path: Where to save the model
        
    Returns:
        Trained ZhangStrategy instance
    """
    strategy = ZhangStrategy(n_top=10)
    strategy.train(data_dict, min_date=min_date)
    strategy.save(save_path)
    return strategy


if __name__ == "__main__":
    print("Zhang 2019 Strategy")
    print("=" * 40)
    print(f"Features: {len(ZHANG_FEATURE_COLUMNS)}")
    print(f"Periods: {ZHANG_PERIODS}")
    print(f"Top N: 10 (long 10, short 10)")
    print(f"Vertical Ensemble: (0.1, 0.3, 0.6)")
