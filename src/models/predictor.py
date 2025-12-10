"""
ML Predictor for Paper Trader

Uses the trained XGBRegressor to predict expected next-day returns.
Outputs:
  - Expected return (float): Predicted percentage return
  - Signal: BUY/SELL/HOLD based on return magnitude thresholds
"""

import joblib
import pandas as pd
import numpy as np
import os
from src.features.indicators import generate_features, FEATURE_COLUMNS

MODEL_PATH = "models"
MODEL_FILE = os.path.join(MODEL_PATH, "xgb_model.joblib")


class Predictor:
    """
    Predicts expected returns using trained XGBoost regression model.
    
    The model was trained to predict next-day percentage returns.
    Higher predicted returns → Stronger BUY signal
    Lower (negative) predicted returns → SELL signal
    """
    
    def __init__(self):
        self.model = None
        self.is_regression = True  # Phase 3: regression model
        self.selected_features = FEATURE_COLUMNS  # Default: use all features
        
        if os.path.exists(MODEL_FILE):
            model_data = joblib.load(MODEL_FILE)
            
            # Handle new model format with metadata
            if isinstance(model_data, dict) and 'model' in model_data:
                self.model = model_data['model']
                self.selected_features = model_data.get('selected_features', FEATURE_COLUMNS)
                print(f"✅ Loaded XGBoost regression model.")
                print(f"   Selected features: {len(self.selected_features)}/{len(FEATURE_COLUMNS)}")
            else:
                # Legacy format: model only
                self.model = model_data
                print("✅ Loaded XGBoost regression model.")
                print("   (Legacy model format - using all features)")
            
            # Detect model type
            if hasattr(self.model, 'predict_proba'):
                self.is_regression = False
                print("   (Legacy classification model detected)")
        else:
            print("⚠️ Warning: No model found. Run training first.")
    
    def predict(self, df):
        """
        Predicts expected return for the latest data point.
        
        Args:
            df: Raw OHLCV DataFrame (at least 200 rows for SMA_200)
            
        Returns:
            float: Expected next-day return (as decimal, e.g., 0.02 = 2%)
                   For legacy classifier: returns probability of up move
        """
        if self.model is None:
            return 0.0  # Neutral
        
        # Generate features (no target for inference)
        processed_df = generate_features(df, include_target=False)
        
        if processed_df.empty:
            return 0.0
        
        # Extract features for the latest date (use selected features only)
        try:
            last_row = processed_df.iloc[[-1]][self.selected_features]
        except KeyError as e:
            print(f"⚠️ Missing feature columns: {e}")
            return 0.0
        
        if self.is_regression:
            # Regression model: predict expected return
            prediction = self.model.predict(last_row)[0]
            return float(prediction)
        else:
            # Legacy classifier: return probability of up move
            proba = self.model.predict_proba(last_row)[0][1]
            return float(proba)
    
    def predict_with_signal(self, df, buy_threshold=0.005, sell_threshold=-0.005):
        """
        Predicts expected return and generates trading signal.
        
        Thresholds:
            - BUY: Expected return > buy_threshold (default: 0.5%)
            - SELL: Expected return < sell_threshold (default: -0.5%)
            - HOLD: Otherwise
            
        Args:
            df: Raw OHLCV DataFrame
            buy_threshold: Minimum expected return to trigger BUY
            sell_threshold: Maximum expected return to trigger SELL
            
        Returns:
            Tuple of (signal, expected_return, confidence)
        """
        expected_return = self.predict(df)
        
        # For legacy classifier, convert probability to return-like scale
        if not self.is_regression:
            # Map 0.5-1.0 → 0-0.02 and 0-0.5 → -0.02-0
            expected_return = (expected_return - 0.5) * 0.04
        
        # Generate signal with confidence
        if expected_return > buy_threshold:
            signal = 'BUY'
            confidence = min(expected_return / 0.02, 1.0)  # Scale to 0-1
        elif expected_return < sell_threshold:
            signal = 'SELL'
            confidence = min(abs(expected_return) / 0.02, 1.0)
        else:
            signal = 'HOLD'
            confidence = 1.0 - abs(expected_return) / buy_threshold
        
        return signal, expected_return, confidence
    
    def predict_batch(self, data_dict):
        """
        Predict expected returns for multiple tickers.
        
        Args:
            data_dict: Dictionary of {ticker: OHLCV DataFrame}
            
        Returns:
            Dictionary of {ticker: expected_return}
        """
        predictions = {}
        
        for ticker, df in data_dict.items():
            try:
                predictions[ticker] = self.predict(df)
            except Exception as e:
                print(f"⚠️ Prediction failed for {ticker}: {e}")
                predictions[ticker] = 0.0
        
        return predictions
