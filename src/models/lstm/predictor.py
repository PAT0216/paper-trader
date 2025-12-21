"""
LSTM Predictor with Monte Carlo Dropout Uncertainty

Provides probability estimates with confidence intervals.
Supports both LSTM (if TF available) and XGBoost fallback.
"""

import numpy as np
import os
from typing import Tuple, Optional
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class LSTMPredictor:
    """
    Threshold classification predictor with uncertainty estimation.
    
    Supports:
    - LSTM with MC Dropout (if TensorFlow available)
    - XGBoost fallback with bootstrap uncertainty
    
    Usage:
        predictor = LSTMPredictor()
        prob, uncertainty = predictor.predict_with_uncertainty(df)
    """
    
    def __init__(
        self, 
        model_path: str = 'models/lstm_model.h5',
        n_mc_samples: int = 50,
        sequence_length: int = 60
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model weights
            n_mc_samples: Number of Monte Carlo samples for uncertainty
            sequence_length: Input sequence length
        """
        self.model_path = model_path
        self.n_mc_samples = n_mc_samples
        self.sequence_length = sequence_length
        self.model = None
        self.model_type = None
        
        self._load_model()
    
    def _load_model(self):
        """Load trained model (LSTM or XGBoost)."""
        # Try XGBoost first (more common fallback)
        xgb_path = self.model_path.replace('.h5', '_xgb.joblib')
        if os.path.exists(xgb_path):
            try:
                import joblib
                self.model = joblib.load(xgb_path)
                self.model_type = 'xgboost'
                print(f"Loaded XGBoost model from {xgb_path}")
                return
            except Exception as e:
                print(f"Could not load XGBoost model: {e}")
        
        # Try LSTM
        if os.path.exists(self.model_path):
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(self.model_path)
                self.model_type = 'lstm'
                print(f"Loaded LSTM model from {self.model_path}")
                return
            except ImportError:
                print("TensorFlow not available")
            except Exception as e:
                print(f"Could not load LSTM model: {e}")
        
        print(f"No model found at {self.model_path} or {xgb_path}")
    
    def predict(self, df: pd.DataFrame) -> float:
        """Get single probability prediction."""
        prob, _ = self.predict_with_uncertainty(df)
        return prob
    
    def predict_with_uncertainty(self, df: pd.DataFrame) -> Tuple[float, float]:
        """
        Predict with uncertainty estimation.
        
        For LSTM: Monte Carlo Dropout
        For XGBoost: Bootstrap-based uncertainty approximation
        
        Returns:
            (mean_probability, uncertainty_std)
        """
        if self.model is None:
            return 0.5, 1.0
        
        from src.models.lstm.features import generate_lstm_features, LSTM_FEATURES
        
        try:
            features_df = generate_lstm_features(df)
            
            if len(features_df) < self.sequence_length:
                return 0.5, 1.0
            
            X = features_df[LSTM_FEATURES].iloc[-self.sequence_length:].values
            
            if self.model_type == 'lstm':
                # Monte Carlo Dropout
                X = np.expand_dims(X, axis=0)
                predictions = []
                for _ in range(self.n_mc_samples):
                    pred = self.model(X, training=True)
                    predictions.append(float(pred.numpy()[0, 0]))
                
                mean_pred = np.mean(predictions)
                std_pred = np.std(predictions)
                
            else:  # XGBoost
                X_flat = X.reshape(1, -1)
                proba = self.model.predict_proba(X_flat)[0]
                mean_pred = float(proba[1])
                # Approximate uncertainty from probability (closer to 0.5 = more uncertain)
                std_pred = 0.5 - abs(mean_pred - 0.5)
            
            return mean_pred, std_pred
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.5, 1.0
    
    def predict_batch(self, data_dict: dict) -> dict:
        """Predict for multiple tickers."""
        results = {}
        for ticker, df in data_dict.items():
            try:
                prob, uncertainty = self.predict_with_uncertainty(df)
                results[ticker] = (prob, uncertainty)
            except Exception:
                pass
        return results


if __name__ == "__main__":
    print("LSTM/XGBoost Predictor Module")
    print("Usage: predictor = LSTMPredictor(); prob, unc = predictor.predict_with_uncertainty(df)")
