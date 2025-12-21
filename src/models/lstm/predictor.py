"""
LSTM Predictor with Monte Carlo Dropout Uncertainty

Provides probability estimates with confidence intervals
by running multiple forward passes with dropout active.
"""

import numpy as np
import os
from typing import Tuple, Optional
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class LSTMPredictor:
    """
    LSTM predictor with Monte Carlo Dropout for uncertainty estimation.
    
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
        Initialize LSTM predictor.
        
        Args:
            model_path: Path to trained model weights
            n_mc_samples: Number of Monte Carlo samples for uncertainty
            sequence_length: Input sequence length
        """
        self.model_path = model_path
        self.n_mc_samples = n_mc_samples
        self.sequence_length = sequence_length
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load trained LSTM model."""
        if not os.path.exists(self.model_path):
            print(f"LSTM model not found at {self.model_path}")
            return
        
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Loaded LSTM model from {self.model_path}")
        except Exception as e:
            print(f"Could not load LSTM model: {e}")
            self.model = None
    
    def predict(self, df: pd.DataFrame) -> float:
        """
        Get single probability prediction.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Probability of return exceeding threshold
        """
        prob, _ = self.predict_with_uncertainty(df)
        return prob
    
    def predict_with_uncertainty(self, df: pd.DataFrame) -> Tuple[float, float]:
        """
        Predict with Monte Carlo Dropout for uncertainty estimation.
        
        Runs multiple forward passes with dropout active to estimate
        both the mean prediction and uncertainty.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            (mean_probability, uncertainty_std)
        """
        if self.model is None:
            return 0.5, 1.0  # Neutral prediction with max uncertainty
        
        from src.models.lstm.features import generate_lstm_features, LSTM_FEATURES
        
        try:
            # Generate features
            features_df = generate_lstm_features(df)
            
            if len(features_df) < self.sequence_length:
                return 0.5, 1.0
            
            # Get last sequence
            X = features_df[LSTM_FEATURES].iloc[-self.sequence_length:].values
            X = np.expand_dims(X, axis=0)
            
            # Monte Carlo sampling (dropout active via training=True)
            predictions = []
            for _ in range(self.n_mc_samples):
                pred = self.model(X, training=True)  # Dropout stays active
                predictions.append(float(pred.numpy()[0, 0]))
            
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            
            return mean_pred, std_pred
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.5, 1.0
    
    def predict_batch(self, data_dict: dict) -> dict:
        """
        Predict for multiple tickers.
        
        Args:
            data_dict: Dictionary of {ticker: DataFrame}
        
        Returns:
            Dictionary of {ticker: (probability, uncertainty)}
        """
        results = {}
        for ticker, df in data_dict.items():
            try:
                prob, uncertainty = self.predict_with_uncertainty(df)
                results[ticker] = (prob, uncertainty)
            except Exception:
                pass
        return results


if __name__ == "__main__":
    print("LSTM Predictor Module")
    print("Usage: predictor = LSTMPredictor(); prob, unc = predictor.predict_with_uncertainty(df)")
