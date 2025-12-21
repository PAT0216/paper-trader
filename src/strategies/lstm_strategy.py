"""
LSTM V4 Strategy - Threshold Classification with Uncertainty

Uses trained LSTM to predict P(return > threshold) and ranks universe
by probability * confidence for cross-sectional selection.
"""

from typing import Dict, Optional
import pandas as pd

from src.strategies.base import BaseStrategy


class LSTMStrategy(BaseStrategy):
    """
    LSTM-based threshold classification strategy.
    
    Features:
    - Conv1D + LSTM architecture with Monte Carlo Dropout
    - Threshold target: return > (mean + 2*sigma)
    - Uncertainty estimation for confidence-weighted ranking
    - Non-overlapping window training (no data leakage)
    - Daily rebalancing
    
    Requires model training before use.
    """
    
    def __init__(
        self, 
        n_mc_samples: int = 50, 
        confidence_threshold: float = 0.5,
        sequence_length: int = 60
    ):
        """
        Initialize LSTM strategy.
        
        Args:
            n_mc_samples: Number of Monte Carlo dropout samples
            confidence_threshold: Minimum probability to consider for trading
            sequence_length: LSTM input sequence length
        """
        self.n_mc_samples = n_mc_samples
        self.confidence_threshold = confidence_threshold
        self.sequence_length = sequence_length
        self.predictor = None
        self._load_predictor()
    
    def _load_predictor(self):
        """Lazy load LSTM predictor."""
        try:
            from src.models.lstm.predictor import LSTMPredictor
            self.predictor = LSTMPredictor(
                n_mc_samples=self.n_mc_samples,
                sequence_length=self.sequence_length
            )
        except Exception as e:
            print(f"Could not load LSTM predictor: {e}")
            self.predictor = None
    
    def get_name(self) -> str:
        """Return strategy identifier."""
        return "lstm"
    
    def get_display_name(self) -> str:
        """Return human-readable name."""
        return "LSTM V4 Threshold"
    
    def needs_training(self) -> bool:
        """LSTM strategy requires model training."""
        return True
    
    def validate_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Validate data for LSTM strategy.
        
        Requires sequence_length + buffer days for feature generation.
        """
        min_days = self.sequence_length + 60  # Buffer for rolling features
        return {
            ticker: df 
            for ticker, df in data_dict.items() 
            if len(df) >= min_days
        }
    
    def rank_universe(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Score all tickers using LSTM predictions.
        
        Score = probability * confidence, where confidence = 1 - uncertainty.
        
        Args:
            data_dict: Dictionary of {ticker: OHLCV DataFrame}
            
        Returns:
            Dictionary of {ticker: score}
        """
        if self.predictor is None or self.predictor.model is None:
            print("No LSTM predictor loaded - returning empty scores")
            return {}
        
        scores = {}
        for ticker, df in data_dict.items():
            try:
                prob, uncertainty = self.predictor.predict_with_uncertainty(df)
                confidence = max(0, 1 - uncertainty)
                
                # Score = probability weighted by confidence
                score = prob * confidence
                
                # Only include if above confidence threshold
                if prob >= self.confidence_threshold:
                    scores[ticker] = float(score)
                    
            except Exception:
                pass
        
        return scores
    
    def generate_signals(
        self, 
        data_dict: Dict[str, pd.DataFrame],
        top_pct: float = 0.02,  # Top 2% = ~10 stocks from 500
        bottom_pct: float = 0.02
    ) -> Dict[str, str]:
        """
        Generate trading signals using probability ranking.
        
        Uses cross-sectional ranking of confidence-weighted probabilities.
        
        Args:
            data_dict: Dictionary of {ticker: OHLCV DataFrame}
            top_pct: Top percentile for BUY signals
            bottom_pct: Bottom percentile for SELL signals
            
        Returns:
            Dictionary of {ticker: 'BUY' | 'SELL' | 'HOLD'}
        """
        scores = self.rank_universe(data_dict)
        return self._cross_sectional_rank(scores, top_pct, bottom_pct)
    
    def train(self, data_dict: Dict[str, pd.DataFrame], **kwargs):
        """
        Train LSTM model.
        
        Args:
            data_dict: Training data dictionary
            **kwargs: Additional training parameters
        """
        from src.models.lstm.trainer import train_lstm
        train_lstm(data_dict, sequence_length=self.sequence_length, **kwargs)
        self._load_predictor()
    
    def get_predictions_with_uncertainty(
        self, 
        data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, tuple]:
        """
        Get predictions with uncertainty for all tickers.
        
        Returns:
            Dictionary of {ticker: (probability, uncertainty)}
        """
        if self.predictor is None:
            return {}
        return self.predictor.predict_batch(data_dict)
