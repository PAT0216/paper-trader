"""
ML Strategy - XGBoost-based Predictive Strategy

Uses trained XGBoost ensemble to predict next-day returns.
Ranks universe by expected return and generates cross-sectional signals.
"""

from typing import Dict, Optional
import pandas as pd

from src.strategies.base import BaseStrategy


class MLStrategy(BaseStrategy):
    """
    Machine Learning strategy using XGBoost ensemble predictor.
    
    Features:
    - Multi-horizon ensemble (1d, 5d, 20d predictions)
    - 15 technical indicator features
    - Cross-sectional ranking for signal generation
    - Daily rebalancing
    
    Requires model training before use.
    """
    
    def __init__(self, use_ensemble: bool = True):
        """
        Initialize ML strategy.
        
        Args:
            use_ensemble: Use multi-horizon ensemble (default: True)
        """
        self.use_ensemble = use_ensemble
        self.predictor = None
        self._load_predictor()
    
    def _load_predictor(self):
        """Lazy load predictor to avoid import issues."""
        try:
            if self.use_ensemble:
                from src.models.predictor import EnsemblePredictor
                self.predictor = EnsemblePredictor()
            else:
                from src.models.predictor import Predictor
                self.predictor = Predictor()
        except Exception as e:
            print(f"⚠️ Could not load ML predictor: {e}")
            self.predictor = None
    
    def get_name(self) -> str:
        """Return strategy identifier."""
        return "ml"
    
    def get_display_name(self) -> str:
        """Return human-readable name."""
        return "ML Ensemble" if self.use_ensemble else "ML Single"
    
    def needs_training(self) -> bool:
        """ML strategy requires model training."""
        return True
    
    def validate_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Validate data for ML strategy.
        
        Requires 200+ days for SMA_200 feature.
        """
        min_days = 200
        valid = {}
        for ticker, df in data_dict.items():
            if len(df) >= min_days:
                valid[ticker] = df
        return valid
    
    def rank_universe(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Score all tickers using ML predictor.
        
        Args:
            data_dict: Dictionary of {ticker: OHLCV DataFrame}
            
        Returns:
            Dictionary of {ticker: expected_return}
        """
        if self.predictor is None:
            print("⚠️ No ML predictor loaded - returning empty scores")
            return {}
        
        scores = {}
        for ticker, df in data_dict.items():
            try:
                if self.use_ensemble:
                    score = self.predictor.predict(df)
                else:
                    score = self.predictor.predict(df)
                scores[ticker] = float(score)
            except Exception as e:
                # Silently skip failed predictions
                pass
        
        return scores
    
    def generate_signals(
        self, 
        data_dict: Dict[str, pd.DataFrame],
        top_pct: float = 0.10,
        bottom_pct: float = 0.10
    ) -> Dict[str, str]:
        """
        Generate trading signals using cross-sectional ranking.
        
        Args:
            data_dict: Dictionary of {ticker: OHLCV DataFrame}
            top_pct: Top percentile for BUY (default 10%)
            bottom_pct: Bottom percentile for SELL (default 10%)
            
        Returns:
            Dictionary of {ticker: 'BUY' | 'SELL' | 'HOLD'}
        """
        scores = self.rank_universe(data_dict)
        return self._cross_sectional_rank(scores, top_pct, bottom_pct)
    
    def train(self, data_dict: Dict[str, pd.DataFrame], **kwargs):
        """
        Train the ML model.
        
        Args:
            data_dict: Training data
            **kwargs: Additional training parameters
        """
        from src.models.trainer import train_ensemble, train_model
        
        if self.use_ensemble:
            train_ensemble(data_dict, **kwargs)
        else:
            train_model(data_dict, **kwargs)
        
        # Reload predictor after training
        self._load_predictor()
