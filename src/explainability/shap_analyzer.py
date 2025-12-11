"""
SHAP-based Model Explainability

Provides interpretability for XGBoost trading model decisions using SHAP
(SHapley Additive exPlanations) values.

Key Features:
- Feature importance for individual predictions
- Waterfall plots showing decision breakdown
- Top contributing features for each trade signal

Usage:
    analyzer = ShapAnalyzer(model)
    shap_values = analyzer.explain_prediction(features)
    top_features = analyzer.get_top_features(shap_values, n=5)
"""

import os
import shap
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import joblib


class ShapAnalyzer:
    """
    SHAP-based explainability for XGBoost trading model.
    
    Attributes:
        model: Trained XGBoost model
        explainer: SHAP TreeExplainer (cached for efficiency)
        feature_names: List of feature column names
    """
    
    def __init__(self, model=None, feature_names=None):
        """
        Initialize SHAP analyzer.
        
        Args:
            model: Trained XGBoost model (optional, can load later)
            feature_names: List of feature names matching model input
        """
        self.model = model
        self.explainer = None
        self.feature_names = feature_names
        
        if model is not None:
            self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize SHAP explainer (computationally expensive, do once)."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print("Initializing SHAP explainer (this may take a moment)...")
        self.explainer = shap.TreeExplainer(self.model)
        print("✅ SHAP explainer ready")
    
    def load_model(self, model_path: str):
        """
        Load a trained XGBoost model from disk.
        
        Args:
            model_path: Path to .joblib model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = joblib.load(model_path)
        self._initialize_explainer()
    
    def explain_prediction(
        self, 
        features: np.ndarray,
        check_additivity: bool = False
    ) -> np.ndarray:
        """
        Calculate SHAP values for a single prediction.
        
        Args:
            features: Feature array (1D or 2D for single sample)
            check_additivity: Verify SHAP values sum to prediction
        
        Returns:
            SHAP values array (same shape as features)
        """
        if self.explainer is None:
            self._initialize_explainer()
        
        # Ensure 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        shap_values = self.explainer.shap_values(features, check_additivity=check_additivity)
        
        return shap_values
    
    def get_top_features(
        self, 
        shap_values: np.ndarray, 
        n: int = 5,
        absolute: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Get the top N most impactful features for a prediction.
        
        Args:
            shap_values: SHAP values from explain_prediction()
            n: Number of top features to return
            absolute: Use absolute value of SHAP (magnitude) vs signed
        
        Returns:
            List of (feature_name, shap_value) tuples, sorted by impact
        """
        if shap_values.ndim == 2:
            shap_values = shap_values[0] # Take first sample if batch
        
        if self.feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(shap_values))]
        else:
            feature_names = self.feature_names
        
        # Get indices of top features
        if absolute:
            top_indices = np.argsort(np.abs(shap_values))[::-1][:n]
        else:
            top_indices = np.argsort(shap_values)[::-1][:n]
        
        top_features = [
            (feature_names[i], shap_values[i])
            for i in top_indices
        ]
        
        return top_features
    
    def generate_waterfall_plot(
        self,
        shap_values: np.ndarray,
        features: np.ndarray,
        expected_value: Optional[float] = None,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> str:
        """
        Generate a SHAP waterfall plot showing how features contribute to prediction.
        
        Args:
            shap_values: SHAP values from explain_prediction()
            features: Original feature values (for display)
            expected_value: Base value (model's expected output)
            save_path: Path to save plot (default: results/explainability/)
            show: Whether to display plot interactively
        
        Returns:
            Path to saved plot file
        """
        if shap_values.ndim == 2:
            shap_values = shap_values[0]
        if features.ndim == 2:
            features = features[0]
        
        # Get expected value from explainer
        if expected_value is None and self.explainer is not None:
            expected_value = self.explainer.expected_value
        
        # Create Explanation object for waterfall plot
        if self.feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(shap_values))]
        else:
            feature_names = self.feature_names
        
        explanation = shap.Explanation(
            values=shap_values,
            base_values=expected_value,
            data=features,
            feature_names=feature_names
        )
        
        # Generate plot
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation, show=show)
        
        # Save
        if save_path is None:
            save_path = "results/explainability/waterfall_latest.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        return save_path
    
    def generate_force_plot(
        self,
        shap_values: np.ndarray,
        features: np.ndarray,
        expected_value: Optional[float] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate a SHAP force plot (alternative visualization).
        
        Args:
            shap_values: SHAP values from explain_prediction()
            features: Original feature values
            expected_value: Base value
            save_path: Path to save plot HTML
        
        Returns:
            Path to saved HTML file
        """
        if shap_values.ndim == 2:
            shap_values = shap_values[0]
        if features.ndim == 2:
            features = features[0]
        
        if expected_value is None and self.explainer is not None:
            expected_value = self.explainer.expected_value
        
        if save_path is None:
            save_path = "results/explainability/force_plot_latest.html"
        
        # Generate force plot
        force_plot = shap.force_plot(
            expected_value,
            shap_values,
            features,
            feature_names=self.feature_names
        )
        
        # Save as HTML
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        shap.save_html(save_path, force_plot)
        
        return save_path
    
    def explain_trade_decision(
        self,
        ticker: str,
        features: np.ndarray,
        prediction: float,
        save_dir: str = "results/explainability/"
    ) -> Dict:
        """
        Full explanation for a single trade decision.
        
        Args:
            ticker: Stock ticker being explained
            features: Feature values for this prediction
            prediction: Model's predicted return
            save_dir: Directory to save visualizations
        
        Returns:
            Dictionary with top features and plot paths
        """
        # Calculate SHAP values
        shap_values = self.explain_prediction(features)
        
        # Get top features
        top_features = self.get_top_features(shap_values, n=5)
        
        # Generate waterfall plot
        waterfall_path = os.path.join(save_dir, f"{ticker}_waterfall.png")
        self.generate_waterfall_plot(shap_values, features, save_path=waterfall_path)
        
        # Build explanation
        explanation = {
            'ticker': ticker,
            'prediction': prediction,
            'top_features': top_features,
            'waterfall_plot': waterfall_path,
            'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values
        }
        
        return explanation


if __name__ == "__main__":
    # Example usage
    print("SHAP Analyzer - Example Usage")
    print("=" * 50)
    
    # Load a model (replace with your actual model path)
    try:
        analyzer = ShapAnalyzer()
        analyzer.load_model("models/xgb_model.joblib")
        
        # Create dummy features for demonstration
        dummy_features = np.random.randn(15)  # 15 features
        
        # Explain prediction
        shap_values = analyzer.explain_prediction(dummy_features)
        
        # Get top features
        top_features = analyzer.get_top_features(shap_values, n=5)
        
        print("\nTop 5 Contributing Features:")
        for i, (name, value) in enumerate(top_features, 1):
            print(f"  {i}. {name}: {value:.4f}")
        
        # Generate waterfall plot
        plot_path = analyzer.generate_waterfall_plot(
            shap_values, 
            dummy_features,
            save_path="results/explainability/example_waterfall.png"
        )
        print(f"\n✅ Waterfall plot saved to: {plot_path}")
        
    except FileNotFoundError:
        print("Model not found. Train a model first with 'make train'")
