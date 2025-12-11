#!/usr/bin/env python3
"""
Demonstration of SHAP explainability

mu to verify SHAP integration works with actual trained models.
"""

import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.explainability.shap_analyzer import ShapAnalyzer
from src.models.trainer import FEATURE_COLUMNS


def demo_shap():
    """Demonstrate SHAP analysis on trained model."""
    print("=" * 70)
    print("SHAP Explainability Demonstration")
    print("=" * 70)
    
    model_path = "models/xgb_model.joblib"
    
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found at {model_path}")
        print("   Run 'make train' first to train a model")
        return
    
    # Load analyzer
    print(f"\n✅ Loading model from {model_path}")
    analyzer = ShapAnalyzer(feature_names=FEATURE_COLUMNS)
    analyzer.load_model(model_path)
    
    # Create sample features (normally from real stock data)
    print("\n📊 Generating sample prediction...")
    sample_features = np.random.randn(len(FEATURE_COLUMNS))
    
    # Get prediction from model
    prediction = analyzer.model.predict(sample_features.reshape(1, -1))[0]
    print(f"   Model prediction: {prediction:.4f} ({prediction*100:.2f}% expected return)")
    
    # Explain prediction
    print("\n🧠 Calculating SHAP values...")
    shap_values = analyzer.explain_prediction(sample_features)
    
    # Get top contributing features
    top_features = analyzer.get_top_features(shap_values, n=5)
    
    print("\n🎯 Top 5 Contributing Features:")
    for i, (feature_name, shap_value) in enumerate(top_features, 1):
        direction = "📈" if shap_value > 0 else "📉"
        print(f"   {i}. {direction} {feature_name:15s}: {shap_value:+.6f}")
    
    #Generate waterfall plot
    print("\n📊 Generating SHAP waterfall plot...")
    plot_path = analyzer.generate_waterfall_plot(
        shap_values,
        sample_features,
        save_path="results/explainability/demo_waterfall.png"
    )
    
    print(f"   ✅ Waterfall plot saved to: {plot_path}")
    
    # Full explanation
    print("\n🔍 Generating full trade explanation...")
    explanation = analyzer.explain_trade_decision(
        ticker="DEMO",
        features=sample_features,
        prediction=prediction,
        save_dir="results/explainability/"
    )
    
    print("   ✅ Complete explanation generated")
    print(f"      - Top features: {len(explanation['top_features'])}")
    print(f"      - Waterfall plot: {explanation['waterfall_plot']}")
    
    print("\n" + "=" * 70)
    print("✅ SHAP Explainability Demo Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  - View plots in results/explainability/")
    print("  - Integrate SHAP into trading pipeline")
    print("  - Use in Streamlit dashboard (Phase 8.2)")


if __name__ == "__main__":
    demo_shap()
