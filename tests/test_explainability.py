"""
Unit tests for SHAP-based explainability
"""

import pytest
import numpy as np
import joblib
import os

# Skip entire module if shap is not installed
shap = pytest.importorskip("shap", reason="shap not installed")

from src.explainability.shap_analyzer import ShapAnalyzer



@pytest.fixture
def mock_model():
    """Create a simple mock XGBoost model for testing."""
    from xgboost import XGBRegressor
    
    # Create simple model with fixed base_score for SHAP compatibility
    X = np.random.randn(100, 15)
    y = np.random.randn(100)
    
    model = XGBRegressor(
        n_estimators=10, 
        max_depth=3, 
        random_state=42,
        base_score=0.5  # Fixed base_score for SHAP compatibility
    )
    model.fit(X, y)
    
    return model


@pytest.fixture
def shap_analyzer(mock_model):
    """Create ShapAnalyzer instance with mock model."""
    feature_names = [f"Feature_{i}" for i in range(15)]
    analyzer = ShapAnalyzer(model=mock_model, feature_names=feature_names)
    return analyzer


def test_shap_analyzer_init(mock_model):
    """Test ShapAnalyzer initialization."""
    analyzer = ShapAnalyzer(model=mock_model)
    
    assert analyzer.model is not None
    assert analyzer.explainer is not None


def test_explain_prediction(shap_analyzer):
    """Test SHAP value calculation for a single prediction."""
    features = np.random.randn(15)
    
    shap_values = shap_analyzer.explain_prediction(features)
    
    assert shap_values is not None
    assert len(shap_values) == 15  # Should match number of features


def test_get_top_features(shap_analyzer):
    """Test extracting top contributing features."""
    features = np.random.randn(15)
    shap_values = shap_analyzer.explain_prediction(features)
    
    top_features = shap_analyzer.get_top_features(shap_values, n=5)
    
    assert len(top_features) == 5
    assert all(isinstance(item, tuple) for item in top_features)
    assert all(len(item) == 2 for item in top_features)  # (name, value) pairs


def test_waterfall_plot_generation(shap_analyzer, tmp_path):
    """Test waterfall plot generation and saving."""
    features = np.random.randn(15)
    shap_values = shap_analyzer.explain_prediction(features)
    
    save_path = tmp_path / "test_waterfall.png"
    
    result_path = shap_analyzer.generate_waterfall_plot(
        shap_values,
        features,
        save_path=str(save_path)
    )
    
    assert os.path.exists(result_path)
    assert result_path == str(save_path)


def test_explain_trade_decision(shap_analyzer, tmp_path):
    """Test full trade decision explanation."""
    features = np.random.randn(15)
    prediction = 0.015
    
    explanation = shap_analyzer.explain_trade_decision(
        ticker="AAPL",
        features=features,
        prediction=prediction,
        save_dir=str(tmp_path)
    )
    
    assert explanation['ticker'] == 'AAPL'
    assert explanation['prediction'] == prediction
    assert len(explanation['top_features']) == 5
    assert os.path.exists(explanation['waterfall_plot'])


def test_load_model_missing_file():
    """Test loading a non-existent model file."""
    analyzer = ShapAnalyzer()
    
    with pytest.raises(FileNotFoundError):
        analyzer.load_model("nonexistent_model.joblib")


def test_shap_values_2d_input(shap_analyzer):
    """Test SHAP values with 2D feature array."""
    features = np.random.randn(1, 15)  # 2D array
    
    shap_values = shap_analyzer.explain_prediction(features)
    
    assert shap_values is not None
    assert shap_values.shape[-1] == 15
