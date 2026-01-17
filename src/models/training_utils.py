"""
Training Utilities - Shared functions for ML model training.

Extracted from trainer.py for reusability across ML and LSTM trainers.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import os
import json
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from src.features.indicators import generate_features, create_target, FEATURE_COLUMNS


# Default paths
MODEL_PATH = "models"
RESULTS_LIVE_DIR = "results/live"

# Number of random noise features for feature selection
N_NOISE_FEATURES = 5


def select_features_better_than_noise(
    X: np.ndarray, 
    y: np.ndarray, 
    feature_names: List[str], 
    n_noise: int = N_NOISE_FEATURES
) -> List[str]:
    """
    Train a quick model with real features + random noise to identify
    which features have importance greater than random baseline.
    
    This is a robust feature selection method that automatically adapts
    to changing market conditions.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector
        feature_names: List of feature names corresponding to X columns
        n_noise: Number of random noise features to add
        
    Returns:
        List of feature names that beat random noise baseline
    """
    # Add random noise features
    np.random.seed(42)  # Fixed seed for reproducible feature selection
    noise = np.random.randn(len(X), n_noise)
    X_with_noise = np.hstack([X, noise])
    
    noise_names = [f'NOISE_{i}' for i in range(n_noise)]
    all_names = list(feature_names) + noise_names
    
    # Train quick model
    model = xgb.XGBRegressor(
        n_estimators=50, 
        learning_rate=0.1, 
        max_depth=4, 
        random_state=42, 
        verbosity=0
    )
    model.fit(X_with_noise, y)
    
    # Calculate noise baseline
    importances = model.feature_importances_
    noise_baseline = np.mean(importances[-n_noise:])
    
    # Select features better than noise
    selected = []
    for i, feat in enumerate(feature_names):
        if importances[i] > noise_baseline:
            selected.append(feat)
    
    # Fallback: if no features beat noise, keep top 8 by importance
    if not selected:
        sorted_idx = np.argsort(importances[:len(feature_names)])[::-1]
        selected = [feature_names[i] for i in sorted_idx[:8]]
    
    return selected


def prepare_training_data(
    data_dict: Dict[str, pd.DataFrame],
    horizon: int = 1,
    min_date: str = '2010-01-01'
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Prepare training data from a dictionary of ticker DataFrames.
    
    Args:
        data_dict: Dictionary of {ticker: OHLCV DataFrame}
        horizon: Forward return horizon (1, 5, or 20 days)
        min_date: Minimum date filter for quality data
        
    Returns:
        Tuple of (X, y, full_df)
    """
    all_features = []
    
    for ticker, df in data_dict.items():
        # Step 1: Generate features ONLY (no target)
        processed_df = generate_features(df, include_target=False)
        
        # Step 2: Add target AFTER feature generation
        processed_df = create_target(processed_df, target_type='regression', horizon=horizon)
        
        # Drop rows where target is NaN
        processed_df = processed_df.dropna(subset=['Target'])
        
        all_features.append(processed_df)
    
    if not all_features:
        return None, None, None
    
    full_df = pd.concat(all_features)
    
    # Sort by date to ensure temporal order
    full_df = full_df.sort_index()
    
    # Filter to recent data
    full_df = full_df[full_df.index >= min_date]
    
    # Prepare X (features) and y (target)
    X = full_df[FEATURE_COLUMNS].values
    y = full_df['Target'].values
    
    # Clean data: replace inf with NaN, then drop NaN rows
    X = np.where(np.isinf(X), np.nan, X)
    
    # Find rows with NaN in X or y
    valid_rows = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid_rows]
    y = y[valid_rows]
    
    return X, y, full_df


def save_model_with_metadata(
    model: xgb.XGBRegressor,
    selected_features: List[str],
    X: np.ndarray,
    cv_metrics: dict,
    model_path: str = MODEL_PATH,
    model_name: str = "xgb_model"
) -> dict:
    """
    Save model with metadata in portable format.
    
    Args:
        model: Trained XGBRegressor
        selected_features: List of selected feature names
        X: Training data (for prediction stats)
        cv_metrics: Cross-validation metrics dict
        model_path: Directory to save model
        model_name: Base name for model files
        
    Returns:
        Model metadata dict
    """
    os.makedirs(model_path, exist_ok=True)
    
    # Calculate expected prediction range from training data
    train_predictions = model.predict(X)
    pred_mean = float(np.mean(train_predictions))
    pred_std = float(np.std(train_predictions))
    
    model_metadata = {
        'selected_features': selected_features,
        'all_features': list(FEATURE_COLUMNS),
        'training_date': datetime.now().isoformat(),
        'training_samples': len(X),
        'xgboost_version': xgb.__version__,
        'prediction_stats': {
            'mean': pred_mean,
            'std': pred_std,
            'min': float(np.min(train_predictions)),
            'max': float(np.max(train_predictions)),
            'expected_range': [-0.10, 0.10]
        },
        'cv_metrics': cv_metrics
    }
    
    # Save model in portable JSON format (XGBoost native)
    model_json_path = os.path.join(model_path, f"{model_name}.json")
    model.save_model(model_json_path)
    
    # Save metadata separately
    metadata_path = os.path.join(model_path, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    # Also keep legacy pickle for backward compatibility
    model_data = {
        'model': model,
        'selected_features': selected_features,
        'all_features': FEATURE_COLUMNS
    }
    legacy_path = os.path.join(model_path, f"{model_name}.joblib")
    joblib.dump(model_data, legacy_path)
    
    print(f"\nModel saved:")
    print(f"   JSON (portable): {model_json_path}")
    print(f"   Metadata: {metadata_path}")
    print(f"   Legacy pickle: {legacy_path}")
    
    return model_metadata
