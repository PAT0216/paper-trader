"""
ML Training Pipeline with Proper Time-Series Cross-Validation

Key improvements over previous version:
1. Uses TimeSeriesSplit for walk-forward validation (no data leakage)
2. XGBRegressor predicts return magnitude (not just direction)
3. Target creation is separate from feature generation
4. Reports metrics across all CV folds for robustness
"""

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import os
import joblib
import pandas as pd
import xgboost as xgb

from src.features.indicators import generate_features, create_target, FEATURE_COLUMNS

MODEL_PATH = "models"
MODEL_FILE = os.path.join(MODEL_PATH, "xgb_model.joblib")


def train_model(data_dict, n_splits=5, save_model=True):
    """
    Trains XGBoost regressor with proper time-series cross-validation.
    
    Key anti-leakage measures:
    1. Features generated WITHOUT target (no look-ahead)
    2. Target added AFTER feature generation
    3. TimeSeriesSplit ensures train data always before test data
    4. No shuffling in any step
    
    Args:
        data_dict: Dictionary of {ticker: OHLCV DataFrame}
        n_splits: Number of CV folds (default: 5)
        save_model: Whether to save the trained model
        
    Returns:
        Trained model
    """
    print("=" * 60)
    print("ML TRAINING PIPELINE (Phase 3 - Anti-Leakage)")
    print("=" * 60)
    
    print("\n📊 Preparing training data...")
    all_features = []
    
    for ticker, df in data_dict.items():
        # Step 1: Generate features ONLY (no target)
        processed_df = generate_features(df, include_target=False)
        
        # Step 2: Add target AFTER feature generation
        processed_df = create_target(processed_df, target_type='regression', horizon=1)
        
        # Drop rows where target is NaN (last row since we're predicting next day)
        processed_df = processed_df.dropna(subset=['Target'])
        
        all_features.append(processed_df)
    
    if not all_features:
        print("❌ No data to train on.")
        return None
    
    full_df = pd.concat(all_features)
    
    # Sort by date to ensure temporal order
    full_df = full_df.sort_index()
    
    print(f"   Total samples: {len(full_df):,}")
    print(f"   Date range: {full_df.index.min()} to {full_df.index.max()}")
    
    # Prepare X (features) and y (target)
    X = full_df[FEATURE_COLUMNS].values
    y = full_df['Target'].values
    
    print(f"   Features: {len(FEATURE_COLUMNS)}")
    print(f"   Target: Next-day return (regression)")
    
    # Time-Series Cross-Validation
    print(f"\n⏱️  Running {n_splits}-fold Time-Series Cross-Validation...")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train XGBoost Regressor
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            objective='reg:squarederror',
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Directional accuracy (did we predict the right direction?)
        direction_correct = ((y_pred > 0) == (y_test > 0)).mean()
        
        fold_metrics.append({
            'fold': fold + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_correct
        })
        
        print(f"   Fold {fold+1}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, Dir.Acc={direction_correct:.2%}")
    
    # Summary across folds
    avg_rmse = np.mean([m['rmse'] for m in fold_metrics])
    avg_mae = np.mean([m['mae'] for m in fold_metrics])
    avg_r2 = np.mean([m['r2'] for m in fold_metrics])
    avg_dir = np.mean([m['direction_accuracy'] for m in fold_metrics])
    
    print(f"\n📈 Cross-Validation Summary:")
    print(f"   Avg RMSE: {avg_rmse:.4f} ({avg_rmse*100:.2f}% return)")
    print(f"   Avg MAE:  {avg_mae:.4f}")
    print(f"   Avg R²:   {avg_r2:.4f}")
    print(f"   Avg Directional Accuracy: {avg_dir:.2%}")
    
    # Final model: train on ALL data
    print(f"\n🏋️ Training final model on all {len(X):,} samples...")
    final_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        objective='reg:squarederror',
        random_state=42
    )
    final_model.fit(X, y)
    
    # Save metrics
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
        f.write("=" * 50 + "\n")
        f.write("ML MODEL METRICS (Phase 3 - Regression)\n")
        f.write("=" * 50 + "\n\n")
        f.write("Cross-Validation Results:\n")
        for m in fold_metrics:
            f.write(f"  Fold {m['fold']}: RMSE={m['rmse']:.4f}, ")
            f.write(f"MAE={m['mae']:.4f}, Dir.Acc={m['direction_accuracy']:.2%}\n")
        f.write(f"\nAverage Metrics:\n")
        f.write(f"  RMSE: {avg_rmse:.4f}\n")
        f.write(f"  MAE:  {avg_mae:.4f}\n")
        f.write(f"  R²:   {avg_r2:.4f}\n")
        f.write(f"  Directional Accuracy: {avg_dir:.2%}\n")
        f.write(f"\n  Training Samples: {len(X):,}\n")
        f.write(f"  CV Folds: {n_splits}\n")
    
    # Feature importance plot
    feature_importance = pd.DataFrame({
        'feature': FEATURE_COLUMNS,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance (XGBoost Regressor)')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "feature_importance.png"))
    plt.close()
    
    # Save model
    if save_model:
        os.makedirs(MODEL_PATH, exist_ok=True)
        joblib.dump(final_model, MODEL_FILE)
        print(f"\n💾 Model saved to {MODEL_FILE}")
    
    print(f"📊 Metrics saved to {results_dir}/metrics.txt")
    print(f"📈 Feature importance saved to {results_dir}/feature_importance.png")
    
    return final_model


def evaluate_model(model, test_df):
    """
    Evaluate model on held-out test data.
    
    Args:
        model: Trained XGBRegressor
        test_df: DataFrame with OHLCV data
        
    Returns:
        Dict with evaluation metrics
    """
    # Generate features (no target)
    processed_df = generate_features(test_df, include_target=False)
    
    # Add target
    processed_df = create_target(processed_df, target_type='regression')
    processed_df = processed_df.dropna(subset=['Target'])
    
    X = processed_df[FEATURE_COLUMNS].values
    y_true = processed_df['Target'].values
    
    y_pred = model.predict(X)
    
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'direction_accuracy': ((y_pred > 0) == (y_true > 0)).mean()
    }
