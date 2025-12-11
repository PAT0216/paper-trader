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
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import numpy as np
import os
import joblib
import pandas as pd
import xgboost as xgb

from src.features.indicators import generate_features, create_target, FEATURE_COLUMNS

MODEL_PATH = "models"
MODEL_FILE = os.path.join(MODEL_PATH, "xgb_model.joblib")
RESULTS_LIVE_DIR = "results/live"  # Training/live metrics go here
RESULTS_BACKTEST_DIR = "results/backtest"  # Backtest results go here


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
    
    # Filter to recent data (2010+) for better quality
    # Older data often has issues: missing values, stock splits, data errors
    MIN_DATE = '2010-01-01'
    full_df = full_df[full_df.index >= MIN_DATE]
    print(f"   After date filter (>= {MIN_DATE}): {len(full_df):,} samples")
    
    # Prepare X (features) and y (target)
    X = full_df[FEATURE_COLUMNS].values
    y = full_df['Target'].values
    
    # Clean data: replace inf with NaN, then drop NaN rows
    # This prevents XGBoost "Input contains inf" error
    X = np.where(np.isinf(X), np.nan, X)
    
    # Find rows with NaN in X or y
    valid_rows = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid_rows]
    y = y[valid_rows]
    
    dropped = len(full_df) - len(X)
    if dropped > 0:
        print(f"   ⚠️ Dropped {dropped:,} rows with inf/NaN values")
    
    print(f"   Final samples: {len(X):,}")
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
        
        # RANKING METRICS (Phase 7 - explains R² paradox)
        # Spearman rank correlation: do predicted ranks match actual ranks?
        spearman_corr, _ = spearmanr(y_pred, y_test)
        
        # Top-10 accuracy: if we pick top 10 predictions, how many are actually top 10?
        n_top = min(10, len(y_pred) // 10)  # 10% or top 10, whichever is more meaningful
        if n_top > 0:
            pred_top_idx = np.argsort(y_pred)[-n_top:]
            actual_top_idx = np.argsort(y_test)[-n_top:]
            top_k_overlap = len(set(pred_top_idx) & set(actual_top_idx)) / n_top
        else:
            top_k_overlap = 0.0
        
        fold_metrics.append({
            'fold': fold + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_correct,
            'spearman': spearman_corr,
            'top_k_accuracy': top_k_overlap
        })
        
        print(f"   Fold {fold+1}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, Dir.Acc={direction_correct:.2%}")
    
    # Summary across folds
    avg_rmse = np.mean([m['rmse'] for m in fold_metrics])
    avg_mae = np.mean([m['mae'] for m in fold_metrics])
    avg_r2 = np.mean([m['r2'] for m in fold_metrics])
    avg_dir = np.mean([m['direction_accuracy'] for m in fold_metrics])
    avg_spearman = np.mean([m['spearman'] for m in fold_metrics])
    avg_top_k = np.mean([m['top_k_accuracy'] for m in fold_metrics])
    
    print(f"\n📈 Cross-Validation Summary:")
    print(f"   Avg RMSE: {avg_rmse:.4f} ({avg_rmse*100:.2f}% return)")
    print(f"   Avg MAE:  {avg_mae:.4f}")
    print(f"   Avg R²:   {avg_r2:.4f}")
    print(f"   Avg Directional Accuracy: {avg_dir:.2%}")
    print(f"")
    print(f"   --- RANKING METRICS (explains R² paradox) ---")
    print(f"   Avg Spearman Rank Corr: {avg_spearman:.4f}")
    print(f"   Avg Top-10% Accuracy:   {avg_top_k:.2%}")
    
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
    
    # ==================== DYNAMIC FEATURE SELECTION ====================
    # Phase 3.5: Automatically filter features with importance < threshold
    IMPORTANCE_THRESHOLD = 0.03  # 3% minimum importance
    
    feature_importance = pd.DataFrame({
        'feature': FEATURE_COLUMNS,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Identify useful vs low-importance features
    useful_features = feature_importance[feature_importance['importance'] >= IMPORTANCE_THRESHOLD]['feature'].tolist()
    dropped_features = feature_importance[feature_importance['importance'] < IMPORTANCE_THRESHOLD]['feature'].tolist()
    
    print(f"\n🔍 Feature Selection (threshold: {IMPORTANCE_THRESHOLD*100:.1f}%):")
    print(f"   Keeping: {len(useful_features)} features")
    if dropped_features:
        print(f"   ⚠️  Dropping: {dropped_features}")
        
        # Retrain with only useful features
        feature_indices = [i for i, f in enumerate(FEATURE_COLUMNS) if f in useful_features]
        X_selected = X[:, feature_indices]
        
        print(f"\n🔄 Retraining with {len(useful_features)} selected features...")
        final_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            objective='reg:squarederror',
            random_state=42
        )
        final_model.fit(X_selected, y)
        
        # Update feature importance for selected features only
        feature_importance = pd.DataFrame({
            'feature': useful_features,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        print("   ✅ All features above threshold - keeping all")
    
    # Store selected features for inference
    selected_features = useful_features
    # ==================== END DYNAMIC FEATURE SELECTION ====================
    
    # Save metrics to LIVE results directory (separate from backtest)
    os.makedirs(RESULTS_LIVE_DIR, exist_ok=True)
    
    with open(os.path.join(RESULTS_LIVE_DIR, "metrics.txt"), "w") as f:
        f.write("=" * 50 + "\n")
        f.write("ML MODEL METRICS (Phase 3.5 - Dynamic Feature Selection)\n")
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
        f.write(f"\n  --- RANKING METRICS ---\n")
        f.write(f"  Spearman Rank Corr: {avg_spearman:.4f}\n")
        f.write(f"  Top-10% Accuracy: {avg_top_k:.2%}\n")
        f.write(f"\n  Training Samples: {len(X):,}\n")
        f.write(f"  CV Folds: {n_splits}\n")
        f.write(f"\nFeature Selection:\n")
        f.write(f"  Threshold: {IMPORTANCE_THRESHOLD*100:.1f}%\n")
        f.write(f"  Selected: {len(selected_features)} of {len(FEATURE_COLUMNS)}\n")
        if dropped_features:
            f.write(f"  Dropped: {dropped_features}\n")
    
    # Feature importance plot (for selected features)
    plt.figure(figsize=(10, 6))
    feat_sorted = feature_importance.sort_values('importance', ascending=True)
    colors = ['green' if imp >= IMPORTANCE_THRESHOLD else 'red' for imp in feat_sorted['importance']]
    plt.barh(feat_sorted['feature'], feat_sorted['importance'], color=colors)
    plt.axvline(x=IMPORTANCE_THRESHOLD, color='red', linestyle='--', label=f'Threshold ({IMPORTANCE_THRESHOLD*100:.0f}%)')
    plt.xlabel('Importance')
    plt.title(f'Feature Importance (Selected: {len(selected_features)} features)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_LIVE_DIR, "feature_importance.png"))
    plt.close()
    
    # Save selected features list for inference
    with open(os.path.join(RESULTS_LIVE_DIR, "selected_features.txt"), "w") as f:
        f.write("# Selected Features (importance >= 3%)\n")
        for feat in selected_features:
            imp = feature_importance[feature_importance['feature'] == feat]['importance'].values[0]
            f.write(f"{feat}: {imp:.4f}\n")
    
    # Save model with metadata
    model_data = {
        'model': final_model,
        'selected_features': selected_features,
        'all_features': FEATURE_COLUMNS
    }
    
    if save_model:
        os.makedirs(MODEL_PATH, exist_ok=True)
        joblib.dump(model_data, MODEL_FILE)
        print(f"\n💾 Model saved to {MODEL_FILE}")
        print(f"   Selected features: {selected_features}")
    
    print(f"📊 Metrics saved to {RESULTS_LIVE_DIR}/metrics.txt")
    print(f"📈 Feature importance saved to {RESULTS_LIVE_DIR}/feature_importance.png")
    
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
        'direction_accuracy': ((y_pred > 0) == (y_true > 0)).mean(),
        'spearman': spearmanr(y_pred, y_true)[0]
    }


# ==================== MULTI-HORIZON ENSEMBLE (Phase 3.6) ====================

ENSEMBLE_FILE = os.path.join(MODEL_PATH, "xgb_ensemble.joblib")
HORIZONS = [1, 5, 20]  # Days ahead to predict
HORIZON_WEIGHTS = {1: 0.5, 5: 0.3, 20: 0.2}  # More weight to short-term


def train_ensemble(data_dict, n_splits=5, save_model=True):
    """
    Train multi-horizon ensemble model.
    
    Trains 3 separate XGBoost models for different prediction horizons:
    - 1-day return (50% weight)
    - 5-day return (30% weight)
    - 20-day return (20% weight)
    
    Final prediction is a weighted blend that provides more stable signals.
    
    Args:
        data_dict: Dictionary of {ticker: OHLCV DataFrame}
        n_splits: Number of CV folds
        save_model: Whether to save the ensemble
        
    Returns:
        Ensemble dict with models and metadata
    """
    print("=" * 60)
    print("ML TRAINING PIPELINE (Phase 3.6 - Multi-Horizon Ensemble)")
    print("=" * 60)
    
    ensemble = {
        'models': {},
        'selected_features': {},
        'weights': HORIZON_WEIGHTS,
        'horizons': HORIZONS,
        'all_features': FEATURE_COLUMNS
    }
    
    for horizon in HORIZONS:
        print(f"\n🎯 Training {horizon}-day horizon model...")
        
        # Prepare data with specific horizon
        all_features = []
        for ticker, df in data_dict.items():
            processed_df = generate_features(df, include_target=False)
            processed_df = create_target(processed_df, target_type='regression', horizon=horizon)
            processed_df = processed_df.dropna(subset=['Target'])
            all_features.append(processed_df)
        
        if not all_features:
            print(f"   ❌ No data for {horizon}-day horizon")
            continue
        
        full_df = pd.concat(all_features).sort_index()
        
        # Filter to recent data (2010+) for better quality
        MIN_DATE = '2010-01-01'
        full_df = full_df[full_df.index >= MIN_DATE]
        
        X = full_df[FEATURE_COLUMNS].values
        y = full_df['Target'].values
        
        # Clean data: replace inf with NaN, then drop NaN rows
        X = np.where(np.isinf(X), np.nan, X)
        valid_rows = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_rows]
        y = y[valid_rows]
        
        print(f"   Samples: {len(full_df):,}")
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                objective='reg:squarederror',
                random_state=42
            )
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            direction_correct = ((y_pred > 0) == (y_test > 0)).mean()
            fold_metrics.append(direction_correct)
        
        avg_dir_acc = np.mean(fold_metrics)
        print(f"   Avg Directional Accuracy: {avg_dir_acc:.2%}")
        
        # Train final model
        final_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            objective='reg:squarederror',
            random_state=42
        )
        final_model.fit(X, y)
        
        # Dynamic feature selection (same as single model)
        IMPORTANCE_THRESHOLD = 0.03
        feature_importance = pd.DataFrame({
            'feature': FEATURE_COLUMNS,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        useful_features = feature_importance[
            feature_importance['importance'] >= IMPORTANCE_THRESHOLD
        ]['feature'].tolist()
        
        dropped = len(FEATURE_COLUMNS) - len(useful_features)
        if dropped > 0:
            print(f"   Dropped {dropped} low-importance features")
            feature_indices = [i for i, f in enumerate(FEATURE_COLUMNS) if f in useful_features]
            X_selected = X[:, feature_indices]
            final_model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                objective='reg:squarederror',
                random_state=42
            )
            final_model.fit(X_selected, y)
        
        ensemble['models'][horizon] = final_model
        ensemble['selected_features'][horizon] = useful_features
        print(f"   ✅ {horizon}-day model ready ({len(useful_features)} features)")
    
    # Save ensemble
    if save_model:
        os.makedirs(MODEL_PATH, exist_ok=True)
        joblib.dump(ensemble, ENSEMBLE_FILE)
        print(f"\n💾 Ensemble saved to {ENSEMBLE_FILE}")
    
    # Also save single-horizon model for backward compatibility
    if 1 in ensemble['models']:
        single_model_data = {
            'model': ensemble['models'][1],
            'selected_features': ensemble['selected_features'][1],
            'all_features': FEATURE_COLUMNS
        }
        joblib.dump(single_model_data, MODEL_FILE)
        print(f"💾 Single model (1-day) saved to {MODEL_FILE}")
    
    print("\n" + "=" * 60)
    print("✅ MULTI-HORIZON ENSEMBLE TRAINING COMPLETE")
    print("=" * 60)
    print(f"   Horizons: {HORIZONS}")
    print(f"   Weights: {list(HORIZON_WEIGHTS.values())}")
    
    return ensemble
