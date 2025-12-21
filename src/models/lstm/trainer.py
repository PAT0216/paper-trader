"""
LSTM Training Pipeline

Implements walk-forward training with non-overlapping windows
to prevent data leakage and overfitting.
"""

import numpy as np
import os
from typing import Dict, Optional
import pandas as pd

from src.models.lstm.features import generate_lstm_features, create_sequences, LSTM_FEATURES
from src.models.lstm.threshold import create_pit_target


def train_lstm(
    data_dict: Dict[str, pd.DataFrame],
    sequence_length: int = 60,
    window_step: int = 60,  # Non-overlapping windows
    horizon: int = 1,       # 1-day forward prediction
    sigma_multiplier: float = 2.0,
    epochs: int = 50,
    batch_size: int = 64,
    validation_split: float = 0.2,
    save_path: str = 'models/lstm_model.h5',
    verbose: int = 1
) -> dict:
    """
    Train threshold classification model.
    
    Uses LSTM if TensorFlow available, otherwise XGBoost fallback.
    Non-overlapping windows prevent data leakage.
    
    Args:
        data_dict: Dictionary of {ticker: OHLCV DataFrame}
        sequence_length: Number of days per sequence
        window_step: Step between windows (same as sequence_length for non-overlapping)
        horizon: Forward return horizon
        sigma_multiplier: Threshold for target classification
        epochs: Maximum training epochs (LSTM only)
        batch_size: Training batch size (LSTM only)
        validation_split: Fraction for validation
        save_path: Where to save trained model
        verbose: Verbosity level
    
    Returns:
        Training history and metrics
    """
    from src.models.lstm.model import is_tensorflow_available, build_lstm_model, get_callbacks, build_xgb_threshold_model
    
    use_lstm = is_tensorflow_available()
    
    print(f"Building training dataset...")
    print(f"  Model: {'LSTM' if use_lstm else 'XGBoost (TF fallback)'}")
    print(f"  Sequence length: {sequence_length} days")
    print(f"  Window step: {window_step} (non-overlapping: {window_step >= sequence_length})")
    print(f"  Threshold: mean + {sigma_multiplier}*sigma")
    
    X_all, y_all = [], []
    
    for ticker, df in data_dict.items():
        try:
            # Generate features
            features_df = generate_lstm_features(df)
            if len(features_df) < sequence_length + horizon + 60:
                continue
            
            # Create target (point-in-time: uses forward returns)
            target = create_pit_target(df, horizon=horizon, sigma_multiplier=sigma_multiplier)
            target = target.loc[features_df.index]
            
            # Create sequences with specified step
            X, y = create_sequences(
                features_df, 
                target, 
                sequence_length=sequence_length,
                step_size=window_step
            )
            
            if len(X) > 0:
                X_all.append(X)
                y_all.append(y)
                
        except Exception as e:
            if verbose:
                print(f"  Skipped {ticker}: {e}")
            continue
    
    if not X_all:
        raise ValueError("No valid training data generated")
    
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    
    print(f"\nTraining samples: {len(X_all)}")
    print(f"  Positive class: {y_all.sum()} ({100*y_all.mean():.1f}%)")
    print(f"  Negative class: {len(y_all) - y_all.sum()} ({100*(1-y_all.mean()):.1f}%)")
    
    # Time-based split (no shuffling!)
    split_idx = int(len(X_all) * (1 - validation_split))
    X_train, X_val = X_all[:split_idx], X_all[split_idx:]
    y_train, y_val = y_all[:split_idx], y_all[split_idx:]
    
    print(f"\nTrain/Val split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    
    if use_lstm:
        # LSTM training
        model = build_lstm_model(
            sequence_length=sequence_length,
            n_features=len(LSTM_FEATURES)
        )
        
        if verbose:
            model.summary()
        
        pos_weight = len(y_train) / (2 * y_train.sum() + 1)
        neg_weight = len(y_train) / (2 * (len(y_train) - y_train.sum()) + 1)
        class_weight = {0: neg_weight, 1: pos_weight}
        
        print(f"\nTraining LSTM for up to {epochs} epochs...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=get_callbacks(patience=15),
            class_weight=class_weight,
            verbose=verbose
        )
        
        model.save(save_path)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        
    else:
        # XGBoost fallback - flatten sequences for tabular model
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        
        print(f"\nTraining XGBoost threshold classifier...")
        model = build_xgb_threshold_model()
        model.fit(
            X_train_flat, y_train,
            eval_set=[(X_val_flat, y_val)],
            verbose=verbose > 0
        )
        
        # Save model
        import joblib
        xgb_save_path = save_path.replace('.h5', '_xgb.joblib')
        os.makedirs(os.path.dirname(xgb_save_path) or '.', exist_ok=True)
        joblib.dump(model, xgb_save_path)
        save_path = xgb_save_path
        
        # Evaluate
        from sklearn.metrics import accuracy_score, log_loss
        y_pred = model.predict(X_val_flat)
        y_prob = model.predict_proba(X_val_flat)[:, 1]
        val_acc = accuracy_score(y_val, y_pred)
        val_loss = log_loss(y_val, y_prob)
        history = {'loss': [], 'val_loss': [val_loss]}
    
    print(f"\nModel saved to {save_path}")
    print(f"\nFinal Validation:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {val_acc:.4f}")
    
    return {
        'history': history if use_lstm else {'val_loss': [val_loss]},
        'val_loss': val_loss,
        'val_accuracy': val_acc,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'model_path': save_path,
        'model_type': 'lstm' if use_lstm else 'xgboost'
    }


if __name__ == "__main__":
    # Quick test
    print("LSTM Trainer Module")
    print("Run: python -c 'from src.models.lstm.trainer import train_lstm'")
