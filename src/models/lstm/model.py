"""
LSTM V4 Model Architecture

Conv1D + LSTM with Monte Carlo Dropout for uncertainty estimation.
Uses TensorFlow/Keras backend.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available - LSTM model will not work")


def build_lstm_model(
    sequence_length: int = 60,
    n_features: int = 12,
    mc_dropout_rate: float = 0.3
):
    """
    Build LSTM model with Monte Carlo Dropout.
    
    Architecture:
        Input -> Conv1D -> LSTM -> LSTM -> Dense -> MC Dropout -> Sigmoid
    
    Args:
        sequence_length: Number of time steps (days)
        n_features: Number of input features
        mc_dropout_rate: Dropout rate (active at inference for uncertainty)
    
    Returns:
        Compiled Keras model
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required for LSTM model")
    
    inputs = layers.Input(shape=(sequence_length, n_features))
    
    # 1D Convolution for local patterns (peaks/valleys)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    
    # Stacked LSTM for temporal dependencies
    x = layers.LSTM(128, return_sequences=True, dropout=0.2)(x)
    x = layers.LSTM(64, return_sequences=False, dropout=0.2)(x)
    
    # Dense layers
    x = layers.Dense(32, activation='relu')(x)
    
    # Monte Carlo Dropout (remains active at inference via training=True)
    x = layers.Dropout(mc_dropout_rate)(x, training=True)
    
    # Output: probability of exceeding threshold
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs, name='lstm_v4_threshold')
    
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_callbacks(patience: int = 15):
    """Get training callbacks."""
    if not TF_AVAILABLE:
        return []
    
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
    ]
