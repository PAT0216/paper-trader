from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from src.features.indicators import generate_features
import os
import joblib
import pandas as pd
import xgboost as xgb

MODEL_PATH = "models"
MODEL_FILE = os.path.join(MODEL_PATH, "xgb_model.joblib")

def train_model(data_dict, test_size=0.2):
    """
    Trains a single XGBoost model on combined data from all tickers.
    """
    print("Preparing training data...")
    all_features = []
    
    for ticker, df in data_dict.items():
        processed_df = generate_features(df)
        processed_df['Ticker'] = ticker # Optional: if we want to use ticker as categorical, but maybe overfits
        # We drop Ticker for now to make a general model
        processed_df = processed_df.drop(columns=['Ticker'])
        all_features.append(processed_df)
        
    if not all_features:
        print("No data to train on.")
        return None
        
    full_df = pd.concat(all_features)
    
    # Define features and target
    target = 'Target'
    # Exclude columns that are not features (like OHLC unless we normalize them)
    # We use only the computed ratios and indicators
    feature_cols = ['RSI', 'MACD', 'MACD_signal', 'MACD_hist', 
                    'BB_Width', 'Dist_SMA50', 'Dist_SMA200', 
                    'Return_1d', 'Return_5d']
    
    X = full_df[feature_cols]
    y = full_df[target]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    print(f"Training XGBoost on {len(X_train)} samples...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {acc:.4f}")
    report = classification_report(y_test, preds)
    print(report)
    
    # Save Metrics
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
        f.write(f"Model Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        
    # Plot Confusion Matrix
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()
    
    # Save Model
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")
    print(f"Metrics saved to {results_dir}/metrics.txt & confusion_matrix.png")
    
    return model
