import joblib
import pandas as pd
import os
from src.features.indicators import generate_features

MODEL_PATH = "models"
MODEL_FILE = os.path.join(MODEL_PATH, "xgb_model.joblib")

class Predictor:
    def __init__(self):
        self.model = None
        if os.path.exists(MODEL_FILE):
            self.model = joblib.load(MODEL_FILE)
            print("Loaded XGBoost model.")
        else:
            print("Warning: No model found. Run training first.")
    
    def predict(self, df):
        """
        Takes a raw dataframe (OHLC), processes features, and returns the last prediction probability.
        """
        if not self.model:
            return 0.5 # Neutral
            
        processed_df = generate_features(df)
        
        if processed_df.empty:
            return 0.5
            
        # Extract features for the latest date
        feature_cols = ['RSI', 'MACD', 'MACD_signal', 'MACD_hist', 
                        'BB_Width', 'Dist_SMA50', 'Dist_SMA200', 
                        'Return_1d', 'Return_5d']
        
        last_row = processed_df.iloc[[-1]][feature_cols]
        
        # Predict class (0 or 1)
        prediction = self.model.predict(last_row)[0]
        # Predict probability
        proba = self.model.predict_proba(last_row)[0][1] # Probability of class 1 (Up)
        
        return proba
