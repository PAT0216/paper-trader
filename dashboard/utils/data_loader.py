"""
Data loading utilities for Streamlit dashboard

Helper functions to load model/backtest/ledger data for visualization.
"""

import os
import pandas as pd
import joblib
import json
from typing import Dict, Optional, Tuple


def load_ledger() -> pd.DataFrame:
    """Load ledger.csv with trade history."""
    ledger_path = "ledger.csv"
    
    if not os.path.exists(ledger_path):
        return pd.DataFrame(columns=['date', 'ticker', 'action', 'quantity', 'price', 'cash', 'total_value'])
    
    df = pd.read_csv(ledger_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    return df


def load_model() -> Tuple[Optional[object], Optional[list]]:
    """
    Load trained model and selected features.
    
    Returns:
        (model, selected_features) tuple
    """
    model_path = "models/xgb_model.joblib"
    
    if not os.path.exists(model_path):
        return None, None
    
    model_data = joblib.load(model_path)
    
    if isinstance(model_data, dict):
        model = model_data.get('model')
        selected_features = model_data.get('selected_features', [])
    else:
        model = model_data
        selected_features = []
    
    return model, selected_features


def load_backtest_metrics() -> Dict:
    """Load backtest metrics from JSON."""
    metrics_path = "results/backtest_metrics.json"
    
    if not os.path.exists(metrics_path):
        return {}
   
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def load_backtest_trades() -> pd.DataFrame:
    """Load backtest trade history."""
    trades_path = "results/backtest_trades.csv"
    
    if not os.path.exists(trades_path):
        return pd.DataFrame()
    
    df = pd.read_csv(trades_path)
    if 'entry_date' in df.columns:
        df['entry_date'] = pd.to_datetime(df['entry_date'])
    if 'exit_date' in df.columns:
        df['exit_date'] = pd.to_datetime(df['exit_date'])
    
    return df


def load_equity_curve() -> Optional[pd.DataFrame]:
    """Load equity curve from backtest."""
    equity_path = "results/equity_curve.csv"
    
    if not os.path.exists(equity_path):
        return None
    
    df = pd.read_csv(equity_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    return df


def calculate_portfolio_stats(ledger_df: pd.DataFrame) -> Dict:
    """Calculate current portfolio statistics from ledger."""
    if ledger_df.empty:
        return {
            'total_value': 0,
            'cash': 0,
            'total_return_pct': 0,
            'num_trades': 0,
            'num_positions': 0
        }
    
    latest = ledger_df.iloc[-1]
    
    # Count trades (BUY/SELL actions)
    num_trades = len(ledger_df[ledger_df['action'].isin(['BUY', 'SELL'])])
    
    # Count current positions (approximate - would need position tracking)
    num_positions = 0  # TODO: implement position tracking
    
    # Calculate return if initial value is known
    if len(ledger_df) > 1:
        initial_value = ledger_df.iloc[0]['total_value']
        total_return_pct = ((latest['total_value'] - initial_value) / initial_value) * 100
    else:
        total_return_pct = 0
    
    return {
        'total_value': latest['total_value'],
        'cash': latest['cash'],
        'total_return_pct': total_return_pct,
        'num_trades': num_trades,
        'num_positions': num_positions
    }


def get_feature_importance(model, selected_features: list) -> pd.DataFrame:
    """Extract feature importance from XGBoost model."""
    if model is None or not hasattr(model, 'feature_importances_'):
        return pd.DataFrame()
    
    importance_df = pd.DataFrame({
        'feature': selected_features,
        'importance': model.feature_importances_
    })
    
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df
