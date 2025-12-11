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
    
    # Count trades (BUY/SELL actions)
    num_trades = len(ledger_df[ledger_df['action'] == 'BUY'])
    
    # Get current cash
    latest = ledger_df.iloc[-1]
    cash = latest['cash_balance'] if 'cash_balance' in ledger_df.columns else latest.get('cash', 0)
    
    # Calculate positions (shares held per ticker)
    positions = {}
    for _, row in ledger_df.iterrows():
        ticker = row['ticker']
        action = row['action']
        
        if action == 'BUY':
            shares = row.get('shares', 0)
            price = row['price']
            if ticker not in positions:
                positions[ticker] = {'shares': 0, 'avg_price': 0}
            positions[ticker]['shares'] += shares
            # Simple average for demo (not weighted)
            positions[ticker]['avg_price'] = price
        elif action == 'SELL':
            shares = row.get('shares', 0)
            if ticker in positions:
                positions[ticker]['shares'] -= shares
    
    # Remove positions with 0 shares
    positions = {t: p for t, p in positions.items() if p['shares'] > 0}
    num_positions = len(positions)
    
    # Calculate portfolio value (cash + positions value)
    # Note: We'd need current prices to get real value, for now use entry prices
    position_value = sum(p['shares'] * p['avg_price'] for p in positions.values())
    total_value = cash + position_value
    
    # Get initial deposit
    initial_value = 10000  # Default
    deposit_rows = ledger_df[ledger_df['action'] == 'DEPOSIT']
    if not deposit_rows.empty:
        initial_value = deposit_rows.iloc[0]['amount']
    
    # Calculate return
    total_return_pct = ((total_value - initial_value) / initial_value) * 100 if initial_value > 0 else 0
    
    return {
        'total_value': total_value,
        'cash': cash,
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
