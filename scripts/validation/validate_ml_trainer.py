#!/usr/bin/env python3
"""
Replication Test: Does ml_improvement_experiments.py match walkforward_fair_comparison.py?

Uses EXACT same config:
- 2017-2025 (8 years)
- Monthly retrain (every 4 weeks)  
- Weekly rebalance
- Full ~505 tickers
- Normalized predictions (divide by horizon)

Only runs baseline experiment to compare with fair_comparison results.
"""

# Disable matplotlib to avoid slow font cache building
import matplotlib
matplotlib.use('Agg')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import warnings
import json
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass

warnings.filterwarnings('ignore')

from src.data.cache import DataCache
from src.features.indicators import generate_features, FEATURE_COLUMNS

# =============================================================================
# CONFIGURATION - MATCHING FAIR COMPARISON
# =============================================================================

@dataclass
class ExperimentConfig:
    name: str
    start_year: int = 2024  # Quick test - change back to 2017 for production
    end_year: int = 2025
    retrain_weeks: int = 4  # Monthly
    top_n: int = 10
    slippage_bps: int = 0  # No slippage for fair comparison
    initial_cash: float = 100000
    horizon_weights: Dict[int, float] = None
    
    def __post_init__(self):
        if self.horizon_weights is None:
            self.horizon_weights = {1: 0.5, 5: 0.3, 20: 0.2}


def load_data():
    """Load all price data from cache."""
    print("📦 Loading data from cache...")
    cache = DataCache()
    all_tickers = cache.get_cached_tickers()
    
    all_price_data = {}
    for ticker in all_tickers:
        df = cache.get_price_data(ticker)
        if df is not None and len(df) > 200:
            df = df[df.index >= '2010-01-01']
            if len(df) > 200:
                all_price_data[ticker] = df.copy()
    
    spy_data = all_price_data.pop('SPY', None)
    print(f"Loaded {len(all_price_data)} tickers")
    return all_price_data, spy_data


def train_standard_ensemble(train_data: Dict[str, pd.DataFrame], config: ExperimentConfig):
    """Train standard ensemble using trainer.py."""
    from src.models.trainer import train_ensemble
    
    try:
        ensemble = train_ensemble(train_data, n_splits=3, save_model=False)
        if ensemble:
            ensemble['weights'] = config.horizon_weights
        return ensemble
    except Exception as e:
        print(f"Training error: {e}")
        return None


def predict_standard_ensemble(ensemble: Dict, df: pd.DataFrame, config: ExperimentConfig) -> float:
    """Generate prediction using standard ensemble with NORMALIZATION."""
    if ensemble is None or not ensemble.get('models'):
        return 0.0
    
    features_df = generate_features(df.copy(), include_target=False)
    if features_df.empty:
        return 0.0
    features_df = features_df.tail(1)
    
    total_pred = 0
    total_weight = 0
    
    for horizon, model in ensemble['models'].items():
        selected = ensemble['selected_features'].get(horizon, FEATURE_COLUMNS)
        weight = config.horizon_weights.get(horizon, 0.33)
        
        try:
            feature_list = [f for f in selected if f in features_df.columns]
            if not feature_list:
                continue
            X = features_df[feature_list].values
            X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
            raw_pred = model.predict(X)[0]
            
            # NORMALIZE to daily return (same as fair comparison)
            daily_pred = raw_pred / horizon
            total_pred += weight * daily_pred
            total_weight += weight
        except:
            continue
    
    return total_pred / total_weight if total_weight > 0 else 0.0


def run_experiment(config: ExperimentConfig, all_price_data: Dict, spy_data: pd.DataFrame):
    """Run baseline experiment with 2017-2025 period."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {config.name}")
    print(f"{'='*70}")
    print(f"Period: {config.start_year}-{config.end_year}")
    print(f"Retrain: Every {config.retrain_weeks} weeks")
    print(f"Universe: {len(all_price_data)} tickers")
    
    # Get trading weeks
    sample_df = list(all_price_data.values())[0]
    all_dates = sorted(sample_df.index)
    all_dates = [d for d in all_dates 
                 if d >= pd.Timestamp(f'{config.start_year}-01-01') 
                 and d < pd.Timestamp(f'{config.end_year+1}-01-01')]
    mondays = [d for d in all_dates if d.weekday() == 0]
    
    print(f"Total weeks: {len(mondays)}")
    
    portfolio_value = config.initial_cash
    weekly_returns = []
    portfolio_history = [(mondays[0], portfolio_value)]
    model = None
    retrain_count = 0
    
    for week_idx in range(len(mondays) - 1):
        monday = mondays[week_idx]
        next_monday = mondays[week_idx + 1]
        
        # Retrain every 4 weeks
        if week_idx % config.retrain_weeks == 0:
            train_data = {t: df[df.index < monday] 
                         for t, df in all_price_data.items() 
                         if len(df[df.index < monday]) > 100}
            
            if len(train_data) >= 50:
                model = train_standard_ensemble(train_data, config)
                retrain_count += 1
                
                if retrain_count % 12 == 0:
                    print(f"  Retrain #{retrain_count}: {monday.date()}")
        
        if model is None:
            continue
        
        # Generate predictions
        predictions = {}
        for ticker, df in all_price_data.items():
            history = df[df.index <= monday]
            if len(history) < 50:
                continue
            
            pred = predict_standard_ensemble(model, history, config)
            predictions[ticker] = pred
        
        if len(predictions) < config.top_n:
            continue
        
        # Sort and select top N
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        selected = [t for t, _ in sorted_preds[:config.top_n]]
        
        # Calculate returns - MATCHING fair_comparison.py
        # Uses Close prices within the same week (not Open across weeks)
        week_returns = []
        for ticker in selected:
            df = all_price_data[ticker]
            df_period = df[(df.index >= monday) & (df.index < next_monday)]
            
            if len(df_period) < 2:  # Need at least 2 days for Close-to-Close
                continue
            
            # Match fair_comparison: Close at end / Close at start
            entry = df_period['Close'].iloc[0]
            exit_price = df_period['Close'].iloc[-1]
            ret = (exit_price / entry) - 1
            week_returns.append(ret)
        
        if week_returns:
            avg_return = np.mean(week_returns)
            weekly_returns.append(avg_return)
            portfolio_value *= (1 + avg_return)
            portfolio_history.append((next_monday, portfolio_value))
    
    # Calculate metrics
    if not weekly_returns:
        return {'name': config.name, 'error': 'No trades'}
    
    total_return = (portfolio_value / config.initial_cash) - 1
    years = len(weekly_returns) / 52
    cagr = ((1 + total_return) ** (1/years) - 1) if years > 0 else 0
    vol = np.std(weekly_returns) * np.sqrt(52)
    sharpe = (np.mean(weekly_returns) * 52) / vol if vol > 0 else 0
    
    # Max drawdown
    values = [v for _, v in portfolio_history]
    peak = values[0]
    max_dd = 0
    for v in values:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd
    
    win_rate = sum(1 for r in weekly_returns if r > 0) / len(weekly_returns)
    
    # SPY comparison
    if spy_data is not None:
        start_date = portfolio_history[0][0]
        end_date = portfolio_history[-1][0]
        spy_period = spy_data[(spy_data.index >= start_date) & (spy_data.index <= end_date)]
        if len(spy_period) > 1:
            spy_return = (spy_period['Close'].iloc[-1] / spy_period['Close'].iloc[0]) - 1
        else:
            spy_return = 0
    else:
        spy_return = 0
    
    results = {
        'name': config.name,
        'total_return': total_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'spy_return': spy_return,
        'weeks_traded': len(weekly_returns),
        'retrains': retrain_count,
        'final_value': portfolio_value
    }
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Total Return: {total_return*100:+.1f}%")
    print(f"CAGR: {cagr*100:+.2f}%")
    print(f"Sharpe: {sharpe:.3f}")
    print(f"Max DD: {max_dd*100:.1f}%")
    print(f"Win Rate: {win_rate*100:.1f}%")
    print(f"Final Value: ${portfolio_value:,.2f}")
    print(f"SPY Return: {spy_return*100:+.1f}%")
    
    return results


def main():
    print("="*70)
    print("REPLICATION TEST: ml_improvement_experiments.py baseline")
    print("Config: 2017-2025, monthly retrain, weekly rebalance, normalized")
    print("="*70)
    print()
    
    # Load data
    all_price_data, spy_data = load_data()
    
    # Run baseline experiment with 2017-2025
    config = ExperimentConfig(
        name="Baseline_2017_2025",
        start_year=2017,  # Production setting
        end_year=2025,
        retrain_weeks=4,  # Monthly
        horizon_weights={1: 0.5, 5: 0.3, 20: 0.2}
    )
    
    result = run_experiment(config, all_price_data, spy_data)
    
    # Save results
    with open('results/ml_experiments_replication.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print()
    print(f"💾 Results saved to results/ml_experiments_replication.json")
    
    # Compare with fair comparison
    print()
    print("="*70)
    print("COMPARISON WITH FAIR_COMPARISON RESULTS")
    print("="*70)
    
    try:
        with open('results/walkforward_fair_comparison.json', 'r') as f:
            fair = json.load(f)
        
        print(f"{'Metric':<20} {'This Script':<15} {'Fair Comparison':<15} {'Diff'}")
        print("-"*65)
        print(f"{'CAGR':<20} {result['cagr']*100:>+.2f}%          {fair['cagr']*100:>+.2f}%          {(result['cagr']-fair['cagr'])*100:+.2f}%")
        print(f"{'Sharpe':<20} {result['sharpe']:>+.3f}          {fair['sharpe']:>+.3f}          {result['sharpe']-fair['sharpe']:+.3f}")
        print(f"{'Max DD':<20} {result['max_drawdown']*100:>.1f}%          {fair['max_drawdown']*100:>.1f}%          {(result['max_drawdown']-fair['max_drawdown'])*100:+.1f}%")
        print(f"{'Win Rate':<20} {result['win_rate']*100:>.1f}%          {fair['win_rate']*100:>.1f}%          {(result['win_rate']-fair['win_rate'])*100:+.1f}%")
    except Exception as e:
        print(f"Could not load fair comparison: {e}")


if __name__ == "__main__":
    main()
