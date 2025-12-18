#!/usr/bin/env python3
"""
ML Model Improvement Experiments
================================
Systematic testing of various ML strategy improvements.

Experiments:
1. Signal Inversion - Buy bottom predictions instead of top
2. Momentum Features - Replace mean-reversion indicators with momentum
3. Horizon Reweighting - Weight longer horizons more heavily
4. ML + Momentum Hybrid - Combine ML ranking with momentum filter
5. Risk Controls - Add stop-loss and diversification

Each experiment runs a 1-year walkforward (2024) for quick iteration.
Best approaches get full 2017-2025 validation.
"""

# Disable matplotlib to avoid slow font cache building
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import warnings
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

warnings.filterwarnings('ignore')

from src.data.cache import DataCache
from src.features.indicators import generate_features, FEATURE_COLUMNS

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for each experiment."""
    name: str
    start_year: int = 2024
    end_year: int = 2025
    retrain_weeks: int = 4  # Monthly for faster iteration
    top_n: int = 10
    slippage_bps: int = 10
    initial_cash: float = 100000
    
    # Experiment-specific settings
    invert_signal: bool = False
    use_momentum_features: bool = False
    horizon_weights: Dict[int, float] = None
    use_momentum_filter: bool = False
    momentum_threshold: float = 0.0  # Minimum 12-1 momentum to buy
    use_stop_loss: bool = False
    stop_loss_pct: float = 0.08
    max_sector_pct: float = 1.0  # 1.0 = no limit
    min_stocks: int = 10  # Minimum diversification
    
    def __post_init__(self):
        if self.horizon_weights is None:
            self.horizon_weights = {1: 0.5, 5: 0.3, 20: 0.2}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data() -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Load all price data from cache."""
    print("Loading data from cache...")
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


# =============================================================================
# MOMENTUM FEATURES (NEW - replaces mean-reversion)
# =============================================================================

MOMENTUM_FEATURE_COLUMNS = [
    'MOM_12_1',      # 12-1 month momentum (academic standard)
    'MOM_6_1',       # 6-1 month momentum
    'MOM_3_1',       # 3-1 month momentum
    'MOM_1M',        # 1 month momentum
    'ROC_20',        # 20-day rate of change
    'ROC_60',        # 60-day rate of change
    'TREND_STRENGTH', # Above/below 200 SMA
    'RELATIVE_STRENGTH', # Normalized rank
    'VOLUME_TREND',  # Volume vs average
    'VOLATILITY',    # 20-day volatility (for sizing)
]


def generate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate momentum-based features instead of mean-reversion indicators.
    
    Academic basis:
    - Jegadeesh & Titman (1993): 12-1 month momentum
    - Fama-French: Momentum factor
    - Asness et al: Value and Momentum everywhere
    """
    df = df.copy()
    close = df['Close']
    volume = df['Volume']
    
    # Core momentum signals (look-back only, no look-ahead)
    df['MOM_12_1'] = close.shift(21) / close.shift(252) - 1  # Skip last month
    df['MOM_6_1'] = close.shift(21) / close.shift(126) - 1
    df['MOM_3_1'] = close.shift(21) / close.shift(63) - 1
    df['MOM_1M'] = close.pct_change(21)
    
    # Rate of change
    df['ROC_20'] = close.pct_change(20)
    df['ROC_60'] = close.pct_change(60)
    
    # Trend strength (distance from 200 SMA)
    sma_200 = close.rolling(200).mean()
    df['TREND_STRENGTH'] = (close / sma_200) - 1
    
    # Relative strength (normalized)
    df['RELATIVE_STRENGTH'] = df['MOM_12_1'].rank(pct=True)
    
    # Volume trend
    vol_avg = volume.rolling(20).mean()
    vol_long_avg = volume.rolling(60).mean()
    df['VOLUME_TREND'] = vol_avg / vol_long_avg - 1
    
    # Volatility (for risk adjustment)
    df['VOLATILITY'] = close.pct_change().rolling(20).std() * np.sqrt(252)
    
    return df.dropna()


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_momentum_model(train_data: Dict[str, pd.DataFrame], config: ExperimentConfig) -> Optional[Dict]:
    """
    Train model using momentum features.
    """
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    
    all_features = []
    
    for ticker, df in train_data.items():
        # Generate momentum features
        processed = generate_momentum_features(df)
        if len(processed) < 100:
            continue
        
        # Target: Next week return (5-day forward)
        processed['Target'] = processed['Close'].pct_change(5).shift(-5)
        processed = processed.dropna()
        
        if len(processed) > 50:
            all_features.append(processed)
    
    if not all_features:
        return None
    
    full_df = pd.concat(all_features).sort_index()
    
    # Prepare X, y
    feature_cols = [c for c in MOMENTUM_FEATURE_COLUMNS if c in full_df.columns]
    X = full_df[feature_cols].values
    y = full_df['Target'].values
    
    # Clean
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    valid = ~np.isnan(y)
    X, y = X[valid], y[valid]
    
    if len(X) < 1000:
        return None
    
    # Train simple model
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        verbosity=0
    )
    model.fit(X, y)
    
    return {
        'model': model,
        'features': feature_cols,
        'type': 'momentum'
    }


def train_standard_ensemble(train_data: Dict[str, pd.DataFrame], config: ExperimentConfig) -> Optional[Dict]:
    """
    Train standard ensemble (existing approach) but with configurable weights.
    """
    from src.models.trainer import train_ensemble
    
    try:
        ensemble = train_ensemble(train_data, n_splits=3, save_model=False)
        if ensemble:
            # Override weights if specified
            ensemble['weights'] = config.horizon_weights
        return ensemble
    except:
        return None


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def predict_momentum_model(model_data: Dict, df: pd.DataFrame) -> float:
    """Generate prediction using momentum model."""
    if model_data is None or 'model' not in model_data:
        return 0.0
    
    processed = generate_momentum_features(df)
    if processed.empty:
        return 0.0
    
    features = model_data['features']
    X = processed[features].tail(1).values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    return model_data['model'].predict(X)[0]


def predict_standard_ensemble(ensemble: Dict, df: pd.DataFrame, config: ExperimentConfig) -> float:
    """Generate prediction using standard ensemble with custom weights."""
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
            daily_pred = raw_pred / horizon
            total_pred += weight * daily_pred
            total_weight += weight
        except:
            continue
    
    return total_pred / total_weight if total_weight > 0 else 0.0


# =============================================================================
# MOMENTUM FILTER
# =============================================================================

def calculate_12_1_momentum(df: pd.DataFrame) -> Optional[float]:
    """Calculate 12-1 month momentum for filtering."""
    if len(df) < 252:
        return None
    close = df['Close']
    return (close.iloc[-21] / close.iloc[-252]) - 1


# =============================================================================
# WALKFORWARD ENGINE
# =============================================================================

def run_experiment(config: ExperimentConfig, all_price_data: Dict, spy_data: pd.DataFrame) -> Dict:
    """
    Run a single experiment with given configuration.
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {config.name}")
    print(f"{'='*60}")
    print(f"Period: {config.start_year}-{config.end_year}")
    print(f"Settings: invert={config.invert_signal}, momentum_features={config.use_momentum_features}")
    print(f"Horizon weights: {config.horizon_weights}")
    
    # Get trading weeks
    sample_df = list(all_price_data.values())[0]
    all_dates = sorted(sample_df.index)
    all_dates = [d for d in all_dates 
                 if d >= pd.Timestamp(f'{config.start_year}-01-01') 
                 and d < pd.Timestamp(f'{config.end_year+1}-01-01')]
    mondays = [d for d in all_dates if d.weekday() == 0]
    
    print(f"Weeks: {len(mondays)}")
    
    # Run walkforward
    portfolio_value = config.initial_cash
    weekly_returns = []
    portfolio_history = [(mondays[0], portfolio_value)]
    model = None
    
    # Track positions for stop-loss
    positions = {}  # {ticker: entry_price}
    
    for week_idx in range(len(mondays) - 1):
        monday = mondays[week_idx]
        next_monday = mondays[week_idx + 1]
        
        # Retrain periodically
        if week_idx % config.retrain_weeks == 0:
            train_data = {t: df[df.index < monday] 
                         for t, df in all_price_data.items() 
                         if len(df[df.index < monday]) > 100}
            
            if len(train_data) >= 50:
                if config.use_momentum_features:
                    model = train_momentum_model(train_data, config)
                else:
                    model = train_standard_ensemble(train_data, config)
        
        if model is None:
            continue
        
        # Generate predictions
        predictions = {}
        momentum_scores = {}  # For filtering
        
        for ticker, df in all_price_data.items():
            history = df[df.index <= monday]
            if len(history) < 50:
                continue
            
            # Get prediction
            if config.use_momentum_features:
                pred = predict_momentum_model(model, history)
            else:
                pred = predict_standard_ensemble(model, history, config)
            
            predictions[ticker] = pred
            
            # Calculate momentum for filtering
            if config.use_momentum_filter:
                mom = calculate_12_1_momentum(history)
                if mom is not None:
                    momentum_scores[ticker] = mom
        
        if len(predictions) < config.top_n:
            continue
        
        # Sort predictions
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        # EXPERIMENT: Signal inversion
        if config.invert_signal:
            sorted_preds = sorted_preds[::-1]  # Reverse order
        
        # Select candidates
        candidates = []
        for ticker, pred in sorted_preds:
            # Apply momentum filter if enabled
            if config.use_momentum_filter:
                mom = momentum_scores.get(ticker, -999)
                if mom < config.momentum_threshold:
                    continue  # Skip stocks with negative momentum
            
            candidates.append((ticker, pred))
            if len(candidates) >= config.top_n * 2:  # Get extra for diversification
                break
        
        # Diversification: limit per sector (simplified - just take top N)
        selected = [t for t, _ in candidates[:config.top_n]]
        
        if len(selected) < config.min_stocks // 2:
            continue
        
        # Calculate returns with stop-loss
        week_returns = []
        new_positions = {}
        
        for ticker in selected:
            df = all_price_data[ticker]
            df_period = df[(df.index >= monday) & (df.index < next_monday)]
            
            if len(df_period) < 1:
                continue
            
            entry = df_period['Open'].iloc[0] * (1 + config.slippage_bps/10000)
            
            # Check stop-loss during the week
            if config.use_stop_loss:
                stopped_out = False
                for idx in range(len(df_period)):
                    low = df_period['Low'].iloc[idx]
                    if (low / entry - 1) < -config.stop_loss_pct:
                        # Stop-loss triggered - exit at stop price
                        exit_price = entry * (1 - config.stop_loss_pct)
                        exit_price *= (1 - config.slippage_bps/10000)
                        ret = (exit_price / entry) - 1
                        week_returns.append(ret)
                        stopped_out = True
                        break
                
                if stopped_out:
                    continue
            
            # Normal exit
            df_next = df[df.index >= next_monday]
            if len(df_next) >= 1:
                exit_price = df_next['Open'].iloc[0] * (1 - config.slippage_bps/10000)
                ret = (exit_price / entry) - 1
                week_returns.append(ret)
                new_positions[ticker] = entry
        
        positions = new_positions
        
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
    sharpe = (np.mean(weekly_returns) * 52 - 0.04) / vol if vol > 0 else 0
    
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
        'excess_return': total_return - spy_return,
        'weeks_traded': len(weekly_returns),
        'final_value': portfolio_value
    }
    
    print(f"\nResults:")
    print(f"  Total Return: {total_return*100:+.1f}% vs SPY {spy_return*100:+.1f}%")
    print(f"  CAGR: {cagr*100:+.2f}%")
    print(f"  Sharpe: {sharpe:.3f}")
    print(f"  Max DD: {max_dd*100:.1f}%")
    print(f"  Win Rate: {win_rate*100:.1f}%")
    
    return results


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def run_all_experiments():
    """Run all experiments and compare results."""
    
    # Load data once
    all_price_data, spy_data = load_data()
    
    # Define experiments
    experiments = [
        # Experiment 0: Baseline (current broken approach)
        ExperimentConfig(
            name="0_Baseline_Current",
            horizon_weights={1: 0.5, 5: 0.3, 20: 0.2}
        ),
        
        # Experiment 1: Signal Inversion (quick test)
        ExperimentConfig(
            name="1_Signal_Inversion",
            invert_signal=True,
            horizon_weights={1: 0.5, 5: 0.3, 20: 0.2}
        ),
        
        # Experiment 2: Momentum Features (replace mean-reversion)
        ExperimentConfig(
            name="2_Momentum_Features",
            use_momentum_features=True
        ),
        
        # Experiment 3: Long Horizon Weighting
        ExperimentConfig(
            name="3_Long_Horizon_Weight",
            horizon_weights={1: 0.1, 5: 0.3, 20: 0.6}
        ),
        
        # Experiment 4: ML + Momentum Filter (hybrid)
        ExperimentConfig(
            name="4_ML_Momentum_Hybrid",
            use_momentum_filter=True,
            momentum_threshold=0.0,  # Only buy stocks with positive 12-1 momentum
            horizon_weights={1: 0.3, 5: 0.3, 20: 0.4}
        ),
        
        # Experiment 5: Risk Controls (stop-loss)
        ExperimentConfig(
            name="5_With_StopLoss",
            use_stop_loss=True,
            stop_loss_pct=0.08,
            horizon_weights={1: 0.5, 5: 0.3, 20: 0.2}
        ),
        
        # Experiment 6: Combined Best (invert + momentum filter + stop-loss)
        ExperimentConfig(
            name="6_Combined_Improvements",
            invert_signal=True,
            use_momentum_filter=True,
            momentum_threshold=0.0,
            use_stop_loss=True,
            stop_loss_pct=0.08,
            horizon_weights={1: 0.2, 5: 0.3, 20: 0.5}
        ),
        
        # Experiment 7: Pure Momentum ML
        ExperimentConfig(
            name="7_Pure_Momentum_ML",
            use_momentum_features=True,
            use_stop_loss=True,
            stop_loss_pct=0.10
        ),
    ]
    
    # Run all experiments
    results = []
    for config in experiments:
        try:
            result = run_experiment(config, all_price_data, spy_data)
            results.append(result)
        except Exception as e:
            print(f"Experiment {config.name} failed: {e}")
            results.append({'name': config.name, 'error': str(e)})
    
    # Summary table
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print(f"{'Experiment':<30} {'Return':>10} {'vs SPY':>10} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8}")
    print("-"*80)
    
    for r in results:
        if 'error' in r:
            print(f"{r['name']:<30} ERROR: {r['error']}")
        else:
            print(f"{r['name']:<30} {r['total_return']*100:>+9.1f}% {r['excess_return']*100:>+9.1f}% "
                  f"{r['sharpe']:>+7.2f} {r['max_drawdown']*100:>7.1f}% {r['win_rate']*100:>7.1f}%")
    
    print("-"*80)
    
    # Find best experiment
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        best_sharpe = max(valid_results, key=lambda x: x['sharpe'])
        best_return = max(valid_results, key=lambda x: x['total_return'])
        
        print(f"\n🏆 Best Sharpe: {best_sharpe['name']} (Sharpe={best_sharpe['sharpe']:.3f})")
        print(f"🏆 Best Return: {best_return['name']} (Return={best_return['total_return']*100:+.1f}%)")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/ml_experiments_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Results saved to results/ml_experiments_results.json")
    
    return results


if __name__ == "__main__":
    run_all_experiments()

