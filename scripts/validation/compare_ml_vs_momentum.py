#!/usr/bin/env python3
"""
Full Validation: Pure Momentum ML vs Non-ML Momentum
=====================================================
Compares:
1. Pure Momentum ML (Experiment 7 winner) - XGBoost with momentum features + 10% stop-loss
2. Pure Momentum (Fama-French 12-1) - Currently in production

Period: 2017-2025 (8 years)
Retraining: Monthly (ML only)
Rebalancing: Weekly (both)
Risk Controls: 10% stop-loss (both)
"""

import matplotlib
matplotlib.use('Agg')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import warnings
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

from src.data.cache import DataCache

# =============================================================================
# CONFIGURATION
# =============================================================================

START_YEAR = 2017
END_YEAR = 2025
RETRAIN_WEEKS = 4  # Monthly retraining for ML
TOP_N = 10
SLIPPAGE_BPS = 10
INITIAL_CASH = 100000
STOP_LOSS_PCT = 0.10  # 10% stop-loss

# =============================================================================
# MOMENTUM FEATURES (from Experiment 7)
# =============================================================================

MOMENTUM_FEATURE_COLUMNS = [
    'MOM_12_1', 'MOM_6_1', 'MOM_3_1', 'MOM_1M',
    'ROC_20', 'ROC_60', 'TREND_STRENGTH',
    'RELATIVE_STRENGTH', 'VOLUME_TREND', 'VOLATILITY'
]


def generate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate momentum-based features."""
    df = df.copy()
    close = df['Close']
    volume = df['Volume']
    
    # Core momentum signals
    df['MOM_12_1'] = close.shift(21) / close.shift(252) - 1
    df['MOM_6_1'] = close.shift(21) / close.shift(126) - 1
    df['MOM_3_1'] = close.shift(21) / close.shift(63) - 1
    df['MOM_1M'] = close.pct_change(21)
    
    # Rate of change
    df['ROC_20'] = close.pct_change(20)
    df['ROC_60'] = close.pct_change(60)
    
    # Trend strength
    sma_200 = close.rolling(200).mean()
    df['TREND_STRENGTH'] = (close / sma_200) - 1
    
    # Relative strength (normalized)
    df['RELATIVE_STRENGTH'] = df['MOM_12_1'].rank(pct=True)
    
    # Volume trend
    vol_avg = volume.rolling(20).mean()
    vol_long_avg = volume.rolling(60).mean()
    df['VOLUME_TREND'] = vol_avg / vol_long_avg - 1
    
    # Volatility
    df['VOLATILITY'] = close.pct_change().rolling(20).std() * np.sqrt(252)
    
    return df.dropna()


def calculate_12_1_momentum(df: pd.DataFrame) -> Optional[float]:
    """Calculate 12-1 month momentum (Fama-French style)."""
    if len(df) < 252:
        return None
    close = df['Close']
    return (close.iloc[-21] / close.iloc[-252]) - 1


# =============================================================================
# ML MODEL TRAINING
# =============================================================================

def train_momentum_ml(train_data: Dict[str, pd.DataFrame]) -> Optional[Dict]:
    """Train XGBoost model with momentum features."""
    import xgboost as xgb
    
    all_features = []
    
    for ticker, df in train_data.items():
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
    
    feature_cols = [c for c in MOMENTUM_FEATURE_COLUMNS if c in full_df.columns]
    X = full_df[feature_cols].values
    y = full_df['Target'].values
    
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    valid = ~np.isnan(y)
    X, y = X[valid], y[valid]
    
    if len(X) < 1000:
        return None
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        verbosity=0
    )
    model.fit(X, y)
    
    return {'model': model, 'features': feature_cols}


def predict_ml(model_data: Dict, df: pd.DataFrame) -> float:
    """Generate ML prediction."""
    if model_data is None:
        return 0.0
    
    processed = generate_momentum_features(df)
    if processed.empty:
        return 0.0
    
    features = model_data['features']
    X = processed[features].tail(1).values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    return model_data['model'].predict(X)[0]


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_strategy(
    strategy_name: str,
    all_price_data: Dict,
    spy_data: pd.DataFrame,
    use_ml: bool = False
) -> Dict:
    """
    Run backtest for a strategy.
    
    Args:
        strategy_name: Name for logging
        all_price_data: Dict of ticker -> DataFrame
        spy_data: SPY benchmark data
        use_ml: True for ML momentum, False for simple momentum
    """
    print(f"\n{'='*70}")
    print(f"STRATEGY: {strategy_name}")
    print(f"{'='*70}")
    print(f"Period: {START_YEAR}-{END_YEAR}")
    print(f"Rebalancing: Weekly | Retraining: {'Monthly' if use_ml else 'N/A'}")
    print(f"Stop-Loss: {STOP_LOSS_PCT*100:.0f}%")
    
    # Get trading weeks
    sample_df = list(all_price_data.values())[0]
    all_dates = sorted(sample_df.index)
    all_dates = [d for d in all_dates 
                 if d >= pd.Timestamp(f'{START_YEAR}-01-01') 
                 and d < pd.Timestamp(f'{END_YEAR+1}-01-01')]
    mondays = [d for d in all_dates if d.weekday() == 0]
    
    print(f"Trading Weeks: {len(mondays)}")
    
    # Initialize
    portfolio_value = INITIAL_CASH
    weekly_returns = []
    portfolio_history = [(mondays[0], portfolio_value)]
    model = None
    positions = {}  # {ticker: entry_price}
    
    # Track yearly performance
    yearly_returns = {}
    current_year = None
    year_start_value = INITIAL_CASH
    
    for week_idx in range(len(mondays) - 1):
        monday = mondays[week_idx]
        next_monday = mondays[week_idx + 1]
        
        # Track yearly
        year = monday.year
        if year != current_year:
            if current_year is not None:
                yearly_returns[current_year] = (portfolio_value / year_start_value) - 1
            current_year = year
            year_start_value = portfolio_value
        
        # Retrain ML model periodically
        if use_ml and (week_idx % RETRAIN_WEEKS == 0):
            train_data = {t: df[df.index < monday] 
                         for t, df in all_price_data.items() 
                         if len(df[df.index < monday]) > 252}
            
            if len(train_data) >= 50:
                model = train_momentum_ml(train_data)
                if week_idx % 52 == 0:  # Log yearly
                    print(f"  {monday.strftime('%Y-%m')}: Retrained ML model")
        
        # Generate predictions/scores
        scores = {}
        
        for ticker, df in all_price_data.items():
            history = df[df.index <= monday]
            if len(history) < 252:
                continue
            
            if use_ml:
                if model is None:
                    continue
                scores[ticker] = predict_ml(model, history)
            else:
                # Simple 12-1 momentum
                mom = calculate_12_1_momentum(history)
                if mom is not None:
                    scores[ticker] = mom
        
        if len(scores) < TOP_N:
            continue
        
        # Select top N
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [t for t, _ in sorted_scores[:TOP_N]]
        
        # Calculate returns with stop-loss
        week_returns = []
        new_positions = {}
        
        for ticker in selected:
            df = all_price_data[ticker]
            df_period = df[(df.index >= monday) & (df.index < next_monday)]
            
            if len(df_period) < 1:
                continue
            
            entry = df_period['Open'].iloc[0] * (1 + SLIPPAGE_BPS/10000)
            
            # Check stop-loss during the week
            stopped_out = False
            for idx in range(len(df_period)):
                low = df_period['Low'].iloc[idx]
                if (low / entry - 1) < -STOP_LOSS_PCT:
                    exit_price = entry * (1 - STOP_LOSS_PCT)
                    exit_price *= (1 - SLIPPAGE_BPS/10000)
                    ret = (exit_price / entry) - 1
                    week_returns.append(ret)
                    stopped_out = True
                    break
            
            if stopped_out:
                continue
            
            # Normal exit
            df_next = df[df.index >= next_monday]
            if len(df_next) >= 1:
                exit_price = df_next['Open'].iloc[0] * (1 - SLIPPAGE_BPS/10000)
                ret = (exit_price / entry) - 1
                week_returns.append(ret)
                new_positions[ticker] = entry
        
        positions = new_positions
        
        if week_returns:
            avg_return = np.mean(week_returns)
            weekly_returns.append(avg_return)
            portfolio_value *= (1 + avg_return)
            portfolio_history.append((next_monday, portfolio_value))
    
    # Final year
    if current_year is not None:
        yearly_returns[current_year] = (portfolio_value / year_start_value) - 1
    
    # Calculate metrics
    if not weekly_returns:
        return {'name': strategy_name, 'error': 'No trades'}
    
    total_return = (portfolio_value / INITIAL_CASH) - 1
    years = (mondays[-1] - mondays[0]).days / 365.25
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
            spy_cagr = ((1 + spy_return) ** (1/years) - 1) if years > 0 else 0
        else:
            spy_return = 0
            spy_cagr = 0
    else:
        spy_return = 0
        spy_cagr = 0
    
    results = {
        'name': strategy_name,
        'total_return': total_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'volatility': vol,
        'win_rate': win_rate,
        'spy_return': spy_return,
        'spy_cagr': spy_cagr,
        'excess_return': total_return - spy_return,
        'excess_cagr': cagr - spy_cagr,
        'weeks_traded': len(weekly_returns),
        'years': years,
        'final_value': portfolio_value,
        'yearly_returns': yearly_returns,
        'portfolio_history': portfolio_history
    }
    
    print(f"\n📊 Results:")
    print(f"  Total Return: {total_return*100:+.1f}%")
    print(f"  CAGR: {cagr*100:+.2f}%")
    print(f"  Sharpe: {sharpe:.3f}")
    print(f"  Max Drawdown: {max_dd*100:.1f}%")
    print(f"  Volatility: {vol*100:.1f}%")
    print(f"  Win Rate: {win_rate*100:.1f}%")
    print(f"\n  SPY: {spy_return*100:+.1f}% total, {spy_cagr*100:+.2f}% CAGR")
    print(f"  Excess CAGR: {(cagr - spy_cagr)*100:+.2f}%")
    
    print(f"\n📅 Yearly Returns:")
    for year, ret in sorted(yearly_returns.items()):
        print(f"  {year}: {ret*100:+.1f}%")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("FULL VALIDATION: Pure Momentum ML vs Simple Momentum")
    print("="*70)
    print(f"Period: {START_YEAR}-{END_YEAR} ({END_YEAR - START_YEAR} years)")
    print(f"Rebalancing: Weekly | ML Retraining: Monthly")
    print(f"Stop-Loss: {STOP_LOSS_PCT*100:.0f}% | Top-N: {TOP_N}")
    
    # Load data
    print("\n📦 Loading data from cache...")
    cache = DataCache()
    all_tickers = cache.get_cached_tickers()
    
    all_price_data = {}
    for ticker in all_tickers:
        df = cache.get_price_data(ticker)
        if df is not None and len(df) > 500:
            df = df[df.index >= '2010-01-01']
            if len(df) > 500:
                all_price_data[ticker] = df.copy()
    
    spy_data = all_price_data.pop('SPY', None)
    print(f"Loaded {len(all_price_data)} tickers")
    
    # Run both strategies
    results = []
    
    # 1. Simple Momentum (Non-ML)
    simple_result = run_strategy(
        "Simple_Momentum_12-1",
        all_price_data, spy_data,
        use_ml=False
    )
    results.append(simple_result)
    
    # 2. ML Momentum
    ml_result = run_strategy(
        "ML_Momentum",
        all_price_data, spy_data,
        use_ml=True
    )
    results.append(ml_result)
    
    # Comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Strategy':<25} {'CAGR':>10} {'Sharpe':>10} {'Max DD':>10} {'Win Rate':>10}")
    print("-"*70)
    
    for r in results:
        if 'error' not in r:
            print(f"{r['name']:<25} {r['cagr']*100:>+9.2f}% {r['sharpe']:>+9.3f} "
                  f"{r['max_drawdown']*100:>9.1f}% {r['win_rate']*100:>9.1f}%")
    
    if spy_data is not None and 'error' not in results[0]:
        print(f"{'SPY (Benchmark)':<25} {results[0]['spy_cagr']*100:>+9.2f}%")
    
    print("-"*70)
    
    # Yearly comparison
    print(f"\n📅 Yearly CAGR Comparison:")
    print(f"{'Year':<8}", end="")
    for r in results:
        if 'error' not in r:
            print(f"{r['name'][:15]:>18}", end="")
    print()
    print("-"*50)
    
    all_years = set()
    for r in results:
        if 'error' not in r:
            all_years.update(r['yearly_returns'].keys())
    
    for year in sorted(all_years):
        print(f"{year:<8}", end="")
        for r in results:
            if 'error' not in r:
                ret = r['yearly_returns'].get(year, 0)
                print(f"{ret*100:>+17.1f}%", end="")
        print()
    
    # Winner
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        best = max(valid_results, key=lambda x: x['sharpe'])
        print(f"\n🏆 Best Risk-Adjusted (Sharpe): {best['name']}")
        print(f"   CAGR: {best['cagr']*100:+.2f}%, Sharpe: {best['sharpe']:.3f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    # Save to JSON (without portfolio history for size)
    save_results = []
    for r in results:
        if 'error' not in r:
            save_r = {k: v for k, v in r.items() if k != 'portfolio_history'}
            save_r['yearly_returns'] = {str(k): v for k, v in r['yearly_returns'].items()}
            save_results.append(save_r)
    
    with open('results/ml_vs_momentum_validation.json', 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to results/ml_vs_momentum_validation.json")
    
    return results


if __name__ == "__main__":
    main()

