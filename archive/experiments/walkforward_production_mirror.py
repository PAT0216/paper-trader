#!/usr/bin/env python3
"""
Production Mirror Walk-Forward Backtest
- Weekly retraining (as close to daily production as feasible)
- Full 508 ticker universe
- Noise-based feature selection via train_ensemble
- 1/5/20 day ensemble model
- 10 bps slippage
- Top 10 cross-sectional picks
- 2017-2025 timeframe
- SPY comparison

CHANGES LOG (for potential revert):
- 2024-12-16: Fixed missing daily return normalization bug (was not dividing by horizon)
- 2024-12-16: Added detailed DEBUG_MODE logging
- 2024-12-16: Added per-horizon prediction breakdown logging
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import warnings
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

from src.data.cache import DataCache
from src.models.trainer import train_ensemble, FEATURE_COLUMNS
from src.features.indicators import generate_features

# =============================================================================
# DEBUG CONFIGURATION
# =============================================================================
DEBUG_MODE = True  # Set to True for detailed logging
LOG_FILE = 'results/walkforward_production_mirror_debug.log'

# Setup logging
if DEBUG_MODE:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
else:
    logging.basicConfig(level=logging.INFO, format='%(message)s')

log = logging.getLogger(__name__)

# =============================================================================
# MAIN SCRIPT
# =============================================================================

log.info('='*70)
log.info('PRODUCTION MIRROR WALK-FORWARD (WEEKLY RETRAINING)')
log.info('='*70)

# Load full universe
cache = DataCache()
all_tickers = cache.get_cached_tickers()
log.info(f'Full universe: {len(all_tickers)} tickers')

# Load all price data from cache
log.info('Loading data from cache...')
all_price_data = {}
for ticker in all_tickers:
    df = cache.get_price_data(ticker)
    if df is not None and len(df) > 200:
        df = df[df.index >= '2010-01-01']
        if len(df) > 200:
            all_price_data[ticker] = df.copy()

log.info(f'Loaded {len(all_price_data)} tickers with data')

# Get SPY for benchmark
spy_data = all_price_data.pop('SPY', None)

# =============================================================================
# PARAMETERS - WEEKLY RETRAINING
# =============================================================================
TEST_MODE = False  # Full run with weekly retraining
START_YEAR = 2017  # Full backtest period
END_YEAR = 2025
RETRAIN_WEEKS = 1  # Weekly retraining (production mirror)
TOP_N = 10
SLIPPAGE_BPS = 10
INITIAL_CASH = 100000

# Ensemble horizon weights (MUST MATCH PRODUCTION)
HORIZON_WEIGHTS = {1: 0.5, 5: 0.3, 20: 0.2}

log.info(f'\n--- PARAMETERS ---')
log.info(f'TEST_MODE: {TEST_MODE}')
log.info(f'START_YEAR: {START_YEAR}')
log.info(f'END_YEAR: {END_YEAR}')
log.info(f'RETRAIN_WEEKS: {RETRAIN_WEEKS}')
log.info(f'TOP_N: {TOP_N}')
log.info(f'SLIPPAGE_BPS: {SLIPPAGE_BPS}')
log.info(f'HORIZON_WEIGHTS: {HORIZON_WEIGHTS}')

# Get trading weeks
sample_df = list(all_price_data.values())[0]
all_dates = sorted(sample_df.index)
all_dates = [d for d in all_dates if d >= pd.Timestamp(f'{START_YEAR}-01-01') and d < pd.Timestamp(f'{END_YEAR+1}-01-01')]
mondays = [d for d in all_dates if d.weekday() == 0]

log.info(f'Walk-forward: {mondays[0].date()} to {mondays[-1].date()}')
log.info(f'Total weeks: {len(mondays)} (WEEKLY retraining)')
log.info('')

# =============================================================================
# RUN WALK-FORWARD
# =============================================================================
portfolio_value = INITIAL_CASH
weekly_returns = []
portfolio_history = [(mondays[0], portfolio_value)]
ensemble = None
week_count = 0

# Track selected features over time
feature_selection_history = []

for week_idx in range(0, len(mondays)-1):
    monday = mondays[week_idx]
    next_monday = mondays[week_idx + 1]
    week_count += 1
    
    # =========================================================================
    # WEEKLY RETRAINING
    # =========================================================================
    if week_idx % RETRAIN_WEEKS == 0:
        train_data = {t: df[df.index < monday] for t, df in all_price_data.items() if len(df[df.index < monday]) > 100}
        
        # Progress every 13 weeks (quarterly summary)
        if week_count % 13 == 1 or week_idx == 0:
            ytd_return = (portfolio_value / INITIAL_CASH - 1) * 100
            log.info(f'Week {week_count} ({monday.date()}): Training... Portfolio=${portfolio_value:,.0f} (YTD: {ytd_return:+.1f}%)')
        
        if len(train_data) >= 50:
            try:
                log.debug(f'Training ensemble with {len(train_data)} tickers...')
                ensemble = train_ensemble(train_data, n_splits=3, save_model=False)
                
                # Log feature selection results
                if DEBUG_MODE and ensemble:
                    for horizon in ensemble.get('horizons', []):
                        selected = ensemble['selected_features'].get(horizon, [])
                        log.debug(f'  Horizon {horizon}d: {len(selected)} features selected')
                        log.debug(f'    Features: {selected[:5]}...')  # First 5
                    
                    feature_selection_history.append({
                        'date': monday,
                        'horizons': {h: len(ensemble['selected_features'].get(h, [])) 
                                   for h in ensemble.get('horizons', [])}
                    })
            except Exception as e:
                log.error(f'Training failed: {e}')
                pass
    
    if ensemble is None or not ensemble.get('models'):
        log.debug(f'Week {week_count}: No ensemble available, skipping')
        continue
    
    # =========================================================================
    # GENERATE PREDICTIONS
    # =========================================================================
    predictions = {}
    prediction_breakdown = {}  # For debugging
    
    for ticker, df in all_price_data.items():
        # Pass FULL history up to Monday so indicators can be calculated
        history = df[df.index <= monday]
        if len(history) < 50:  # Need enough data for indicators
            continue
        
        # Generate features on full history, then take LAST row only
        features_df = generate_features(history.copy(), include_target=False)
        if features_df.empty:
            continue
        features_df = features_df.tail(1)  # Get only the last row (monday's features)
        
        # Ensemble prediction with DAILY RETURN NORMALIZATION
        # FIX: Must divide by horizon to normalize to daily returns (as in EnsemblePredictor)
        total_pred = 0
        total_weight = 0
        ticker_breakdown = {}
        
        for horizon, model in ensemble['models'].items():
            selected = ensemble['selected_features'].get(horizon, FEATURE_COLUMNS)
            weight = ensemble['weights'].get(horizon, HORIZON_WEIGHTS.get(horizon, 0.33))
            
            try:
                feature_list = [f for f in selected if f in features_df.columns]
                if not feature_list:
                    continue
                X = features_df[feature_list].values
                X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
                raw_pred = model.predict(X)[0]
                
                # CRITICAL FIX: Normalize to daily return
                # Production EnsemblePredictor does: daily_pred = pred / horizon
                daily_pred = raw_pred / horizon
                
                total_pred += weight * daily_pred
                total_weight += weight
                
                ticker_breakdown[horizon] = {
                    'raw': raw_pred,
                    'daily': daily_pred,
                    'weight': weight,
                    'weighted': weight * daily_pred
                }
            except Exception as e:
                log.debug(f'Prediction failed for {ticker} horizon {horizon}: {e}')
                continue
        
        if total_weight > 0:
            final_pred = total_pred / total_weight
            predictions[ticker] = final_pred
            prediction_breakdown[ticker] = {
                'final': final_pred,
                'total_weight': total_weight,
                'horizons': ticker_breakdown
            }
    
    if len(predictions) < TOP_N:
        log.debug(f'Week {week_count}: Only {len(predictions)} predictions, need {TOP_N}')
        continue
    
    # =========================================================================
    # SELECT TOP N
    # =========================================================================
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    selected = [t for t, _ in sorted_preds[:TOP_N]]
    
    # Debug: Log top and bottom picks
    if DEBUG_MODE:
        log.debug(f'\n--- Week {week_count} ({monday.date()}) ---')
        log.debug(f'Top {TOP_N} picks:')
        for i, (ticker, pred) in enumerate(sorted_preds[:TOP_N]):
            breakdown = prediction_breakdown.get(ticker, {})
            horizon_strs = []
            for h in [1, 5, 20]:
                if h in breakdown.get('horizons', {}):
                    hd = breakdown['horizons'][h]
                    horizon_strs.append(f'{h}d:{hd["daily"]*100:+.2f}%')
            log.debug(f'  {i+1}. {ticker}: {pred*100:+.3f}% | {" | ".join(horizon_strs)}')
        
        log.debug(f'Bottom 3:')
        for ticker, pred in sorted_preds[-3:]:
            log.debug(f'  {ticker}: {pred*100:+.3f}%')
    
    # =========================================================================
    # CALCULATE WEEKLY RETURN
    # =========================================================================
    week_returns_list = []
    trade_details = []
    
    for ticker in selected:
        df = all_price_data[ticker]
        df_period = df[(df.index >= monday) & (df.index < next_monday)]
        if len(df_period) >= 1:
            entry = df_period['Open'].iloc[0] * (1 + SLIPPAGE_BPS/10000)
            df_next = df[df.index >= next_monday]
            if len(df_next) >= 1:
                exit_price = df_next['Open'].iloc[0] * (1 - SLIPPAGE_BPS/10000)
                ret = (exit_price / entry) - 1
                week_returns_list.append(ret)
                trade_details.append({
                    'ticker': ticker,
                    'entry': entry,
                    'exit': exit_price,
                    'return': ret,
                    'predicted': predictions.get(ticker, 0)
                })
    
    if week_returns_list:
        avg_return = np.mean(week_returns_list)
        weekly_returns.append(avg_return)
        portfolio_value *= (1 + avg_return)
        portfolio_history.append((next_monday, portfolio_value))
        
        # Log every week in TEST_MODE
        if TEST_MODE or DEBUG_MODE:
            winners = sum(1 for r in week_returns_list if r > 0)
            best_trade = max(trade_details, key=lambda x: x['return'])
            worst_trade = min(trade_details, key=lambda x: x['return'])
            log.info(f'  Week {week_count}: {len(week_returns_list)} trades, return={avg_return*100:+.2f}%, '
                    f'winners={winners}/{len(week_returns_list)}, portfolio=${portfolio_value:,.0f}')
            if DEBUG_MODE:
                log.debug(f'    Best:  {best_trade["ticker"]} {best_trade["return"]*100:+.2f}% (pred: {best_trade["predicted"]*100:+.2f}%)')
                log.debug(f'    Worst: {worst_trade["ticker"]} {worst_trade["return"]*100:+.2f}% (pred: {worst_trade["predicted"]*100:+.2f}%)')

# =============================================================================
# FINAL RESULTS
# =============================================================================
log.info('')
log.info('='*70)
log.info('FINAL RESULTS (WEEKLY RETRAINING)')
log.info('='*70)

if len(weekly_returns) > 0:
    total_return = (portfolio_value / INITIAL_CASH) - 1
    years = len(weekly_returns) / 52
    cagr = ((1 + total_return) ** (1/years) - 1) if years > 0 else 0
    vol = np.std(weekly_returns) * np.sqrt(52)
    sharpe = (np.mean(weekly_returns) * 52 - 0.04) / vol if vol > 0 else 0
    
    portfolio_df = pd.DataFrame(portfolio_history, columns=['date', 'value'])
    max_dd = (portfolio_df['value'] / portfolio_df['value'].cummax() - 1).min()
    win_rate = sum(1 for r in weekly_returns if r > 0) / len(weekly_returns)
    
    # SPY comparison
    if spy_data is not None:
        start_date, end_date = portfolio_df['date'].iloc[0], portfolio_df['date'].iloc[-1]
        spy_period = spy_data[(spy_data.index >= start_date) & (spy_data.index <= end_date)]
        if len(spy_period) > 1:
            spy_return = (spy_period['Close'].iloc[-1] / spy_period['Close'].iloc[0]) - 1
            spy_years = len(spy_period) / 252
            spy_cagr = ((1 + spy_return) ** (1/spy_years) - 1) if spy_years > 0 else 0
            spy_daily = spy_period['Close'].pct_change().dropna()
            spy_vol = spy_daily.std() * np.sqrt(252)
            spy_sharpe = (spy_cagr - 0.04) / spy_vol if spy_vol > 0 else 0
            spy_dd = (spy_period['Close'] / spy_period['Close'].cummax() - 1).min()
        else:
            spy_return = spy_cagr = spy_sharpe = spy_dd = 0
    else:
        spy_return = spy_cagr = spy_sharpe = spy_dd = 0
    
    log.info('')
    log.info(f"{'Metric':<25} {'MODEL':<15} {'SPY':<15} {'Delta':>10}")
    log.info('-'*65)
    log.info(f"{'Total Return':<25} {total_return*100:>+12.1f}% {spy_return*100:>+12.1f}% {(total_return-spy_return)*100:>+10.1f}%")
    log.info(f"{'CAGR':<25} {cagr*100:>+12.2f}% {spy_cagr*100:>+12.2f}% {(cagr-spy_cagr)*100:>+10.2f}%")
    log.info(f"{'Sharpe Ratio':<25} {sharpe:>+12.3f}  {spy_sharpe:>+12.3f}  {(sharpe-spy_sharpe):>+10.3f}")
    log.info(f"{'Max Drawdown':<25} {max_dd*100:>+12.1f}% {spy_dd*100:>+12.1f}% {(max_dd-spy_dd)*100:>+10.1f}%")
    log.info(f"{'Win Rate':<25} {win_rate*100:>+12.1f}%")
    log.info(f"{'Final Value ($100k)':<25} ${portfolio_value:>11,.0f}  ${INITIAL_CASH*(1+spy_return):>11,.0f}  ${portfolio_value-INITIAL_CASH*(1+spy_return):>+9,.0f}")
    log.info(f"{'Weeks Traded':<25} {len(weekly_returns)}")
    
    log.info('')
    log.info('='*70)
    if cagr > spy_cagr and sharpe > spy_sharpe:
        log.info('✅ MODEL BEATS SPY on both CAGR and Sharpe!')
    elif cagr > spy_cagr:
        log.info('✅ MODEL BEATS SPY on CAGR')
    elif sharpe > spy_sharpe:
        log.info('✅ MODEL BEATS SPY on Sharpe')
    else:
        log.info('❌ SPY outperforms MODEL')
    log.info('='*70)
    
    # Save results
    portfolio_df.to_csv('results/walkforward_production_mirror.csv', index=False)
    log.info('💾 Saved to results/walkforward_production_mirror.csv')
    
    if DEBUG_MODE:
        log.info(f'📋 Debug log saved to {LOG_FILE}')
        
        # Save feature selection history
        if feature_selection_history:
            fs_df = pd.DataFrame(feature_selection_history)
            fs_df.to_csv('results/walkforward_feature_selection.csv', index=False)
            log.info('📋 Feature selection history saved to results/walkforward_feature_selection.csv')
else:
    log.error('❌ No trades executed - check data')
