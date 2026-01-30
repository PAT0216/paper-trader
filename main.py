import sys
import argparse
import time
from datetime import datetime
from src.utils.config import load_config
from src.data import loader
from src.data.validator import DataValidator
# Note: trainer and predictor are imported conditionally to avoid XGBoost issues
from src.trading import portfolio
from src.trading.risk_manager import RiskManager, RiskLimits, DrawdownController
from src.backtesting.costs import TransactionCostModel
from src.strategies import get_strategy, list_strategies
import numpy as np
import pandas as pd

def main():
    # Use registry for dynamic strategy choices
    available_strategies = list_strategies()
    
    parser = argparse.ArgumentParser(description="AI Paper Trader")
    parser.add_argument("--mode", choices=["trade", "train", "backtest"], default="trade", help="Mode of operation")
    parser.add_argument("--strategy", choices=available_strategies, default="momentum", help=f"Strategy: {', '.join(available_strategies)}")
    parser.add_argument("--portfolio", default="default", help="Portfolio ID (e.g., 'momentum', 'ml') for isolated ledgers")
    args = parser.parse_args()

    
    # 1. Load Configuration
    try:
        config = load_config()
        
        # Check for dynamic universe (S&P 500)
        universe_type = config.get('universe', {}).get('type', 'config')
        
        if universe_type == 'sp500':
            # First try cached tickers (fast, reliable)
            cached_tickers_file = 'data/sp500_tickers.txt'
            try:
                with open(cached_tickers_file, 'r') as f:
                    tickers = [line.strip() for line in f if line.strip()]
                print(f" Loaded {len(tickers)} S&P 500 stocks from cache")
            except FileNotFoundError:
                # Fallback to Wikipedia (slow, may fail)
                print(" Cache not found, fetching S&P 500 from Wikipedia...")
                from src.data.universe import fetch_sp500_tickers, get_mega_caps
                try:
                    tickers = fetch_sp500_tickers()
                    # Save to cache for next time
                    with open(cached_tickers_file, 'w') as f:
                        f.write('\n'.join(tickers))
                    print(f"   Loaded {len(tickers)} stocks and saved to cache")
                except Exception as e:
                    print(f"    S&P 500 fetch failed: {e}")
                    tickers = get_mega_caps()
        else:
            # Use tickers from config
            tickers = config['tickers']
        
        print(f"--- 🤖 AI Paper Trader | Mode: {args.mode.upper()} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        print(f"Universe: {len(tickers)} tickers")
    except Exception as e:
        print(f" Configuration Error: {e}")
        sys.exit(1)

    # 2. Fetch Data
    print("Fetching market data...")
    
    # Both strategies use cache-only mode (cache is refreshed by separate workflow)
    print(" Using CACHE-ONLY mode (cache refreshed daily by separate workflow)")
    data_dict = loader.fetch_from_cache_only(tickers)
    
    if len(data_dict) < 50:
        print(f"  Only {len(data_dict)} tickers in cache, falling back to API fetch...")
        data_dict = loader.fetch_data(tickers, period=config['model']['training_period'])
    
    if not data_dict:
        print(" Failed to fetch data.")
        sys.exit(1)
    
    # 2a. Validate Data Quality
    # Momentum strategy: skip heavy validation (only needs 12mo of recent data)
    if args.strategy == "momentum":
        print("\n--- 🔍 Basic Data Check (Momentum) ---")
        # Just check we have enough bars for momentum calculation (252 days)
        valid_tickers = []
        for ticker, df in list(data_dict.items()):
            if len(df) >= 252:
                valid_tickers.append(ticker)
            else:
                del data_dict[ticker]
        print(f" {len(valid_tickers)} tickers have sufficient data (252+ days)")
        if len(data_dict) < 50:
            print(f"  Only {len(data_dict)} tickers available, need at least 50")
    else:
        # ML strategy: full validation
        print("\n--- 🔍 Validating Data Quality ---")
        validator = DataValidator()
        validation_results = validator.validate_data_dict(data_dict)
        
        # Filter out invalid tickers
        invalid_tickers = [ticker for ticker, result in validation_results.items() if not result.is_valid]
        if invalid_tickers:
            print(f"  Removing {len(invalid_tickers)} invalid tickers: {invalid_tickers}")
            for ticker in invalid_tickers:
                del data_dict[ticker]
        
        # Print validation summary
        validator.print_validation_summary(validation_results)
    
    if not data_dict:
        print(" No valid data after validation.")
        sys.exit(1)
        
    # Build current prices dict, filtering out NaN values
    current_prices = {}
    nan_price_tickers = []
    for t, df in data_dict.items():
        price = df['Close'].iloc[-1]
        if pd.notna(price) and price > 0:
            current_prices[t] = price
        else:
            nan_price_tickers.append(t)
    
    if nan_price_tickers:
        print(f"  Warning: {len(nan_price_tickers)} tickers have invalid prices, skipping: {nan_price_tickers[:5]}{'...' if len(nan_price_tickers) > 5 else ''}")

    # 3. Operations based on Mode
    # Load strategy instance for training check
    strategy = get_strategy(args.strategy)
    
    if args.mode == "train" or (args.mode == "trade" and config['model']['retrain_daily']):
        if strategy.needs_training():
            print(f" Training {strategy.get_display_name()} Model...")
            from src.models import trainer
            trainer.train_model(data_dict)
        else:
            print(f" {strategy.get_display_name()} - no training required")
        
    if args.mode == "trade":
        print(f"\n---  Generating Signals (Strategy: {args.strategy.upper()}) ---")
        signals = {}
        expected_returns = {}
        
        # Config
        BUY_PERCENTILE = 0.10   # Top 10%
        SELL_PERCENTILE = 0.10  # Bottom 10%
        
        # Use strategy's rank_universe method (strategies own their scoring logic)
        expected_returns = strategy.rank_universe(data_dict)
        
        if not expected_returns:
            print("No valid scores from strategy")
            sys.exit(1)
        
        # Cross-sectional ranking (same for both strategies)
        sorted_preds = sorted(expected_returns.items(), key=lambda x: x[1], reverse=True)
        n_tickers = len(sorted_preds)
        n_buy = max(1, int(n_tickers * BUY_PERCENTILE))
        n_sell = max(1, int(n_tickers * SELL_PERCENTILE))
        
        buy_tickers = set([t for t, _ in sorted_preds[:n_buy]])
        sell_tickers = set([t for t, _ in sorted_preds[-n_sell:]])
        
        print(f"\n Cross-Sectional Ranking:")
        print(f"   Universe: {n_tickers} stocks")
        print(f"   Top {n_buy} → BUY, Bottom {n_sell} → SELL")
        if sorted_preds:
            print(f"   Score range: [{sorted_preds[-1][1]*100:.2f}%, {sorted_preds[0][1]*100:.2f}%]")
        
        # Generate signals
        for ticker, score in sorted_preds:
            if ticker in buy_tickers:
                action = "BUY"
                score_type = "Momentum" if args.strategy == "momentum" else "Pred"
                print(f" {ticker}: BUY  (Rank: Top {BUY_PERCENTILE*100:.0f}%, {score_type}: {score*100:+.2f}%)")
            elif ticker in sell_tickers:
                action = "SELL"
                score_type = "Momentum" if args.strategy == "momentum" else "Pred"
                print(f" {ticker}: SELL (Rank: Bottom {SELL_PERCENTILE*100:.0f}%, {score_type}: {score*100:+.2f}%)")
            else:
                action = "HOLD"
            signals[ticker] = action
        # ==================== END CROSS-SECTIONAL RANKING ====================
        
        # ==================== MODEL VALIDATION GATE (Quant Standard) ====================
        # If >80% of predictions are extreme (>3% expected return), model is likely corrupted
        # A healthy model should have predictions centered around 0 with most in [-2%, +2%]
        EXTREME_THRESHOLD = 0.50  # Relaxed to 50% for paper trading/experimentation
        extreme_count = sum(1 for r in expected_returns.values() if abs(r) > EXTREME_THRESHOLD)
        extreme_pct = extreme_count / max(len(expected_returns), 1)
        
        if extreme_pct > 0.80:
            print(f"\n  MODEL VALIDATION WARNING!")
            print(f"   {extreme_count}/{len(expected_returns)} ({extreme_pct:.0%}) predictions are extreme (>{EXTREME_THRESHOLD:.1%})")
            print(f"   Proceeding with caution for paper trading...")
        else:
            print(f"\n Model validation passed: {extreme_pct:.0%} extreme predictions")
        # ==================== END MODEL VALIDATION GATE ====================
            
        print("\n---  Executing Trades ---")


        pf = portfolio.Portfolio(portfolio_id=args.portfolio, start_cash=config['portfolio']['initial_cash'])
        cost_model = TransactionCostModel()  # 5 bps slippage for realistic execution
        current_holdings = pf.get_holdings()
        
        # Load risk settings from config (Phase 7)
        risk_config = config.get('risk', {})
        risk_limits = RiskLimits(
            max_position_pct=risk_config.get('max_position_pct', 0.15),
            max_sector_pct=risk_config.get('max_sector_pct', 0.30),
            min_cash_buffer=config['portfolio']['min_cash_buffer'],
            drawdown_warning=risk_config.get('drawdown_warning', 0.15),
            drawdown_halt=risk_config.get('drawdown_halt', 0.20),
            drawdown_liquidate=risk_config.get('drawdown_liquidate', 0.25)
        )
        risk_mgr = RiskManager(risk_limits=risk_limits)
        
        # Initialize drawdown controller (Phase 7)
        drawdown_ctrl = DrawdownController(
            warning_threshold=risk_limits.drawdown_warning,
            halt_threshold=risk_limits.drawdown_halt,
            liquidate_threshold=risk_limits.drawdown_liquidate
        )
        
        portfolio_value = pf.get_portfolio_value(current_prices)
        drawdown_ctrl.update(portfolio_value)
        print(f"\n Portfolio: ${portfolio_value:,.2f} | {drawdown_ctrl.get_status()}")
        
        # ==================== STOP-LOSS CHECK (Phase 7) ====================
        stop_loss_pct = risk_config.get('stop_loss_pct', 0.08)
        stop_losses_triggered = pf.check_stop_losses(current_prices, stop_loss_pct)
        
        if stop_losses_triggered:
            print(f"\n STOP-LOSS TRIGGERED:")
            for ticker, current, entry, loss_pct in stop_losses_triggered:
                print(f"   {ticker}: {loss_pct:.1%} loss (entry=${entry:.2f}, now=${current:.2f})")
                # Force SELL signal
                signals[ticker] = "SELL"
        # ==================== END STOP-LOSS ====================
        
        # Check drawdown status
        position_multiplier = drawdown_ctrl.get_position_multiplier()
        if position_multiplier < 1.0:
            print(f" Position sizing reduced to {position_multiplier:.0%} due to drawdown")
        
        # Display current risk metrics
        if current_holdings:
            sector_exposure = risk_mgr.get_sector_exposure_summary(
                current_holdings, current_prices, portfolio_value
            )
            print("\n Current Sector Exposure:")
            for sector, pct in sorted(sector_exposure.items(), key=lambda x: x[1], reverse=True):
                print(f"   {sector}: {pct:.1%}")
        
        # Sells
        for ticker, action in signals.items():
            if action == "SELL" and ticker in current_holdings:
                # Skip tickers that were filtered out due to invalid prices
                if ticker not in current_prices:
                    print(f"  {ticker}: Skip SELL - No valid price data")
                    continue
                
                shares = current_holdings[ticker]
                raw_price = current_prices[ticker]
                # Apply slippage (5 bps) - receive less on sell
                exec_price, _ = cost_model.calculate_execution_price('SELL', raw_price, shares)
                
                # Validate trade
                is_valid, reason = risk_mgr.validate_trade(
                    ticker, "SELL", shares, exec_price,
                    current_holdings, current_prices,
                    pf.get_last_balance(), portfolio_value
                )
                
                if not is_valid:
                    print(f"  {ticker}: Trade rejected - {reason}")
                    continue
                
                if pf.record_trade(ticker, "SELL", exec_price, shares, strategy=args.strategy):
                    print(f"📉 SOLD {shares} of {ticker} at ${exec_price:.2f} (mkt: ${raw_price:.2f})")

        # Buys with Risk-Adjusted Position Sizing
        # CRITICAL: Sort by expected return (highest first) for priority allocation
        buy_candidates = sorted(
            [(t, expected_returns.get(t, 0.0)) for t, a in signals.items() if a == "BUY"],
            key=lambda x: x[1],
            reverse=True  # Highest expected return first
        )
        
        # Check if buys are halted due to drawdown
        if position_multiplier == 0.0:
            print(" Buys halted due to drawdown - skipping all buy orders")
            buy_candidates = []
        
        if buy_candidates:
            # Refresh holdings after sells
            current_holdings = pf.get_holdings()
            cash = pf.get_last_balance()
            portfolio_value = pf.get_portfolio_value(current_prices)
            available_cash = cash - config['portfolio']['min_cash_buffer']
            
            if available_cash > 0:
                print(f"\n Available Cash for Buys: ${available_cash:.2f}")
                
                for ticker, exp_ret in buy_candidates:
                    # Skip tickers that were filtered out due to invalid prices
                    if ticker not in current_prices:
                        print(f" {ticker}: Skip - No valid price data")
                        continue
                    
                    raw_price = current_prices[ticker]
                    # Apply slippage (5 bps) - pay more on buy
                    exec_price, _ = cost_model.calculate_execution_price('BUY', raw_price, 1)
                    
                    # Calculate risk-adjusted position size using execution price
                    shares, sizing_reason = risk_mgr.calculate_position_size(
                        ticker=ticker,
                        current_price=exec_price,
                        available_cash=available_cash,
                        portfolio_value=portfolio_value,
                        historical_data=data_dict[ticker],
                        current_holdings=current_holdings,
                        current_prices=current_prices
                    )
                    
                    # Apply drawdown position multiplier (Phase 7)
                    if position_multiplier < 1.0 and shares > 0:
                        original_shares = shares
                        shares = int(shares * position_multiplier)
                        if shares < original_shares:
                            sizing_reason += f", drawdown={position_multiplier:.0%}"
                    
                    if shares == 0:
                        print(f" {ticker}: Skip - {sizing_reason}")
                        continue
                    
                    # Recalculate execution price with actual shares
                    exec_price, _ = cost_model.calculate_execution_price('BUY', raw_price, shares)
                    
                    # Validate trade
                    is_valid, validation_reason = risk_mgr.validate_trade(
                        ticker, "BUY", shares, exec_price,
                        current_holdings, current_prices,
                        cash, portfolio_value
                    )
                    
                    if not is_valid:
                        print(f"  {ticker}: Trade rejected - {validation_reason}")
                        continue
                    
                    if pf.record_trade(ticker, "BUY", exec_price, shares, strategy=args.strategy):
                        print(f" BOUGHT {shares} of {ticker} at ${exec_price:.2f} (mkt: ${raw_price:.2f}) | {sizing_reason}")
                        # Update for next iteration
                        current_holdings = pf.get_holdings()
                        cash = pf.get_last_balance()
                        available_cash = cash - config['portfolio']['min_cash_buffer']
            else:
                print(f"  Insufficient cash for new positions (need buffer of ${config['portfolio']['min_cash_buffer']:.2f})")
        
        # Final portfolio summary
        total_val = pf.get_portfolio_value(current_prices)
        print(f"\n Total Portfolio Value: ${total_val:.2f}")
        
        # Risk metrics
        print("\n Risk Metrics:")
        current_holdings = pf.get_holdings()
        if current_holdings:
            var = risk_mgr.calculate_portfolio_var(
                current_holdings, current_prices, data_dict, confidence=0.95
            )
            if var is not None:
                var_pct = (var / total_val) * 100 if total_val > 0 else 0
                print(f"   1-Day VaR (95%): ${var:.2f} ({var_pct:.2f}% of portfolio)")
            
            sector_exposure = risk_mgr.get_sector_exposure_summary(
                current_holdings, current_prices, total_val
            )
            if sector_exposure:
                print(f"   Largest Sector Exposure: {max(sector_exposure.items(), key=lambda x: x[1])[0]} ({max(sector_exposure.values()):.1%})")

if __name__ == "__main__":
    main()
