import sys
import argparse
import time
from datetime import datetime
from src.utils.config import load_config
from src.data import loader
from src.data.validator import DataValidator
from src.models import trainer, predictor
from src.trading import portfolio
from src.trading.risk_manager import RiskManager, RiskLimits, DrawdownController
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="AI Paper Trader")
    parser.add_argument("--mode", choices=["trade", "train", "backtest"], default="trade", help="Mode of operation")
    args = parser.parse_args()
    
    # 1. Load Configuration
    try:
        config = load_config()
        
        # Check for dynamic universe (S&P 500)
        universe_type = config.get('universe', {}).get('type', 'config')
        
        if universe_type == 'sp500':
            print("📊 Fetching S&P 500 universe from Wikipedia...")
            from src.data.universe import fetch_sp500_tickers, get_mega_caps
            try:
                tickers = fetch_sp500_tickers()
                print(f"   Loaded {len(tickers)} S&P 500 stocks")
            except Exception as e:
                print(f"   ⚠️ S&P 500 fetch failed: {e}")
                print(f"   Falling back to mega-caps...")
                tickers = get_mega_caps()
        else:
            # Use tickers from config
            tickers = config['tickers']
        
        print(f"--- 🤖 AI Paper Trader | Mode: {args.mode.upper()} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        print(f"Universe: {len(tickers)} tickers")
    except Exception as e:
        print(f"❌ Configuration Error: {e}")
        sys.exit(1)

    # 2. Fetch Data
    print("Fetching market data...")
    # Fetch enough data for training or trading
    data_dict = loader.fetch_data(tickers, period=config['model']['training_period'])
    
    if not data_dict:
        print("❌ Failed to fetch data.")
        sys.exit(1)
    
    # 2a. Validate Data Quality
    print("\n--- 🔍 Validating Data Quality ---")
    validator = DataValidator()
    validation_results = validator.validate_data_dict(data_dict)
    
    # Filter out invalid tickers
    invalid_tickers = [ticker for ticker, result in validation_results.items() if not result.is_valid]
    if invalid_tickers:
        print(f"⚠️  Removing {len(invalid_tickers)} invalid tickers: {invalid_tickers}")
        for ticker in invalid_tickers:
            del data_dict[ticker]
    
    # Print validation summary
    validator.print_validation_summary(validation_results)
    
    if not data_dict:
        print("❌ No valid data after validation.")
        sys.exit(1)
        
    current_prices = {t: df['Close'].iloc[-1] for t, df in data_dict.items()}

    # 3. Operations based on Mode
    if args.mode == "train" or (args.mode == "trade" and config['model']['retrain_daily']):
        print("🧠 Training Model...")
        trainer.train_model(data_dict)
        
    if args.mode == "trade":
        print("\n--- 🔮 Generating Predictions ---")
        ai_predictor = predictor.Predictor()
        signals = {}
        expected_returns = {}
        
        # ==================== QUANT STRATEGY: MOMENTUM-ENHANCED RANKING ====================
        # Phase 2: Combine ML predictions with 12-1 month momentum factor
        # Momentum is the most documented alpha source (Jegadeesh-Titman 1993)
        # 12-1 skips last month to avoid short-term reversal
        
        # Config
        BUY_PERCENTILE = 0.10   # Top 10%
        SELL_PERCENTILE = 0.10  # Bottom 10%
        ML_WEIGHT = 0.5         # 50% ML, 50% Momentum
        MOMENTUM_WEIGHT = 0.5
        
        momentum_scores = {}
        composite_scores = {}
        
        for ticker in tickers:
            if ticker not in data_dict:
                 continue
            df = data_dict[ticker]
            
            # Get ML prediction
            expected_ret = ai_predictor.predict(df)
            expected_returns[ticker] = expected_ret
            
            # Calculate 12-1 month momentum (skip last 21 days, use 252 days before that)
            if len(df) >= 273:  # Need at least 12 months + 1 month
                close = df['Close']
                # 12 month return ending 1 month ago
                momentum_12_1 = (close.iloc[-22] / close.iloc[-273] - 1) if close.iloc[-273] != 0 else 0
                momentum_scores[ticker] = momentum_12_1
            else:
                momentum_scores[ticker] = 0.0
        
        # Z-score normalize both signals for fair combination
        import numpy as np
        from scipy.stats import zscore
        
        # Convert to arrays for z-scoring
        tickers_list = list(expected_returns.keys())
        ml_values = np.array([expected_returns[t] for t in tickers_list])
        mom_values = np.array([momentum_scores.get(t, 0) for t in tickers_list])
        
        # Z-score (handle edge cases)
        if len(ml_values) > 1 and ml_values.std() > 0:
            ml_z = zscore(ml_values)
        else:
            ml_z = np.zeros_like(ml_values)
            
        if len(mom_values) > 1 and mom_values.std() > 0:
            mom_z = zscore(mom_values)
        else:
            mom_z = np.zeros_like(mom_values)
        
        # Combine signals
        composite = ML_WEIGHT * ml_z + MOMENTUM_WEIGHT * mom_z
        composite_scores = {tickers_list[i]: composite[i] for i in range(len(tickers_list))}
        
        # Sort by composite score (descending)
        sorted_composite = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        n_tickers = len(sorted_composite)
        n_buy = max(1, int(n_tickers * BUY_PERCENTILE))
        n_sell = max(1, int(n_tickers * SELL_PERCENTILE))
        
        buy_tickers = set([t for t, _ in sorted_composite[:n_buy]])
        sell_tickers = set([t for t, _ in sorted_composite[-n_sell:]])
        
        print(f"\n📊 Momentum-Enhanced Ranking (50% ML + 50% Momentum):")
        print(f"   Universe: {n_tickers} stocks")
        print(f"   Top {n_buy} → BUY, Bottom {n_sell} → SELL")
        
        # Generate signals based on composite ranking
        for ticker, score in sorted_composite:
            if ticker in buy_tickers:
                action = "BUY"
                mom_pct = momentum_scores.get(ticker, 0) * 100
                ml_pct = expected_returns.get(ticker, 0) * 100
                print(f"🟢 {ticker}: BUY  (Composite: {score:+.2f}, ML: {ml_pct:+.2f}%, Mom: {mom_pct:+.1f}%)")
            elif ticker in sell_tickers:
                action = "SELL"
                mom_pct = momentum_scores.get(ticker, 0) * 100
                ml_pct = expected_returns.get(ticker, 0) * 100
                print(f"🔴 {ticker}: SELL (Composite: {score:+.2f}, ML: {ml_pct:+.2f}%, Mom: {mom_pct:+.1f}%)")
            else:
                action = "HOLD"
            signals[ticker] = action
        # ==================== END MOMENTUM-ENHANCED RANKING ====================
        
        # ==================== MODEL VALIDATION GATE (Quant Standard) ====================
        # If >80% of predictions are extreme (>3% expected return), model is likely corrupted
        # A healthy model should have predictions centered around 0 with most in [-2%, +2%]
        EXTREME_THRESHOLD = 0.03  # 3% daily return is extreme
        extreme_count = sum(1 for r in expected_returns.values() if abs(r) > EXTREME_THRESHOLD)
        extreme_pct = extreme_count / max(len(expected_returns), 1)
        
        if extreme_pct > 0.80:
            print(f"\n🚨 MODEL VALIDATION FAILED!")
            print(f"   {extreme_count}/{len(expected_returns)} ({extreme_pct:.0%}) predictions are extreme (>{EXTREME_THRESHOLD*100}%)")
            print(f"   This likely indicates model corruption or version mismatch.")
            print(f"   HALTING TRADES - please check model files.")
            sys.exit(1)
        elif extreme_pct > 0.30:
            print(f"\n⚠️  MODEL WARNING: {extreme_pct:.0%} of predictions are extreme (>{EXTREME_THRESHOLD*100}%)")
        else:
            print(f"\n✅ Model validation passed: {extreme_pct:.0%} extreme predictions")
        # ==================== END MODEL VALIDATION GATE ====================
            
        print("\n--- 💼 Executing Trades ---")


        pf = portfolio.Portfolio(start_cash=config['portfolio']['initial_cash'])
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
        print(f"\n📈 Portfolio: ${portfolio_value:,.2f} | {drawdown_ctrl.get_status()}")
        
        # ==================== STOP-LOSS CHECK (Phase 7) ====================
        stop_loss_pct = risk_config.get('stop_loss_pct', 0.08)
        stop_losses_triggered = pf.check_stop_losses(current_prices, stop_loss_pct)
        
        if stop_losses_triggered:
            print(f"\n🛑 STOP-LOSS TRIGGERED:")
            for ticker, current, entry, loss_pct in stop_losses_triggered:
                print(f"   {ticker}: {loss_pct:.1%} loss (entry=${entry:.2f}, now=${current:.2f})")
                # Force SELL signal
                signals[ticker] = "SELL"
        # ==================== END STOP-LOSS ====================
        
        # Check drawdown status
        position_multiplier = drawdown_ctrl.get_position_multiplier()
        if position_multiplier < 1.0:
            print(f"⚠️ Position sizing reduced to {position_multiplier:.0%} due to drawdown")
        
        # Display current risk metrics
        if current_holdings:
            sector_exposure = risk_mgr.get_sector_exposure_summary(
                current_holdings, current_prices, portfolio_value
            )
            print("\n📊 Current Sector Exposure:")
            for sector, pct in sorted(sector_exposure.items(), key=lambda x: x[1], reverse=True):
                print(f"   {sector}: {pct:.1%}")
        
        # Sells
        for ticker, action in signals.items():
            if action == "SELL" and ticker in current_holdings:
                shares = current_holdings[ticker]
                price = current_prices[ticker]
                
                # Validate trade
                is_valid, reason = risk_mgr.validate_trade(
                    ticker, "SELL", shares, price,
                    current_holdings, current_prices,
                    pf.get_last_balance(), portfolio_value
                )
                
                if not is_valid:
                    print(f"⚠️  {ticker}: Trade rejected - {reason}")
                    continue
                
                if pf.record_trade(ticker, "SELL", price, shares):
                    print(f"📉 SOLD {shares} of {ticker} at ${price:.2f}")

        # Buys with Risk-Adjusted Position Sizing
        # CRITICAL: Sort by expected return (highest first) for priority allocation
        buy_candidates = sorted(
            [(t, expected_returns.get(t, 0.0)) for t, a in signals.items() if a == "BUY"],
            key=lambda x: x[1],
            reverse=True  # Highest expected return first
        )
        
        # Check if buys are halted due to drawdown
        if position_multiplier == 0.0:
            print("🛑 Buys halted due to drawdown - skipping all buy orders")
            buy_candidates = []
        
        if buy_candidates:
            # Refresh holdings after sells
            current_holdings = pf.get_holdings()
            cash = pf.get_last_balance()
            portfolio_value = pf.get_portfolio_value(current_prices)
            available_cash = cash - config['portfolio']['min_cash_buffer']
            
            if available_cash > 0:
                print(f"\n💵 Available Cash for Buys: ${available_cash:.2f}")
                
                for ticker, exp_ret in buy_candidates:
                    price = current_prices[ticker]
                    
                    # Calculate risk-adjusted position size
                    shares, sizing_reason = risk_mgr.calculate_position_size(
                        ticker=ticker,
                        current_price=price,
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
                        print(f"⚪️ {ticker}: Skip - {sizing_reason}")
                        continue
                    
                    # Validate trade
                    is_valid, validation_reason = risk_mgr.validate_trade(
                        ticker, "BUY", shares, price,
                        current_holdings, current_prices,
                        cash, portfolio_value
                    )
                    
                    if not is_valid:
                        print(f"⚠️  {ticker}: Trade rejected - {validation_reason}")
                        continue
                    
                    if pf.record_trade(ticker, "BUY", price, shares):
                        print(f"📈 BOUGHT {shares} of {ticker} at ${price:.2f} | {sizing_reason}")
                        # Update for next iteration
                        current_holdings = pf.get_holdings()
                        cash = pf.get_last_balance()
                        available_cash = cash - config['portfolio']['min_cash_buffer']
            else:
                print(f"⚠️  Insufficient cash for new positions (need buffer of ${config['portfolio']['min_cash_buffer']:.2f})")
        
        # Final portfolio summary
        total_val = pf.get_portfolio_value(current_prices)
        print(f"\n💰 Total Portfolio Value: ${total_val:.2f}")
        
        # Risk metrics
        print("\n📈 Risk Metrics:")
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
