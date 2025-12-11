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
        
        # Thresholds for return-based decisions (Phase 3)
        # BUY if expected return > 0.5%, SELL if < -0.5%
        return_buy_thresh = 0.005   # 0.5% expected return
        return_sell_thresh = -0.005  # -0.5% expected return
        
        for ticker in tickers:
            if ticker not in data_dict:
                 continue
            df = data_dict[ticker]
            
            # Get expected return from regression model
            expected_ret = ai_predictor.predict(df)
            expected_returns[ticker] = expected_ret
        
        # ==================== CROSS-SECTIONAL NORMALIZATION (Phase 7) ====================
        # Z-score normalize predictions to compare apples-to-apples
        if expected_returns and ai_predictor.is_regression:
            returns_array = np.array(list(expected_returns.values()))
            mu = np.mean(returns_array)
            sigma = np.std(returns_array)
            
            if sigma > 0:
                print(f"\n📊 Cross-sectional normalization: μ={mu*100:.2f}%, σ={sigma*100:.2f}%")
                normalized_returns = {t: (r - mu) / sigma for t, r in expected_returns.items()}
            else:
                normalized_returns = {t: 0.0 for t in expected_returns}
        else:
            normalized_returns = expected_returns
        # ==================== END NORMALIZATION ====================
        
        # Generate signals using z-scores (threshold = 1.0 = top ~16%)
        z_buy_thresh = 1.0   # Buy if z-score > 1.0
        z_sell_thresh = -1.0  # Sell if z-score < -1.0
        
        for ticker, expected_ret in expected_returns.items():
            z_score = normalized_returns.get(ticker, 0.0)
            
            # Legacy classifier compatibility: check if output looks like probability
            if ai_predictor.is_regression:
                # Use z-score thresholds for ranking-based signals
                if z_score > z_buy_thresh:
                    action = "BUY"
                    print(f"🟢 {ticker}: BUY  (Exp.Ret: {expected_ret*100:+.2f}%, z={z_score:.2f})")
                elif z_score < z_sell_thresh:
                    action = "SELL"
                    print(f"🔴 {ticker}: SELL (Exp.Ret: {expected_ret*100:+.2f}%, z={z_score:.2f})")
                else:
                    action = "HOLD"
                    # Only print if notable
                    if abs(z_score) > 0.5:
                        print(f"⚪️ {ticker}: HOLD (Exp.Ret: {expected_ret*100:+.2f}%, z={z_score:.2f})")
            else:
                # Legacy classifier: expected_ret is probability 0-1
                prob = expected_ret
                thresh_buy = config['model']['threshold_buy']
                thresh_sell = config['model']['threshold_sell']
                
                if prob > thresh_buy:
                    action = "BUY"
                    print(f"🟢 {ticker}: BUY  (Prob: {prob:.4f})")
                elif prob < thresh_sell:
                    action = "SELL"
                    print(f"🔴 {ticker}: SELL (Prob: {prob:.4f})")
                else:
                    action = "HOLD"
                    print(f"⚪️ {ticker}: HOLD (Prob: {prob:.4f})")
                
            signals[ticker] = action
            
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
