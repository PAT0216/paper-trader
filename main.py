import sys
import argparse
import time
from datetime import datetime
from src.utils.config import load_config
from src.data import loader
from src.data.validator import DataValidator
from src.models import trainer, predictor
from src.trading import portfolio
from src.trading.risk_manager import RiskManager, RiskLimits

def main():
    parser = argparse.ArgumentParser(description="AI Paper Trader")
    parser.add_argument("--mode", choices=["trade", "train", "backtest"], default="trade", help="Mode of operation")
    args = parser.parse_args()
    
    # 1. Load Configuration
    try:
        config = load_config()
        tickers = config['tickers']
        print(f"--- 🤖 AI Paper Trader | Mode: {args.mode.upper()} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        print(f"Universe: {tickers}")
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
            
            # Legacy classifier compatibility: check if output looks like probability
            if ai_predictor.is_regression:
                # Regression: expected_ret is in decimal (e.g., 0.02 = 2%)
                if expected_ret > return_buy_thresh:
                    action = "BUY"
                    print(f"🟢 {ticker}: BUY  (Exp.Ret: {expected_ret*100:+.2f}%)")
                elif expected_ret < return_sell_thresh:
                    action = "SELL"
                    print(f"🔴 {ticker}: SELL (Exp.Ret: {expected_ret*100:+.2f}%)")
                else:
                    action = "HOLD"
                    print(f"⚪️ {ticker}: HOLD (Exp.Ret: {expected_ret*100:+.2f}%)")
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
        
        # Initialize Risk Manager
        risk_limits = RiskLimits(
            max_position_pct=0.15,
            max_sector_pct=0.40,
            min_cash_buffer=config['portfolio']['min_cash_buffer']
        )
        risk_mgr = RiskManager(risk_limits=risk_limits)
        
        portfolio_value = pf.get_portfolio_value(current_prices)
        
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
        buy_candidates = [t for t, a in signals.items() if a == "BUY"]
        if buy_candidates:
            # Refresh holdings after sells
            current_holdings = pf.get_holdings()
            cash = pf.get_last_balance()
            portfolio_value = pf.get_portfolio_value(current_prices)
            available_cash = cash - config['portfolio']['min_cash_buffer']
            
            if available_cash > 0:
                print(f"\n💵 Available Cash for Buys: ${available_cash:.2f}")
                
                for ticker in buy_candidates:
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
