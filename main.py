import sys
import argparse
import time
from datetime import datetime
from src.utils.config import load_config
from src.data import loader
from src.models import trainer, predictor
from src.trading import portfolio

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
        
    current_prices = {t: df['Close'].iloc[-1] for t, df in data_dict.items()}

    # 3. Operations based on Mode
    if args.mode == "train" or (args.mode == "trade" and config['model']['retrain_daily']):
        print("🧠 Training Model...")
        trainer.train_model(data_dict)
        
    if args.mode == "trade":
        print("\n--- 🔮 Generating Predictions ---")
        ai_predictor = predictor.Predictor()
        signals = {}
        
        for ticker in tickers:
            if ticker not in data_dict:
                 continue
            df = data_dict[ticker]
            prob = ai_predictor.predict(df)
            
            thresh_buy = config['model']['threshold_buy']
            thresh_sell = config['model']['threshold_sell']
            
            action = "HOLD"
            if prob > thresh_buy:
                action = "BUY"
                print(f"🟢 {ticker}: BUY  (Prob: {prob:.4f})")
            elif prob < thresh_sell:
                action = "SELL"
                print(f"🔴 {ticker}: SELL (Prob: {prob:.4f})")
            else:
                print(f"⚪️ {ticker}: HOLD (Prob: {prob:.4f})")
                
            signals[ticker] = action
            
        print("\n--- 💼 Executing Trades ---")
        pf = portfolio.Portfolio(start_cash=config['portfolio']['initial_cash'])
        current_holdings = pf.get_holdings()
        
        # Sells
        for ticker, action in signals.items():
            if action == "SELL" and ticker in current_holdings:
                shares = current_holdings[ticker]
                price = current_prices[ticker]
                if pf.record_trade(ticker, "SELL", price, shares):
                    print(f"📉 SOLD {shares} of {ticker} at ${price:.2f}")

        # Buys
        buy_candidates = [t for t, a in signals.items() if a == "BUY"]
        if buy_candidates:
            cash = pf.get_last_balance()
            min_buffer = config['portfolio']['min_cash_buffer']
            
            if cash > min_buffer:
                available_cash = cash - min_buffer
                budget_per_asset = available_cash / len(buy_candidates)
                
                for ticker in buy_candidates:
                    price = current_prices[ticker]
                    shares = int(budget_per_asset // price)
                    
                    if shares > 0:
                         if pf.record_trade(ticker, "BUY", price, shares):
                            print(f"📈 BOUGHT {shares} of {ticker} at ${price:.2f}")
        
        total_val = pf.get_portfolio_value(current_prices)
        print(f"\n💰 Total Portfolio Value: ${total_val:.2f}")

if __name__ == "__main__":
    main()
