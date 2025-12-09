import sys
import argparse
from datetime import datetime
from src.ml import data_loader, model_trainer, predictor
from src.trading import portfolio

# Configuration
TICKERS = ["SPY", "QQQ", "IWM", "DIA"] 
# Universe of liquid ETFs

def run():
    print(f"--- Running AI Paper Trader for {datetime.now().strftime('%Y-%m-%d')} ---")
    
    # 1. Initialize Portfolio
    pf = portfolio.Portfolio()
    
    # 2. Fetch Data
    # Fetch 2 years to ensure enough for indicators + training
    data_dict = data_loader.fetch_data(TICKERS, period="2y")
    
    if not data_dict:
        print("Failed to fetch data.")
        sys.exit(1)
        
    current_prices = {t: df['Close'].iloc[-1] for t, df in data_dict.items()}
    print(f"Current Prices: {current_prices}")

    # 3. Train Model (Simplified: Retrain daily on latest data)
    # In production, we might load an existing model and only retrain weekly.
    # But XGBoost on this small data is fast (<10s).
    print("Retraining model on latest data...")
    model_trainer.train_model(data_dict)
    
    # 4. Predict & Generate Signals
    ai_predictor = predictor.Predictor()
    signals = {}
    
    print("\n--- Generating Predictions ---")
    for ticker in TICKERS:
        df = data_dict[ticker]
        prob = ai_predictor.predict(df)
        print(f"{ticker}: Up Probability = {prob:.4f}")
        
        if prob > 0.55:
            signals[ticker] = "BUY"
        elif prob < 0.45:
            signals[ticker] = "SELL"
        else:
            signals[ticker] = "HOLD"
            
    # 5. Execute Trades (Simple Rebalance Logic)
    # Strategy: 
    # - Sell if signal is SELL.
    # - Buy if signal is BUY and we have cash.
    # - Evenly split available cash among BUY signals.
    
    print("\n--- Executing Trades ---")
    current_holdings = pf.get_holdings()
    
    # First, Process Sells to free up cash
    for ticker, action in signals.items():
        if action == "SELL" and ticker in current_holdings:
            shares = current_holdings[ticker]
            price = current_prices[ticker]
            if pf.record_trade(ticker, "SELL", price, shares):
                print(f"SOLD {shares} of {ticker} at ${price:.2f}")

    # Then, Process Buys
    buy_candidates = [t for t, a in signals.items() if a == "BUY"]
    if buy_candidates:
        cash = pf.get_last_balance()
        if cash > 100: # Minimum cash buffer
            budget_per_asset = cash / len(buy_candidates)
            
            for ticker in buy_candidates:
                price = current_prices[ticker]
                # Calculate shares to buy (floor)
                shares = int(budget_per_asset // price)
                
                if shares > 0:
                     if pf.record_trade(ticker, "BUY", price, shares):
                        print(f"BOUGHT {shares} of {ticker} at ${price:.2f}")
    else:
        print("No BUY signals today.")

    # 6. Report Value
    total_val = pf.get_portfolio_value(current_prices)
    print(f"\nTotal Portfolio Value: ${total_val:.2f}")

if __name__ == "__main__":
    run()
