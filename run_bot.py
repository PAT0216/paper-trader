import sys
import market_data
import strategy
import portfolio
from datetime import datetime

# Configuration
TICKER = "SPY"
SHARES_TO_TRADE = 1 # Simple Logic: Buy/Sell fixed amount

def run():
    print(f"--- Running Paper Trader Bot for {datetime.now().strftime('%Y-%m-%d')} ---")
    
    # 1. Initialize Portfolio
    pf = portfolio.Portfolio()
    
    # 2. Fetch Data
    try:
        df = market_data.fetch_data(TICKER)
    except Exception as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)
        
    # 3. Calculate Indicators
    df = strategy.calculate_indicators(df)
    current_price = market_data.get_current_price(TICKER)
    print(f"Current Price of {TICKER}: ${current_price:.2f}")

    # 4. Generate Signal
    current_holdings = pf.get_holdings(TICKER)
    signal = strategy.generate_signal(df, current_holdings)
    print(f"Signal: {signal} (Holdings: {current_holdings})")

    # 5. Execute Trade
    if signal == "BUY":
        success = pf.record_trade(TICKER, "BUY", current_price, SHARES_TO_TRADE)
        if success:
            print(f"Executed BUY of {SHARES_TO_TRADE} shares at ${current_price:.2f}")
    elif signal == "SELL":
        if current_holdings > 0:
            # Sell all
            success = pf.record_trade(TICKER, "SELL", current_price, current_holdings)
            if success:
                print(f"Executed SELL of {current_holdings} shares at ${current_price:.2f}")
    else:
        print("No trade execution needed.")
        
    # 6. Update Total Portfolio Value (Optional for now, but good for logs)
    total_val = pf.get_portfolio_value({TICKER: current_price})
    print(f"Total Portfolio Value: ${total_val:.2f}")
    
    
    # 7. Generate Plot
    try:
        import visualizer
        visualizer.plot_performance(df)
    except Exception as e:
        print(f"Error plotting: {e}")
    
if __name__ == "__main__":
    run()
