
import yfinance as yf
import numpy as np

def get_benchmark():
    # Fetch SPY data
    spy = yf.download("SPY", start="2015-01-01", end="2024-12-10", progress=False)
    
    # Calculate Buy & Hold Return
    start_price = spy['Close'].iloc[0].item()  # Extract scalar value at [0]
    end_price = spy['Close'].iloc[-1].item()   # Extract scalar value at [-1]
    
    total_return = (end_price / start_price) - 1
    
    # Calculate CAGR
    days = (spy.index[-1] - spy.index[0]).days
    years = days / 365.25
    cagr = (end_price / start_price) ** (1/years) - 1
    
    # Calculate Sharpe
    daily_returns = spy['Close'].pct_change().dropna()
    avg_return = daily_returns.mean().item()  # Convert to scalar
    std_dev = daily_returns.std().item()      # Convert to scalar
    sharpe = (avg_return * 252 - 0.04) / (std_dev * np.sqrt(252))
    
    # Max Drawdown
    rolling_max = spy['Close'].cummax()
    drawdown = (spy['Close'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min().item()     # Convert to scalar
    
    print(f"SPY Metrics (2015-2024):")
    print(f"Total Return: {total_return:.2%}")
    print(f"CAGR: {cagr:.2%}")
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")

if __name__ == "__main__":
    get_benchmark()
