import yfinance as yf
import pandas as pd

def fetch_data(tickers, period="2y", interval="1d"):
    """
    Fetches historical data for multiple tickers.
    Returns a dictionary of DataFrames {ticker: df}.
    """
    data = {}
    print(f"Fetching data for: {tickers}")
    
    # We fetch individually to ensure clean DataFrames without MultiIndex complexity for now
    for ticker in tickers:
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval)
            if df.empty:
                print(f"Warning: No data for {ticker}")
                continue
            
            # Ensure index is datetime and sorted
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            data[ticker] = df
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            
    return data
