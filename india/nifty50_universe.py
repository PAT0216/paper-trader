# NIFTY 50 Stocks (NSE - National Stock Exchange of India)
# These are the 50 largest stocks on the Indian market
# yfinance tickers use .NS suffix for NSE

NIFTY_50 = [
    "RELIANCE.NS",    # Reliance Industries
    "TCS.NS",         # Tata Consultancy Services
    "HDFCBANK.NS",    # HDFC Bank
    "INFY.NS",        # Infosys
    "ICICIBANK.NS",   # ICICI Bank
    "HINDUNILVR.NS",  # Hindustan Unilever
    "SBIN.NS",        # State Bank of India
    "BHARTIARTL.NS",  # Bharti Airtel
    "ITC.NS",         # ITC Limited
    "KOTAKBANK.NS",   # Kotak Mahindra Bank
    "LT.NS",          # Larsen & Toubro
    "AXISBANK.NS",    # Axis Bank
    "ASIANPAINT.NS",  # Asian Paints
    "MARUTI.NS",      # Maruti Suzuki
    "HCLTECH.NS",     # HCL Technologies
    "SUNPHARMA.NS",   # Sun Pharma
    "TITAN.NS",       # Titan Company
    "BAJFINANCE.NS",  # Bajaj Finance
    "WIPRO.NS",       # Wipro
    "ULTRACEMCO.NS",  # UltraTech Cement
    "ONGC.NS",        # Oil & Natural Gas Corp
    "NTPC.NS",        # NTPC Ltd
    "TATAMOTORS.NS",  # Tata Motors
    "POWERGRID.NS",   # Power Grid Corp
    "M&M.NS",         # Mahindra & Mahindra
    "JSWSTEEL.NS",    # JSW Steel
    "TATASTEEL.NS",   # Tata Steel
    "ADANIENT.NS",    # Adani Enterprises
    "ADANIPORTS.NS",  # Adani Ports
    "COALINDIA.NS",   # Coal India
    "BAJAJFINSV.NS",  # Bajaj Finserv
    "NESTLEIND.NS",   # Nestle India
    "GRASIM.NS",      # Grasim Industries
    "TECHM.NS",       # Tech Mahindra
    "INDUSINDBK.NS",  # IndusInd Bank
    "HINDALCO.NS",    # Hindalco Industries
    "DRREDDY.NS",     # Dr. Reddy's Labs
    "DIVISLAB.NS",    # Divi's Laboratories
    "CIPLA.NS",       # Cipla
    "EICHERMOT.NS",   # Eicher Motors
    "APOLLOHOSP.NS",  # Apollo Hospitals
    "HEROMOTOCO.NS",  # Hero MotoCorp
    "BPCL.NS",        # Bharat Petroleum
    "SBILIFE.NS",     # SBI Life Insurance
    "BRITANNIA.NS",   # Britannia Industries
    "TATACONSUM.NS",  # Tata Consumer Products
    "HDFCLIFE.NS",    # HDFC Life Insurance
    "SHREECEM.NS",    # Shree Cement
    "BAJAJ-AUTO.NS",  # Bajaj Auto (note the hyphen)
    "UPL.NS",         # UPL Limited
]

# Benchmark index
NIFTY_INDEX = "^NSEI"  # NIFTY 50 Index
SENSEX_INDEX = "^BSESN"  # BSE SENSEX

def get_nifty50_tickers():
    """Return list of NIFTY 50 tickers."""
    return NIFTY_50.copy()

def get_benchmark():
    """Return NIFTY 50 index ticker."""
    return NIFTY_INDEX
