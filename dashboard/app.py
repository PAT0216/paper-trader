"""
Paper Trader Dashboard - Main App

Interactive Streamlit dashboard for comparing portfolio strategies.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Paper Trader Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for clean styling - theme-aware
st.markdown("""
<style>
    /* Remove fixed background colors - let Streamlit handle theme */
    .stMetric {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    /* Ensure text is visible in both modes */
    .stMetric label {
        font-weight: 600 !important;
    }
    /* Header styling */
    h1, h2, h3 {
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


def load_portfolio_data(portfolio_id: str) -> pd.DataFrame:
    """Load ledger data for a portfolio."""
    if portfolio_id == "default":
        path = "../ledger.csv"
    else:
        path = f"../ledger_{portfolio_id}.csv"
    
    # Try relative paths
    for base in ["", "../", "../../"]:
        full_path = base + path.lstrip("../")
        if os.path.exists(full_path):
            return pd.read_csv(full_path)
    
    return pd.DataFrame()


def get_portfolio_value(df: pd.DataFrame) -> float:
    """Get current portfolio value."""
    if df.empty:
        return 0
    return df.iloc[-1]['total_value']


def get_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """Get current holdings from ledger."""
    if df.empty:
        return pd.DataFrame()
    
    holdings = {}
    for _, row in df.iterrows():
        if row['ticker'] == 'CASH':
            continue
        ticker = row['ticker']
        if row['action'] == 'BUY':
            holdings[ticker] = holdings.get(ticker, 0) + row['shares']
        elif row['action'] == 'SELL':
            holdings[ticker] = holdings.get(ticker, 0) - row['shares']
    
    return pd.DataFrame([
        {'Ticker': k, 'Shares': v}
        for k, v in holdings.items() if v > 0
    ])


# Main app
st.title("📈 Paper Trader Dashboard")
st.markdown("Compare **Momentum** vs **ML** strategy performance in real-time.")

# Sidebar
st.sidebar.title("Portfolios")
portfolios = st.sidebar.multiselect(
    "Select portfolios to compare",
    ["momentum", "ml", "default"],
    default=["momentum", "ml"]
)

# Load data
data = {}
for pid in portfolios:
    df = load_portfolio_data(pid)
    if not df.empty:
        data[pid] = df

if not data:
    st.warning("No portfolio data found. Run the trading workflows first.")
    st.info("""
    To get started:
    1. Run `python main.py --portfolio momentum --strategy momentum`
    2. Or trigger the GitHub Actions workflow
    """)
    st.stop()

# Metrics row
st.markdown("### 📊 Portfolio Overview")
cols = st.columns(len(data))

for i, (pid, df) in enumerate(data.items()):
    with cols[i]:
        value = get_portfolio_value(df)
        start = df[df['action'] == 'DEPOSIT']['amount'].sum()
        if start == 0:
            start = 100000
        pnl = value - start
        pnl_pct = (pnl / start) * 100 if start > 0 else 0
        
        st.metric(
            label=f"**{pid.upper()}** Portfolio",
            value=f"${value:,.0f}",
            delta=f"{pnl_pct:+.1f}% (${pnl:+,.0f})"
        )

# Comparison table
st.markdown("### 📋 Strategy Comparison")
comparison_data = []
for pid, df in data.items():
    value = get_portfolio_value(df)
    start = df[df['action'] == 'DEPOSIT']['amount'].sum() or 100000
    trades = len(df[df['ticker'] != 'CASH'])
    
    comparison_data.append({
        'Portfolio': pid.upper(),
        'Current Value': f"${value:,.0f}",
        'Return': f"{((value/start)-1)*100:+.1f}%",
        'Trades': trades,
        'Start Date': df['date'].iloc[0] if not df.empty else '-'
    })

st.dataframe(
    pd.DataFrame(comparison_data),
    use_container_width=True,
    hide_index=True
)

# Holdings comparison
st.markdown("### 💼 Current Holdings")
cols = st.columns(len(data))

for i, (pid, df) in enumerate(data.items()):
    with cols[i]:
        st.markdown(f"**{pid.upper()}**")
        holdings = get_holdings(df)
        if holdings.empty:
            st.info("No positions")
        else:
            st.dataframe(holdings, hide_index=True, use_container_width=True)

# Trade history
st.markdown("### 📜 Recent Trades")
selected_portfolio = st.selectbox("Portfolio", list(data.keys()))
if selected_portfolio:
    trades_df = data[selected_portfolio][data[selected_portfolio]['ticker'] != 'CASH'].tail(10)
    if trades_df.empty:
        st.info("No trades yet")
    else:
        st.dataframe(
            trades_df[['date', 'ticker', 'action', 'price', 'shares', 'amount']],
            hide_index=True,
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown(
    "Built with Streamlit | "
    "[GitHub](https://github.com/PAT0216/paper-trader) | "
    f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
)
