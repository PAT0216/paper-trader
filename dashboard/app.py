"""
Paper Trader Dashboard - Strategy Comparison
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
    page_title="Paper Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium dark theme CSS
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
    }
    
    /* Metric values */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }
    
    /* Metric labels */
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 2.5rem !important;
    }
    
    h2, h3 {
        color: #e0e0e0 !important;
        font-weight: 600 !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Radio buttons - clean look */
    .stRadio > div {
        gap: 0.5rem;
    }
    
    .stRadio > div > label {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 8px 16px;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .stRadio > div > label:hover {
        background: rgba(102, 126, 234, 0.2);
        border-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)


def load_portfolio_data(portfolio_id: str) -> pd.DataFrame:
    """Load ledger data for a portfolio."""
    if portfolio_id == "default":
        path = "ledger.csv"
    else:
        path = f"ledger_{portfolio_id}.csv"
    
    # Try multiple paths
    for base in ["", "../", "../../"]:
        full_path = base + path
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
        {'Ticker': k, 'Shares': int(v)}
        for k, v in holdings.items() if v > 0
    ])


# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("### 🎯 Strategy Selection")
    st.markdown("---")
    
    # Use radio buttons instead of multiselect (no typing)
    strategy = st.radio(
        "Select strategy to view:",
        options=["Both", "Momentum", "ML"],
        index=0,
        horizontal=True
    )
    
    # Map selection to portfolio IDs
    if strategy == "Both":
        portfolios = ["momentum", "ml"]
    elif strategy == "Momentum":
        portfolios = ["momentum"]
    else:
        portfolios = ["ml"]
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Momentum**: 12-1 month factor strategy  
    **ML**: XGBoost ensemble predictions
    """)
    
    st.markdown("---")
    st.markdown(f"*Last refresh: {pd.Timestamp.now().strftime('%H:%M:%S')}*")


# ============ MAIN CONTENT ============
st.markdown("# 📈 Paper Trader")
st.markdown("##### Real-time strategy comparison dashboard")
st.markdown("---")

# Load data
data = {}
for pid in portfolios:
    df = load_portfolio_data(pid)
    if not df.empty:
        data[pid] = df

if not data:
    st.warning("⚠️ No portfolio data found. Run the trading workflows first.")
    st.info("""
    **Getting started:**
    1. Run `python main.py --strategy momentum --portfolio momentum`
    2. Or trigger GitHub Actions workflows
    """)
    st.stop()

# ============ METRICS ROW ============
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
        
        # Custom metric card
        color = "#10b981" if pnl >= 0 else "#ef4444"
        arrow = "↑" if pnl >= 0 else "↓"
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">
                {pid.upper()} PORTFOLIO
            </div>
            <div style="font-size: 2.5rem; font-weight: 700; color: #fff;">
                ${value:,.0f}
            </div>
            <div style="font-size: 1rem; color: {color}; margin-top: 4px;">
                {arrow} {pnl_pct:+.2f}% (${pnl:+,.0f})
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ============ COMPARISON TABLE ============
st.markdown("### 📋 Strategy Comparison")

comparison_data = []
for pid, df in data.items():
    value = get_portfolio_value(df)
    start = df[df['action'] == 'DEPOSIT']['amount'].sum() or 100000
    trades = len(df[df['ticker'] != 'CASH'])
    pnl_pct = ((value / start) - 1) * 100
    
    comparison_data.append({
        'Strategy': pid.upper(),
        'Value': f"${value:,.0f}",
        'Return': f"{pnl_pct:+.2f}%",
        'Trades': trades,
        'Started': df['date'].iloc[0] if not df.empty else '-'
    })

st.dataframe(
    pd.DataFrame(comparison_data),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# ============ HOLDINGS ============
st.markdown("### 💼 Current Holdings")

cols = st.columns(len(data))
for i, (pid, df) in enumerate(data.items()):
    with cols[i]:
        st.markdown(f"**{pid.upper()}**")
        holdings = get_holdings(df)
        if holdings.empty:
            st.info("No positions")
        else:
            st.dataframe(
                holdings, 
                hide_index=True, 
                use_container_width=True,
                height=min(400, 50 + len(holdings) * 35)
            )

st.markdown("---")

# ============ RECENT TRADES ============
st.markdown("### 📜 Recent Trades")

# Use radio instead of selectbox (no typing/cursor)
selected = st.radio(
    "Select portfolio:",
    options=list(data.keys()),
    format_func=lambda x: x.upper(),
    horizontal=True
)

if selected:
    trades_df = data[selected][data[selected]['ticker'] != 'CASH'].tail(15)
    if trades_df.empty:
        st.info("No trades yet")
    else:
        # Style the action column
        def style_action(val):
            if val == 'BUY':
                return 'color: #10b981'
            elif val == 'SELL':
                return 'color: #ef4444'
            return ''
        
        display_df = trades_df[['date', 'ticker', 'action', 'price', 'shares', 'amount']].copy()
        display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}")
        display_df['amount'] = display_df['amount'].apply(lambda x: f"${x:,.0f}")
        display_df.columns = ['Date', 'Ticker', 'Action', 'Price', 'Shares', 'Amount']
        
        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True
        )

# ============ FOOTER ============
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 0.85rem;">
        Built with Streamlit • 
        <a href="https://github.com/PAT0216/paper-trader" style="color: #667eea;">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
