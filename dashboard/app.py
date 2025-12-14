"""
Paper Trader Dashboard - Strategy Comparison
Professional Financial Dashboard
"""

import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Paper Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Finance Theme CSS
st.markdown("""
<style>
    /* Main background - dark slate */
    .stApp {
        background: #0f172a;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #1e293b;
        border-right: 1px solid #334155;
    }
    
    /* Clean card styling */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
    }
    
    /* Headers - clean white */
    h1, h2, h3 {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }
    
    h1 {
        font-size: 2rem !important;
    }
    
    /* Text */
    p, span, div {
        color: #cbd5e1;
    }
    
    /* Dividers */
    hr {
        border-color: #334155 !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Radio buttons */
    .stRadio > div > label {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        color: #e2e8f0 !important;
    }
    
    .stRadio > div > label:hover {
        border-color: #10b981 !important;
    }
    
    /* Positive/Negative colors */
    .positive { color: #10b981; }
    .negative { color: #ef4444; }
    
    /* Info boxes */
    .stAlert {
        background: #1e293b;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)


def load_portfolio_data(portfolio_id: str) -> pd.DataFrame:
    """Load ledger data for a portfolio."""
    if portfolio_id == "default":
        path = "ledger.csv"
    else:
        path = f"ledger_{portfolio_id}.csv"
    
    for base in ["", "../", "../../"]:
        full_path = base + path
        if os.path.exists(full_path):
            return pd.read_csv(full_path)
    
    return pd.DataFrame()


def get_portfolio_value(df: pd.DataFrame) -> float:
    if df.empty:
        return 0
    return df.iloc[-1]['total_value']


def get_holdings(df: pd.DataFrame) -> pd.DataFrame:
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
    # Logo centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
        if os.path.exists(logo_path):
            st.image(logo_path, width=60)
    
    st.markdown("<h2 style='text-align: center; margin-top: 0;'>Paper Trader</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center;'>"
        "<a href='https://github.com/PAT0216/paper-trader' style='color: #64748b;'>GitHub →</a>"
        "</p>",
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # Strategy selection
    st.markdown("**Strategy**")
    strategy = st.radio(
        "Select strategy:",
        options=["Both", "Momentum", "ML"],
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if strategy == "Both":
        portfolios = ["momentum", "ml"]
    elif strategy == "Momentum":
        portfolios = ["momentum"]
    else:
        portfolios = ["ml"]
    
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("""
    <small>
    <b>Momentum</b>: 12-1 month factor<br>
    <b>ML</b>: XGBoost ensemble
    </small>
    """, unsafe_allow_html=True)


# ============ MAIN CONTENT ============
st.markdown("# Paper Trader")
st.caption("Real-time strategy comparison")
st.markdown("---")

# Load data
data = {}
for pid in portfolios:
    df = load_portfolio_data(pid)
    if not df.empty:
        data[pid] = df

if not data:
    st.warning("No portfolio data found. Run the trading workflows first.")
    st.stop()

# ============ METRICS ============
st.markdown("### Portfolio Overview")

cols = st.columns(len(data))
for i, (pid, df) in enumerate(data.items()):
    with cols[i]:
        value = get_portfolio_value(df)
        start = df[df['action'] == 'DEPOSIT']['amount'].sum()
        if start == 0:
            start = 100000
        pnl = value - start
        pnl_pct = (pnl / start) * 100 if start > 0 else 0
        
        color = "#10b981" if pnl >= 0 else "#ef4444"
        arrow = "↑" if pnl >= 0 else "↓"
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px;">
                {pid.upper()}
            </div>
            <div style="font-size: 2rem; font-weight: 600; color: #f1f5f9; margin: 8px 0;">
                ${value:,.0f}
            </div>
            <div style="font-size: 0.9rem; color: {color};">
                {arrow} {pnl_pct:+.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ============ COMPARISON ============
st.markdown("### Performance")

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
    })

st.dataframe(
    pd.DataFrame(comparison_data),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# ============ HOLDINGS ============
st.markdown("### Holdings")

cols = st.columns(len(data))
for i, (pid, df) in enumerate(data.items()):
    with cols[i]:
        st.markdown(f"**{pid.upper()}**")
        holdings = get_holdings(df)
        if holdings.empty:
            st.caption("No positions")
        else:
            st.dataframe(
                holdings, 
                hide_index=True, 
                use_container_width=True,
                height=min(300, 40 + len(holdings) * 35)
            )

st.markdown("---")

# ============ TRADES ============
st.markdown("### Recent Trades")

selected = st.radio(
    "Portfolio:",
    options=list(data.keys()),
    format_func=lambda x: x.upper(),
    horizontal=True,
    label_visibility="collapsed"
)

if selected:
    trades_df = data[selected][data[selected]['ticker'] != 'CASH'].tail(10)
    if trades_df.empty:
        st.caption("No trades yet")
    else:
        display_df = trades_df[['date', 'ticker', 'action', 'price', 'shares']].copy()
        display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}")
        display_df.columns = ['Date', 'Ticker', 'Action', 'Price', 'Shares']
        st.dataframe(display_df, hide_index=True, use_container_width=True)
