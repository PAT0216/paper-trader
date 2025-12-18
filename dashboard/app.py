"""
Paper Trader Dashboard - Strategy Comparison
Professional Financial Dashboard with Modern UI
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime
import json
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Paper Trader AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ PROFESSIONAL THEME ============
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Main background - deep slate with subtle gradient */
    .stApp {
        background: linear-gradient(135deg, #0a0f1a 0%, #111827 50%, #0f172a 100%);
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Keep Streamlit header/toolbar visible (it contains the mobile sidebar hamburger). */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Footer only (toolbar/header is allowed and should stay visible) */
    footer {visibility: hidden !important;}
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    section[data-testid="stSidebar"] > div {
        background: #0f172a !important;
        padding-top: 1rem;
    }
    
    /* Sidebar collapse button */
    button[data-testid="baseButton-header"] {
        color: #94a3b8 !important;
    }
    
    /* Typography */
    h1 {
        font-family: 'Inter', sans-serif !important;
        font-size: 2.25rem !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
        letter-spacing: -0.02em !important;
    }
    
    h2 {
        font-family: 'Inter', sans-serif !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #f1f5f9 !important;
    }
    
    h3 {
        font-family: 'Inter', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    
    /* IMPORTANT:
       Do NOT force fonts on all spans — Streamlit uses Material icon fonts that are spans.
       If we override them, icons render as text like `keyboard_double_arrow_right`. */
    p, label, li {
        font-family: 'Inter', sans-serif !important;
        color: #94a3b8;
    }

    /* Body text containers */
    .stMarkdown, .stText, .stCaption, .stAlert, .stRadio, .stSelectbox, .stMultiSelect, .stCheckbox, .stDataFrame {
        font-family: 'Inter', sans-serif !important;
        color: #94a3b8;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    
    .metric-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: #f8fafc;
        line-height: 1.2;
    }
    
    .metric-delta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        font-weight: 500;
        margin-top: 8px;
    }
    
    .positive { color: #10b981 !important; }
    .negative { color: #ef4444 !important; }
    
    /* Section dividers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 32px 0 16px 0;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    .section-icon {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
    }
    
    /* DataFrames */
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
    }
    
    /* Radio buttons - pill style */
    .stRadio > div {
        gap: 8px !important;
    }
    
    .stRadio > div > label {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important;
        padding: 8px 20px !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    
    .stRadio > div > label:hover {
        border-color: #10b981 !important;
        color: #f1f5f9 !important;
    }
    
    .stRadio > div > label[data-checked="true"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        border-color: #10b981 !important;
        color: #ffffff !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 24px;
        color: #94a3b8;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        border-color: rgba(16, 185, 129, 0.5);
        color: #f1f5f9;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%) !important;
        border-color: #10b981 !important;
        color: #10b981 !important;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .status-live {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .status-pulse {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }
    
    /* Hide default hr */
    hr {
        border-color: rgba(255, 255, 255, 0.06) !important;
        margin: 24px 0 !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(100, 116, 139, 0.4);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(100, 116, 139, 0.6);
    }
    
    /* Sidebar toggle button - make visible */
    [data-testid="collapsedControl"] {
        color: #f1f5f9 !important;
    }
    
    /* Charts container */
    .chart-container {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 16px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============ DATA LOADING ============
def load_portfolio_data(portfolio_id: str) -> pd.DataFrame:
    """Load ledger data for a portfolio."""
    if portfolio_id == "default":
        path = "ledger.csv"
    else:
        path = f"ledger_{portfolio_id}.csv"
    
    for base in ["", "../", "../../"]:
        full_path = base + path
        if os.path.exists(full_path):
            df = pd.read_csv(full_path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df
    
    return pd.DataFrame()


def load_backtest_results() -> dict:
    """Load backtest metrics if available."""
    for base in ["", "../", "../../"]:
        path = base + "results/backtest_metrics.json"
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    return {}


def load_portfolio_snapshot() -> dict:
    """Load current portfolio snapshot for live values."""
    for base in ["", "../", "../../"]:
        path = base + "data/portfolio_snapshot.json"
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    return {}


def load_spy_benchmark() -> pd.DataFrame:
    """Load SPY benchmark price history from JSON file."""
    for base in ["", "../", "../../"]:
        path = base + "data/spy_benchmark.json"
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                if 'portfolio_history' in data:
                    df = pd.DataFrame(data['portfolio_history'], columns=['date', 'value'])
                    df['date'] = pd.to_datetime(df['date'])
                    return df
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


def get_portfolio_history(df: pd.DataFrame) -> pd.DataFrame:
    """Get portfolio value over time."""
    if df.empty or 'date' not in df.columns:
        return pd.DataFrame()
    
    # Get unique dates with their total values
    history = df.groupby('date').last().reset_index()[['date', 'total_value']]
    return history


def get_ledger_date_range(dfs: dict[str, pd.DataFrame]):
    """Return (min_date, max_date) across all ledgers that have a date column."""
    dates = []
    for df in dfs.values():
        if not df.empty and "date" in df.columns:
            try:
                d = pd.to_datetime(df["date"], errors="coerce")
                d = d.dropna()
                if not d.empty:
                    dates.append(d.min())
                    dates.append(d.max())
            except Exception:
                continue
    if not dates:
        return None, None
    return min(dates), max(dates)


def load_benchmark_series(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    db_path: str,
) -> pd.DataFrame:
    """Load benchmark adj_close series from the local SQLite cache."""
    if not (start_date and end_date):
        return pd.DataFrame()
    if not os.path.exists(db_path):
        return pd.DataFrame()

    start_s = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end_s = pd.to_datetime(end_date).strftime("%Y-%m-%d")

    con = sqlite3.connect(db_path)
    try:
        q = """
            SELECT date, COALESCE(adj_close, close) AS px
            FROM price_data
            WHERE ticker = ?
              AND date >= ?
              AND date <= ?
            ORDER BY date ASC
        """
        df = pd.read_sql_query(q, con, params=(ticker, start_s, end_s))
    finally:
        con.close()

    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "px"])
    return df


# ============ SIDEBAR ============
with st.sidebar:
    # Logo & Title
    st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <div style="font-size: 2.5rem; margin-bottom: 8px;">📈</div>
            <h2 style="margin: 0; font-size: 1.5rem; color: #f8fafc;">Paper Trader AI</h2>
            <p style="margin: 8px 0 0 0; font-size: 0.8rem; color: #64748b;">Algorithmic Trading System</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <span class="status-badge status-live">
                <span class="status-pulse"></span>
                Paper Trading
            </span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Strategy selection
    st.markdown("##### 🎯 Strategy")
    strategy = st.radio(
        "Select strategy:",
        options=["Compare", "Momentum", "ML"],
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if strategy == "Compare":
        portfolios = ["momentum", "ml"]
    elif strategy == "Momentum":
        portfolios = ["momentum"]
    else:
        portfolios = ["ml"]
    
    st.markdown("---")
    
    # Strategy Info
    st.markdown("##### 📚 Strategies")
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 16px; font-size: 0.85rem;">
        <div style="margin-bottom: 12px;">
            <strong style="color: #10b981;">Momentum</strong><br>
            <span style="color: #94a3b8;">12-1 month factor • Monthly rebalance • 15% stop-loss</span>
        </div>
        <div>
            <strong style="color: #3b82f6;">ML Ensemble</strong><br>
            <span style="color: #94a3b8;">XGBoost • 15 features • VIX regime detection</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Links
    st.markdown("""
        <div style="text-align: center; padding: 10px 0;">
            <a href="https://github.com/PAT0216/paper-trader" target="_blank" 
               style="color: #64748b; text-decoration: none; font-size: 0.85rem;">
                <span style="margin-right: 6px;">⭐</span> View on GitHub
            </a>
        </div>
    """, unsafe_allow_html=True)


# ============ MAIN CONTENT ============
# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("# Dashboard")
    st.markdown(f"<p style='color: #64748b; margin-top: -10px;'>Last updated: {datetime.now().strftime('%B %d, %Y')}</p>", unsafe_allow_html=True)

# Load data
data = {}
for pid in portfolios:
    df = load_portfolio_data(pid)
    if not df.empty:
        data[pid] = df

if not data:
    st.markdown("""
        <div style="text-align: center; padding: 60px 20px;">
            <div style="font-size: 3rem; margin-bottom: 16px;">📊</div>
            <h3 style="color: #f1f5f9; margin-bottom: 8px;">No Trading Data Yet</h3>
            <p style="color: #64748b;">Run the trading workflows to see portfolio performance.</p>
        </div>
    """, unsafe_allow_html=True)
    st.stop()


# ============ KEY METRICS ============
st.markdown("""
    <div class="section-header">
        <div class="section-icon">💰</div>
        <h3 style="margin: 0;">Portfolio Overview</h3>
    </div>
""", unsafe_allow_html=True)

# Load snapshot for live values
snapshot = load_portfolio_snapshot()
snapshot_portfolios = snapshot.get('portfolios', {})
price_date = snapshot.get('price_date', '')

# Metrics cards
cols = st.columns(len(data) + 1)  # +1 for benchmark comparison

for i, (pid, df) in enumerate(data.items()):
    with cols[i]:
        # Use snapshot value if available, otherwise fallback to ledger
        if pid in snapshot_portfolios:
            value = snapshot_portfolios[pid]['value']
            pnl_pct = snapshot_portfolios[pid]['return_pct']
            start = snapshot.get('initial_capital', 10000)
            pnl = value - start
        else:
            value = get_portfolio_value(df)
            start = df[df['action'] == 'DEPOSIT']['amount'].sum()
            if start == 0:
                start = 10000
            pnl = value - start
            pnl_pct = (pnl / start) * 100 if start > 0 else 0
        
        color_class = "positive" if pnl >= 0 else "negative"
        arrow = "↑" if pnl >= 0 else "↓"
        
        strategy_name = "Momentum" if pid == "momentum" else "ML Ensemble"
        strategy_emoji = "🚀" if pid == "momentum" else "🤖"
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{strategy_emoji} {strategy_name}</div>
            <div class="metric-value">${value:,.0f}</div>
            <div class="metric-delta {color_class}">
                {arrow} ${abs(pnl):,.0f} ({pnl_pct:+.2f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)

# S&P 500 benchmark (uses snapshot first, fallback to database)
with cols[-1]:
    # Try snapshot first (works on deployed site)
    benchmark = snapshot.get('benchmark', {})
    
    if benchmark:
        bench_value = benchmark['value']
        bench_ret = benchmark['return_pct']
        period_label = f"{benchmark.get('start_date', '')} → {benchmark.get('end_date', '')}"
        bench_arrow = "↑" if bench_ret >= 0 else "↓"
        bench_color = "#10b981" if bench_ret >= 0 else "#ef4444"
        
        st.markdown(f"""
        <div class="metric-card" style="border-color: rgba(100, 116, 139, 0.3);">
            <div class="metric-label">📊 S&P 500 (SPY)</div>
            <div class="metric-value" style="color: #94a3b8;">${bench_value:,.0f}</div>
            <div class="metric-delta" style="color: {bench_color};">
                {bench_arrow} {bench_ret:+.2f}% <span style="color:#64748b; font-family: 'Inter', sans-serif;">({period_label})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback to database (works on localhost)
        min_d, max_d = get_ledger_date_range(data)
        first_df = next(iter(data.values()))
        base_start = first_df[first_df["action"] == "DEPOSIT"]["amount"].sum() if "action" in first_df.columns else 0
        if not base_start:
            base_start = 10000

        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "market.db")
        spy = load_benchmark_series("SPY", min_d, max_d, db_path)

        if spy.empty:
            st.markdown(f"""
            <div class="metric-card" style="border-color: rgba(100, 116, 139, 0.3);">
                <div class="metric-label">📊 S&P 500 (Benchmark)</div>
                <div class="metric-value" style="color: #94a3b8;">N/A</div>
                <div class="metric-delta" style="color: #64748b;">
                    Missing SPY data for {min_d.strftime('%Y-%m-%d') if min_d else '—'} → {max_d.strftime('%Y-%m-%d') if max_d else '—'}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            start_px = float(spy.iloc[0]["px"])
            end_px = float(spy.iloc[-1]["px"])
            bench_value = base_start * (end_px / start_px) if start_px > 0 else base_start
            bench_ret = ((end_px / start_px) - 1) * 100 if start_px > 0 else 0.0
            bench_arrow = "↑" if bench_ret >= 0 else "↓"
            bench_color = "#10b981" if bench_ret >= 0 else "#ef4444"
            period_label = f"{min_d.strftime('%b %d, %Y')} → {max_d.strftime('%b %d, %Y')}" if min_d and max_d else "Ledger period"

            st.markdown(f"""
            <div class="metric-card" style="border-color: rgba(100, 116, 139, 0.3);">
                <div class="metric-label">📊 S&P 500 (SPY)</div>
                <div class="metric-value" style="color: #94a3b8;">${bench_value:,.0f}</div>
                <div class="metric-delta" style="color: {bench_color};">
                    {bench_arrow} {bench_ret:+.2f}% <span style="color:#64748b; font-family: 'Inter', sans-serif;">({period_label})</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ============ PERFORMANCE CHART ============
st.markdown("""
    <div class="section-header">
        <div class="section-icon">📈</div>
        <h3 style="margin: 0;">Performance</h3>
    </div>
""", unsafe_allow_html=True)

# Create performance chart
fig = go.Figure()

colors = {
    'momentum': '#10b981',
    'ml': '#3b82f6'
}

for pid, df in data.items():
    history = get_portfolio_history(df)
    if not history.empty:
        fig.add_trace(go.Scatter(
            x=history['date'],
            y=history['total_value'],
            name=pid.upper(),
            line=dict(color=colors.get(pid, '#10b981'), width=2.5),
            fill='tozeroy',
            fillcolor=f"rgba({int(colors.get(pid, '#10b981')[1:3], 16)}, {int(colors.get(pid, '#10b981')[3:5], 16)}, {int(colors.get(pid, '#10b981')[5:7], 16)}, 0.1)"
        ))

# Add SPY benchmark line
min_d, max_d = get_ledger_date_range(data)
spy_chart_data = None

# Try database first (localhost)
if min_d and max_d:
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "market.db")
    spy_data = load_benchmark_series("SPY", min_d, max_d, db_path)
    
    if not spy_data.empty:
        start_price = spy_data['px'].iloc[0]
        spy_data['value'] = (spy_data['px'] / start_price) * 10000
        spy_chart_data = spy_data[['date', 'value']]

# Fallback to JSON file (deployed site)
if spy_chart_data is None or spy_chart_data.empty:
    spy_chart_data = load_spy_benchmark()

if spy_chart_data is not None and not spy_chart_data.empty:
    fig.add_trace(go.Scatter(
        x=spy_chart_data['date'],
        y=spy_chart_data['value'],
        name='SPY',
        line=dict(color='#64748b', width=2, dash='dash'),
        opacity=0.8
    ))

fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', color='#94a3b8'),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1,
        bgcolor='rgba(0,0,0,0)'
    ),
    margin=dict(l=0, r=0, t=40, b=0),
    xaxis=dict(
        showgrid=True,
        gridcolor='rgba(255,255,255,0.05)',
        linecolor='rgba(255,255,255,0.1)'
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='rgba(255,255,255,0.05)',
        linecolor='rgba(255,255,255,0.1)',
        tickprefix='$',
        tickformat=','
    ),
    hovermode='x unified',
    hoverlabel=dict(
        bgcolor='#1e293b',
        bordercolor='#334155',
        font=dict(color='#f1f5f9')
    )
)

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})



# ============ DETAILED METRICS ============
st.markdown("""
    <div class="section-header">
        <div class="section-icon">📊</div>
        <h3 style="margin: 0;">Detailed Metrics</h3>
    </div>
""", unsafe_allow_html=True)

# Create comparison table
comparison_data = []
for pid, df in data.items():
    # Use snapshot values if available
    if pid in snapshot_portfolios:
        value = snapshot_portfolios[pid]['value']
        pnl_pct = snapshot_portfolios[pid]['return_pct']
        holdings_count = snapshot_portfolios[pid].get('positions', 0)
    else:
        value = get_portfolio_value(df)
        start = df[df['action'] == 'DEPOSIT']['amount'].sum() or 10000
        pnl_pct = ((value / start) - 1) * 100
        holdings_count = len(get_holdings(df))
    
    # Filter out PORTFOLIO VALUE rows for trade counts
    actual_trades = df[(df['ticker'] != 'CASH') & (df['action'] != 'VALUE')]
    trades = len(actual_trades)
    buys = len(df[df['action'] == 'BUY'])
    sells = len(df[df['action'] == 'SELL'])
    
    comparison_data.append({
        'Strategy': pid.upper(),
        'Current Value': f"${value:,.0f}",
        'Total Return': f"{pnl_pct:+.2f}%",
        'Total Trades': trades,
        'Buys': buys,
        'Sells': sells,
        'Holdings': holdings_count
    })

df_comparison = pd.DataFrame(comparison_data)

# Style the dataframe
st.dataframe(
    df_comparison,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Strategy": st.column_config.TextColumn("Strategy", width="medium"),
        "Current Value": st.column_config.TextColumn("Value", width="medium"),
        "Total Return": st.column_config.TextColumn("Return", width="small"),
        "Total Trades": st.column_config.NumberColumn("Trades", width="small"),
        "Holdings": st.column_config.NumberColumn("Positions", width="small")
    }
)


# ============ HOLDINGS & TRADES ============
tab1, tab2 = st.tabs(["📦 Current Holdings", "📜 Recent Trades"])

with tab1:
    cols = st.columns(len(data))
    for i, (pid, df) in enumerate(data.items()):
        with cols[i]:
            strategy_name = "Momentum" if pid == "momentum" else "ML Ensemble"
            st.markdown(f"**{strategy_name}**")
            
            # First try to get holdings from ledger
            holdings = get_holdings(df)
            
            # Fallback to snapshot if ledger has no trades
            if holdings.empty and snapshot and 'portfolios' in snapshot:
                if pid in snapshot['portfolios'] and 'holdings' in snapshot['portfolios'][pid]:
                    snap_holdings = snapshot['portfolios'][pid]['holdings']
                    if snap_holdings:
                        holdings = pd.DataFrame([
                            {'Ticker': k, 'Shares': int(v)}
                            for k, v in snap_holdings.items()
                        ])
            
            if holdings.empty:
                st.markdown("<p style='color: #64748b; font-style: italic;'>No current positions</p>", unsafe_allow_html=True)
            else:
                st.dataframe(
                    holdings,
                    hide_index=True,
                    use_container_width=True,
                    height=min(300, 40 + len(holdings) * 35)
                )

with tab2:
    if len(data) > 1:
        selected = st.radio(
            "Select portfolio:",
            options=list(data.keys()),
            format_func=lambda x: "Momentum" if x == "momentum" else "ML Ensemble",
            horizontal=True,
            label_visibility="collapsed"
        )
    else:
        selected = list(data.keys())[0]
    
    if selected:
        # Filter out CASH and PORTFOLIO VALUE rows - only show actual trades
        actual_trades = data[selected][(data[selected]['ticker'] != 'CASH') & (data[selected]['action'] != 'VALUE')]
        trades_df = actual_trades.tail(15)
        if trades_df.empty:
            st.markdown("<p style='color: #64748b; font-style: italic;'>No trades recorded yet (ML uses simulated backtest values)</p>", unsafe_allow_html=True)
        else:
            display_df = trades_df[['date', 'ticker', 'action', 'price', 'shares']].copy()
            display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
            display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}")
            display_df.columns = ['Date', 'Ticker', 'Action', 'Price', 'Shares']
            
            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Action": st.column_config.TextColumn(
                        "Action",
                        help="BUY or SELL"
                    )
                }
            )


# ============ FOOTER ============
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 20px 0; color: #475569;">
        <p style="margin: 0; font-size: 0.8rem;">
            Paper Trader AI • Built by 
            <a href="https://pat0216.github.io" target="_blank" style="color: #10b981; text-decoration: none;">Prabuddha Tamhane</a>
        </p>
        <p style="margin: 4px 0 0 0; font-size: 0.7rem; color: #334155;">
            ⚠️ Backtested results. Not financial advice.
        </p>
    </div>
""", unsafe_allow_html=True)
