"""
Paper Trader Dashboard - Strategy Comparison
Professional Financial Dashboard with Modern UI
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from _data_version import LAST_DATA_UPDATE
except ImportError:
    LAST_DATA_UPDATE = None
import os
import sys
from datetime import datetime
import json
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Paper Trader AI",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ REFINED THEME ============
st.markdown("""
<style>
    /* Import Google Font - Geist for modern feel */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Main background - pure dark with subtle texture */
    .stApp {
        background: #09090b;
        font-family: 'Space Grotesk', -apple-system, sans-serif;
    }
    
    /* Header transparent */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    footer {visibility: hidden !important;}
    
    /* Sidebar - minimal */
    section[data-testid="stSidebar"] {
        background: #09090b !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    section[data-testid="stSidebar"] > div {
        background: #09090b !important;
        padding-top: 1.5rem;
    }
    button[data-testid="baseButton-header"] {
        color: #71717a !important;
    }
    
    /* Typography - clean hierarchy */
    h1 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: #fafafa !important;
        letter-spacing: -0.03em !important;
    }
    h2 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.25rem !important;
        font-weight: 500 !important;
        color: #fafafa !important;
    }
    h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        color: #71717a !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
    }
    p, label, li {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #a1a1aa;
    }
    .stMarkdown, .stText, .stCaption, .stAlert, .stRadio, .stSelectbox, .stMultiSelect, .stCheckbox, .stDataFrame {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #a1a1aa;
    }
    
    /* Hero Metric Cards - big & bold */
    .hero-metric {
        background: linear-gradient(135deg, rgba(24, 24, 27, 0.8) 0%, rgba(9, 9, 11, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 20px;
        padding: 28px 24px;
        margin: 0;
        position: relative;
        overflow: hidden;
    }
    .hero-metric::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(16, 185, 129, 0.5), transparent);
    }
    .hero-metric.momentum::before {
        background: linear-gradient(90deg, transparent, rgba(16, 185, 129, 0.6), transparent);
    }
    .hero-metric.ml::before {
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.6), transparent);
    }
    .hero-metric.lstm::before {
        background: linear-gradient(90deg, transparent, rgba(168, 85, 247, 0.6), transparent);
    }
    .hero-metric.benchmark::before {
        background: linear-gradient(90deg, transparent, rgba(113, 113, 122, 0.4), transparent);
    }
    .metric-label {
        font-size: 0.7rem;
        font-weight: 500;
        color: #52525b;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 12px;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem;
        font-weight: 600;
        color: #fafafa;
        line-height: 1;
        letter-spacing: -0.02em;
    }
    .metric-delta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        font-weight: 500;
        margin-top: 12px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .positive { color: #22c55e !important; }
    .negative { color: #ef4444 !important; }
    .neutral { color: #71717a !important; }
    
    /* Section headers - minimal */
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.75rem;
        font-weight: 500;
        color: #52525b;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin: 48px 0 20px 0;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }
    
    /* DataFrames - cleaner */
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background: rgba(24, 24, 27, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-radius: 12px !important;
    }
    
    /* Radio buttons - compact list style */
    .stRadio > div {
        gap: 4px !important;
        flex-direction: column !important;
    }
    .stRadio > div > label {
        background: transparent !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
        color: #52525b !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
        transition: all 0.15s ease !important;
        margin: 0 !important;
    }
    .stRadio > div > label:hover {
        background: rgba(255, 255, 255, 0.03) !important;
        color: #a1a1aa !important;
    }
    .stRadio > div > label[data-checked="true"] {
        background: rgba(22, 163, 74, 0.1) !important;
        color: #22c55e !important;
    }
    
    /* Tabs - minimal */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        border-radius: 0;
        padding: 12px 20px;
        color: #52525b;
        font-weight: 500;
        font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #a1a1aa;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        border-bottom: 2px solid #22c55e !important;
        color: #fafafa !important;
    }
    
    /* Status badge - subtle */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 14px;
        border-radius: 8px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        background: rgba(22, 163, 74, 0.1);
        color: #22c55e;
        border: 1px solid rgba(22, 163, 74, 0.2);
    }
    .status-pulse {
        width: 6px;
        height: 6px;
        background: #22c55e;
        border-radius: 50%;
        animation: pulse 2s infinite;
        box-shadow: 0 0 8px rgba(34, 197, 94, 0.6);
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    
    /* HR - subtle */
    hr {
        border-color: rgba(255, 255, 255, 0.04) !important;
        margin: 20px 0 !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(113, 113, 122, 0.3); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(113, 113, 122, 0.5); }
    
    [data-testid="collapsedControl"] { color: #a1a1aa !important; }
    
    /* Strategy indicator dots */
    .strategy-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    .dot-momentum { background: #22c55e; box-shadow: 0 0 8px rgba(34, 197, 94, 0.4); }
    .dot-ml { background: #3b82f6; box-shadow: 0 0 8px rgba(59, 130, 246, 0.4); }
    .dot-lstm { background: #a855f7; box-shadow: 0 0 8px rgba(168, 85, 247, 0.4); }
</style>
""", unsafe_allow_html=True)


# ============ DATA LOADING ============
def load_portfolio_data(portfolio_id: str) -> pd.DataFrame:
    """Load ledger data for a portfolio."""
    if portfolio_id == "default":
        path = "ledger.csv"
    else:
        path = f"data/ledgers/ledger_{portfolio_id}.csv"
    
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
    """Get latest portfolio value from VALUE entries (stocks + cash)."""
    if df.empty:
        return 0
    # Use VALUE entries which contain correct portfolio value (stocks + cash)
    if 'action' in df.columns:
        value_rows = df[df['action'] == 'VALUE']
        if not value_rows.empty:
            return value_rows.iloc[-1]['total_value']
    # Fallback to last row if no VALUE entries
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
    """Get portfolio value over time - prefer PORTFOLIO VALUE rows."""
    if df.empty or 'date' not in df.columns:
        return pd.DataFrame()
    
    # Prefer PORTFOLIO VALUE rows (end-of-day snapshots) if they exist
    if 'action' in df.columns:
        value_rows = df[df['action'] == 'VALUE']
        if not value_rows.empty:
            history = value_rows.groupby('date').last().reset_index()[['date', 'total_value']]
            history = history.sort_values('date').reset_index(drop=True)
            return history
    
    # Fallback: use last entry per date, sorted
    history = df.groupby('date').last().reset_index()[['date', 'total_value']]
    history = history.sort_values('date').reset_index(drop=True)
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
    # Logo with different font (JetBrains Mono for terminal feel)
    st.markdown("""
        <div style="padding: 0 0 20px 0;">
            <h1 style="margin: 0; font-family: 'JetBrains Mono', monospace; font-size: 1rem; color: #fafafa; font-weight: 600; letter-spacing: -0.02em; white-space: nowrap;">
                Paper<span style="color: #22c55e;">Trader</span>
            </h1>
            <p style="margin: 4px 0 0 0; font-size: 0.6rem; color: #52525b; text-transform: uppercase; letter-spacing: 0.08em;">
                AI Trading
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="margin-bottom: 24px;">
            <span class="status-badge">
                <span class="status-pulse"></span>
                Live
            </span>
        </div>
    """, unsafe_allow_html=True)
    
    # Strategy selection - vertical for sidebar
    st.markdown("<p style='font-size: 0.7rem; color: #52525b; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;'>View</p>", unsafe_allow_html=True)
    strategy = st.radio(
        "Select strategy:",
        options=["All", "Momentum", "ML", "LSTM"],
        index=0,
        horizontal=False,
        label_visibility="collapsed"
    )
    
    if strategy == "All":
        portfolios = ["momentum", "ml", "lstm"]
    elif strategy == "Momentum":
        portfolios = ["momentum"]
    elif strategy == "ML":
        portfolios = ["ml"]
    else:
        portfolios = ["lstm"]
    
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    
    # Strategy Info - minimal
    st.markdown("<p style='font-size: 0.7rem; color: #52525b; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px;'>Strategies</p>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.8rem; line-height: 1.8;">
        <div style="margin-bottom: 10px;">
            <span class="strategy-dot dot-momentum"></span>
            <span style="color: #a1a1aa;">Momentum</span>
            <span style="color: #52525b; font-size: 0.75rem;"> · Monthly</span>
        </div>
        <div style="margin-bottom: 10px;">
            <span class="strategy-dot dot-ml"></span>
            <span style="color: #a1a1aa;">XGBoost</span>
            <span style="color: #52525b; font-size: 0.75rem;"> · Daily</span>
        </div>
        <div>
            <span class="strategy-dot dot-lstm"></span>
            <span style="color: #a1a1aa;">LSTM</span>
            <span style="color: #52525b; font-size: 0.75rem;"> · Daily</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Spacer to push links to bottom
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    
    # Links - minimal
    st.markdown("""
        <div style="padding: 16px 0; border-top: 1px solid rgba(255,255,255,0.04);">
            <a href="https://github.com/PAT0216/paper-trader" target="_blank" 
               style="color: #52525b; text-decoration: none; font-size: 0.75rem; display: flex; align-items: center; gap: 6px;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                Source Code
            </a>
            <a href="https://pat0216.github.io" target="_blank" 
               style="color: #52525b; text-decoration: none; font-size: 0.75rem; display: flex; align-items: center; gap: 6px; margin-top: 8px;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                About Me
            </a>
        </div>
    """, unsafe_allow_html=True)


# ============ MAIN CONTENT ============
# Header - minimal
if LAST_DATA_UPDATE:
    from datetime import datetime as _dt
    try:
        _data_ts = _dt.fromisoformat(LAST_DATA_UPDATE.replace('Z', '+00:00'))
        _updated_str = _data_ts.strftime('%b %d, %Y')
    except Exception:
        _updated_str = datetime.now().strftime('%b %d, %Y')
else:
    _updated_str = datetime.now().strftime('%b %d, %Y')

st.markdown(f"""
    <div style="margin-bottom: 32px;">
        <h1 style="margin: 0 0 4px 0;">Dashboard</h1>
        <p style="color: #52525b; font-size: 0.8rem; margin: 0;">
            Updated {_updated_str} · Live since Oct 2025
        </p>
    </div>
""", unsafe_allow_html=True)

# Load data
data = {}
for pid in portfolios:
    df = load_portfolio_data(pid)
    if not df.empty:
        data[pid] = df

if not data:
    st.markdown("""
        <div style="text-align: center; padding: 80px 20px;">
            <p style="font-size: 0.9rem; color: #52525b; margin: 0;">No trading data yet</p>
            <p style="font-size: 0.8rem; color: #3f3f46; margin: 8px 0 0 0;">Run the trading workflows to begin</p>
        </div>
    """, unsafe_allow_html=True)
    st.stop()


# ============ KEY METRICS ============
# Load snapshot for live values
snapshot = load_portfolio_snapshot()
snapshot_portfolios = snapshot.get('portfolios', {})
price_date = snapshot.get('price_date', '')

# Hero metrics - big and bold
cols = st.columns(len(data) + 1)  # +1 for benchmark comparison

for i, (pid, df) in enumerate(data.items()):
    with cols[i]:
        # ALWAYS use ledger VALUE entries as source of truth (most accurate)
        value = get_portfolio_value(df)
        start = df[df['action'] == 'DEPOSIT']['amount'].sum()
        if start == 0:
            start = 10000
        pnl = value - start
        pnl_pct = (pnl / start) * 100 if start > 0 else 0

        
        color_class = "positive" if pnl >= 0 else "negative"
        arrow = "+" if pnl >= 0 else ""
        
        strategy_name = "Momentum" if pid == "momentum" else "XGBoost" if pid == "ml" else "LSTM"
        
        st.markdown(f"""
        <div class="hero-metric {pid}">
            <div class="metric-label"><span class="strategy-dot dot-{pid}"></span>{strategy_name}</div>
            <div class="metric-value">${value:,.0f}</div>
            <div class="metric-delta {color_class}">
                {arrow}{pnl_pct:.2f}%
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
        bench_arrow = "+" if bench_ret >= 0 else ""
        bench_color = "positive" if bench_ret >= 0 else "negative"
        
        st.markdown(f"""
        <div class="hero-metric benchmark">
            <div class="metric-label">SPY Benchmark</div>
            <div class="metric-value" style="color: #71717a;">${bench_value:,.0f}</div>
            <div class="metric-delta {bench_color}">
                {bench_arrow}{bench_ret:.2f}%
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
            <div class="hero-metric benchmark">
                <div class="metric-label">SPY Benchmark</div>
                <div class="metric-value" style="color: #71717a;">—</div>
                <div class="metric-delta neutral">No data</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            start_px = float(spy.iloc[0]["px"])
            end_px = float(spy.iloc[-1]["px"])
            bench_value = base_start * (end_px / start_px) if start_px > 0 else base_start
            bench_ret = ((end_px / start_px) - 1) * 100 if start_px > 0 else 0.0
            bench_arrow = "+" if bench_ret >= 0 else ""
            bench_color = "positive" if bench_ret >= 0 else "negative"

            st.markdown(f"""
            <div class="hero-metric benchmark">
                <div class="metric-label">SPY Benchmark</div>
                <div class="metric-value" style="color: #71717a;">${bench_value:,.0f}</div>
                <div class="metric-delta {bench_color}">
                    {bench_arrow}{bench_ret:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)


# ============ PERFORMANCE CHART ============
st.markdown('<div class="section-title">Performance</div>', unsafe_allow_html=True)

# Create performance chart - cleaner design
fig = go.Figure()

colors = {
    'momentum': '#22c55e',  # Green
    'ml': '#3b82f6',         # Blue
    'lstm': '#a855f7'        # Purple
}

# Collect all portfolio dates for SPY alignment
all_portfolio_dates = set()
portfolio_histories = {}

for pid, df in data.items():
    history = get_portfolio_history(df)
    if not history.empty:
        portfolio_histories[pid] = history
        all_portfolio_dates.update(history['date'].tolist())


# Add portfolio traces
for pid, history in portfolio_histories.items():
    color = colors.get(pid, '#22c55e')
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    
    fig.add_trace(go.Scatter(
        x=history['date'],
        y=history['total_value'],
        name=pid.upper(),
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=f'rgba({r}, {g}, {b}, 0.08)',
        hovertemplate='%{y:$,.0f}<extra></extra>'
    ))

# Add SPY benchmark line - aligned to portfolio dates for consistent hover
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
    # Align SPY data to portfolio dates so hover shows SPY on all dates
    if all_portfolio_dates:
        spy_chart_data = spy_chart_data.copy()
        spy_chart_data['date'] = pd.to_datetime(spy_chart_data['date'])
        spy_chart_data = spy_chart_data.set_index('date').sort_index()
        
        # Create date range covering all portfolio dates
        all_dates = pd.to_datetime(list(all_portfolio_dates))
        date_range = pd.date_range(start=all_dates.min(), end=all_dates.max(), freq='D')
        
        # Reindex and forward-fill to cover all dates
        spy_chart_data = spy_chart_data.reindex(date_range).ffill().bfill()
        spy_chart_data = spy_chart_data.reset_index().rename(columns={'index': 'date'})
        
        # Filter to only portfolio dates
        spy_chart_data = spy_chart_data[spy_chart_data['date'].isin(all_dates)]
    
    fig.add_trace(go.Scatter(
        x=spy_chart_data['date'],
        y=spy_chart_data['value'],
        name='SPY',
        line=dict(color='#52525b', width=1.5, dash='dot'),
        hovertemplate='SPY: %{y:$,.0f}<extra></extra>'
    ))

fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Space Grotesk, sans-serif', color='#71717a', size=11),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1,
        bgcolor='rgba(0,0,0,0)',
        font=dict(size=11)
    ),
    margin=dict(l=0, r=0, t=30, b=0),
    xaxis=dict(
        showgrid=False,
        showline=False,
        zeroline=False,
        tickfont=dict(size=10)
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='rgba(255,255,255,0.03)',
        showline=False,
        zeroline=False,
        tickprefix='$',
        tickformat=',',
        tickfont=dict(size=10)
    ),
    hovermode='x unified',
    hoverlabel=dict(
        bgcolor='#18181b',
        bordercolor='#27272a',
        font=dict(color='#fafafa', family='JetBrains Mono', size=12)
    ),
    height=350
)

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})



# ============ DETAILED METRICS ============
st.markdown('<div class="section-title">Metrics</div>', unsafe_allow_html=True)

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
tab1, tab2 = st.tabs(["Current Holdings", "Recent Trades"])

with tab1:
    cols = st.columns(len(data))
    for i, (pid, df) in enumerate(data.items()):
        with cols[i]:
            strategy_name = "Momentum" if pid == "momentum" else "XGBoost" if pid == "ml" else "LSTM"
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
                st.markdown("<p style='color: #52525b; font-size: 0.85rem;'>No positions</p>", unsafe_allow_html=True)
            else:
                st.dataframe(
                    holdings,
                    hide_index=True,
                    use_container_width=True,
                    height=min(300, 40 + len(holdings) * 35)
                )

with tab2:
    if len(data) > 1:
        # Pill buttons for strategy selection (no typing)
        options = list(data.keys())
        selected = st.pills(
            "Select portfolio:",
            options=options,
            format_func=lambda x: "Momentum" if x == "momentum" else "XGBoost" if x == "ml" else "LSTM",
            label_visibility="collapsed",
            default=options[0]
        )
        if selected is None:
            selected = options[0]
    else:
        selected = list(data.keys())[0]
    
    if selected:
        # Filter out CASH and PORTFOLIO VALUE rows - only show actual trades
        actual_trades = data[selected][(data[selected]['ticker'] != 'CASH') & (data[selected]['action'] != 'VALUE')]
        trades_df = actual_trades.tail(15)
        if trades_df.empty:
            st.markdown("<p style='color: #52525b; font-size: 0.85rem;'>No trades yet</p>", unsafe_allow_html=True)
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
st.markdown("""
    <div style="text-align: center; padding: 40px 0 20px 0; margin-top: 40px; border-top: 1px solid rgba(255,255,255,0.04);">
        <p style="margin: 0 0 8px 0; font-size: 0.8rem; color: #52525b;">
            Built by <a href="https://pat0216.github.io" target="_blank" style="color: #22c55e; text-decoration: none; font-weight: 500;">Prabuddha Tamhane</a>
        </p>
        <p style="margin: 0; font-size: 0.65rem; color: #3f3f46;">
            Paper trading only · Not financial advice
        </p>
    </div>
""", unsafe_allow_html=True)
