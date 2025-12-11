"""
Paper Trader AI - Interactive Dashboard

Interview-ready showcase of the trading system with:
- Portfolio overview and performance
- Model insights and feature importance
- Backtest results and metrics
- Live performance tracking

Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page config
st.set_page_config(
    page_title="Paper Trader AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    /* White background for metrics (dark mode compatible) */
    [data-testid="metric-container"] {
        background-color: white !important;
        padding: 15px !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    /* Make metric values DARK and visible */
    [data-testid="stMetricValue"] {
        color: #0e1117 !important;
        font-size: 2rem !important;
        font-weight: bold !important;
    }
    /* Make metric labels dark */
    [data-testid="stMetricLabel"] {
        color: #31333F !important;
        font-weight: 500 !important;
    }
    /* Green delta */
    [data-testid="stMetricDelta"] {
        color: #09ab3b !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    # Logo as styled text instead of image
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
        <h2 style="color: white; margin: 0; font-weight: bold;">📈 Paper Trader AI</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.title("Navigation")
    
    page = st.radio(
        "Go to:",
        ["🏠 Home", "📊 Portfolio", "🧠 Model Insights", "📈 Backtest Results"],
        label_visibility="collapsed"
    )
    
    
    st.markdown("---")
    st.markdown("""
    **Phase 7: Risk Controls**
    - 🛡️ Stop-Loss: **15%**
    - 📉 Max Drawdown: **-20%**
    """)
    
    st.markdown("---")
    st.caption("📚 [Documentation](https://github.com/PAT0216/paper-trader)")

# Main content
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">Paper Trader AI</h1>', unsafe_allow_html=True)
    st.markdown("### Automated ML-Driven Stock Trading System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**🎯 Phase 7 Complete**\n\nRisk controls, stop-loss, drawdown management")
    
    with col2:
        st.success("**📈 Walk-Forward Validated**\n\nRigorous out-of-sample testing")
    
    with col3:
        st.warning("**🇮🇳 Cross-Market Tested**\n\nValidated on Indian markets (NIFTY 50)")
    
    st.markdown("---")
    
    # Quick Stats
    st.subheader("📊 Quick Stats")
    
    from dashboard.utils.data_loader import load_ledger, load_backtest_metrics, calculate_portfolio_stats
    
    ledger = load_ledger()
    backtest_metrics = load_backtest_metrics()
    portfolio_stats = calculate_portfolio_stats(ledger)
    
    # Only show metrics if we have data
    if ledger.empty and not backtest_metrics:
        st.info("📝 **No trading data yet** - Run `make train && make trade` to start generating data")
    else:
        # Use custom HTML metrics for dark mode visibility
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="color: #666; font-size: 0.875rem; margin-bottom: 5px;">Total Trades</div>
                <div style="color: #0e1117; font-size: 2rem; font-weight: bold;">{portfolio_stats['num_trades']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="color: #666; font-size: 0.875rem; margin-bottom: 5px;">Portfolio Value</div>
                <div style="color: #0e1117; font-size: 2rem; font-weight: bold;">${portfolio_stats['total_value']:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            return_color = "#09ab3b" if portfolio_stats['total_return_pct'] >= 0 else "#ff4b4b"
            st.markdown(f"""
            <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="color: #666; font-size: 0.875rem; margin-bottom: 5px;">Return</div>
                <div style="color: {return_color}; font-size: 2rem; font-weight: bold;">{portfolio_stats['total_return_pct']:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Try to get sharpe from backtest metrics
            sharpe_value = None
            if backtest_metrics:
                if 'sharpe_ratio' in backtest_metrics:
                    sharpe_value = backtest_metrics['sharpe_ratio']
                elif 'risk_adjusted' in backtest_metrics and 'sharpe_ratio' in backtest_metrics['risk_adjusted']:
                    sharpe_value = backtest_metrics['risk_adjusted']['sharpe_ratio']
            
            if sharpe_value is not None:
                st.markdown(f"""
                <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="color: #666; font-size: 0.875rem; margin-bottom: 5px;">Backtest Sharpe</div>
                    <div style="color: #0e1117; font-size: 2rem; font-weight: bold;">{sharpe_value:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Show placeholder encouraging backtest
                st.markdown(f"""
                <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="color: #666; font-size: 0.875rem; margin-bottom: 5px;">Backtest Sharpe</div>
                    <div style="color: #999; font-size: 2rem; font-weight: bold;">--</div>
                </div>
                """, unsafe_allow_html=True)
    
    
    st.markdown("---")
    
    # System Overview
    st.subheader("🏗️ System Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Data Pipeline**
        - 📥 Yahoo Finance API → Market Data
        - ✅ 10 Validation Checks
        - 🔧 15 Technical Features
        - 📊 Volume & Volatility Indicators
        """)
        
        st.markdown("""
        **Model**
        - 🧠 XGBoost Regressor (predicts next-day return)
        - 📉 Time-Series CV (no data leakage)
        - 🎯 Dynamic Feature Selection (top 12-15 features)
        - 📈 Spearman IC: ~0.12 (ranking accuracy)
        """)
    
    with col2:
        st.markdown("""
        **Trading**
        - 🎲 Fixed Threshold Signals (+0.5% buy, -0.5% sell)
        - 🛡️ 15% Stop-Loss (A/B tested)
        - 📊 Drawdown Circuit Breakers (-15%/-20%/-25%)
        - 💼 Position Limits (max 15% per stock)
        """)
        
        st.markdown("""
        **Validation**
        - 🔬 Walk-Forward Backtesting
        - 🎯 Double Holdout (unseen tickers)
        - 🇮🇳 Cross-Market (Indian NIFTY 50)
        - ✅ 55+ Unit Tests
        """)
    
    st.markdown("---")
    
    # Performance Highlights
    st.subheader("🏆 Performance Highlights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **US Market (Full Backtest)**
        - Total Return: 219% 
        - CAGR: 15.65%
        - Sharpe: 0.75
        - Max Drawdown: -21%
        """)
        
        st.caption("⚠️ Likely inflated due to survivorship bias")
    
    with col2:
        st.markdown("""
        **Realistic (Double Holdout - Unseen Tickers)**
        - 2-Year Return: 43.8%
        - Annual: ~20%
        - Vs S&P 500: +58% (underperformed in bull market)
        - Win Rate: 71%
        """)
        
        st.caption("✅ True generalization test")
    
    st.markdown("---")
    st.caption("Built with ❤️ using Python, XGBoost, Streamlit | Phase 8: Dashboard & Explainability")

elif page == "📊 Portfolio":
    st.title("📊 Portfolio Overview")
    st.write("Portfolio details page - coming in next update")

elif page == "🧠 Model Insights":
    st.title("🧠 Model Insights")
    st.write("Model insights page - coming in next update")

elif page == "📈 Backtest Results":
    st.title("📈 Backtest Results")
    st.write("Backtest results page - coming in next update")
