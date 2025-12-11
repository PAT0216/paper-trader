# Paper Trader AI - Interactive Dashboard

Professional Streamlit dashboard for visualizing trading system performance, model insights, and portfolio analytics.

## 🚀 Quick Start

```bash
# Install dependencies
pip install streamlit plotly

# Run dashboard
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📊 Features

### 🏠 Home
- System overview and architecture
- Quick stats (portfolio value, trades, returns)
- Performance highlights
- Phase 7 status (risk controls)

### 📊 Portfolio Overview (Coming Soon)
- Current positions
- Sector allocation
- P&L breakdown
- Trade history

### 🧠 Model Insights (Coming Soon)
- Feature importance
- Model metrics
- Prediction explanations

### 📈 Backtest Results (Coming Soon)
- Equity curve
- Drawdown chart
- Performance metrics
- Trade analysis

##  Structure

```
dashboard/
├── app.py                  # Main Streamlit app
├── pages/                  # Multi-page sections
├── utils/
│   └── data_loader.py     # Data loading utilities
└── assets/
    └── screenshots/        # Dashboard screenshots
```

## 🎯 Interview-Ready

This dashboard is designed to showcase the project in technical interviews:
- Clean, professional UI
- Real-time data updates
- Interactive visualizations
- Easy to demonstrate

## 🛠️ Dependencies

- `streamlit>=1.28.0` - Dashboard framework
- `plotly>=5.17.0` - Interactive charts
- `pandas` - Data manipulation

## 📸 Screenshots

(Coming soon - take screenshots after completion)

## ⚠️ Current Status

**Phase 8.2: In Progress**
- ✅ Main app structure
- ✅ Home page with system overview
- ✅ Data loading utilities
- ⏳ Portfolio page
- ⏳ Model insights page
- ⏳ Backtest results page
