# 🤖 AI Paper Trader

An automated, containerized Artificial Intelligence that trades a paper portfolio on the US Stock Market (SPY, QQQ, IWM, DIA).

![Performance](performance.png)

## 🚀 Quick Start (Locally)

1.  **Install**:
    ```bash
    git clone https://github.com/PAT0216/paper-trader.git
    cd paper-trader
    ```

2.  **Run (with Docker)**:
    This is the easiest way. It handles all dependencies for you.
    ```bash
    docker-compose up --build
    ```
    *That's it!* The bot will fetch data, retrain its brain, and execute trades.

## ⚙️ Configuration

Want to trade different stocks? Easy.
Open `config/settings.yaml` and edit the tickers:

```yaml
# config/settings.yaml
tickers:
  - SPY
  - AAPL  # <--- Added Apple!
  - MSFT
```

## 🧠 How it Works

1.  **Data**: It downloads 2 years of daily price history from Yahoo Finance.
2.  **Learning**: It feeds this data into an **XGBoost Algorithm** which learns patterns (RSI, MACD, Volatility).
3.  **Predicting**: The AI predicts the probability of the price closing HIGHER tomorrow.
    - `> 55%` -> **BUY** 🟢
    - `< 45%` -> **SELL** 🔴
4.  **Trading**: It connects to your local `ledger.csv` and updates your virtual portfolio.

## ☁️ Automation

This project is configured to run automatically on **GitHub Actions** every weekday at market close.
Check the [Actions Tab](https://github.com/PAT0216/paper-trader/actions) to see it living its life!

---
*Disclaimer: This is for educational purposes only. Do not use this logic for real money trading without serious risk assessment.*
