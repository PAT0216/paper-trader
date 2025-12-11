# Paper Trader - India Market Testing

This branch contains the infrastructure for testing the Paper Trader model on Indian markets (NSE/BSE).

## 🇮🇳 Purpose

Test if the US-designed technical indicators generalize to the Indian market:
- Different market microstructure
- Different trading hours (9:15 AM - 3:30 PM IST)
- Different currency (INR)
- Different economic drivers (emerging market)

## 📁 Structure

```
india/
├── data/              # Indian market SQLite cache (gitignored)
├── models/            # India-trained models
├── results/           # Test results
├── nifty50_universe.py     # NIFTY 50 stock list
├── fetch_india_data.py     # Data fetcher (SQLite cache)
└── run_india_test.py       # Walk-forward test vs NIFTY 50
```

## 🚀 Usage

### Option 1: Use GitHub Actions (Recommended - No Rate Limits)

1. Go to Actions tab on GitHub
2. Run "Fetch India Market Data" workflow
3. Download the `india-market-data` artifact
4. Extract `india_cache.db` to `india/data/`
5. Run the test:
   ```bash
   python india/run_india_test.py
   ```

### Option 2: Fetch locally (May hit rate limits)

```bash
python india/fetch_india_data.py
python india/run_india_test.py
```

## 📊 What Gets Tested

- **Training**: NIFTY 50 stocks (2018-2022)
- **Testing**: 2023-2024
- **Benchmark**: NIFTY 50 Index (^NSEI)
- **Comparison**: Strategy vs Buy-and-Hold NIFTY

## 🏛️ Why This Matters

If the model works on Indian stocks:
- Features capture **universal market dynamics**
- Not just overfitting to US market quirks
- Model has true **cross-market generalization**

## ⚠️ Important Notes

- Data uses `.NS` suffix (NSE - National Stock Exchange)
- Prices are in INR (Indian Rupees)
- Different from main branch (isolated experiment)
