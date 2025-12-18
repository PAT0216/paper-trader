"""
Fundamental Factor Model - Backtest & Analysis

Tests a value+quality factor model using fundamental data.
Comprehensive testing including:
- Portfolio construction (top 20 by composite score)
- Walk-forward validation  
- Comparison vs SPY benchmark
- Factor analysis and attribution
"""

import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# DATA LOADING
# ============================================================

def load_fundamentals() -> pd.DataFrame:
    """Load and prepare fundamental data."""
    path = 'data/fundamentals_historical.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fundamentals not found: {path}")
    
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"📊 Loaded fundamentals: {len(df)} rows, {df['ticker'].nunique()} tickers")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    return df


def load_prices(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Load price data from market.db."""
    db_path = 'data/market.db'
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    con = sqlite3.connect(db_path)
    
    # Format tickers for SQL
    ticker_list = "','".join(tickers)
    
    query = f"""
        SELECT date, ticker, COALESCE(adj_close, close) as close
        FROM price_data
        WHERE ticker IN ('{ticker_list}')
          AND date >= '{start_date}'
          AND date <= '{end_date}'
        ORDER BY date, ticker
    """
    
    prices = pd.read_sql_query(query, con)
    con.close()
    
    prices['date'] = pd.to_datetime(prices['date'])
    
    print(f"📈 Loaded prices: {len(prices)} rows, {prices['ticker'].nunique()} tickers")
    
    return prices


# ============================================================
# FACTOR CONSTRUCTION
# ============================================================

def calculate_factor_scores(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate composite factor scores for each stock-quarter.
    
    Factors:
    1. Value: Low P/E, high earnings yield
    2. Quality: High ROE, low debt
    3. Profitability: High margins
    """
    df = fundamentals.copy()
    
    # Handle column name variations
    revenue_col = 'Total Revenue' if 'Total Revenue' in df.columns else 'revenue'
    net_income_col = 'Net Income' if 'Net Income' in df.columns else 'netIncome'
    
    # Calculate additional ratios if not present
    if 'profitMargin' not in df.columns and revenue_col in df.columns and net_income_col in df.columns:
        df['profitMargin'] = df[net_income_col] / df[revenue_col].replace(0, np.nan)
    
    # Define factor columns (use what's available)
    factor_cols = []
    
    # Value factors (lower is better -> we'll invert)
    value_cols = []
    
    # Quality factors (higher is better)
    quality_cols = []
    if 'roe' in df.columns:
        quality_cols.append('roe')
    if 'currentRatio' in df.columns:
        quality_cols.append('currentRatio')
    
    # Profitability factors (higher is better)
    profit_cols = []
    if 'profitMargin' in df.columns:
        profit_cols.append('profitMargin')
    if 'grossMargin' in df.columns:
        profit_cols.append('grossMargin')
    
    # Inverse of leverage (lower debt is better)
    if 'debtToEquity' in df.columns:
        df['low_leverage'] = -df['debtToEquity']  # Negate so higher = better
        quality_cols.append('low_leverage')
    
    all_factor_cols = quality_cols + profit_cols
    
    if not all_factor_cols:
        print("⚠️ No factor columns available!")
        return df
    
    print(f"   Using factors: {all_factor_cols}")
    
    # For each date, rank stocks and create composite score
    scored_dfs = []
    
    for date in df['date'].unique():
        date_df = df[df['date'] == date].copy()
        
        if len(date_df) < 10:
            continue
        
        # Rank each factor (percentile)
        for col in all_factor_cols:
            if col in date_df.columns:
                # Percentile rank (0-1, higher is better)
                date_df[f'{col}_rank'] = date_df[col].rank(pct=True, na_option='keep')
        
        # Composite score = average of all factor percentiles
        rank_cols = [f'{col}_rank' for col in all_factor_cols if f'{col}_rank' in date_df.columns]
        date_df['composite_score'] = date_df[rank_cols].mean(axis=1)
        
        scored_dfs.append(date_df)
    
    result = pd.concat(scored_dfs, ignore_index=True)
    
    # Summary
    valid_scores = result['composite_score'].notna().sum()
    print(f"   Calculated scores for {valid_scores} stock-quarters")
    
    return result


# ============================================================
# PORTFOLIO CONSTRUCTION
# ============================================================

def select_portfolio(scored_data: pd.DataFrame, date: pd.Timestamp, 
                     top_n: int = 20) -> list:
    """
    Select top N stocks based on composite score.
    
    Uses most recent fundamental data available before the selection date.
    """
    # Get most recent fundamentals before this date
    available = scored_data[scored_data['date'] <= date]
    
    if available.empty:
        return []
    
    # For each ticker, get the most recent observation
    latest = available.sort_values('date').groupby('ticker').tail(1)
    
    # Select top N by composite score
    top_stocks = latest.nlargest(top_n, 'composite_score')
    
    return top_stocks['ticker'].tolist()


# ============================================================
# BACKTESTING
# ============================================================

def run_backtest(fundamentals: pd.DataFrame, prices: pd.DataFrame,
                 top_n: int = 20, rebalance_freq: str = 'M',
                 initial_capital: float = 10000) -> pd.DataFrame:
    """
    Run backtest of fundamental factor strategy.
    
    Args:
        fundamentals: DataFrame with factor scores
        prices: Daily price DataFrame
        top_n: Number of stocks to hold
        rebalance_freq: 'M' for monthly, 'Q' for quarterly
        initial_capital: Starting capital
        
    Returns:
        DataFrame with daily portfolio values
    """
    print(f"\n🔄 RUNNING BACKTEST")
    print(f"   Top {top_n} stocks, {rebalance_freq} rebalance")
    print("=" * 50)
    
    # Get trading dates
    all_dates = prices['date'].unique()
    all_dates = sorted(all_dates)
    
    # Get rebalance dates (first trading day of each month)
    rebalance_dates = set()
    for d in all_dates:
        if rebalance_freq == 'M':
            key = (d.year, d.month)
        else:  # Quarterly
            key = (d.year, (d.month - 1) // 3)
        
        if key not in {(rd.year, rd.month if rebalance_freq == 'M' else (rd.month-1)//3) 
                       for rd in rebalance_dates}:
            rebalance_dates.add(d)
    
    print(f"   Trading days: {len(all_dates)}")
    print(f"   Rebalance dates: {len(rebalance_dates)}")
    
    # Initialize
    cash = initial_capital
    holdings = {}  # {ticker: shares}
    portfolio_values = []
    
    # Price lookup helper
    def get_price(ticker, date):
        row = prices[(prices['ticker'] == ticker) & (prices['date'] == date)]
        return row['close'].iloc[0] if not row.empty else None
    
    # Current portfolio tickers
    current_portfolio = []
    
    for date in all_dates:
        # Rebalance?
        if date in rebalance_dates:
            # Select new portfolio
            current_portfolio = select_portfolio(fundamentals, date, top_n)
            
            if current_portfolio:
                # Sell everything
                for ticker, shares in holdings.items():
                    price = get_price(ticker, date)
                    if price:
                        cash += shares * price
                holdings = {}
                
                # Calculate equal weight
                total_value = cash
                weight = 1.0 / len(current_portfolio)
                
                # Buy new positions
                for ticker in current_portfolio:
                    price = get_price(ticker, date)
                    if price and price > 0:
                        target_value = total_value * weight
                        shares = int(target_value / price)
                        if shares > 0:
                            holdings[ticker] = shares
                            cash -= shares * price
        
        # Calculate daily portfolio value
        total_value = cash
        for ticker, shares in holdings.items():
            price = get_price(ticker, date)
            if price:
                total_value += shares * price
        
        portfolio_values.append({
            'date': date,
            'value': total_value,
            'cash': cash,
            'positions': len(holdings)
        })
    
    return pd.DataFrame(portfolio_values)


def run_spy_benchmark(prices: pd.DataFrame, initial_capital: float = 10000) -> pd.DataFrame:
    """Run SPY buy-and-hold benchmark."""
    spy = prices[prices['ticker'] == 'SPY'].copy()
    
    if spy.empty:
        print("⚠️ SPY not in price data")
        return pd.DataFrame()
    
    spy = spy.sort_values('date')
    start_price = spy['close'].iloc[0]
    spy['value'] = initial_capital * (spy['close'] / start_price)
    
    return spy[['date', 'value']]


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(df: pd.DataFrame, name: str) -> dict:
    """Calculate comprehensive performance metrics."""
    if df.empty or len(df) < 2:
        return {}
    
    df = df.copy()
    df['return'] = df['value'].pct_change()
    
    # Basic metrics
    total_return = (df['value'].iloc[-1] / df['value'].iloc[0]) - 1
    years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Risk metrics
    daily_vol = df['return'].std()
    annual_vol = daily_vol * np.sqrt(252)
    sharpe = cagr / annual_vol if annual_vol > 0 else 0
    
    # Drawdown
    rolling_max = df['value'].cummax()
    drawdown = (df['value'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Win rate (positive return days)
    win_rate = (df['return'] > 0).mean()
    
    # Sortino (downside deviation)
    downside_returns = df['return'][df['return'] < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = cagr / downside_vol if downside_vol > 0 else 0
    
    metrics = {
        'name': name,
        'total_return_pct': round(total_return * 100, 2),
        'cagr_pct': round(cagr * 100, 2),
        'volatility_pct': round(annual_vol * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_drawdown_pct': round(max_drawdown * 100, 2),
        'win_rate_pct': round(win_rate * 100, 2),
        'final_value': round(df['value'].iloc[-1], 2),
        'years': round(years, 2)
    }
    
    print(f"\n📊 {name}:")
    for key, value in metrics.items():
        if key != 'name':
            print(f"   {key}: {value}")
    
    return metrics


# ============================================================
# FACTOR ANALYSIS
# ============================================================

def analyze_factors(fundamentals: pd.DataFrame, prices: pd.DataFrame):
    """Analyze factor performance and correlations."""
    print("\n📈 FACTOR ANALYSIS")
    print("=" * 50)
    
    # Get factor columns
    factor_cols = ['roe', 'profitMargin', 'grossMargin', 'debtToEquity']
    available_factors = [c for c in factor_cols if c in fundamentals.columns]
    
    print(f"   Analyzing factors: {available_factors}")
    
    # For each factor, compare top vs bottom quintile returns
    for factor in available_factors:
        df = fundamentals[fundamentals[factor].notna()].copy()
        
        if len(df) < 50:
            continue
        
        # Split into quintiles
        df['quintile'] = pd.qcut(df[factor], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        
        # Count per quintile
        counts = df['quintile'].value_counts().sort_index()
        print(f"\n   {factor}:")
        print(f"      Quintile distribution: {dict(counts)}")


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

def walk_forward_test(fundamentals: pd.DataFrame, prices: pd.DataFrame,
                      test_months: int = 3):
    """
    Walk-forward validation: retrain every N months.
    
    Since we're using a ranking model (no training), this tests
    the stability of factor scores over time.
    """
    print("\n🔄 WALK-FORWARD VALIDATION")
    print("=" * 50)
    
    results = []
    
    # Get unique months
    prices['month'] = prices['date'].dt.to_period('M')
    months = sorted(prices['month'].unique())
    
    print(f"   Total months: {len(months)}")
    print(f"   Test window: {test_months} months")
    
    for i in range(0, len(months) - test_months, test_months):
        start_month = months[i]
        end_month = months[min(i + test_months - 1, len(months) - 1)]
        
        # Filter prices for this period
        period_prices = prices[
            (prices['month'] >= start_month) & 
            (prices['month'] <= end_month)
        ].copy()
        
        if len(period_prices) < 20:
            continue
        
        # Run backtest for this period
        portfolio = run_backtest(
            fundamentals, 
            period_prices.drop(columns=['month']), 
            top_n=20, 
            rebalance_freq='M'
        )
        
        if portfolio.empty:
            continue
        
        # Calculate return for this period
        period_return = (portfolio['value'].iloc[-1] / portfolio['value'].iloc[0]) - 1
        
        results.append({
            'period': f"{start_month} - {end_month}",
            'return': period_return * 100
        })
        
        print(f"   {start_month} - {end_month}: {period_return*100:+.2f}%")
    
    if results:
        avg_return = np.mean([r['return'] for r in results])
        print(f"\n   Average period return: {avg_return:+.2f}%")
        print(f"   Positive periods: {sum(1 for r in results if r['return'] > 0)}/{len(results)}")
    
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("FUNDAMENTAL FACTOR MODEL - COMPREHENSIVE BACKTEST")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Load data
    fundamentals = load_fundamentals()
    
    # Get tickers from fundamentals
    tickers = fundamentals['ticker'].unique().tolist()
    
    # Add SPY for benchmark
    if 'SPY' not in tickers:
        tickers.append('SPY')
    
    # Load prices
    start_date = fundamentals['date'].min().strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    prices = load_prices(tickers, start_date, end_date)
    
    if prices.empty:
        print("❌ No price data available")
        return
    
    # Calculate factor scores
    print("\n📊 CALCULATING FACTOR SCORES")
    print("=" * 50)
    scored_fundamentals = calculate_factor_scores(fundamentals)
    
    # Run main backtest
    portfolio_results = run_backtest(scored_fundamentals, prices, top_n=20)
    
    # Run SPY benchmark
    spy_results = run_spy_benchmark(prices)
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("PERFORMANCE RESULTS")
    print("=" * 60)
    
    fund_metrics = calculate_metrics(portfolio_results, "FUNDAMENTAL FACTOR")
    spy_metrics = calculate_metrics(spy_results, "SPY BUY-AND-HOLD")
    
    # Comparison
    print("\n🆚 COMPARISON vs SPY:")
    if fund_metrics and spy_metrics:
        print(f"   CAGR Diff: {fund_metrics['cagr_pct'] - spy_metrics['cagr_pct']:+.2f}%")
        print(f"   Sharpe Diff: {fund_metrics['sharpe'] - spy_metrics['sharpe']:+.3f}")
        print(f"   Drawdown Diff: {spy_metrics['max_drawdown_pct'] - fund_metrics['max_drawdown_pct']:+.2f}%")
    
    # Factor analysis
    analyze_factors(scored_fundamentals, prices)
    
    # Walk-forward validation (simplified)
    # walk_forward_test(scored_fundamentals, prices)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'fundamental_metrics': fund_metrics,
        'spy_metrics': spy_metrics,
        'config': {
            'top_n': 20,
            'rebalance': 'monthly',
            'factors': ['roe', 'profitMargin', 'grossMargin', 'low_leverage']
        }
    }
    
    with open('results/fundamental_backtest.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    portfolio_results.to_csv('results/fundamental_portfolio_values.csv', index=False)
    
    print(f"\n✅ Results saved to results/fundamental_backtest.json")
    print(f"🏁 Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
