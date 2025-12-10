"""
SQLite Data Cache for Paper Trader - Phase 4

Caches OHLCV and macro data locally to:
1. Avoid rate limits from yfinance
2. Enable fast backtesting
3. Support incremental daily updates

Database: data/market.db
Tables: price_data, macro_data, cache_metadata
"""

import sqlite3
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

# Database path
DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "market.db")


class DataCache:
    """
    SQLite-based cache for market data.
    
    Usage:
        cache = DataCache()
        
        # Store data
        cache.update_price_data('AAPL', df)
        
        # Retrieve data
        df = cache.get_price_data('AAPL', '2020-01-01', '2024-12-09')
        
        # Check what's cached
        last_date = cache.get_last_date('AAPL')
    """
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._ensure_db_dir()
        self._init_db()
    
    def _ensure_db_dir(self):
        """Create database directory if it doesn't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _init_db(self):
        """Initialize database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # OHLCV price data for all tickers
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_data (
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    adj_close REAL,
                    PRIMARY KEY (ticker, date)
                )
            """)
            
            # Macro data (VIX, yields, etc.)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS macro_data (
                    series TEXT NOT NULL,
                    date TEXT NOT NULL,
                    value REAL,
                    PRIMARY KEY (series, date)
                )
            """)
            
            # Cache metadata for tracking updates
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    ticker TEXT PRIMARY KEY,
                    first_date TEXT,
                    last_date TEXT,
                    rows_cached INTEGER,
                    last_update TEXT
                )
            """)
            
            # Create indices for fast queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_price_ticker_date 
                ON price_data(ticker, date)
            """)
            
            conn.commit()
    
    # ==================== PRICE DATA ====================
    
    def get_price_data(
        self, 
        ticker: str, 
        start_date: str = None, 
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Retrieve cached OHLCV data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD), optional
            end_date: End date (YYYY-MM-DD), optional
            
        Returns:
            DataFrame with OHLCV data, DatetimeIndex
        """
        query = "SELECT * FROM price_data WHERE ticker = ?"
        params = [ticker]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        if df.empty:
            return pd.DataFrame()
        
        # Convert to proper format
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df.drop(columns=['ticker'])
        
        # Rename columns to match yfinance format
        df.columns = [c.title() for c in df.columns]
        df = df.rename(columns={'Adj_close': 'Adj Close'})
        
        return df
    
    def update_price_data(self, ticker: str, df: pd.DataFrame):
        """
        Update cache with new price data.
        
        Uses INSERT OR REPLACE to handle duplicates gracefully.
        
        Args:
            ticker: Stock ticker symbol
            df: DataFrame with OHLCV data (DatetimeIndex)
        """
        if df.empty:
            return
        
        # Prepare data for insertion
        df = df.copy()
        df = df.reset_index()
        
        # Normalize column names
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        
        # Handle index column name
        date_col = 'date' if 'date' in df.columns else df.columns[0]
        df = df.rename(columns={date_col: 'date'})
        
        # Convert date to string
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df['ticker'] = ticker
        
        # Ensure required columns exist
        required = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                df[col] = None
        
        if 'adj_close' not in df.columns:
            df['adj_close'] = df['close']
        
        with sqlite3.connect(self.db_path) as conn:
            # Insert or replace
            df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']].to_sql(
                'price_data',
                conn,
                if_exists='append',
                index=False,
                method='multi'
            )
            
            # Handle duplicates by keeping latest
            conn.execute("""
                DELETE FROM price_data 
                WHERE rowid NOT IN (
                    SELECT MIN(rowid) 
                    FROM price_data 
                    GROUP BY ticker, date
                )
            """)
            
            # Update metadata
            self._update_metadata(ticker, conn)
            conn.commit()
    
    def _update_metadata(self, ticker: str, conn):
        """Update cache metadata for a ticker."""
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT MIN(date), MAX(date), COUNT(*) 
            FROM price_data 
            WHERE ticker = ?
        """, (ticker,))
        
        result = cursor.fetchone()
        first_date, last_date, row_count = result
        
        cursor.execute("""
            INSERT OR REPLACE INTO cache_metadata 
            (ticker, first_date, last_date, rows_cached, last_update)
            VALUES (?, ?, ?, ?, ?)
        """, (ticker, first_date, last_date, row_count, datetime.now().isoformat()))
    
    def get_last_date(self, ticker: str) -> Optional[str]:
        """
        Get the last cached date for a ticker.
        
        Returns:
            Date string (YYYY-MM-DD) or None if not cached
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT last_date FROM cache_metadata WHERE ticker = ?", 
                (ticker,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
    
    def get_first_date(self, ticker: str) -> Optional[str]:
        """Get the first cached date for a ticker."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT first_date FROM cache_metadata WHERE ticker = ?", 
                (ticker,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
    
    def get_cached_tickers(self) -> List[str]:
        """Get list of all cached tickers."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT ticker FROM cache_metadata")
            return [row[0] for row in cursor.fetchall()]
    
    def get_cache_stats(self) -> pd.DataFrame:
        """Get cache statistics for all tickers."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                "SELECT * FROM cache_metadata ORDER BY ticker", 
                conn
            )
    
    # ==================== MACRO DATA ====================
    
    def get_macro_data(
        self, 
        series: str, 
        start_date: str = None, 
        end_date: str = None
    ) -> pd.DataFrame:
        """Retrieve cached macro data for a series (VIX, yields, etc.)."""
        query = "SELECT date, value FROM macro_data WHERE series = ?"
        params = [series]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        if df.empty:
            return pd.DataFrame()
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df.columns = [series]
        
        return df
    
    def update_macro_data(self, series: str, df: pd.DataFrame):
        """Update cache with new macro data."""
        if df.empty:
            return
        
        df = df.copy().reset_index()
        df.columns = ['date', 'value']
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df['series'] = series
        
        with sqlite3.connect(self.db_path) as conn:
            df[['series', 'date', 'value']].to_sql(
                'macro_data',
                conn,
                if_exists='append',
                index=False
            )
            
            # Remove duplicates
            conn.execute("""
                DELETE FROM macro_data 
                WHERE rowid NOT IN (
                    SELECT MIN(rowid) 
                    FROM macro_data 
                    GROUP BY series, date
                )
            """)
            conn.commit()
    
    # ==================== UTILITIES ====================
    
    def clear_ticker(self, ticker: str):
        """Remove all data for a ticker from cache."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM price_data WHERE ticker = ?", (ticker,))
            conn.execute("DELETE FROM cache_metadata WHERE ticker = ?", (ticker,))
            conn.commit()
    
    def clear_all(self):
        """Clear entire cache (use with caution)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM price_data")
            conn.execute("DELETE FROM macro_data")
            conn.execute("DELETE FROM cache_metadata")
            conn.commit()
    
    def vacuum(self):
        """Optimize database file size."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("VACUUM")


# Convenience function
def get_cache() -> DataCache:
    """Get default cache instance."""
    return DataCache()
