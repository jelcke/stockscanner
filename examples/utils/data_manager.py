"""
Data Manager Utilities
=====================
Utilities for managing and processing stock data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
from functools import lru_cache
import asyncio
import aiohttp

logger = logging.getLogger(__name__)


class DataCache:
    """Simple in-memory cache for frequently accessed data"""
    
    def __init__(self, ttl: int = 300):
        """
        Initialize cache.
        
        Args:
            ttl: Time to live in seconds
        """
        self.ttl = ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        if key in self._cache:
            item = self._cache[key]
            if datetime.now() < item['expires']:
                return item['data']
            else:
                del self._cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache"""
        self._cache[key] = {
            'data': value,
            'expires': datetime.now() + timedelta(seconds=self.ttl)
        }
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self._cache.clear()
    
    def cleanup(self) -> None:
        """Remove expired entries"""
        now = datetime.now()
        expired_keys = [
            key for key, item in self._cache.items()
            if now >= item['expires']
        ]
        for key in expired_keys:
            del self._cache[key]


class DataValidator:
    """Validate and clean stock data"""
    
    @staticmethod
    def validate_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate OHLCV data.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            Cleaned DataFrame
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Check required columns
        missing_cols = set(required_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove duplicates
        data = data[~data.index.duplicated(keep='first')]
        
        # Sort by date
        data = data.sort_index()
        
        # Clean invalid values
        data = data[data['high'] >= data['low']]
        data = data[data['high'] >= data['open']]
        data = data[data['high'] >= data['close']]
        data = data[data['low'] <= data['open']]
        data = data[data['low'] <= data['close']]
        
        # Remove zero or negative prices
        data = data[(data[['open', 'high', 'low', 'close']] > 0).all(axis=1)]
        
        # Handle missing values
        data['volume'] = data['volume'].fillna(0)
        
        # Forward fill price data
        price_cols = ['open', 'high', 'low', 'close']
        data[price_cols] = data[price_cols].fillna(method='ffill')
        
        return data
    
    @staticmethod
    def detect_outliers(data: pd.Series, n_std: float = 3) -> pd.Series:
        """
        Detect outliers using z-score method.
        
        Args:
            data: Series to check
            n_std: Number of standard deviations
            
        Returns:
            Boolean series marking outliers
        """
        mean = data.mean()
        std = data.std()
        z_scores = np.abs((data - mean) / std)
        return z_scores > n_std
    
    @staticmethod
    def clean_outliers(data: pd.DataFrame, columns: List[str], n_std: float = 3) -> pd.DataFrame:
        """
        Clean outliers from specified columns.
        
        Args:
            data: DataFrame to clean
            columns: Columns to check for outliers
            n_std: Number of standard deviations
            
        Returns:
            Cleaned DataFrame
        """
        clean_data = data.copy()
        
        for col in columns:
            if col in clean_data.columns:
                outliers = DataValidator.detect_outliers(clean_data[col], n_std)
                clean_data.loc[outliers, col] = np.nan
                clean_data[col] = clean_data[col].fillna(method='ffill')
        
        return clean_data


class DataManager:
    """Manage stock data storage and retrieval"""
    
    def __init__(self, db_path: str = "data/scanner.db", cache_ttl: int = 300):
        """
        Initialize data manager.
        
        Args:
            db_path: Path to SQLite database
            cache_ttl: Cache time to live in seconds
        """
        self.db_path = db_path
        self.cache = DataCache(ttl=cache_ttl)
        self.validator = DataValidator()
        
        # Create database directory if needed
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_data (
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (symbol, date)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_info (
                    symbol TEXT PRIMARY KEY,
                    company_name TEXT,
                    sector TEXT,
                    industry TEXT,
                    market_cap REAL,
                    last_updated TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scan_results (
                    scan_id TEXT,
                    timestamp TIMESTAMP,
                    symbol TEXT,
                    criteria TEXT,
                    score REAL,
                    metadata TEXT
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_stock_data_symbol ON stock_data(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_stock_data_date ON stock_data(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_scan_results_timestamp ON scan_results(timestamp)")
    
    def save_stock_data(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Save stock data to database.
        
        Args:
            symbol: Stock symbol
            data: DataFrame with OHLCV data
        """
        # Validate data
        data = self.validator.validate_ohlcv(data)
        
        # Prepare data for insertion
        records = []
        for date, row in data.iterrows():
            records.append((
                symbol,
                date.date() if hasattr(date, 'date') else date,
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['volume']
            ))
        
        # Insert data
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO stock_data 
                (symbol, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, records)
        
        # Clear cache for this symbol
        cache_key = f"stock_data_{symbol}"
        if cache_key in self.cache._cache:
            del self.cache._cache[cache_key]
    
    @lru_cache(maxsize=100)
    def load_stock_data(self, 
                       symbol: str, 
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Load stock data from database.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            DataFrame with OHLCV data or None
        """
        # Check cache first
        cache_key = f"stock_data_{symbol}_{start_date}_{end_date}"
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Build query
        query = "SELECT date, open, high, low, close, volume FROM stock_data WHERE symbol = ?"
        params = [symbol]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date.date() if hasattr(start_date, 'date') else start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date.date() if hasattr(end_date, 'date') else end_date)
        
        query += " ORDER BY date"
        
        # Execute query
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])
        
        if df.empty:
            return None
        
        # Set date as index
        df.set_index('date', inplace=True)
        
        # Cache result
        self.cache.set(cache_key, df)
        
        return df
    
    def save_stock_info(self, symbol: str, info: Dict[str, Any]) -> None:
        """
        Save stock information.
        
        Args:
            symbol: Stock symbol
            info: Dictionary with stock information
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO stock_info
                (symbol, company_name, sector, industry, market_cap, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                info.get('company_name'),
                info.get('sector'),
                info.get('industry'),
                info.get('market_cap'),
                datetime.now()
            ))
    
    def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get stock information.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with stock info or None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT company_name, sector, industry, market_cap, last_updated
                FROM stock_info WHERE symbol = ?
            """, (symbol,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'symbol': symbol,
                    'company_name': row[0],
                    'sector': row[1],
                    'industry': row[2],
                    'market_cap': row[3],
                    'last_updated': row[4]
                }
        
        return None
    
    def save_scan_results(self, scan_id: str, results: List[Dict[str, Any]]) -> None:
        """
        Save scan results.
        
        Args:
            scan_id: Unique scan identifier
            results: List of scan results
        """
        records = []
        timestamp = datetime.now()
        
        for result in results:
            records.append((
                scan_id,
                timestamp,
                result['symbol'],
                json.dumps(result.get('criteria', {})),
                result.get('score', 0),
                json.dumps(result.get('metadata', {}))
            ))
        
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT INTO scan_results
                (scan_id, timestamp, symbol, criteria, score, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, records)
    
    def get_scan_history(self, 
                        symbol: Optional[str] = None,
                        days_back: int = 30) -> pd.DataFrame:
        """
        Get scan history.
        
        Args:
            symbol: Filter by symbol (optional)
            days_back: Number of days to look back
            
        Returns:
            DataFrame with scan history
        """
        query = """
            SELECT scan_id, timestamp, symbol, criteria, score, metadata
            FROM scan_results
            WHERE timestamp >= ?
        """
        params = [datetime.now() - timedelta(days=days_back)]
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        query += " ORDER BY timestamp DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['timestamp'])
        
        # Parse JSON fields
        if not df.empty:
            df['criteria'] = df['criteria'].apply(json.loads)
            df['metadata'] = df['metadata'].apply(json.loads)
        
        return df
    
    async def fetch_external_data(self, 
                                 symbol: str, 
                                 source: str = "yahoo") -> Optional[pd.DataFrame]:
        """
        Fetch data from external sources.
        
        Args:
            symbol: Stock symbol
            source: Data source
            
        Returns:
            DataFrame with stock data or None
        """
        # This is a placeholder for external data fetching
        # In practice, you would implement actual API calls here
        
        if source == "yahoo":
            # Example Yahoo Finance API call
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            # Parse and return data
                            # This is simplified - actual implementation would be more complex
                            return None
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
                    return None
        
        return None
    
    def export_to_csv(self, 
                     symbols: List[str], 
                     output_dir: str = "exports",
                     include_indicators: bool = True) -> None:
        """
        Export stock data to CSV files.
        
        Args:
            symbols: List of symbols to export
            output_dir: Output directory
            include_indicators: Whether to include calculated indicators
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for symbol in symbols:
            data = self.load_stock_data(symbol)
            if data is not None:
                if include_indicators:
                    # Add basic indicators
                    data['sma_20'] = data['close'].rolling(20).mean()
                    data['sma_50'] = data['close'].rolling(50).mean()
                    data['volume_sma'] = data['volume'].rolling(20).mean()
                
                filename = output_path / f"{symbol}_data.csv"
                data.to_csv(filename)
                logger.info(f"Exported {symbol} to {filename}")
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> None:
        """
        Clean up old data from database.
        
        Args:
            days_to_keep: Number of days of data to keep
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with sqlite3.connect(self.db_path) as conn:
            # Clean old stock data
            conn.execute("""
                DELETE FROM stock_data WHERE date < ?
            """, (cutoff_date.date(),))
            
            # Clean old scan results
            conn.execute("""
                DELETE FROM scan_results WHERE timestamp < ?
            """, (cutoff_date,))
            
            # Vacuum database to reclaim space
            conn.execute("VACUUM")
        
        logger.info(f"Cleaned up data older than {cutoff_date.date()}")