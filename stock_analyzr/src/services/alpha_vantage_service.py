import time
import pandas as pd
from collections import deque
from typing import Optional, Dict, Any, List
from datetime import datetime
from services.log_service import get_logger

logger = get_logger()

try:
    from alpha_vantage.techindicators import TechIndicators
    from alpha_vantage.timeseries import TimeSeries
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False
    logger.warning("Alpha Vantage library not installed. Falling back to simulated services.")

class AlphaVantageService:
    """Manages queries to Alpha Vantage with dynamic API rate limiting (5 calls/min)."""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        # Queue to track request timestamps for rate limiting
        self.request_queue = deque(maxlen=5)

    def _wait_for_rate_limit(self):
        """Enforces at most 5 requests per 60 seconds (minimum 12s between calls if queue full)."""
        now = time.time()
        self.request_queue.append(now)
        
        if len(self.request_queue) >= 5:
            elapsed = now - self.request_queue[0]
            if elapsed < 60:
                sleep_duration = 60.5 - elapsed
                logger.info(f"Alpha Vantage rate limit threshold hit. Pausing for {sleep_duration:.2f} seconds...")
                time.sleep(sleep_duration)

    def fetch_price_history(self, symbol: str) -> pd.DataFrame:
        """
        Fetch daily price history from Alpha Vantage.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            DataFrame with Open, High, Low, Close, Volume, Date columns
        """
        if not ALPHA_VANTAGE_AVAILABLE or not self.api_key or self.api_key == "YOUR_KEY_HERE":
            logger.warning("Alpha Vantage API Key missing or library not available. Skipping daily history fetch.")
            return pd.DataFrame()
            
        self._wait_for_rate_limit()
        logger.info(f"Fetching daily price history from Alpha Vantage for: {symbol}")
        
        try:
            ts = TimeSeries(key=self.api_key, output_format='pandas')
            # get_daily returns (df, meta_data)
            df, _ = ts.get_daily(symbol=symbol, outputsize='full')
            if df.empty:
                return pd.DataFrame()
                
            # Alpha Vantage daily returns index named 'date' and columns:
            # '1. open', '2. high', '3. low', '4. close', '5. volume'
            df = df.rename_axis('Date').reset_index()
            df = df.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            })
            
            df['Date'] = pd.to_datetime(df['Date'])
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].sort_values('Date').reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Alpha Vantage price download failed: {e}")
            return pd.DataFrame()

    def fetch_technical_indicators(self, symbol: str) -> pd.DataFrame:
        """
        Query Alpha Vantage for multiple technical indicators (RSI, EMA, MACD, SMA) and merge them.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            DataFrame with columns: Date, RSI, EMA_20, MACD, MACD_Signal, MACD_Hist, SMA_50
        """
        if not ALPHA_VANTAGE_AVAILABLE or not self.api_key or self.api_key == "YOUR_KEY_HERE":
            logger.warning("Alpha Vantage indicators skip (missing API Key/library).")
            return pd.DataFrame()
            
        logger.info(f"Querying technical indicators from Alpha Vantage for: {symbol}")
        try:
            self._wait_for_rate_limit()
            ti = TechIndicators(key=self.api_key, output_format='pandas')
            
            # Fetch RSI
            rsi_df, _ = ti.get_rsi(symbol=symbol, time_period=14, series_type='close')
            self._wait_for_rate_limit()
            # Fetch EMA
            ema_df, _ = ti.get_ema(symbol=symbol, time_period=20, series_type='close')
            self._wait_for_rate_limit()
            # Fetch MACD
            macd_df, _ = ti.get_macd(symbol=symbol, series_type='close')
            
            # Reset indices and format
            rsi_df = rsi_df.sort_index().reset_index().rename(columns={'date': 'Date', 'RSI': 'RSI'})
            ema_df = ema_df.sort_index().reset_index().rename(columns={'date': 'Date', 'EMA': 'EMA_20'})
            macd_df = macd_df.sort_index().reset_index().rename(columns={
                'date': 'Date', 
                'MACD': 'MACD', 
                'MACD_Signal': 'MACD_Signal', 
                'MACD_Hist': 'MACD_Hist'
            })
            
            # Standardize date types for merging
            for d in [rsi_df, ema_df, macd_df]:
                d['Date'] = pd.to_datetime(d['Date'])
                
            # Merge
            merged = pd.merge(rsi_df, ema_df, on='Date', how='outer')
            merged = pd.merge(merged, macd_df, on='Date', how='outer')
            
            logger.info(f"Successfully retrieved and merged Alpha Vantage indicators for {symbol}.")
            return merged
            
        except Exception as e:
            logger.error(f"Failed to fetch Alpha Vantage indicators: {e}")
            return pd.DataFrame()
