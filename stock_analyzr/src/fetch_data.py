"""
Data fetching module for stock analysis using Yahoo Finance and Alpha Vantage APIs.
Refactored to integrate with Clean Architecture Service Layer while preserving original interface.
"""

import pandas as pd
import numpy as np
import os
import heapq
from collections import deque
from typing import Dict, List, Tuple, Optional

# Add path compatibility
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.yahoo_service import YahooService
from services.alpha_vantage_service import AlphaVantageService
from services.log_service import get_logger

logger = get_logger()


class DataFetcher:
    """
    Data fetcher class demonstrating explicit data structure usage:
    - HashMap (dict): Store per-symbol data
    - Queue (deque): Buffer for streaming data
    - Stack (list): Track price peaks for reversal detection
    - Priority Queue (heapq): Rank stocks by performance
    
    Refactored to wrap and delegate to YahooService and AlphaVantageService.
    """
    
    def __init__(self, alpha_vantage_key: Optional[str] = None):
        """
        Initialize data fetcher with API key.
        
        Args:
            alpha_vantage_key: Alpha Vantage API key. If None, uses environment variable.
        """
        self.alpha_key = alpha_vantage_key or os.getenv("ALPHA_VANTAGE_API_KEY", "GQ1C7TJRZ4ANOZM9")
        self.data_cache = {}  # HashMap: symbol -> DataFrame
        self.request_queue = deque()  # Queue: manage API request timing
        self.price_stack = []  # Stack: track recent price peaks
        self.performance_heap = []  # Priority Queue: rank stocks by performance
        
        # Instantiate clean services
        self.yahoo_service = YahooService()
        self.alpha_service = AlphaVantageService(self.alpha_key)
        
    def fetch_yahoo_data(self, symbol: str, start: Optional[str] = None, 
                        end: Optional[str] = None, period: str = "1y", 
                        interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical data from Yahoo Finance.
        """
        try:
            df = self.yahoo_service.fetch_price_history(
                symbol=symbol, start=start, end=end, period=period, interval=interval
            )
            if not df.empty:
                # Add Symbol column for compatibility
                df['Symbol'] = symbol
                # Cache internally
                self.data_cache[symbol] = df
            return df
        except Exception as e:
            logger.error(f"Error in DataFetcher.fetch_yahoo_data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_alpha_indicators(self, symbol: str, interval: str = 'daily') -> pd.DataFrame:
        """
        Fetch technical indicators from Alpha Vantage.
        """
        try:
            df = self.alpha_service.fetch_technical_indicators(symbol)
            if not df.empty:
                # Rename columns for compatibility with old pipeline
                # Date, RSI, EMA_20, MACD, MACD_HIST, MACD_SIGNAL, SMA_50
                df = df.rename(columns={
                    'MACD_Signal': 'MACD_SIGNAL',
                    'MACD_Hist': 'MACD_HIST'
                })
                # Check for SMA_50, if missing calculate a simple SMA 50
                if 'SMA_50' not in df.columns:
                    # Fetch price history to compute SMA 50
                    hist_df = self.data_cache.get(symbol)
                    if hist_df is not None and not hist_df.empty:
                        df = df.merge(hist_df[['Date', 'Close']], on='Date', how='outer')
                        df['SMA_50'] = df['Close'].rolling(window=50).mean()
                        df = df.drop(columns=['Close'])
                    else:
                        df['SMA_50'] = np.nan
                        
                df['Symbol'] = symbol
                
            return df
        except Exception as e:
            logger.error(f"Error in DataFetcher.fetch_alpha_indicators for {symbol}: {e}")
            return pd.DataFrame()
    
    def merge_data(self, yahoo_df: pd.DataFrame, indicators_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge Yahoo Finance data with Alpha Vantage indicators.
        """
        if yahoo_df.empty:
            return pd.DataFrame()
        if indicators_df.empty:
            return yahoo_df.copy()
            
        # Convert dates to datetime.date
        y_df = yahoo_df.copy()
        ind_df = indicators_df.copy()
        
        y_df['Date'] = pd.to_datetime(y_df['Date']).dt.date
        ind_df['Date'] = pd.to_datetime(ind_df['Date']).dt.date
        
        # Merge on Date
        df = pd.merge(y_df, ind_df.drop(columns=['Symbol'], errors='ignore'), on='Date', how='left')
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df
    
    def detect_price_reversals(self, prices: List[float], threshold: float = 0.02) -> List[Tuple[int, float]]:
        """
        Detect price reversals using stack data structure.
        """
        reversals = []
        self.price_stack = []  # Reset stack
        
        for i, price in enumerate(prices):
            if not self.price_stack:
                self.price_stack.append((i, price))
            else:
                last_idx, last_price = self.price_stack[-1]
                
                if price > last_price * (1 + threshold):
                    self.price_stack.append((i, price))
                elif price < last_price * (1 - threshold):
                    while self.price_stack and self.price_stack[-1][1] > price:
                        popped_idx, popped_price = self.price_stack.pop()
                        reversals.append((popped_idx, popped_price))
        
        return reversals
    
    def rank_stocks_by_performance(self, symbols: List[str], lookback_days: int = 30) -> List[Tuple[str, float]]:
        """
        Rank stocks by recent performance using priority queue (heap).
        """
        self.performance_heap = []  # Reset heap
        
        for symbol in symbols:
            df = self.data_cache.get(symbol)
            if df is not None and len(df) >= lookback_days:
                recent_price = df['Close'].iloc[-1]
                old_price = df['Close'].iloc[-lookback_days]
                return_pct = (recent_price - old_price) / old_price
                heapq.heappush(self.performance_heap, (-return_pct, symbol, return_pct))
        
        top_performers = []
        while self.performance_heap:
            neg_return, symbol, actual_return = heapq.heappop(self.performance_heap)
            top_performers.append((symbol, actual_return))
        
        return top_performers
    
    def get_cached_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get cached data for symbol using HashMap lookup.
        """
        return self.data_cache.get(symbol)
    
    def clear_cache(self):
        """Clear all cached data."""
        self.data_cache.clear()
        self.request_queue.clear()
        self.price_stack.clear()
        self.performance_heap.clear()


def fetch_multiple_symbols(symbols: List[str], fetcher: DataFetcher) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for multiple symbols demonstrating HashMap usage.
    """
    data_map = {}
    for symbol in symbols:
        logger.info(f"Fetching data for {symbol}...")
        yahoo_data = fetcher.fetch_yahoo_data(symbol, period="2y")
        indicators = fetcher.fetch_alpha_indicators(symbol)
        merged_data = fetcher.merge_data(yahoo_data, indicators)
        
        if not merged_data.empty:
            data_map[symbol] = merged_data
            logger.info(f"Successfully fetched {len(merged_data)} records for {symbol}")
        else:
            logger.error(f"Failed to fetch data for {symbol}")
    return data_map
