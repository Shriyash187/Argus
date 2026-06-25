"""
Data Structure Helpers Module
Demonstrates explicit implementation and usage of core data structures:
- Queue (deque) for sliding windows
- Stack for pattern detection
- HashMap (dict) for efficient lookups
- Priority Queue (heapq) for ranking
"""

from collections import deque
import heapq
from typing import Any, List, Tuple, Optional, Dict
import pandas as pd
import numpy as np


class SlidingWindowQueue:
    """
    Queue-based sliding window implementation for time series data.
    Demonstrates FIFO behavior and fixed capacity.
    """
    
    def __init__(self, max_size: int):
        """
        Initialize sliding window queue.
        
        Args:
            max_size: Maximum number of elements in the queue
        """
        self.queue = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add(self, item: Any) -> None:
        """
        Add item to queue. Automatically removes oldest if at capacity.
        
        Args:
            item: Item to add
        """
        self.queue.append(item)
    
    def get_window(self) -> List[Any]:
        """
        Get current window as list.
        
        Returns:
            List of items in current window
        """
        return list(self.queue)
    
    def is_full(self) -> bool:
        """
        Check if queue is at maximum capacity.
        
        Returns:
            True if queue is full
        """
        return len(self.queue) == self.max_size
    
    def size(self) -> int:
        """
        Get current size of queue.
        
        Returns:
            Number of items in queue
        """
        return len(self.queue)
    
    def clear(self) -> None:
        """Clear all items from queue."""
        self.queue.clear()


class PricePatternStack:
    """
    Stack-based implementation for detecting price patterns and reversals.
    Demonstrates LIFO behavior for pattern tracking.
    """
    
    def __init__(self):
        """Initialize empty stack."""
        self.stack = []
    
    def push_peak(self, index: int, price: float, timestamp: Optional[str] = None) -> None:
        """
        Push a price peak onto the stack.
        
        Args:
            index: Index of the price point
            price: Price value
            timestamp: Optional timestamp
        """
        self.stack.append({
            'index': index,
            'price': price,
            'timestamp': timestamp
        })
    
    def pop_peak(self) -> Optional[Dict[str, Any]]:
        """
        Pop the most recent peak from the stack.
        
        Returns:
            Dictionary with peak information or None if stack is empty
        """
        if self.stack:
            return self.stack.pop()
        return None
    
    def peek_peak(self) -> Optional[Dict[str, Any]]:
        """
        Peek at the most recent peak without removing it.
        
        Returns:
            Dictionary with peak information or None if stack is empty
        """
        if self.stack:
            return self.stack[-1]
        return None
    
    def detect_reversal(self, current_price: float, threshold: float = 0.02) -> bool:
        """
        Detect if current price represents a reversal from recent peak.
        
        Args:
            current_price: Current price to check
            threshold: Minimum percentage change to consider reversal
            
        Returns:
            True if reversal detected
        """
        if not self.stack:
            return False
        
        recent_peak = self.stack[-1]['price']
        return current_price < recent_peak * (1 - threshold)
    
    def get_all_peaks(self) -> List[Dict[str, Any]]:
        """
        Get all peaks in the stack.
        
        Returns:
            List of all peak dictionaries
        """
        return self.stack.copy()
    
    def clear(self) -> None:
        """Clear all peaks from stack."""
        self.stack.clear()


class StockDataHashMap:
    """
    HashMap-based storage for stock data with efficient lookups.
    Demonstrates O(1) average case lookup time.
    """
    
    def __init__(self):
        """Initialize empty hash map."""
        self.data_map = {}
        self.metadata = {}
    
    def store_data(self, symbol: str, data: pd.DataFrame, 
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store stock data in hash map.
        
        Args:
            symbol: Stock symbol as key
            data: DataFrame containing stock data
            metadata: Optional metadata about the data
        """
        self.data_map[symbol] = data.copy()
        self.metadata[symbol] = metadata or {}
    
    def get_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Retrieve stock data by symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame if found, None otherwise
        """
        return self.data_map.get(symbol)
    
    def get_metadata(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Metadata dictionary if found, None otherwise
        """
        return self.metadata.get(symbol)
    
    def has_symbol(self, symbol: str) -> bool:
        """
        Check if symbol exists in hash map.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if symbol exists
        """
        return symbol in self.data_map
    
    def get_all_symbols(self) -> List[str]:
        """
        Get all symbols in the hash map.
        
        Returns:
            List of all symbols
        """
        return list(self.data_map.keys())
    
    def remove_symbol(self, symbol: str) -> bool:
        """
        Remove symbol and its data from hash map.
        
        Args:
            symbol: Stock symbol to remove
            
        Returns:
            True if symbol was removed, False if not found
        """
        if symbol in self.data_map:
            del self.data_map[symbol]
            if symbol in self.metadata:
                del self.metadata[symbol]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all data from hash map."""
        self.data_map.clear()
        self.metadata.clear()
    
    def size(self) -> int:
        """
        Get number of symbols in hash map.
        
        Returns:
            Number of symbols stored
        """
        return len(self.data_map)


class StockRankingPriorityQueue:
    """
    Priority Queue-based ranking system for stocks.
    Demonstrates heap operations for efficient ranking.
    """
    
    def __init__(self):
        """Initialize empty priority queue."""
        self.heap = []
        self.ranking_criteria = {}
    
    def add_stock(self, symbol: str, score: float, criteria: str = "performance") -> None:
        """
        Add stock to priority queue with score.
        
        Args:
            symbol: Stock symbol
            score: Score for ranking (higher is better)
            criteria: Criteria used for scoring
        """
        # Use negative score for max-heap behavior (heapq is min-heap)
        heapq.heappush(self.heap, (-score, symbol, criteria))
        self.ranking_criteria[symbol] = criteria
    
    def get_top_stocks(self, n: int = 10) -> List[Tuple[str, float, str]]:
        """
        Get top N stocks from priority queue.
        
        Args:
            n: Number of top stocks to return
            
        Returns:
            List of (symbol, score, criteria) tuples
        """
        top_stocks = []
        temp_heap = self.heap.copy()
        
        for _ in range(min(n, len(temp_heap))):
            if temp_heap:
                neg_score, symbol, criteria = heapq.heappop(temp_heap)
                actual_score = -neg_score
                top_stocks.append((symbol, actual_score, criteria))
        
        return top_stocks
    
    def update_stock_score(self, symbol: str, new_score: float) -> bool:
        """
        Update score for existing stock.
        
        Args:
            symbol: Stock symbol
            new_score: New score
            
        Returns:
            True if stock was updated, False if not found
        """
        if symbol in self.ranking_criteria:
            # Remove old entry and add new one
            self.remove_stock(symbol)
            self.add_stock(symbol, new_score, self.ranking_criteria[symbol])
            return True
        return False
    
    def remove_stock(self, symbol: str) -> bool:
        """
        Remove stock from priority queue.
        
        Args:
            symbol: Stock symbol to remove
            
        Returns:
            True if stock was removed, False if not found
        """
        if symbol in self.ranking_criteria:
            # Rebuild heap without the symbol
            new_heap = []
            for neg_score, sym, criteria in self.heap:
                if sym != symbol:
                    new_heap.append((neg_score, sym, criteria))
            
            heapq.heapify(new_heap)
            self.heap = new_heap
            del self.ranking_criteria[symbol]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all stocks from priority queue."""
        self.heap.clear()
        self.ranking_criteria.clear()
    
    def size(self) -> int:
        """
        Get number of stocks in priority queue.
        
        Returns:
            Number of stocks
        """
        return len(self.heap)


def create_sequences_with_queue(data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences using sliding window queue.
    
    Args:
        data: Input data array (n_samples, n_features)
        sequence_length: Length of each sequence
        
    Returns:
        Tuple of (X, y) where X is sequences and y is targets
    """
    queue = SlidingWindowQueue(sequence_length)
    X, y = [], []
    
    for i in range(len(data)):
        queue.add(data[i])
        
        if queue.is_full() and i + 1 < len(data):
            X.append(np.array(queue.get_window()))
            y.append(data[i + 1, 0])  # Predict next day's close price
    
    return np.array(X), np.array(y)


def detect_support_resistance_levels(prices: List[float], window_size: int = 20) -> Tuple[List[float], List[float]]:
    """
    Detect support and resistance levels using stack-based approach.
    
    Args:
        prices: List of price values
        window_size: Window size for local extrema detection
        
    Returns:
        Tuple of (support_levels, resistance_levels)
    """
    stack = PricePatternStack()
    support_levels = []
    resistance_levels = []
    
    for i in range(len(prices)):
        current_price = prices[i]
        
        # Check for local maxima (resistance)
        if i >= window_size // 2 and i < len(prices) - window_size // 2:
            window_prices = prices[i - window_size // 2:i + window_size // 2 + 1]
            if current_price == max(window_prices):
                stack.push_peak(i, current_price)
                resistance_levels.append(current_price)
        
        # Check for local minima (support)
        if i >= window_size // 2 and i < len(prices) - window_size // 2:
            window_prices = prices[i - window_size // 2:i + window_size // 2 + 1]
            if current_price == min(window_prices):
                # Check if this breaks through previous resistance
                if stack.peek_peak() and current_price < stack.peek_peak()['price'] * 0.95:
                    support_levels.append(current_price)
    
    return support_levels, resistance_levels


def rank_stocks_by_multiple_criteria(data_map: Dict[str, pd.DataFrame]) -> Dict[str, List[Tuple[str, float]]]:
    """
    Rank stocks by multiple criteria using priority queues.
    
    Args:
        data_map: Dictionary mapping symbols to their data
        
    Returns:
        Dictionary with rankings by different criteria
    """
    rankings = {
        'performance': StockRankingPriorityQueue(),
        'volatility': StockRankingPriorityQueue(),
        'volume': StockRankingPriorityQueue()
    }
    
    for symbol, df in data_map.items():
        if len(df) < 30:  # Need enough data
            continue
        
        # Performance ranking (30-day return)
        recent_return = (df['Close'].iloc[-1] - df['Close'].iloc[-30]) / df['Close'].iloc[-30]
        rankings['performance'].add_stock(symbol, recent_return, '30_day_return')
        
        # Volatility ranking (30-day standard deviation)
        volatility = df['Close'].tail(30).std()
        rankings['volatility'].add_stock(symbol, volatility, '30_day_volatility')
        
        # Volume ranking (average volume)
        avg_volume = df['Volume'].tail(30).mean()
        rankings['volume'].add_stock(symbol, avg_volume, 'avg_volume')
    
    # Get top 5 for each criteria
    result = {}
    for criteria, pq in rankings.items():
        result[criteria] = pq.get_top_stocks(5)
    
    return result


# Example usage and testing
if __name__ == "__main__":
    print("Testing Data Structure Implementations")
    
    # Test SlidingWindowQueue
    print("\n1. Testing SlidingWindowQueue:")
    queue = SlidingWindowQueue(5)
    for i in range(10):
        queue.add(i)
        print(f"Added {i}, window: {queue.get_window()}")
    
    # Test PricePatternStack
    print("\n2. Testing PricePatternStack:")
    stack = PricePatternStack()
    prices = [100, 105, 110, 108, 115, 112, 120, 118, 125, 122]
    
    for i, price in enumerate(prices):
        if i > 0 and price > prices[i-1] * 1.02:  # 2% increase
            stack.push_peak(i, price)
            print(f"Pushed peak at index {i}, price {price}")
        
        if stack.detect_reversal(price):
            peak = stack.pop_peak()
            print(f"Reversal detected! Popped peak: {peak}")
    
    # Test StockDataHashMap
    print("\n3. Testing StockDataHashMap:")
    hash_map = StockDataHashMap()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=10),
        'Close': [100 + i for i in range(10)],
        'Volume': [1000 + i*100 for i in range(10)]
    })
    
    hash_map.store_data('AAPL', sample_data, {'source': 'test'})
    print(f"Stored AAPL data: {hash_map.has_symbol('AAPL')}")
    print(f"Retrieved data shape: {hash_map.get_data('AAPL').shape}")
    
    # Test StockRankingPriorityQueue
    print("\n4. Testing StockRankingPriorityQueue:")
    pq = StockRankingPriorityQueue()
    
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    scores = [0.15, 0.08, 0.12, 0.25]
    
    for stock, score in zip(stocks, scores):
        pq.add_stock(stock, score)
    
    top_stocks = pq.get_top_stocks(3)
    print(f"Top 3 stocks: {top_stocks}")
    
    print("\nAll tests completed successfully!")
