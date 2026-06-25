"""
Technical Indicators Module
Computes various technical indicators for stock analysis.
Demonstrates data structure usage in indicator calculations.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from collections import deque
import heapq


class TechnicalIndicators:
    """
    Technical indicators calculator with explicit data structure usage.
    """
    
    def __init__(self):
        """Initialize technical indicators calculator."""
        self.price_history = deque(maxlen=1000)  # Queue for price history
        self.volume_history = deque(maxlen=1000)  # Queue for volume history
        
    def add_price_data(self, price: float, volume: float) -> None:
        """
        Add new price and volume data to history queues.
        
        Args:
            price: Closing price
            volume: Trading volume
        """
        self.price_history.append(price)
        self.volume_history.append(volume)
    
    def calculate_sma(self, period: int = 20) -> Optional[float]:
        """
        Calculate Simple Moving Average using queue.
        
        Args:
            period: Period for SMA calculation
            
        Returns:
            SMA value or None if insufficient data
        """
        if len(self.price_history) < period:
            return None
        
        # Use last 'period' prices from queue
        recent_prices = list(self.price_history)[-period:]
        return sum(recent_prices) / period
    
    def calculate_ema(self, period: int = 20, previous_ema: Optional[float] = None) -> Optional[float]:
        """
        Calculate Exponential Moving Average.
        
        Args:
            period: Period for EMA calculation
            previous_ema: Previous EMA value for recursive calculation
            
        Returns:
            EMA value or None if insufficient data
        """
        if len(self.price_history) < period:
            return None
        
        current_price = self.price_history[-1]
        
        if previous_ema is None:
            # Calculate SMA for first EMA value
            sma = self.calculate_sma(period)
            return sma
        
        # EMA formula: EMA = (Price - Previous_EMA) * Multiplier + Previous_EMA
        multiplier = 2 / (period + 1)
        ema = (current_price - previous_ema) * multiplier + previous_ema
        
        return ema
    
    def calculate_rsi(self, period: int = 14) -> Optional[float]:
        """
        Calculate Relative Strength Index using queue.
        
        Args:
            period: Period for RSI calculation
            
        Returns:
            RSI value or None if insufficient data
        """
        if len(self.price_history) < period + 1:
            return None
        
        # Get price changes from queue
        prices = list(self.price_history)[-(period + 1):]
        price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Separate gains and losses
        gains = [change if change > 0 else 0 for change in price_changes]
        losses = [-change if change < 0 else 0 for change in price_changes]
        
        # Calculate average gains and losses
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100
        
        # Calculate RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Optional[Dict[str, float]]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            
        Returns:
            Dictionary with MACD, Signal, and Histogram values
        """
        if len(self.price_history) < slow_period:
            return None
        
        # Calculate EMAs
        ema_fast = self._calculate_ema_recursive(fast_period)
        ema_slow = self._calculate_ema_recursive(slow_period)
        
        if ema_fast is None or ema_slow is None:
            return None
        
        macd_line = ema_fast - ema_slow
        
        # For signal line, we need MACD history
        # This is simplified - in practice, you'd maintain MACD history
        signal_line = macd_line * 0.9  # Simplified signal calculation
        histogram = macd_line - signal_line
        
        return {
            'MACD': macd_line,
            'MACD_SIGNAL': signal_line,
            'MACD_HIST': histogram
        }
    
    def _calculate_ema_recursive(self, period: int) -> Optional[float]:
        """Calculate EMA recursively using all available data."""
        if len(self.price_history) < period:
            return None
        
        prices = list(self.price_history)
        multiplier = 2 / (period + 1)
        
        # Start with SMA
        ema = sum(prices[:period]) / period
        
        # Calculate EMA for remaining prices
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def calculate_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> Optional[Dict[str, float]]:
        """
        Calculate Bollinger Bands using queue.
        
        Args:
            period: Period for calculation
            std_dev: Standard deviation multiplier
            
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        if len(self.price_history) < period:
            return None
        
        recent_prices = list(self.price_history)[-period:]
        
        # Calculate SMA (middle band)
        sma = sum(recent_prices) / period
        
        # Calculate standard deviation
        variance = sum((price - sma) ** 2 for price in recent_prices) / period
        std = np.sqrt(variance)
        
        # Calculate bands
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return {
            'BB_UPPER': upper_band,
            'BB_MIDDLE': sma,
            'BB_LOWER': lower_band
        }
    
    def calculate_stochastic(self, k_period: int = 14, d_period: int = 3) -> Optional[Dict[str, float]]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            k_period: %K period
            d_period: %D period (SMA of %K)
            
        Returns:
            Dictionary with %K and %D values
        """
        if len(self.price_history) < k_period:
            return None
        
        recent_prices = list(self.price_history)[-k_period:]
        
        # Calculate %K
        lowest_low = min(recent_prices)
        highest_high = max(recent_prices)
        current_price = recent_prices[-1]
        
        if highest_high == lowest_low:
            k_percent = 50  # Neutral when no range
        else:
            k_percent = ((current_price - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Calculate %D (simplified - would need %K history for proper calculation)
        d_percent = k_percent  # Simplified
        
        return {
            'STOCH_K': k_percent,
            'STOCH_D': d_percent
        }
    
    def detect_support_resistance(self, window_size: int = 20) -> Dict[str, List[float]]:
        """
        Detect support and resistance levels using stack-based approach.
        
        Args:
            window_size: Window size for local extrema detection
            
        Returns:
            Dictionary with support and resistance levels
        """
        if len(self.price_history) < window_size:
            return {'support': [], 'resistance': []}
        
        prices = list(self.price_history)
        support_levels = []
        resistance_levels = []
        
        # Use stack to track peaks and troughs
        peak_stack = []
        trough_stack = []
        
        for i in range(window_size // 2, len(prices) - window_size // 2):
            current_price = prices[i]
            
            # Check for local maximum (resistance)
            window_prices = prices[i - window_size // 2:i + window_size // 2 + 1]
            if current_price == max(window_prices):
                peak_stack.append(current_price)
                resistance_levels.append(current_price)
            
            # Check for local minimum (support)
            if current_price == min(window_prices):
                trough_stack.append(current_price)
                support_levels.append(current_price)
        
        return {
            'support': support_levels,
            'resistance': resistance_levels
        }
    
    def calculate_volume_indicators(self) -> Optional[Dict[str, float]]:
        """
        Calculate volume-based indicators.
        
        Returns:
            Dictionary with volume indicators
        """
        if len(self.volume_history) < 20:
            return None
        
        recent_volumes = list(self.volume_history)[-20:]
        current_volume = recent_volumes[-1]
        
        # Volume moving average
        volume_ma = sum(recent_volumes) / len(recent_volumes)
        
        # Volume ratio
        volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1.0
        
        # On-Balance Volume (simplified)
        obv = 0
        prices = list(self.price_history)[-20:]
        volumes = recent_volumes
        
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                obv += volumes[i]
            elif prices[i] < prices[i-1]:
                obv -= volumes[i]
        
        return {
            'VOLUME_MA': volume_ma,
            'VOLUME_RATIO': volume_ratio,
            'OBV': obv
        }


class IndicatorRanker:
    """
    Priority queue-based ranking system for indicators.
    """
    
    def __init__(self):
        """Initialize indicator ranker."""
        self.indicator_heap = []
        self.ranking_criteria = {}
    
    def add_indicator_score(self, symbol: str, indicator: str, score: float, 
                          criteria: str = "performance") -> None:
        """
        Add indicator score to ranking system.
        
        Args:
            symbol: Stock symbol
            indicator: Indicator name
            score: Score value
            criteria: Ranking criteria
        """
        key = f"{symbol}_{indicator}"
        # Use negative score for max-heap behavior
        heapq.heappush(self.indicator_heap, (-score, key, symbol, indicator, score, criteria))
        self.ranking_criteria[key] = criteria
    
    def get_top_indicators(self, n: int = 10) -> List[Tuple[str, str, float, str]]:
        """
        Get top N indicators by score.
        
        Args:
            n: Number of top indicators to return
            
        Returns:
            List of (symbol, indicator, score, criteria) tuples
        """
        top_indicators = []
        temp_heap = self.indicator_heap.copy()
        
        for _ in range(min(n, len(temp_heap))):
            if temp_heap:
                neg_score, key, symbol, indicator, actual_score, criteria = heapq.heappop(temp_heap)
                top_indicators.append((symbol, indicator, actual_score, criteria))
        
        return top_indicators
    
    def rank_by_rsi_signals(self, symbol_indicator_data: Dict[str, Dict[str, float]]) -> List[Tuple[str, float]]:
        """
        Rank stocks by RSI signal strength.
        
        Args:
            symbol_indicator_data: Dictionary with symbol -> indicators mapping
            
        Returns:
            List of (symbol, rsi_signal_strength) tuples
        """
        rsi_rankings = []
        
        for symbol, indicators in symbol_indicator_data.items():
            if 'RSI' in indicators:
                rsi = indicators['RSI']
                
                # Calculate signal strength based on RSI position
                if rsi < 30:  # Oversold
                    signal_strength = (30 - rsi) / 30  # Higher is stronger oversold signal
                elif rsi > 70:  # Overbought
                    signal_strength = (rsi - 70) / 30  # Higher is stronger overbought signal
                else:  # Neutral
                    signal_strength = 0.0
                
                rsi_rankings.append((symbol, signal_strength))
        
        # Sort by signal strength (descending)
        rsi_rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rsi_rankings


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators for a DataFrame.
    
    Args:
        df: DataFrame with 'Close' and 'Volume' columns
        
    Returns:
        DataFrame with added indicator columns
    """
    df = df.copy()
    indicator_calc = TechnicalIndicators()
    
    # Initialize indicator columns
    df['SMA_20'] = np.nan
    df['EMA_20'] = np.nan
    df['RSI'] = np.nan
    df['MACD'] = np.nan
    df['MACD_SIGNAL'] = np.nan
    df['MACD_HIST'] = np.nan
    df['BB_UPPER'] = np.nan
    df['BB_MIDDLE'] = np.nan
    df['BB_LOWER'] = np.nan
    df['STOCH_K'] = np.nan
    df['STOCH_D'] = np.nan
    df['VOLUME_MA'] = np.nan
    df['VOLUME_RATIO'] = np.nan
    df['OBV'] = np.nan
    
    # Calculate indicators row by row
    for i in range(len(df)):
        close_price = df.iloc[i]['Close']
        volume = df.iloc[i]['Volume']
        
        # Add data to calculator
        indicator_calc.add_price_data(close_price, volume)
        
        # Calculate indicators
        sma = indicator_calc.calculate_sma(20)
        if sma is not None:
            df.iloc[i, df.columns.get_loc('SMA_20')] = sma
        
        ema = indicator_calc.calculate_ema(20)
        if ema is not None:
            df.iloc[i, df.columns.get_loc('EMA_20')] = ema
        
        rsi = indicator_calc.calculate_rsi(14)
        if rsi is not None:
            df.iloc[i, df.columns.get_loc('RSI')] = rsi
        
        macd_data = indicator_calc.calculate_macd()
        if macd_data is not None:
            df.iloc[i, df.columns.get_loc('MACD')] = macd_data['MACD']
            df.iloc[i, df.columns.get_loc('MACD_SIGNAL')] = macd_data['MACD_SIGNAL']
            df.iloc[i, df.columns.get_loc('MACD_HIST')] = macd_data['MACD_HIST']
        
        bb_data = indicator_calc.calculate_bollinger_bands()
        if bb_data is not None:
            df.iloc[i, df.columns.get_loc('BB_UPPER')] = bb_data['BB_UPPER']
            df.iloc[i, df.columns.get_loc('BB_MIDDLE')] = bb_data['BB_MIDDLE']
            df.iloc[i, df.columns.get_loc('BB_LOWER')] = bb_data['BB_LOWER']
        
        stoch_data = indicator_calc.calculate_stochastic()
        if stoch_data is not None:
            df.iloc[i, df.columns.get_loc('STOCH_K')] = stoch_data['STOCH_K']
            df.iloc[i, df.columns.get_loc('STOCH_D')] = stoch_data['STOCH_D']
        
        volume_data = indicator_calc.calculate_volume_indicators()
        if volume_data is not None:
            df.iloc[i, df.columns.get_loc('VOLUME_MA')] = volume_data['VOLUME_MA']
            df.iloc[i, df.columns.get_loc('VOLUME_RATIO')] = volume_data['VOLUME_RATIO']
            df.iloc[i, df.columns.get_loc('OBV')] = volume_data['OBV']
    
    return df


# Example usage and testing
if __name__ == "__main__":
    print("Testing Technical Indicators Module")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Date': dates,
        'Close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'Volume': np.random.randint(1000, 10000, 100)
    })
    
    print(f"Sample data shape: {sample_data.shape}")
    
    # Test individual indicator calculator
    indicator_calc = TechnicalIndicators()
    
    print("\nTesting individual indicators:")
    for i in range(len(sample_data)):
        close_price = sample_data.iloc[i]['Close']
        volume = sample_data.iloc[i]['Volume']
        
        indicator_calc.add_price_data(close_price, volume)
        
        if i >= 20:  # After enough data
            sma = indicator_calc.calculate_sma(20)
            rsi = indicator_calc.calculate_rsi(14)
            print(f"Day {i}: SMA={sma:.2f}, RSI={rsi:.2f}")
    
    # Test indicator ranker
    print("\nTesting indicator ranker:")
    ranker = IndicatorRanker()
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    for symbol in symbols:
        rsi_value = np.random.uniform(20, 80)
        ranker.add_indicator_score(symbol, 'RSI', rsi_value)
    
    top_indicators = ranker.get_top_indicators(3)
    print("Top RSI indicators:")
    for symbol, indicator, score, criteria in top_indicators:
        print(f"  {symbol}: {score:.2f}")
    
    # Test complete indicator calculation
    print("\nTesting complete indicator calculation:")
    df_with_indicators = calculate_all_indicators(sample_data)
    print(f"DataFrame with indicators shape: {df_with_indicators.shape}")
    print(f"Indicator columns: {[col for col in df_with_indicators.columns if col not in ['Date', 'Close', 'Volume']]}")
    
    print("\nTechnical indicators module testing completed successfully!")
