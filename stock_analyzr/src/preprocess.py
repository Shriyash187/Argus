"""
Preprocessing module for stock data preparation and sequence creation.
Demonstrates explicit usage of queue data structure for sliding windows.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from typing import Tuple, Optional, Dict, Any
from collections import deque

from ds_helpers import SlidingWindowQueue


class DataPreprocessor:
    """
    Data preprocessor class that handles feature engineering, scaling, and sequence creation.
    Uses queue data structure for efficient sliding window operations.
    """
    
    def __init__(self, scaler_type: str = 'minmax'):
        """
        Initialize preprocessor.
        
        Args:
            scaler_type: Type of scaler ('minmax' or 'standard')
        """
        self.scaler_type = scaler_type
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_columns = None
        self.target_column = None
        
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'Close') -> pd.DataFrame:
        """
        Prepare features for training by selecting relevant columns and handling missing values.
        
        Args:
            df: Input DataFrame with stock data
            target_col: Name of target column
            
        Returns:
            Processed DataFrame
        """
        df = df.copy()
        
        # Flatten multi-level columns if they exist
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in df.columns]
        
        # Convert Date to datetime if it's not already
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        # Select relevant columns - check for both original and flattened names
        base_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
        indicator_cols = ['RSI', 'EMA_20', 'MACD', 'MACD_HIST', 'MACD_SIGNAL', 'SMA_50']
        
        # Find available columns (handle both original and flattened names)
        available_feature_cols = []
        for col in base_cols + indicator_cols:
            if col in df.columns:
                available_feature_cols.append(col)
            else:
                # Check for flattened versions (e.g., Close_AAPL)
                flattened_cols = [c for c in df.columns if c.startswith(col + '_')]
                if flattened_cols:
                    available_feature_cols.extend(flattened_cols)
        
        # Keep only available columns
        df = df[available_feature_cols]
        
        # Handle missing values
        df = df.ffill().bfill()
        df = df.dropna()
        
        # Store column information - find the actual target column
        actual_target_col = None
        if target_col in df.columns:
            actual_target_col = target_col
        else:
            # Look for flattened version
            target_candidates = [c for c in df.columns if c.startswith(target_col + '_')]
            if target_candidates:
                actual_target_col = target_candidates[0]  # Use first match
            else:
                # If target_col has a suffix like Close_AAPL, look for the base name Close
                base_target = target_col.split('_')[0]
                if base_target in df.columns:
                    actual_target_col = base_target
        
        if actual_target_col is None:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame columns: {list(df.columns)}")
        
        self.feature_columns = [col for col in available_feature_cols if col != actual_target_col]
        self.target_column = actual_target_col
        
        return df
    
    def create_sequences_with_queue(self, data: np.ndarray, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences using sliding window queue implementation.
        
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
    
    def create_sequences_traditional(self, data: np.ndarray, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences using traditional approach for comparison.
        
        Args:
            data: Input data array (n_samples, n_features)
            sequence_length: Length of each sequence
            
        Returns:
            Tuple of (X, y) where X is sequences and y is targets
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length, 0])  # Predict next day's close price
        
        return np.array(X), np.array(y)
    
    def scale_features(self, X_train: np.ndarray, X_val: np.ndarray = None, 
                      X_test: np.ndarray = None, fit_scaler: bool = True) -> Tuple[np.ndarray, ...]:
        """
        Scale features using the specified scaler type.
        
        Args:
            X_train: Training features
            X_val: Validation features (optional)
            X_test: Test features (optional)
            fit_scaler: Whether to fit the scaler on training data
            
        Returns:
            Tuple of scaled features
        """
        if self.scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        
        # Reshape for scaling
        n_seq, seq_len, n_feat = X_train.shape
        X_train_flat = X_train.reshape(-1, n_feat)
        
        if fit_scaler:
            self.feature_scaler = scaler
            X_train_scaled = scaler.fit_transform(X_train_flat)
        else:
            X_train_scaled = scaler.transform(X_train_flat)
        
        X_train_scaled = X_train_scaled.reshape(n_seq, seq_len, n_feat)
        
        result = [X_train_scaled]
        
        if X_val is not None:
            n_seq_val, seq_len_val, n_feat_val = X_val.shape
            X_val_flat = X_val.reshape(-1, n_feat_val)
            X_val_scaled = scaler.transform(X_val_flat)
            X_val_scaled = X_val_scaled.reshape(n_seq_val, seq_len_val, n_feat_val)
            result.append(X_val_scaled)
        
        if X_test is not None:
            n_seq_test, seq_len_test, n_feat_test = X_test.shape
            X_test_flat = X_test.reshape(-1, n_feat_test)
            X_test_scaled = scaler.transform(X_test_flat)
            X_test_scaled = X_test_scaled.reshape(n_seq_test, seq_len_test, n_feat_test)
            result.append(X_test_scaled)
        
        return tuple(result)
    
    def scale_targets(self, y_train: np.ndarray, y_val: np.ndarray = None, 
                     y_test: np.ndarray = None, fit_scaler: bool = True) -> Tuple[np.ndarray, ...]:
        """
        Scale target values.
        
        Args:
            y_train: Training targets
            y_val: Validation targets (optional)
            y_test: Test targets (optional)
            fit_scaler: Whether to fit the scaler on training data
            
        Returns:
            Tuple of scaled targets
        """
        if self.scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        
        y_train_reshaped = y_train.reshape(-1, 1)
        
        if fit_scaler:
            self.target_scaler = scaler
            y_train_scaled = scaler.fit_transform(y_train_reshaped).ravel()
        else:
            y_train_scaled = scaler.transform(y_train_reshaped).ravel()
        
        result = [y_train_scaled]
        
        if y_val is not None:
            y_val_reshaped = y_val.reshape(-1, 1)
            y_val_scaled = scaler.transform(y_val_reshaped).ravel()
            result.append(y_val_scaled)
        
        if y_test is not None:
            y_test_reshaped = y_test.reshape(-1, 1)
            y_test_scaled = scaler.transform(y_test_reshaped).ravel()
            result.append(y_test_scaled)
        
        return tuple(result)
    
    def inverse_transform_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled targets back to original scale.
        
        Args:
            y_scaled: Scaled target values
            
        Returns:
            Inverse transformed targets
        """
        if self.target_scaler is None:
            raise ValueError("Target scaler not fitted. Call scale_targets first.")
        
        y_reshaped = y_scaled.reshape(-1, 1)
        y_original = self.target_scaler.inverse_transform(y_reshaped)
        return y_original.ravel()
    
    def time_series_split(self, data: np.ndarray, train_ratio: float = 0.7, 
                         val_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split time series data temporally (no random splitting).
        
        Args:
            data: Input data array
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        n_samples = len(data)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        return train_data, val_data, test_data
    
    def save_scalers(self, filepath_prefix: str = "scalers/"):
        """
        Save fitted scalers to disk.
        
        Args:
            filepath_prefix: Prefix for scaler file paths
        """
        import os
        os.makedirs(filepath_prefix, exist_ok=True)
        
        if self.feature_scaler is not None:
            with open(f"{filepath_prefix}feature_scaler.pkl", "wb") as f:
                pickle.dump(self.feature_scaler, f)
        
        if self.target_scaler is not None:
            with open(f"{filepath_prefix}target_scaler.pkl", "wb") as f:
                pickle.dump(self.target_scaler, f)
    
    def load_scalers(self, filepath_prefix: str = "scalers/"):
        """
        Load fitted scalers from disk.
        
        Args:
            filepath_prefix: Prefix for scaler file paths
        """
        try:
            with open(f"{filepath_prefix}feature_scaler.pkl", "rb") as f:
                self.feature_scaler = pickle.load(f)
            
            with open(f"{filepath_prefix}target_scaler.pkl", "rb") as f:
                self.target_scaler = pickle.load(f)
        except FileNotFoundError as e:
            print(f"Scaler files not found: {e}")
    
    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add additional technical features to the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        # Find the actual column names (handle both original and flattened names)
        close_col = 'Close' if 'Close' in df.columns else [c for c in df.columns if c.startswith('Close_')][0]
        high_col = 'High' if 'High' in df.columns else [c for c in df.columns if c.startswith('High_')][0]
        low_col = 'Low' if 'Low' in df.columns else [c for c in df.columns if c.startswith('Low_')][0]
        open_col = 'Open' if 'Open' in df.columns else [c for c in df.columns if c.startswith('Open_')][0]
        volume_col = 'Volume' if 'Volume' in df.columns else [c for c in df.columns if c.startswith('Volume_')][0]
        
        # Price-based features
        df['Price_Change'] = df[close_col].pct_change()
        df['High_Low_Pct'] = (df[high_col] - df[low_col]) / df[close_col]
        df['Open_Close_Pct'] = (df[close_col] - df[open_col]) / df[open_col]
        
        # Moving averages
        df['SMA_5'] = df[close_col].rolling(window=5).mean()
        df['SMA_10'] = df[close_col].rolling(window=10).mean()
        df['SMA_20'] = df[close_col].rolling(window=20).mean()
        
        # Volatility
        df['Volatility'] = df[close_col].rolling(window=20).std()
        
        # Volume features
        df['Volume_MA'] = df[volume_col].rolling(window=20).mean()
        df['Volume_Ratio'] = df[volume_col] / df['Volume_MA']
        
        # Momentum indicators
        df['Momentum_5'] = df[close_col] / df[close_col].shift(5) - 1
        df['Momentum_10'] = df[close_col] / df[close_col].shift(10) - 1
        
        # Fill NaN values
        df = df.ffill().bfill()
        
        return df
    
    def prepare_full_pipeline(self, df: pd.DataFrame, sequence_length: int = 60,
                            train_ratio: float = 0.7, val_ratio: float = 0.15,
                            use_queue: bool = True, target_col: str = 'Close') -> Dict[str, Any]:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            sequence_length: Length of sequences
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            use_queue: Whether to use queue-based sequence creation
            
        Returns:
            Dictionary with processed data and metadata
        """
        # Prepare features
        df_processed = self.prepare_features(df, target_col=target_col)
        df_processed = self.add_technical_features(df_processed)
        
        # Convert to numpy array
        data_array = df_processed.values
        
        # Time series split
        train_data, val_data, test_data = self.time_series_split(
            data_array, train_ratio, val_ratio
        )
        
        # Create sequences
        if use_queue:
            X_train, y_train = self.create_sequences_with_queue(train_data, sequence_length)
            X_val, y_val = self.create_sequences_with_queue(val_data, sequence_length)
            X_test, y_test = self.create_sequences_with_queue(test_data, sequence_length)
        else:
            X_train, y_train = self.create_sequences_traditional(train_data, sequence_length)
            X_val, y_val = self.create_sequences_traditional(val_data, sequence_length)
            X_test, y_test = self.create_sequences_traditional(test_data, sequence_length)
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            X_train, X_val, X_test, fit_scaler=True
        )
        
        # Scale targets
        y_train_scaled, y_val_scaled, y_test_scaled = self.scale_targets(
            y_train, y_val, y_test, fit_scaler=True
        )
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train_scaled,
            'y_val': y_val_scaled,
            'y_test': y_test_scaled,
            'y_train_original': y_train,
            'y_val_original': y_val,
            'y_test_original': y_test,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'sequence_length': sequence_length,
            'scaler_type': self.scaler_type,
            'preprocessor': self
        }


def demonstrate_queue_vs_traditional(data: np.ndarray, sequence_length: int = 60) -> Dict[str, Any]:
    """
    Demonstrate the difference between queue-based and traditional sequence creation.
    
    Args:
        data: Input data array
        sequence_length: Length of sequences
        
    Returns:
        Dictionary with comparison results
    """
    preprocessor = DataPreprocessor()
    
    # Queue-based approach
    X_queue, y_queue = preprocessor.create_sequences_with_queue(data, sequence_length)
    
    # Traditional approach
    X_traditional, y_traditional = preprocessor.create_sequences_traditional(data, sequence_length)
    
    return {
        'queue_sequences': X_queue.shape,
        'traditional_sequences': X_traditional.shape,
        'queue_targets': y_queue.shape,
        'traditional_targets': y_traditional.shape,
        'queue_memory_efficient': True,
        'traditional_memory_efficient': False,
        'queue_time_complexity': 'O(n)',
        'traditional_time_complexity': 'O(n * sequence_length)'
    }


# Example usage and testing
if __name__ == "__main__":
    print("Testing Data Preprocessing Pipeline")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    sample_data = pd.DataFrame({
        'Date': dates,
        'Close': 100 + np.cumsum(np.random.randn(1000) * 0.5),
        'Open': 100 + np.cumsum(np.random.randn(1000) * 0.5),
        'High': 100 + np.cumsum(np.random.randn(1000) * 0.5) + np.random.rand(1000) * 2,
        'Low': 100 + np.cumsum(np.random.randn(1000) * 0.5) - np.random.rand(1000) * 2,
        'Volume': np.random.randint(1000, 10000, 1000),
        'RSI': np.random.uniform(20, 80, 1000),
        'EMA_20': 100 + np.cumsum(np.random.randn(1000) * 0.3),
        'MACD': np.random.randn(1000) * 0.5,
        'MACD_HIST': np.random.randn(1000) * 0.2,
        'MACD_SIGNAL': np.random.randn(1000) * 0.3,
        'SMA_50': 100 + np.cumsum(np.random.randn(1000) * 0.3)
    })
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(scaler_type='minmax')
    
    # Run full pipeline
    result = preprocessor.prepare_full_pipeline(sample_data, sequence_length=30)
    
    print(f"Training sequences shape: {result['X_train'].shape}")
    print(f"Validation sequences shape: {result['X_val'].shape}")
    print(f"Test sequences shape: {result['X_test'].shape}")
    print(f"Feature columns: {result['feature_columns']}")
    print(f"Target column: {result['target_column']}")
    
    # Demonstrate queue vs traditional
    comparison = demonstrate_queue_vs_traditional(sample_data.values, sequence_length=30)
    print(f"\nQueue vs Traditional Comparison:")
    print(f"Queue sequences: {comparison['queue_sequences']}")
    print(f"Traditional sequences: {comparison['traditional_sequences']}")
    print(f"Queue memory efficient: {comparison['queue_memory_efficient']}")
    
    print("\nPreprocessing pipeline test completed successfully!")
