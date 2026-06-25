# 📈 LSTM Stock Analyzer & Predictor

A complete, step-by-step LSTM-based stock analysis and prediction system that integrates Yahoo Finance and Alpha Vantage data sources with explicit data structure usage (Queue, Stack, HashMap, Priority Queue) for academic Data Structures projects.

## 🎯 Project Overview

This project demonstrates a comprehensive stock analysis pipeline using:
- **LSTM neural networks** for price prediction
- **Yahoo Finance** for historical stock data
- **Alpha Vantage** for technical indicators
- **Explicit data structure usage** for academic demonstration
- **Walk-forward backtesting** for model validation
- **Interactive Streamlit web interface**

## 🏗️ Data Structures Used

### 1. Queue (deque) - Sliding Window
- **Purpose**: Efficient sliding window for LSTM sequence creation
- **Implementation**: `collections.deque` with fixed capacity
- **Benefits**: O(1) insertion/deletion, memory efficient
- **Usage**: Creating overlapping time series sequences for LSTM training

### 2. Stack (list) - Pattern Detection
- **Purpose**: Detect price peaks and reversals
- **Implementation**: Python list with push/pop operations
- **Benefits**: LIFO behavior for pattern tracking
- **Usage**: Identifying support/resistance levels and trend reversals

### 3. HashMap (dict) - Data Storage
- **Purpose**: Efficient storage and retrieval of stock data
- **Implementation**: Python dictionary
- **Benefits**: O(1) average case lookup time
- **Usage**: Symbol-based data organization and caching

### 4. Priority Queue (heapq) - Stock Ranking
- **Purpose**: Rank stocks by performance metrics
- **Implementation**: Python heapq module
- **Benefits**: Efficient insertion and extraction of top performers
- **Usage**: Ranking stocks by returns, volatility, volume, etc.

## 📁 Project Structure

```
stock_analyzr/
├── data/                      # Cached CSV files
├── notebooks/                 # Jupyter notebooks for experiments
├── models/                    # Saved models and scalers
├── src/                      # Source code modules
│   ├── fetch_data.py         # Data fetching with DS usage
│   ├── ds_helpers.py         # Data structure implementations
│   ├── preprocess.py         # Preprocessing with queue
│   ├── model.py              # LSTM model architectures
│   ├── backtest.py           # Walk-forward backtesting
│   └── app.py                # Streamlit web interface
├── requirements.txt          # Python dependencies
├── run_lstm_example.py       # Complete runnable example
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd stock_analyzr

# Install dependencies
pip install -r requirements.txt

# Set up Alpha Vantage API key (optional but recommended)
export ALPHA_VANTAGE_API_KEY="your_api_key_here"
```

### 2. Get Alpha Vantage API Key (Optional)

1. Visit [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Sign up for a free account
3. Get your API key
4. Set it as an environment variable:
   ```bash
   export ALPHA_VANTAGE_API_KEY="your_key_here"
   ```

### 3. Run the Complete Example

```bash
python run_lstm_example.py
```

This will:
- Demonstrate all data structures
- Fetch stock data (AAPL by default)
- Train an LSTM model
- Run walk-forward backtesting
- Generate comprehensive visualizations
- Save all results

### 4. Run the Interactive Web Interface

```bash
streamlit run src/app.py
```

## 📊 Features

### Data Fetching (`fetch_data.py`)
- **Yahoo Finance Integration**: Historical OHLCV data
- **Alpha Vantage Integration**: Technical indicators (RSI, EMA, MACD, SMA)
- **Rate Limiting**: Queue-based API request management
- **Data Merging**: Efficient combination of multiple data sources

### Data Structures (`ds_helpers.py`)
- **SlidingWindowQueue**: Fixed-capacity deque for sequences
- **PricePatternStack**: LIFO stack for peak detection
- **StockDataHashMap**: O(1) lookup data storage
- **StockRankingPriorityQueue**: Heap-based stock ranking

### Preprocessing (`preprocess.py`)
- **Queue-based Sequence Creation**: Memory-efficient sliding windows
- **Feature Engineering**: Technical indicators and price features
- **Scaling**: MinMax and Standard scaling options
- **Temporal Splitting**: Proper time series train/validation/test splits

### Model Training (`model.py`)
- **Multiple Architectures**: Standard, Deep, Bidirectional, Attention LSTM
- **Training Callbacks**: Early stopping, learning rate reduction, model checkpointing
- **Comprehensive Metrics**: RMSE, MAPE, directional accuracy
- **Model Persistence**: Save/load trained models

### Backtesting (`backtest.py`)
- **Walk-Forward Validation**: Proper temporal validation
- **Trading Metrics**: Returns, Sharpe ratio, maximum drawdown
- **Performance Analysis**: Window-wise and overall metrics
- **Model Comparison**: Compare multiple architectures

### Web Interface (`app.py`)
- **Interactive Data Analysis**: Real-time data fetching and visualization
- **Model Training Interface**: Configure and train models through UI
- **Backtesting Dashboard**: Run and visualize backtesting results
- **Data Structure Demonstrations**: Interactive examples of all DS usage

## 🔧 Usage Examples

### Basic Data Fetching

```python
from src.fetch_data import DataFetcher

# Initialize fetcher
fetcher = DataFetcher(alpha_vantage_key="your_key")

# Fetch data
yahoo_data = fetcher.fetch_yahoo_data("AAPL", period="1y")
indicators = fetcher.fetch_alpha_indicators("AAPL")
merged_data = fetcher.merge_data(yahoo_data, indicators)
```

### Data Structure Usage

```python
from src.ds_helpers import SlidingWindowQueue, PricePatternStack

# Queue for sliding window
queue = SlidingWindowQueue(max_size=60)
for data_point in time_series_data:
    queue.add(data_point)
    if queue.is_full():
        sequence = queue.get_window()

# Stack for pattern detection
stack = PricePatternStack()
for price in prices:
    if price > previous_peak * 1.02:
        stack.push_peak(index, price)
    if stack.detect_reversal(price):
        peak = stack.pop_peak()
```

### Model Training

```python
from src.preprocess import DataPreprocessor
from src.model import StockLSTMModel

# Prepare data
preprocessor = DataPreprocessor()
processed_data = preprocessor.prepare_full_pipeline(data, use_queue=True)

# Train model
model = StockLSTMModel(input_shape=(60, 10), model_type="standard")
model.build_model()
history = model.train(processed_data['X_train'], processed_data['y_train'])
```

### Backtesting

```python
from src.backtest import WalkForwardBacktester

# Run backtesting
backtester = WalkForwardBacktester(model, preprocessor)
results = backtester.walk_forward_validation(
    data, train_window=252, test_window=21, step_size=21
)
```

## 📈 Performance Metrics

The system provides comprehensive evaluation metrics:

### Regression Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error

### Trading Metrics
- **Directional Accuracy**: Percentage of correct direction predictions
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

### Data Structure Efficiency
- **Queue**: O(1) insertion/deletion, fixed memory usage
- **Stack**: O(1) push/pop operations
- **HashMap**: O(1) average case lookup
- **Priority Queue**: O(log n) insertion, O(log n) extraction

## 🎓 Academic Integration

This project is designed for Data Structures courses and includes:

### Explicit Data Structure Usage
- Clear implementation of each data structure
- Performance analysis and complexity discussion
- Real-world application examples
- Interactive demonstrations

### Educational Components
- Detailed comments and documentation
- Step-by-step explanations
- Performance comparisons
- Best practices demonstration

### Report-Ready Features
- Comprehensive visualizations
- Performance metrics
- Code structure analysis
- Data structure efficiency comparisons

## 🔍 Troubleshooting

### Common Issues

1. **Alpha Vantage API Rate Limits**
   - Free tier: 5 calls per minute
   - Solution: The system includes automatic rate limiting

2. **Memory Issues with Large Datasets**
   - Solution: Use queue-based sequence creation (more memory efficient)

3. **Model Training Time**
   - Solution: Reduce epochs or use smaller models for testing

4. **Missing Dependencies**
   - Solution: Ensure all packages from requirements.txt are installed

### Error Messages

- `"No Alpha Vantage API key found"`: Set the environment variable
- `"No data found for symbol"`: Check symbol spelling and availability
- `"Model not trained"`: Ensure model training completed successfully

## 📚 Dependencies

### Core Libraries
- `yfinance>=0.2.18`: Yahoo Finance data
- `alpha_vantage>=2.3.1`: Alpha Vantage API
- `pandas>=1.5.0`: Data manipulation
- `numpy>=1.21.0`: Numerical computing
- `scikit-learn>=1.1.0`: Machine learning utilities

### Deep Learning
- `tensorflow>=2.10.0`: LSTM model implementation

### Visualization
- `matplotlib>=3.5.0`: Static plotting
- `seaborn>=0.11.0`: Statistical visualization
- `plotly>=5.0.0`: Interactive plots

### Web Interface
- `streamlit>=1.25.0`: Web application framework

## 🤝 Contributing

This project is designed for educational purposes. Contributions are welcome for:
- Additional data structure implementations
- New model architectures
- Enhanced visualizations
- Documentation improvements

## 📄 License

This project is provided for educational use. Please ensure compliance with:
- Yahoo Finance Terms of Service
- Alpha Vantage API Terms
- Any applicable academic integrity policies

## 🎯 Learning Objectives

After completing this project, students will understand:

1. **Data Structure Implementation**
   - Queue operations and applications
   - Stack-based pattern detection
   - HashMap efficiency and usage
   - Priority queue algorithms

2. **Time Series Analysis**
   - LSTM architecture and training
   - Sequence creation and preprocessing
   - Walk-forward validation
   - Performance evaluation

3. **Financial Data Processing**
   - API integration and rate limiting
   - Technical indicator calculation
   - Data merging and cleaning
   - Backtesting methodologies

4. **Software Engineering**
   - Modular code organization
   - Error handling and validation
   - Documentation and testing
   - Web interface development

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code comments and documentation
3. Test with the provided example script
4. Verify all dependencies are installed correctly

---

**Happy Learning! 🚀📈**

*This project demonstrates the practical application of data structures in real-world machine learning and financial analysis scenarios.*
