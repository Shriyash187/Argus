"""
Backtesting module for walk-forward validation and performance evaluation.
Implements comprehensive backtesting strategies for LSTM stock prediction models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

from model import StockLSTMModel
from preprocess import DataPreprocessor


class WalkForwardBacktester:
    """
    Walk-forward backtesting class for time series models.
    Implements proper temporal validation without data leakage.
    """
    
    def __init__(self, model: StockLSTMModel, preprocessor: DataPreprocessor):
        """
        Initialize backtester.
        
        Args:
            model: Trained LSTM model
            preprocessor: Data preprocessor with fitted scalers
        """
        self.model = model
        self.preprocessor = preprocessor
        self.results = {}
        self.predictions = []
        self.actuals = []
        self.dates = []
        
    def walk_forward_validation(self, data: pd.DataFrame, 
                              train_window: int = 252,  # 1 year
                              test_window: int = 21,    # 1 month
                              step_size: int = 21,      # 1 month
                              sequence_length: int = 60) -> Dict[str, Any]:
        """
        Perform walk-forward validation on time series data.
        
        Args:
            data: Input DataFrame with stock data
            train_window: Size of training window (days)
            test_window: Size of test window (days)
            step_size: Step size for moving window (days)
            sequence_length: Length of sequences for LSTM
            
        Returns:
            Dictionary with backtesting results
        """
        print(f"Starting walk-forward validation...")
        print(f"Train window: {train_window} days, Test window: {test_window} days")
        print(f"Step size: {step_size} days, Sequence length: {sequence_length}")
        
        # Prepare data
        df_processed = self.preprocessor.prepare_features(data)
        df_processed = self.preprocessor.add_technical_features(df_processed)
        data_array = df_processed.values
        
        # Get dates for plotting
        self.dates = df_processed.index.tolist()
        
        # Initialize results
        all_predictions = []
        all_actuals = []
        all_metrics = []
        window_results = []
        
        # Walk-forward validation
        start_idx = train_window + sequence_length
        end_idx = len(data_array) - test_window
        
        for i in range(start_idx, end_idx, step_size):
            # Define windows
            train_start = i - train_window - sequence_length
            train_end = i - sequence_length
            test_start = i
            test_end = min(i + test_window, len(data_array))
            
            if test_end - test_start < test_window:
                break
            
            print(f"Window {len(window_results) + 1}: Training on {train_start}:{train_end}, Testing on {test_start}:{test_end}")
            
            # Extract training and test data
            train_data = data_array[train_start:train_end]
            test_data = data_array[test_start - sequence_length:test_end]
            
            # Create sequences
            X_train, y_train = self.preprocessor.create_sequences_with_queue(
                train_data, sequence_length
            )
            X_test, y_test = self.preprocessor.create_sequences_with_queue(
                test_data, sequence_length
            )
            
            if len(X_train) == 0 or len(X_test) == 0:
                print(f"Skipping window {len(window_results) + 1}: insufficient data")
                continue
            
            # Scale features (fit on training, transform test)
            X_train_scaled, X_test_scaled = self.preprocessor.scale_features(
                X_train, X_test, fit_scaler=True
            )
            
            # Scale targets
            y_train_scaled, y_test_scaled = self.preprocessor.scale_targets(
                y_train, y_test, fit_scaler=True
            )
            
            # Train model for this window
            window_model = StockLSTMModel(
                input_shape=X_train_scaled.shape[1:],
                model_type=self.model.model_type
            )
            window_model.build_model()
            
            # Train with early stopping (no disk checkpoints during validation)
            history = window_model.train(
                X_train_scaled, y_train_scaled,
                epochs=50, batch_size=32, verbose=0,
                callbacks=[]
            )
            
            # Make predictions
            y_pred_scaled = window_model.predict(X_test_scaled)
            
            # Inverse transform predictions
            y_pred = self.preprocessor.inverse_transform_targets(y_pred_scaled)
            
            # Calculate metrics for this window
            window_metrics = self._calculate_metrics(y_test, y_pred)
            
            # Store results
            all_predictions.extend(y_pred)
            all_actuals.extend(y_test)
            all_metrics.append(window_metrics)
            
            window_results.append({
                'window': len(window_results) + 1,
                'train_period': (train_start, train_end),
                'test_period': (test_start, test_end),
                'metrics': window_metrics,
                'predictions': y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred),
                'actuals': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)
            })
            
            print(f"Window {len(window_results)} RMSE: {window_metrics['rmse']:.4f}")
        
        # Calculate overall metrics
        overall_metrics = self._calculate_metrics(all_actuals, all_predictions)
        
        # Store results
        self.results = {
            'overall_metrics': overall_metrics,
            'window_metrics': all_metrics,
            'window_results': window_results,
            'predictions': all_predictions,
            'actuals': all_actuals,
            'config': {
                'train_window': train_window,
                'test_window': test_window,
                'step_size': step_size,
                'sequence_length': sequence_length
            }
        }
        
        self.predictions = all_predictions
        self.actuals = all_actuals
        
        print(f"\nWalk-forward validation completed!")
        print(f"Total windows: {len(window_results)}")
        print(f"Overall RMSE: {overall_metrics['rmse']:.4f}")
        print(f"Overall MAPE: {overall_metrics['mape']:.4f}")
        print(f"Overall Directional Accuracy: {overall_metrics['directional_accuracy']:.4f}")
        
        return self.results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metrics
        """
        # Convert to numpy arrays to support operations
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
        # Directional accuracy
        if len(y_true) > 1:
            y_true_diff = np.diff(y_true)
            y_pred_diff = np.diff(y_pred)
            directional_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff))
        else:
            directional_accuracy = 0.0
        
        # Additional metrics
        max_error = np.max(np.abs(y_true - y_pred))
        mean_error = np.mean(y_true - y_pred)
        
        # Sharpe-like metric (return/volatility)
        returns = np.diff(y_true) / y_true[:-1]
        pred_returns = np.diff(y_pred) / y_pred[:-1]
        
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns)
            pred_sharpe_ratio = np.mean(pred_returns) / np.std(pred_returns)
        else:
            sharpe_ratio = 0.0
            pred_sharpe_ratio = 0.0
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'directional_accuracy': float(directional_accuracy),
            'max_error': float(max_error),
            'mean_error': float(mean_error),
            'sharpe_ratio': float(sharpe_ratio),
            'pred_sharpe_ratio': float(pred_sharpe_ratio)
        }
    
    def plot_results(self, save_path: str = None) -> None:
        """
        Plot backtesting results.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.results:
            raise ValueError("No results to plot. Run walk_forward_validation first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Predictions vs Actuals
        axes[0, 0].plot(self.actuals, label='Actual', alpha=0.7)
        axes[0, 0].plot(self.predictions, label='Predicted', alpha=0.7)
        axes[0, 0].set_title('Predictions vs Actual Values')
        axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot
        axes[0, 1].scatter(self.actuals, self.predictions, alpha=0.6)
        min_val = min(min(self.actuals), min(self.predictions))
        max_val = max(max(self.actuals), max(self.predictions))
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[0, 1].set_xlabel('Actual Values')
        axes[0, 1].set_ylabel('Predicted Values')
        axes[0, 1].set_title('Prediction Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Residuals
        residuals = np.array(self.actuals) - np.array(self.predictions)
        axes[1, 0].plot(residuals, alpha=0.7)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[1, 0].set_title('Residuals (Actual - Predicted)')
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Residual')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Window-wise RMSE
        window_rmses = [w['metrics']['rmse'] for w in self.results['window_results']]
        axes[1, 1].plot(window_rmses, marker='o', alpha=0.7)
        axes[1, 1].set_title('RMSE by Window')
        axes[1, 1].set_xlabel('Window Number')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_window_performance(self, window_idx: int = 0, save_path: str = None) -> None:
        """
        Plot performance for a specific window.
        
        Args:
            window_idx: Index of window to plot
            save_path: Path to save the plot
        """
        if not self.results or window_idx >= len(self.results['window_results']):
            raise ValueError(f"Invalid window index: {window_idx}")
        
        window_result = self.results['window_results'][window_idx]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot predictions vs actuals for this window
        axes[0].plot(window_result['actuals'], label='Actual', marker='o', alpha=0.7)
        axes[0].plot(window_result['predictions'], label='Predicted', marker='s', alpha=0.7)
        axes[0].set_title(f'Window {window_result["window"]} - Predictions vs Actual')
        axes[0].set_xlabel('Time Steps')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot residuals for this window
        residuals = np.array(window_result['actuals']) - np.array(window_result['predictions'])
        axes[1].plot(residuals, marker='o', alpha=0.7)
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[1].set_title(f'Window {window_result["window"]} - Residuals')
        axes[1].set_xlabel('Time Steps')
        axes[1].set_ylabel('Residual')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_performance_summary(self) -> pd.DataFrame:
        """
        Get performance summary as DataFrame.
        
        Returns:
            DataFrame with performance metrics
        """
        if not self.results:
            raise ValueError("No results available. Run walk_forward_validation first.")
        
        # Overall metrics
        overall_df = pd.DataFrame([self.results['overall_metrics']])
        overall_df['window'] = 'Overall'
        
        # Window-wise metrics
        window_dfs = []
        for i, window_result in enumerate(self.results['window_results']):
            window_df = pd.DataFrame([window_result['metrics']])
            window_df['window'] = f'Window {i + 1}'
            window_dfs.append(window_df)
        
        # Combine
        summary_df = pd.concat([overall_df] + window_dfs, ignore_index=True)
        
        return summary_df
    
    def calculate_trading_metrics(self, initial_capital: float = 10000) -> Dict[str, float]:
        """
        Calculate trading performance metrics.
        
        Args:
            initial_capital: Initial capital for trading simulation
            
        Returns:
            Dictionary with trading metrics
        """
        if len(self.predictions) < 2:
            return {}
        
        # Convert to returns
        actual_returns = np.diff(self.actuals) / np.array(self.actuals[:-1])
        pred_returns = np.diff(self.predictions) / np.array(self.predictions[:-1])
        
        # Simple trading strategy: buy if predicted return > 0, sell otherwise
        positions = np.where(pred_returns > 0, 1, -1)
        strategy_returns = positions * actual_returns
        
        # Calculate metrics
        total_return = np.prod(1 + strategy_returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = np.std(strategy_returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        win_rate = np.mean(strategy_returns > 0)
        
        return {
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate)
        }


class ModelComparison:
    """
    Class for comparing multiple models using walk-forward validation.
    """
    
    def __init__(self, models: List[StockLSTMModel], preprocessor: DataPreprocessor):
        """
        Initialize model comparison.
        
        Args:
            models: List of trained LSTM models
            preprocessor: Data preprocessor with fitted scalers
        """
        self.models = models
        self.preprocessor = preprocessor
        self.comparison_results = {}
    
    def compare_models(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Compare multiple models using walk-forward validation.
        
        Args:
            data: Input DataFrame with stock data
            **kwargs: Additional arguments for walk_forward_validation
            
        Returns:
            Dictionary with comparison results
        """
        results = {}
        
        for i, model in enumerate(self.models):
            print(f"\nEvaluating Model {i + 1}: {model.model_type}")
            
            backtester = WalkForwardBacktester(model, self.preprocessor)
            model_results = backtester.walk_forward_validation(data, **kwargs)
            
            results[f'model_{i + 1}_{model.model_type}'] = {
                'model': model,
                'results': model_results,
                'backtester': backtester
            }
        
        self.comparison_results = results
        return results
    
    def plot_comparison(self, save_path: str = None) -> None:
        """
        Plot comparison of all models.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.comparison_results:
            raise ValueError("No comparison results available.")
        
        n_models = len(self.comparison_results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: RMSE comparison
        model_names = []
        rmses = []
        
        for name, result in self.comparison_results.items():
            model_names.append(name.replace('model_', '').replace('_', ' '))
            rmses.append(result['results']['overall_metrics']['rmse'])
        
        axes[0, 0].bar(model_names, rmses, alpha=0.7)
        axes[0, 0].set_title('RMSE Comparison')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: MAPE comparison
        mapes = [result['results']['overall_metrics']['mape'] for result in self.comparison_results.values()]
        axes[0, 1].bar(model_names, mapes, alpha=0.7)
        axes[0, 1].set_title('MAPE Comparison')
        axes[0, 1].set_ylabel('MAPE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Directional Accuracy comparison
        dir_accs = [result['results']['overall_metrics']['directional_accuracy'] for result in self.comparison_results.values()]
        axes[1, 0].bar(model_names, dir_accs, alpha=0.7)
        axes[1, 0].set_title('Directional Accuracy Comparison')
        axes[1, 0].set_ylabel('Directional Accuracy')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Predictions comparison (first model)
        first_model_name = list(self.comparison_results.keys())[0]
        first_result = self.comparison_results[first_model_name]
        
        axes[1, 1].plot(first_result['results']['actuals'], label='Actual', alpha=0.7)
        axes[1, 1].plot(first_result['results']['predictions'], label=f'{first_model_name} Predicted', alpha=0.7)
        axes[1, 1].set_title(f'Predictions vs Actual ({first_model_name})')
        axes[1, 1].set_xlabel('Time Steps')
        axes[1, 1].set_ylabel('Price')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_comparison_summary(self) -> pd.DataFrame:
        """
        Get comparison summary as DataFrame.
        
        Returns:
            DataFrame with comparison metrics
        """
        if not self.comparison_results:
            raise ValueError("No comparison results available.")
        
        summary_data = []
        
        for name, result in self.comparison_results.items():
            metrics = result['results']['overall_metrics']
            summary_data.append({
                'Model': name,
                'RMSE': metrics['rmse'],
                'MAPE': metrics['mape'],
                'Directional_Accuracy': metrics['directional_accuracy'],
                'MAE': metrics['mae'],
                'Max_Error': metrics['max_error']
            })
        
        return pd.DataFrame(summary_data)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Backtesting Module")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
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
    
    # Create a simple model for testing
    model = StockLSTMModel(input_shape=(60, 10), model_type='standard')
    model.build_model()
    
    # Initialize backtester
    backtester = WalkForwardBacktester(model, preprocessor)
    
    # Run walk-forward validation (short for testing)
    results = backtester.walk_forward_validation(
        sample_data, 
        train_window=100, 
        test_window=20, 
        step_size=20,
        sequence_length=30
    )
    
    print(f"Backtesting completed with {len(results['window_results'])} windows")
    print(f"Overall RMSE: {results['overall_metrics']['rmse']:.4f}")
    
    print("\nBacktesting module testing completed successfully!")
