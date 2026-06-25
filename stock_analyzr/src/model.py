"""
LSTM Model module for stock price prediction.
Implements various LSTM architectures with training and evaluation capabilities.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, Dict, Any, Optional, List
import json


class StockLSTMModel:
    """
    LSTM model class for stock price prediction with various architectures.
    """
    
    def __init__(self, input_shape: Tuple[int, int], model_type: str = 'standard'):
        """
        Initialize LSTM model.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            model_type: Type of model architecture ('standard', 'deep', 'bidirectional', 'attention')
        """
        self.input_shape = input_shape
        self.model_type = model_type
        self.model = None
        self.history = None
        self.training_config = {}
        
    def build_standard_lstm(self, units: int = 64, dropout: float = 0.2, 
                          num_layers: int = 2, learning_rate: float = 0.001) -> Model:
        """
        Build standard LSTM model.
        
        Args:
            units: Number of LSTM units per layer
            dropout: Dropout rate
            num_layers: Number of LSTM layers
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units, return_sequences=(num_layers > 1), 
                      input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        
        # Additional LSTM layers
        for i in range(1, num_layers):
            return_sequences = (i < num_layers - 1)
            model.add(LSTM(units, return_sequences=return_sequences))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))
        
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', 
                     metrics=['mae', 'mape'])
        
        return model
    
    def build_deep_lstm(self, units_list: List[int] = [128, 64, 32], 
                       dropout: float = 0.3, learning_rate: float = 0.001) -> Model:
        """
        Build deep LSTM model with varying layer sizes.
        
        Args:
            units_list: List of units for each LSTM layer
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        for i, units in enumerate(units_list):
            return_sequences = (i < len(units_list) - 1)
            
            if i == 0:
                model.add(LSTM(units, return_sequences=return_sequences, 
                             input_shape=self.input_shape))
            else:
                model.add(LSTM(units, return_sequences=return_sequences))
            
            model.add(BatchNormalization())
            model.add(Dropout(dropout))
        
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', 
                     metrics=['mae', 'mape'])
        
        return model
    
    def build_bidirectional_lstm(self, units: int = 64, dropout: float = 0.2,
                               num_layers: int = 2, learning_rate: float = 0.001) -> Model:
        """
        Build bidirectional LSTM model.
        
        Args:
            units: Number of LSTM units per layer
            dropout: Dropout rate
            num_layers: Number of LSTM layers
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled Keras model
        """
        from tensorflow.keras.layers import Bidirectional
        
        model = Sequential()
        
        # First bidirectional LSTM layer
        model.add(Bidirectional(LSTM(units, return_sequences=(num_layers > 1)), 
                              input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        
        # Additional bidirectional LSTM layers
        for i in range(1, num_layers):
            return_sequences = (i < num_layers - 1)
            model.add(Bidirectional(LSTM(units, return_sequences=return_sequences)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))
        
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', 
                     metrics=['mae', 'mape'])
        
        return model
    
    def build_attention_lstm(self, units: int = 64, dropout: float = 0.2,
                          learning_rate: float = 0.001) -> Model:
        """
        Build LSTM model with attention mechanism.
        
        Args:
            units: Number of LSTM units
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled Keras model
        """
        from tensorflow.keras.layers import Attention, MultiHeadAttention
        
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # LSTM layer
        lstm_out = LSTM(units, return_sequences=True)(inputs)
        lstm_out = BatchNormalization()(lstm_out)
        lstm_out = Dropout(dropout)(lstm_out)
        
        # Attention mechanism
        attention_out = MultiHeadAttention(num_heads=4, key_dim=units//4)(lstm_out, lstm_out)
        attention_out = Dropout(dropout)(attention_out)
        
        # Global average pooling
        pooled = tf.keras.layers.GlobalAveragePooling1D()(attention_out)
        
        # Dense layers
        dense1 = Dense(units//2, activation='relu')(pooled)
        dense1 = Dropout(dropout)(dense1)
        
        # Output layer
        outputs = Dense(1, activation='linear')(dense1)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', 
                     metrics=['mae', 'mape'])
        
        return model
    
    def build_model(self, **kwargs) -> Model:
        """
        Build model based on specified type.
        
        Args:
            **kwargs: Additional arguments for model building
            
        Returns:
            Compiled Keras model
        """
        if self.model_type == 'standard':
            self.model = self.build_standard_lstm(**kwargs)
        elif self.model_type == 'deep':
            self.model = self.build_deep_lstm(**kwargs)
        elif self.model_type == 'bidirectional':
            self.model = self.build_bidirectional_lstm(**kwargs)
        elif self.model_type == 'attention':
            self.model = self.build_attention_lstm(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 100, batch_size: int = 32,
              callbacks: List = None, verbose: int = 1) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of Keras callbacks
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Default callbacks
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        # Training configuration
        self.training_config = {
            'epochs': epochs,
            'batch_size': batch_size,
            'validation_data': (X_val, y_val) if X_val is not None else None,
            'callbacks': callbacks,
            'verbose': verbose
        }
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val) if X_val is not None else None,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history.history
    
    def _get_default_callbacks(self) -> List:
        """
        Get default callbacks for training.
        
        Returns:
            List of Keras callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_lstm_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        mse = tf.reduce_mean(tf.square(y - y_pred)).numpy()
        mae = tf.reduce_mean(tf.abs(y - y_pred)).numpy()
        mape = tf.reduce_mean(tf.abs((y - y_pred) / tf.maximum(tf.abs(y), 1e-8))).numpy()
        
        # Calculate additional metrics
        rmse = np.sqrt(mse)
        
        # Directional accuracy
        y_diff = np.diff(y.flatten())
        y_pred_diff = np.diff(y_pred.flatten())
        directional_accuracy = np.mean(np.sign(y_diff) == np.sign(y_pred_diff))
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'directional_accuracy': float(directional_accuracy)
        }
    
    def plot_training_history(self, save_path: str = None) -> None:
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            raise ValueError("No training history available.")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # MAE
        axes[0, 1].plot(self.history.history['mae'], label='Training MAE')
        if 'val_mae' in self.history.history:
            axes[0, 1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Model MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        
        # MAPE
        axes[1, 0].plot(self.history.history['mape'], label='Training MAPE')
        if 'val_mape' in self.history.history:
            axes[1, 0].plot(self.history.history['val_mape'], label='Validation MAPE')
        axes[1, 0].set_title('Model MAPE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAPE')
        axes[1, 0].legend()
        
        # Learning Rate
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'], label='Learning Rate')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'No Learning Rate Data', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Learning Rate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # plt.show()
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        
        # Save training configuration
        config_path = filepath.replace('.h5', '_config.json')
        serializable_config = {}
        if hasattr(self, 'training_config') and self.training_config:
            for k, v in self.training_config.items():
                if k in ['epochs', 'batch_size', 'verbose']:
                    serializable_config[k] = v
                elif k == 'validation_data' and v is not None:
                    # Do not serialize raw validation arrays
                    serializable_config[k] = "validation data present"
        with open(config_path, 'w') as f:
            json.dump(serializable_config, f, indent=2)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model.
        
        Args:
            filepath: Path to the model file
        """
        self.model = tf.keras.models.load_model(filepath)
        
        # Load training configuration
        config_path = filepath.replace('.h5', '_config.json')
        try:
            with open(config_path, 'r') as f:
                self.training_config = json.load(f)
        except FileNotFoundError:
            print("No training configuration found.")
    
    def get_model_summary(self) -> str:
        """
        Get model summary.
        
        Returns:
            Model summary string
        """
        if self.model is None:
            return "No model built yet."
        
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.model.summary()
        sys.stdout = old_stdout
        
        return buffer.getvalue()


def create_model_comparison(models_config: List[Dict[str, Any]], 
                          X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
    """
    Compare different LSTM model architectures.
    
    Args:
        models_config: List of model configurations
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    for config in models_config:
        model_type = config['model_type']
        print(f"\nTraining {model_type} model...")
        
        # Create model
        lstm_model = StockLSTMModel(
            input_shape=X_train.shape[1:],
            model_type=model_type
        )
        
        # Build model
        lstm_model.build_model(**config.get('model_params', {}))
        
        # Train model
        history = lstm_model.train(
            X_train, y_train, X_val, y_val,
            epochs=config.get('epochs', 50),
            batch_size=config.get('batch_size', 32)
        )
        
        # Evaluate model
        val_metrics = lstm_model.evaluate(X_val, y_val)
        
        results[model_type] = {
            'model': lstm_model,
            'history': history,
            'val_metrics': val_metrics,
            'config': config
        }
        
        print(f"{model_type} validation RMSE: {val_metrics['rmse']:.4f}")
    
    return results


# Example usage and testing
if __name__ == "__main__":
    print("Testing LSTM Model Implementation")
    
    # Create sample data
    np.random.seed(42)
    sequence_length = 60
    n_features = 10
    n_samples = 1000
    
    X_sample = np.random.randn(n_samples, sequence_length, n_features)
    y_sample = np.random.randn(n_samples, 1)
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train = X_sample[:split_idx]
    X_val = X_sample[split_idx:]
    y_train = y_sample[:split_idx]
    y_val = y_sample[split_idx:]
    
    # Test different model types
    model_types = ['standard', 'deep', 'bidirectional']
    
    for model_type in model_types:
        print(f"\nTesting {model_type} model:")
        
        # Create model
        lstm_model = StockLSTMModel(
            input_shape=(sequence_length, n_features),
            model_type=model_type
        )
        
        # Build model
        model = lstm_model.build_model()
        print(f"Model built successfully: {model_type}")
        
        # Train model (short training for testing)
        history = lstm_model.train(
            X_train, y_train, X_val, y_val,
            epochs=5, batch_size=16, verbose=0
        )
        
        # Evaluate model
        metrics = lstm_model.evaluate(X_val, y_val)
        print(f"Validation RMSE: {metrics['rmse']:.4f}")
        print(f"Validation MAE: {metrics['mae']:.4f}")
    
    print("\nLSTM model testing completed successfully!")
