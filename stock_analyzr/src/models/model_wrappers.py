import os
import sys
import pickle
import numpy as np
from typing import Dict, Any, Optional

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# Add path compatibility
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base_model import BaseStockModel
from model import StockLSTMModel

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Helper to calculate performance metrics (RMSE, MAE, MAPE, R2, Directional Accuracy)."""
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    if len(y_true) > 1:
        y_true_diff = np.diff(y_true)
        y_pred_diff = np.diff(y_pred)
        directional_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff))
    else:
        directional_accuracy = 0.0
        
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "r2": float(r2),
        "directional_accuracy": float(directional_accuracy)
    }


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """Helper to flatten a 3D sequence array (N, S, F) into a 2D array (N, S*F)."""
    if len(X.shape) == 3:
        return X.reshape(X.shape[0], -1)
    return X


class LSTMModelWrapper(BaseStockModel):
    """Wrapper for Keras-based LSTM stock model."""
    
    def __init__(self, input_shape: Optional[tuple] = None, model_type: str = 'standard'):
        self.input_shape = input_shape
        self.model_type = model_type
        self.model = None
        
    def build(self, **kwargs) -> None:
        if self.input_shape is None:
            raise ValueError("input_shape must be provided to build LSTMModelWrapper.")
        self.model = StockLSTMModel(input_shape=self.input_shape, model_type=self.model_type)
        self.model.build_model(**kwargs)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, 
              **kwargs) -> Dict[str, Any]:
        if self.model is None:
            self.input_shape = X_train.shape[1:]
            self.build()
        epochs = kwargs.get('epochs', 20)
        batch_size = kwargs.get('batch_size', 32)
        history = self.model.train(
            X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size, verbose=0
        )
        return {"history": history}
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        return self.model.predict(X).ravel()
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)
        return calculate_metrics(y, y_pred)
        
    def save(self, filepath: str) -> None:
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        if not filepath.endswith('.h5'):
            filepath += '.h5'
        self.model.save_model(filepath)
        
    def load(self, filepath: str) -> None:
        if not filepath.endswith('.h5'):
            filepath += '.h5'
        # Set dummy shape to initialize class, load will overwrite actual model structure
        self.model = StockLSTMModel(input_shape=(10, 10))
        self.model.load_model(filepath)
        self.input_shape = self.model.model.input_shape[1:]


class RFModelWrapper(BaseStockModel):
    """Wrapper for Scikit-Learn RandomForestRegressor."""
    
    def __init__(self, **kwargs):
        self.params = kwargs
        self.model = None
        
    def build(self, **kwargs) -> None:
        merged_params = {**self.params, **kwargs}
        self.model = RandomForestRegressor(**merged_params)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, 
              **kwargs) -> Dict[str, Any]:
        if self.model is None:
            self.build()
        X_train_2d = flatten_sequences(X_train)
        self.model.fit(X_train_2d, y_train)
        return {}
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        X_2d = flatten_sequences(X)
        return self.model.predict(X_2d)
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)
        return calculate_metrics(y, y_pred)
        
    def save(self, filepath: str) -> None:
        if self.model is None:
            raise ValueError("Model not trained.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)


class XGBModelWrapper(BaseStockModel):
    """Wrapper for XGBoost Regressor."""
    
    def __init__(self, **kwargs):
        self.params = kwargs
        self.model = None
        
    def build(self, **kwargs) -> None:
        if not XGB_AVAILABLE:
            raise ImportError("xgboost library is required to run XGBModelWrapper.")
        merged_params = {**self.params, **kwargs}
        self.model = xgb.XGBRegressor(**merged_params)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, 
              **kwargs) -> Dict[str, Any]:
        if self.model is None:
            self.build()
        X_train_2d = flatten_sequences(X_train)
        
        if X_val is not None:
            X_val_2d = flatten_sequences(X_val)
            self.model.fit(
                X_train_2d, y_train,
                eval_set=[(X_val_2d, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train_2d, y_train)
        return {}
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained.")
        X_2d = flatten_sequences(X)
        return self.model.predict(X_2d)
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)
        return calculate_metrics(y, y_pred)
        
    def save(self, filepath: str) -> None:
        if self.model is None:
            raise ValueError("Model not trained.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)


class LGBMModelWrapper(BaseStockModel):
    """Wrapper for LightGBM Regressor (Optional)."""
    
    def __init__(self, **kwargs):
        self.params = kwargs
        self.model = None
        
    def build(self, **kwargs) -> None:
        if not LGBM_AVAILABLE:
            raise ImportError("lightgbm library is not installed.")
        merged_params = {**self.params, **kwargs}
        self.model = lgb.LGBMRegressor(**merged_params)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, 
              **kwargs) -> Dict[str, Any]:
        if not LGBM_AVAILABLE:
            # Silent fallback to RF wrapper
            fallback = RFModelWrapper()
            fallback.build()
            self.model = fallback.model
            
        if self.model is None:
            self.build()
            
        X_train_2d = flatten_sequences(X_train)
        if LGBM_AVAILABLE and X_val is not None:
            X_val_2d = flatten_sequences(X_val)
            self.model.fit(
                X_train_2d, y_train,
                eval_set=[(X_val_2d, y_val)],
                callbacks=[lgb.early_stopping(5, verbose=False)]
            )
        else:
            self.model.fit(X_train_2d, y_train)
        return {}
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained.")
        X_2d = flatten_sequences(X)
        return self.model.predict(X_2d)
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)
        return calculate_metrics(y, y_pred)
        
    def save(self, filepath: str) -> None:
        if self.model is None:
            raise ValueError("Model not trained.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)


class LinearRegressionWrapper(BaseStockModel):
    """Wrapper for LinearRegression baseline."""
    
    def __init__(self, **kwargs):
        self.model = None
        
    def build(self, **kwargs) -> None:
        self.model = LinearRegression()
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, 
              **kwargs) -> Dict[str, Any]:
        if self.model is None:
            self.build()
        X_train_2d = flatten_sequences(X_train)
        self.model.fit(X_train_2d, y_train)
        return {}
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained.")
        X_2d = flatten_sequences(X)
        return self.model.predict(X_2d)
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)
        return calculate_metrics(y, y_pred)
        
    def save(self, filepath: str) -> None:
        if self.model is None:
            raise ValueError("Model not trained.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
