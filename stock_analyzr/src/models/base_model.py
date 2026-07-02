from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional

class BaseStockModel(ABC):
    """
    Abstract Base Class for all Stock Prediction Models.
    Ensures unified interface across different algorithms (LSTM, RF, XGB, etc.).
    """
    
    @abstractmethod
    def build(self, **kwargs) -> None:
        """Build and compile the underlying model."""
        pass
        
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, 
              **kwargs) -> Dict[str, Any]:
        """Train the model on the preprocessed dataset."""
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for input sequence data."""
        pass
        
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model and return key metrics (RMSE, MAE, MAPE, R2, Directional Accuracy)."""
        pass
        
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Serialize and save the model to disk."""
        pass
        
    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load the model from disk."""
        pass
