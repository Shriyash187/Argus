import os
import sys
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# Add path compatibility
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base_model import BaseStockModel
from models.model_wrappers import (
    LSTMModelWrapper, RFModelWrapper, XGBModelWrapper, LGBMModelWrapper, 
    LinearRegressionWrapper, LGBM_AVAILABLE, XGB_AVAILABLE
)
from models.explainability import calculate_shap_feature_importance

class ModelSelector:
    """
    Handles cross-model training, evaluation, comparison, and versioned persistence
    of the best-performing model to the Model Registry.
    """
    
    def __init__(self, db_service):
        self.db = db_service
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
        
    def train_and_compare(self, ticker: str, processed_data: Dict[str, Any], 
                          epochs: int = 15, batch_size: int = 32) -> Tuple[Dict[str, Dict[str, float]], str]:
        """
        Train all models on the preprocessed dataset and compare their performance.
        
        Args:
            ticker: Ticker symbol
            processed_data: Output dictionary from DataPreprocessor.prepare_full_pipeline
            epochs: Training epochs for LSTM
            batch_size: Batch size for LSTM
            
        Returns:
            Tuple of (comparison_results, best_model_type)
        """
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_val = processed_data['X_val']
        y_val = processed_data['y_val']
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        input_shape = X_train.shape[1:]
        
        # Instantiate wrappers
        wrappers = {
            "LSTM": LSTMModelWrapper(input_shape=input_shape),
            "Random Forest": RFModelWrapper(),
            "Linear Regression": LinearRegressionWrapper()
        }
        
        if XGB_AVAILABLE:
            wrappers["XGBoost"] = XGBModelWrapper()
        if LGBM_AVAILABLE:
            wrappers["LightGBM"] = LGBMModelWrapper()
            
        comparison_results = {}
        
        for name, wrapper in wrappers.items():
            try:
                print(f"Training model: {name} for {ticker}...")
                # Train model
                if name == "LSTM":
                    wrapper.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
                else:
                    wrapper.train(X_train, y_train, X_val, y_val)
                    
                # Evaluate model
                metrics = wrapper.evaluate(X_test, y_test)
                comparison_results[name] = metrics
                print(f"Model {name} evaluated. RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f}")
            except Exception as e:
                print(f"Failed to train {name}: {e}")
            
        if not comparison_results:
            raise ValueError("All models failed to train.")
            
        # Determine the best model based on lowest RMSE
        best_model_name = min(comparison_results, key=lambda k: comparison_results[k]['rmse'])
        best_metrics = comparison_results[best_model_name]
        
        # Save and register the best model
        best_wrapper = wrappers[best_model_name]
        
        # Determine next version number
        records = self.db.list_models(ticker)
        next_version = (records[0]['version'] + 1) if records else 1
        
        # Replace space for file path compatibility
        clean_model_name = best_model_name.lower().replace(" ", "_")
        ext = ".h5" if best_model_name == "LSTM" else ".pkl"
        filepath = os.path.join(self.models_dir, f"{ticker}_{clean_model_name}_v{next_version}{ext}")
        
        # Save model to disk
        best_wrapper.save(filepath)
        
        # Save preprocessor using pickle
        preprocessor_path = filepath.replace(ext, "_preprocessor.pkl")
        import pickle
        try:
            with open(preprocessor_path, "wb") as f:
                pickle.dump(processed_data['preprocessor'], f)
        except Exception as e:
            print(f"Failed to save preprocessor: {e}")
        
        # Register in Database Model Registry (deactivates others for this ticker)
        self.db.register_model(
            ticker=ticker,
            model_type=best_model_name,
            filepath=filepath,
            metrics=best_metrics
        )
        
        return comparison_results, best_model_name

    def load_best_model(self, ticker: str) -> Tuple[Optional[BaseStockModel], Optional[Dict[str, Any]], Optional[Any]]:
        """
        Load the active model from the registry for a ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Tuple of (model_wrapper, model_record_dict, preprocessor)
        """
        record = self.db.get_active_model(ticker)
        if not record:
            return None, None, None
            
        model_type = record['model_type']
        filepath = record['filepath']
        
        # Instantiate correct wrapper
        if model_type == "LSTM":
            wrapper = LSTMModelWrapper()
        elif model_type == "Random Forest":
            wrapper = RFModelWrapper()
        elif model_type == "XGBoost":
            wrapper = XGBModelWrapper()
        elif model_type == "LightGBM":
            wrapper = LGBMModelWrapper()
        elif model_type == "Linear Regression":
            wrapper = LinearRegressionWrapper()
        else:
            raise ValueError(f"Unknown model type in registry: {model_type}")
            
        # Load weights/binary from disk
        wrapper.load(filepath)
        
        # Load preprocessor
        preprocessor = None
        ext = ".h5" if model_type == "LSTM" else ".pkl"
        preprocessor_path = filepath.replace(ext, "_preprocessor.pkl")
        if os.path.exists(preprocessor_path):
            import pickle
            try:
                with open(preprocessor_path, "rb") as f:
                    preprocessor = pickle.load(f)
            except Exception as e:
                print(f"Failed to load preprocessor from {preprocessor_path}: {e}")
                
        return wrapper, record, preprocessor

    def get_feature_importances(self, wrapper: BaseStockModel, X_val: np.ndarray, 
                                 feature_names: List[str]) -> Dict[str, float]:
        """
        Get feature importances for a model wrapper.
        Uses SHAP for tree models, coefficients for linear regression, and equal weights for LSTM.
        """
        model_type = type(wrapper).__name__
        
        if model_type in ["RFModelWrapper", "XGBModelWrapper", "LGBMModelWrapper"]:
            return calculate_shap_feature_importance(wrapper, X_val, feature_names)
            
        elif model_type == "LinearRegressionWrapper":
            if hasattr(wrapper.model, 'coef_'):
                coefs = np.abs(wrapper.model.coef_)
                num_features = len(feature_names)
                feature_imp = {name: 0.0 for name in feature_names}
                for idx, coef in enumerate(coefs):
                    feat_idx = idx % num_features
                    if feat_idx < len(feature_names):
                        feature_imp[feature_names[feat_idx]] += float(coef)
                return dict(sorted(feature_imp.items(), key=lambda item: item[1], reverse=True))
                
        # Fallback for LSTM or uninitialized models
        return {name: 1.0 / len(feature_names) for name in feature_names}
