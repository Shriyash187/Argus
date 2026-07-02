import numpy as np
from typing import Dict, List, Tuple, Any

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

def calculate_shap_feature_importance(model_wrapper, X_val: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    """
    Calculate SHAP feature importances for tree-based models.
    Aggregates feature importances over the lookback sequence windows to provide 
    a clean, concise view of which baseline indicators matter most.
    
    Args:
        model_wrapper: RFModelWrapper, XGBModelWrapper, or LGBMModelWrapper
        X_val: Input validation array (3D sequence or 2D flattened)
        feature_names: List of base feature names
        
    Returns:
        Sorted Dict mapping feature names to their average absolute SHAP values
    """
    if not SHAP_AVAILABLE:
        # Fallback: Use Scikit-Learn built-in feature importances if available
        if hasattr(model_wrapper, 'model') and model_wrapper.model is not None:
            if hasattr(model_wrapper.model, 'feature_importances_'):
                importances = model_wrapper.model.feature_importances_
                num_features = len(feature_names)
                feature_imp = {name: 0.0 for name in feature_names}
                for idx, imp in enumerate(importances):
                    feat_idx = idx % num_features
                    if feat_idx < len(feature_names):
                        feature_imp[feature_names[feat_idx]] += float(imp)
                return dict(sorted(feature_imp.items(), key=lambda item: item[1], reverse=True))
        return {name: 0.0 for name in feature_names}
        
    try:
        from models.model_wrappers import flatten_sequences
        X_2d = flatten_sequences(X_val)
        
        # Instantiate TreeExplainer
        explainer = shap.TreeExplainer(model_wrapper.model)
        shap_values = explainer.shap_values(X_2d)
        
        # Calculate mean absolute SHAP values per columns
        if isinstance(shap_values, list):
            # Take average over multi-outputs if list
            shap_values = np.mean(np.abs(shap_values), axis=0)
        else:
            shap_values = np.abs(shap_values)
            
        mean_shap = np.mean(shap_values, axis=0)
        
        num_features = len(feature_names)
        feature_shap = {name: 0.0 for name in feature_names}
        
        for idx, val in enumerate(mean_shap):
            feat_idx = idx % num_features
            if feat_idx < len(feature_names):
                feature_shap[feature_names[feat_idx]] += float(val)
                
        # Return sorted dictionary
        sorted_shap = dict(sorted(feature_shap.items(), key=lambda item: item[1], reverse=True))
        return sorted_shap
        
    except Exception as e:
        # Fallback to feature_importances_
        if hasattr(model_wrapper, 'model') and model_wrapper.model is not None:
            if hasattr(model_wrapper.model, 'feature_importances_'):
                importances = model_wrapper.model.feature_importances_
                num_features = len(feature_names)
                feature_imp = {name: 0.0 for name in feature_names}
                for idx, imp in enumerate(importances):
                    feat_idx = idx % num_features
                    if feat_idx < len(feature_names):
                        feature_imp[feature_names[feat_idx]] += float(imp)
                return dict(sorted(feature_imp.items(), key=lambda item: item[1], reverse=True))
        return {"Error calculating SHAP": 0.0}
