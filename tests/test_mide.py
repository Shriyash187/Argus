import os
import sys
import pytest
import numpy as np
import pandas as pd
from datetime import datetime

# Add path compatibility
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stock_analyzr", "src"))

from services.db_service import DatabaseService
from models.model_wrappers import RFModelWrapper, LinearRegressionWrapper
from models.model_selector import ModelSelector
from services.news_service import NewsService
from services.recommendation_service import RecommendationService

@pytest.fixture
def temp_db():
    """Create a temporary database service using SQLite in-memory database."""
    db_service = DatabaseService("sqlite:///:memory:")
    # Seed watched companies
    db_service.cache_company("AAPL", {"name": "Apple Inc."})
    db_service.add_company("AAPL", {"name": "Apple Inc."})
    return db_service

def test_db_portfolio(temp_db):
    """Test portfolio database operations."""
    state = temp_db.get_portfolio_state()
    assert state["cash"] == 100000.0
    assert state["initial_capital"] == 100000.0
    
    # Update cash
    temp_db.update_portfolio_cash(-5000.0)
    assert temp_db.get_portfolio_state()["cash"] == 95000.0
    
    # Update holding
    temp_db.update_holding("AAPL", 10, 150.0, "BUY")
    holding = temp_db.get_holding("AAPL")
    assert holding is not None
    assert holding["shares"] == 10
    assert holding["avg_purchase_price"] == 150.0
    
    # Record transaction
    temp_db.record_transaction("AAPL", "BUY", 10, 150.0)
    txs = temp_db.get_transactions()
    assert len(txs) == 1
    assert txs[0]["ticker"] == "AAPL"
    assert txs[0]["action"] == "BUY"

def test_model_wrappers():
    """Test model wrappers and shape adaptions."""
    X_train = np.random.randn(10, 5, 3) # 10 samples, 5 days, 3 features
    y_train = np.random.randn(10)
    
    # Linear Regression wrapper
    lr = LinearRegressionWrapper()
    lr.train(X_train, y_train)
    pred = lr.predict(X_train)
    assert len(pred) == 10
    
    # RandomForest wrapper
    rf = RFModelWrapper(n_estimators=5)
    rf.train(X_train, y_train)
    pred_rf = rf.predict(X_train)
    assert len(pred_rf) == 10

def test_model_registry(temp_db):
    """Test model registration and setting active version."""
    metrics = {"rmse": 0.12, "r2": 0.85}
    filepath = "models/AAPL_test_v1.pkl"
    
    version1 = temp_db.register_model("AAPL", "Random Forest", filepath, metrics)
    assert version1 == 1
    
    active = temp_db.get_active_model("AAPL")
    assert active["version"] == 1
    assert active["model_type"] == "Random Forest"
    
    # Register another one
    version2 = temp_db.register_model("AAPL", "XGBoost", "models/AAPL_test_v2.pkl", {"rmse": 0.08, "r2": 0.90})
    assert version2 == 2
    
    active_now = temp_db.get_active_model("AAPL")
    assert active_now["version"] == 2
    assert active_now["model_type"] == "XGBoost"
    
    # Manually switch active version back
    temp_db.set_active_model("AAPL", 1)
    assert temp_db.get_active_model("AAPL")["version"] == 1

def test_news_sentiment_fallback(temp_db):
    """Test that NewsService falls back to lexicon-based sentiment if FinBERT is unavailable."""
    # Force FinBERT to None
    news_svc = NewsService(temp_db, None)
    news_svc.finbert_pipeline = None
    
    score, conf = news_svc.analyze_sentiment("Tesla shares surge on beat profit growth")
    assert score > 0.0
    assert conf > 0.0

def test_recommendation_logic(temp_db):
    """Test that recommendation service parses signals correctly."""
    rec_svc = RecommendationService(temp_db)
    # Generate empty recommendation because no data in DB
    rec = rec_svc.generate_recommendation("AAPL")
    assert rec["action"] == "HOLD" # default fallback
    assert rec["confidence_score"] == 0.5


def test_inference_scaling_leak_free():
    """Test that preprocessor scaling for inference does not modify scaling parameters."""
    from preprocess import DataPreprocessor
    
    # Create random historical training data
    df_train = pd.DataFrame({
        "Open": np.linspace(100, 150, 100),
        "High": np.linspace(102, 152, 100),
        "Low": np.linspace(98, 148, 100),
        "Close": np.linspace(101, 151, 100),
        "Volume": np.linspace(1000, 5000, 100)
    })
    
    preprocessor = DataPreprocessor(scaler_type='minmax')
    processed = preprocessor.prepare_full_pipeline(df_train, sequence_length=10, train_ratio=0.7)
    
    # Verify that feature scaler has been fitted
    assert preprocessor.feature_scaler is not None
    
    # Get current scaling params (e.g. data_max_ of the close price at index 0)
    orig_max_close = preprocessor.feature_scaler.data_max_[0]
    
    # Now create extreme dummy inference data (e.g. Close is 1000)
    df_inference = pd.DataFrame({
        "Open": [1000.0] * 15,
        "High": [1000.0] * 15,
        "Low": [1000.0] * 15,
        "Close": [1000.0] * 15,
        "Volume": [10000.0] * 15
    })
    
    # Call prepare_inference_data
    X_inference = preprocessor.prepare_inference_data(df_inference, sequence_length=10)
    
    # Verify that the scaling parameters of the preprocessor have NOT changed!
    assert preprocessor.feature_scaler.data_max_[0] == orig_max_close
    
    # Verify the output shape is (1, sequence_length, features)
    assert X_inference.shape == (1, 10, len(preprocessor.feature_columns) + 1)

