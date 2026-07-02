import os
import sys
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Add path compatibility
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.db_service import DatabaseService
from services.yahoo_service import YahooService
from services.news_service import NewsService
from services.recommendation_service import RecommendationService
from services.advisor_service import InvestmentAdvisorService
from models.model_selector import ModelSelector
from preprocess import DataPreprocessor

app = FastAPI(
    title="M.I.D.E. Platform API",
    description="Market Intelligence & Investment Decision Engine Backend",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Services
db = DatabaseService()
yahoo = YahooService()
news_service = NewsService(db, yahoo)
rec_service = RecommendationService(db)
advisor_service = InvestmentAdvisorService(db, rec_service)
model_selector = ModelSelector(db)

class TradeRequest(BaseModel):
    ticker: str
    action: str  # BUY or SELL
    shares: float
    price: float

@app.get("/health")
def health_check():
    """Return health status, database diagnostics, and active loaded model details."""
    db_ok = False
    active_models_summary = []
    
    session = None
    try:
        session = db.SessionLocal()
        # Query active models
        from services.db_service import ModelRecord, Company
        db_ok = True
        companies = session.query(Company).all()
        for c in companies:
            act_model = db.get_active_model(c.ticker)
            if act_model:
                active_models_summary.append({
                    "ticker": c.ticker,
                    "model_type": act_model["model_type"],
                    "version": act_model["version"],
                    "created_at": act_model["created_at"]
                })
    except Exception as e:
        db_ok = False
    finally:
        if session:
            session.close()
        
    return {
        "status": "healthy" if db_ok else "degraded",
        "database_connected": db_ok,
        "active_models": active_models_summary
    }

@app.get("/predict")
def predict_price(ticker: str = Query(..., description="Stock symbol")):
    """Predict next-day close using the registered active model for a ticker."""
    session = db.SessionLocal()
    try:
        from services.db_service import PriceHistory
        model_wrapper, record, preprocessor = model_selector.load_best_model(ticker)
        if not model_wrapper or not record:
            raise HTTPException(
                status_code=404, 
                detail=f"No active model found in registry for {ticker}. Please train models first."
            )
            
        # Get lookback window dynamically from preprocessor metadata
        seq_len = preprocessor.sequence_length if preprocessor else 30
        hist_prices = session.query(PriceHistory).filter(PriceHistory.ticker == ticker).order_by(PriceHistory.date.desc()).limit(seq_len + 15).all()
        
        if len(hist_prices) < seq_len:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient price history for prediction. Need at least {seq_len} records, got {len(hist_prices)}."
            )
            
        hist_prices.reverse()
        df_prices = pd.DataFrame([{
            "Date": p.date,
            "Open": p.open,
            "High": p.high,
            "Low": p.low,
            "Close": p.close,
            "Volume": p.volume
        } for p in hist_prices])
        
        if preprocessor is not None:
            # Safe, leak-free scaling with pre-fitted registry scaler
            X_latest = preprocessor.prepare_inference_data(df_prices, sequence_length=seq_len)
            pred_scaled = model_wrapper.predict(X_latest)
            pred_price = preprocessor.inverse_transform_targets(pred_scaled)[0]
        else:
            # Fallback to local fit if preprocessor file is missing
            fallback_prep = DataPreprocessor(scaler_type='minmax')
            processed = fallback_prep.prepare_full_pipeline(
                df_prices, sequence_length=seq_len, use_queue=True
            )
            X_latest = processed['X_test'][-1:]
            fallback_prep.scale_targets(np.array([p.close for p in hist_prices]))
            pred_scaled = model_wrapper.predict(X_latest)
            pred_price = fallback_prep.inverse_transform_targets(pred_scaled)[0]
        
        current_price = hist_prices[-1].close
        pct_change = (pred_price - current_price) / current_price
        
        return {
            "ticker": ticker,
            "current_price": float(current_price),
            "predicted_price": float(pred_price),
            "expected_return_pct": float(pct_change),
            "model_used": record["model_type"],
            "model_version": record["version"],
            "model_metrics": record["metrics"]
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    finally:
        session.close()

@app.get("/news")
def get_news(ticker: str = Query(..., description="Stock symbol"), limit: int = 15):
    """Retrieve news articles and computed FinBERT/lexicon sentiment for a ticker."""
    try:
        articles = db.get_news_articles(ticker, limit=limit)
        return [
            {
                "headline": a.headline,
                "summary": a.summary,
                "source": a.source,
                "url": a.url,
                "published_date": a.published_date.strftime("%Y-%m-%d %H:%M:%S"),
                "sentiment_score": a.sentiment_score,
                "sentiment_confidence": a.sentiment_confidence
            }
            for a in articles
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/events")
def get_events(ticker: str = Query(..., description="Stock symbol"), limit: int = 15):
    """Retrieve classified corporate events for a ticker."""
    try:
        events = db.get_events(ticker, limit=limit)
        return [
            {
                "event_type": e.event_type,
                "sentiment": e.sentiment,
                "confidence": e.confidence,
                "severity": e.severity,
                "description": e.description,
                "date": e.date.strftime("%Y-%m-%d %H:%M:%S")
            }
            for e in events
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend")
def get_recommendation(ticker: str = Query(..., description="Stock symbol")):
    """Retrieve action recommendation (BUY/HOLD/SELL) and detailed signal breakdown."""
    try:
        return rec_service.generate_recommendation(ticker)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/backtest")
def backtest_strategy(ticker: str = Query(..., description="Stock symbol"), 
                      lookback_days: int = Query(120, description="Historical lookback days")):
    """Run strategy backtesting of recommendation signals over historical periods."""
    try:
        res = rec_service.run_strategy_backtest(ticker, lookback_days)
        if "error" in res:
            raise HTTPException(status_code=400, detail=res["error"])
        return res
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio")
def get_portfolio():
    """Retrieve portfolio cash balance, initial capital, current holdings, and total portfolio valuation."""
    try:
        state = db.get_portfolio_state()
        holdings = db.get_holdings()
        
        # Calculate current valuation of holdings using yfinance latest prices
        total_holdings_val = 0.0
        parsed_holdings = []
        
        for h in holdings:
            ticker = h["ticker"]
            shares = h["shares"]
            avg_price = h["avg_purchase_price"]
            
            # Fetch latest close price from yfinance
            latest_price = avg_price
            try:
                import yfinance as yf
                t = yf.Ticker(ticker)
                # Use fast_info if available
                if hasattr(t, 'fast_info'):
                    latest_price = t.fast_info.get('last_price', avg_price)
                if not latest_price or latest_price == avg_price:
                    # Fallback to history close
                    hist = t.history(period="1d")
                    if not hist.empty:
                        latest_price = hist['Close'].iloc[-1]
            except Exception:
                pass
                
            current_value = shares * latest_price
            cost_basis = shares * avg_price
            pl = current_value - cost_basis
            pl_pct = (pl / cost_basis) if cost_basis > 0 else 0.0
            
            total_holdings_val += current_value
            parsed_holdings.append({
                "ticker": ticker,
                "shares": shares,
                "avg_purchase_price": avg_price,
                "current_price": float(latest_price),
                "current_value": float(current_value),
                "profit_loss": float(pl),
                "profit_loss_pct": float(pl_pct)
            })
            
        total_value = state["cash"] + total_holdings_val
        total_pl = total_value - state["initial_capital"]
        total_pl_pct = (total_pl / state["initial_capital"]) if state["initial_capital"] > 0 else 0.0
        
        # Calculate allocations
        for ph in parsed_holdings:
            ph["allocation_pct"] = ph["current_value"] / total_value if total_value > 0 else 0.0
            
        return {
            "cash": state["cash"],
            "initial_capital": state["initial_capital"],
            "total_holdings_value": total_holdings_val,
            "total_portfolio_value": total_value,
            "total_profit_loss": total_pl,
            "total_profit_loss_pct": total_pl_pct,
            "holdings": parsed_holdings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trade")
def execute_trade(trade: TradeRequest):
    """Execute a BUY or SELL transaction in Paper Trading mode, updating portfolio balance and holdings."""
    ticker = trade.ticker.upper()
    action = trade.action.upper()
    shares = trade.shares
    price = trade.price
    
    if action not in ["BUY", "SELL"]:
        raise HTTPException(status_code=400, detail="Invalid action. Must be 'BUY' or 'SELL'.")
    if shares <= 0:
        raise HTTPException(status_code=400, detail="Shares count must be positive.")
    if price <= 0:
        raise HTTPException(status_code=400, detail="Price must be positive.")
        
    state = db.get_portfolio_state()
    total_cost = shares * price
    
    if action == "BUY":
        if state["cash"] < total_cost:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient virtual cash. Needed: ${total_cost:.2f}, Available: ${state['cash']:.2f}."
            )
        # Deduct cash
        db.update_portfolio_cash(-total_cost)
        # Add to holding
        db.update_holding(ticker, shares, price, "BUY")
        # Log transaction
        db.record_transaction(ticker, "BUY", shares, price)
        
    elif action == "SELL":
        holding = db.get_holding(ticker)
        if not holding or holding["shares"] < shares:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient holdings of {ticker}. Hold: {holding['shares'] if holding else 0}, Sell Request: {shares}."
            )
        # Add cash
        db.update_portfolio_cash(total_cost)
        # Deduct from holding
        db.update_holding(ticker, shares, price, "SELL")
        # Log transaction
        db.record_transaction(ticker, "SELL", shares, price)
        
    return {
        "status": "success",
        "message": f"Successfully executed trade: {action} {shares} shares of {ticker} at ${price:.2f}"
    }

@app.get("/advisor")
def get_advisor_memo(ticker: str = Query(..., description="Stock symbol"), 
                     gemini_key: Optional[str] = Query(None, description="Optional Gemini API key")):
    """Generate a structured Markdown Investment Advisor Memo (Investment Thesis, Technicals, Sentiments, Forecast)."""
    try:
        memo = advisor_service.generate_investment_memo(ticker, gemini_api_key=gemini_key)
        return {"ticker": ticker, "memo_markdown": memo}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
