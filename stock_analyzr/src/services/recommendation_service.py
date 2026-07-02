import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

# Add path compatibility
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.log_service import get_logger
from models.model_selector import ModelSelector
from preprocess import DataPreprocessor

logger = get_logger()

class RecommendationService:
    """
    Combines ML predictions, technical indicators, news sentiment, and corporate events
    to generate structured trading recommendations (BUY/HOLD/SELL).
    Includes a strategy backtester to simulate historical recommendation signals.
    """
    
    def __init__(self, db_service):
        self.db = db_service
        self.model_selector = ModelSelector(db_service)
        
    def generate_recommendation(self, ticker: str, date_val: Optional[datetime.date] = None) -> Dict[str, Any]:
        """
        Generate recommendation for a given ticker on a specific date (defaults to latest available).
        
        Args:
            ticker: Stock symbol
            date_val: Date to generate recommendation for (defaults to today)
            
        Returns:
            Dict containing action, confidence, risk, expected return, horizon, and positive/negative signals.
        """
        session = self.db.SessionLocal()
        try:
            from services.db_service import PriceHistory, TechnicalFeature, NewsArticle, Event
            
            # 1. Fetch latest prices
            prices_query = session.query(PriceHistory).filter(PriceHistory.ticker == ticker)
            if date_val:
                prices_query = prices_query.filter(PriceHistory.date <= date_val)
            prices = prices_query.order_by(PriceHistory.date.desc()).limit(35).all()
            
            if not prices:
                return self._empty_recommendation("No price data available.")
                
            prices.reverse() # Sort chronologically
            current_price = prices[-1].close
            prev_price = prices[-2].close if len(prices) > 1 else current_price
            
            # 2. Fetch technical features
            tech_query = session.query(TechnicalFeature).filter(TechnicalFeature.ticker == ticker)
            if date_val:
                tech_query = tech_query.filter(TechnicalFeature.date <= date_val)
            tech = tech_query.order_by(TechnicalFeature.date.desc()).limit(1).first()
            
            # 3. Fetch latest news sentiment
            end_date = datetime.combine(date_val or datetime.utcnow().date(), datetime.max.time())
            start_date = end_date - timedelta(days=7)
            news = session.query(NewsArticle).filter(
                NewsArticle.ticker == ticker,
                NewsArticle.published_date.between(start_date, end_date)
            ).all()
            
            avg_news_sentiment = np.mean([n.sentiment_score for n in news]) if news else 0.0
            
            # 4. Fetch corporate events
            events = session.query(Event).filter(
                Event.ticker == ticker,
                Event.date.between(end_date - timedelta(days=14), end_date)
            ).all()
            
            # 5. Load ML forecast (Optional/Fallback)
            predicted_return_pct = 0.0
            ml_available = False
            
            # Try to load best active model from registry
            try:
                model_wrapper, record, preprocessor = self.model_selector.load_best_model(ticker)
                if model_wrapper and record:
                    # Fetch lookback features
                    lookback_query = session.query(PriceHistory).filter(PriceHistory.ticker == ticker)
                    if date_val:
                        lookback_query = lookback_query.filter(PriceHistory.date <= date_val)
                    # Get required sequence lookback
                    seq_len = preprocessor.sequence_length if preprocessor else 30
                    hist_prices = lookback_query.order_by(PriceHistory.date.desc()).limit(seq_len + 15).all()
                    
                    if len(hist_prices) >= seq_len:
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
                            # Leak-free scaling with pre-fitted registry scaler
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
                            
                        predicted_return_pct = (pred_price - current_price) / current_price
                        ml_available = True
            except Exception as e:
                logger.warning(f"ML forecast prediction bypassed: {e}")
                
            # Compile signals
            pos_signals = []
            neg_signals = []
            score = 0.0
            
            # Evaluate technicals
            if tech:
                # RSI
                if tech.rsi:
                    if tech.rsi < 30:
                        pos_signals.append(f"RSI is oversold ({tech.rsi:.1f})")
                        score += 1.5
                    elif tech.rsi > 70:
                        neg_signals.append(f"RSI is overbought ({tech.rsi:.1f})")
                        score -= 1.5
                    elif tech.rsi < 45:
                        pos_signals.append(f"RSI shows positive momentum ({tech.rsi:.1f})")
                        score += 0.5
                    elif tech.rsi > 55:
                        neg_signals.append(f"RSI shows cooling momentum ({tech.rsi:.1f})")
                        score -= 0.5
                
                # MACD Crossover
                if tech.macd and tech.macd_signal:
                    if tech.macd > tech.macd_signal:
                        pos_signals.append("MACD is bullish (above signal line)")
                        score += 1.0
                    else:
                        neg_signals.append("MACD is bearish (below signal line)")
                        score -= 1.0
                        
                # Bollinger Bands
                if tech.bollinger_lower and tech.bollinger_upper:
                    if current_price <= tech.bollinger_lower:
                        pos_signals.append("Price is below Bollinger Lower Band (reversion setup)")
                        score += 1.0
                    elif current_price >= tech.bollinger_upper:
                        neg_signals.append("Price is above Bollinger Upper Band (overextended)")
                        score -= 1.0
                        
                # Trend analysis (EMA)
                if tech.ema_20:
                    if current_price > tech.ema_20:
                        pos_signals.append("Price is trading above 20-day EMA (uptrend)")
                        score += 1.0
                    else:
                        neg_signals.append("Price is trading below 20-day EMA (downtrend)")
                        score -= 1.0
            else:
                # Simple moving average fallback using prices
                if len(prices) >= 20:
                    ma_20 = np.mean([p.close for p in prices[-20:]])
                    if current_price > ma_20:
                        pos_signals.append("Price is above 20-day SMA")
                        score += 0.5
                    else:
                        neg_signals.append("Price is below 20-day SMA")
                        score -= 0.5
            
            # Evaluate News
            if news:
                if avg_news_sentiment > 0.15:
                    pos_signals.append(f"Positive news sentiment ({avg_news_sentiment:+.2f} over {len(news)} articles)")
                    score += 1.0
                elif avg_news_sentiment < -0.15:
                    neg_signals.append(f"Negative news sentiment ({avg_news_sentiment:.2f} over {len(news)} articles)")
                    score -= 1.0
            
            # Evaluate Events
            for ev in events:
                if ev.sentiment == "Positive":
                    pos_signals.append(f"Corporate Event: {ev.event_type} (Severity: {ev.severity:.2f})")
                    score += ev.severity * 1.5
                elif ev.sentiment == "Negative":
                    neg_signals.append(f"Corporate Event: {ev.event_type} (Severity: {ev.severity:.2f})")
                    score -= ev.severity * 1.5
                    
            # Evaluate ML Model
            if ml_available:
                if predicted_return_pct > 0.015:
                    pos_signals.append(f"ML Model forecasts price increase (+{predicted_return_pct*100:.2f}%)")
                    score += 1.5
                elif predicted_return_pct < -0.015:
                    neg_signals.append(f"ML Model forecasts price decrease ({predicted_return_pct*100:.2f}%)")
                    score -= 1.5
                elif predicted_return_pct > 0:
                    pos_signals.append(f"ML Model forecasts minor rise (+{predicted_return_pct*100:.2f}%)")
                    score += 0.5
                else:
                    neg_signals.append(f"ML Model forecasts minor dip ({predicted_return_pct*100:.2f}%)")
                    score -= 0.5
            else:
                # Fallback direction using standard returns
                hist_return = (current_price - prev_price) / prev_price
                predicted_return_pct = hist_return * 0.1 # Conservative drift
                
            # Determine Action
            if score >= 2.0:
                action = "BUY"
            elif score <= -2.0:
                action = "SELL"
            else:
                action = "HOLD"
                
            # Confidence Score
            # Percentage based on size of absolute score relative to max possible
            max_possible_score = 6.0
            confidence = min(0.95, 0.5 + (abs(score) / max_possible_score) * 0.45)
            
            # Risk Level
            # Combine price volatility + news density
            vol = tech.volatility if (tech and tech.volatility) else 0.02
            event_risk = max([ev.severity for ev in events]) if events else 0.0
            
            risk_val = vol * 15 + event_risk * 0.5
            if risk_val < 0.3:
                risk_level = "Low"
            elif risk_val < 0.6:
                risk_level = "Medium"
            else:
                risk_level = "High"
                
            # Investment Horizon
            if action == "BUY":
                horizon = "Medium-term (3-12 months)" if score < 3.5 else "Short-term (1-3 months)"
            elif action == "SELL":
                horizon = "Short-term (1-3 months)"
            else:
                horizon = "Long-term (12+ months)"
                
            # Expected Return
            expected_return = predicted_return_pct * 3.0 if action == "BUY" else (predicted_return_pct * 1.5 if action == "SELL" else 0.0)
            
            return {
                "ticker": ticker,
                "date": (date_val or datetime.utcnow().date()).strftime("%Y-%m-%d"),
                "action": action,
                "confidence_score": round(confidence, 2),
                "expected_return": round(expected_return, 4),
                "risk_level": risk_level,
                "investment_horizon": horizon,
                "key_positive_signals": pos_signals if pos_signals else ["No major positive signals"],
                "key_negative_signals": neg_signals if neg_signals else ["No major negative signals"]
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return self._empty_recommendation(str(e))
        finally:
            session.close()
            
    def _empty_recommendation(self, reason: str) -> Dict[str, Any]:
        return {
            "ticker": "N/A",
            "date": datetime.utcnow().date().strftime("%Y-%m-%d"),
            "action": "HOLD",
            "confidence_score": 0.5,
            "expected_return": 0.0,
            "risk_level": "Medium",
            "investment_horizon": "N/A",
            "key_positive_signals": ["None"],
            "key_negative_signals": [f"Bypassed: {reason}"]
        }
        
    def run_strategy_backtest(self, ticker: str, lookback_days: int = 120) -> Dict[str, Any]:
        """
        Evaluate recommendation logic over historical price data.
        
        Args:
            ticker: Ticker symbol
            lookback_days: Number of days to run backtest
            
        Returns:
            Dict containing cumulative return, Sharpe ratio, drawdown, win rate, and trade log.
        """
        session = self.db.SessionLocal()
        try:
            from services.db_service import PriceHistory
            
            # Fetch prices
            start_date = datetime.utcnow().date() - timedelta(days=lookback_days + 30) # get extra cushion
            prices = session.query(PriceHistory).filter(
                PriceHistory.ticker == ticker,
                PriceHistory.date >= start_date
            ).order_by(PriceHistory.date.asc()).all()
            
            if len(prices) < 15:
                return {"error": "Insufficient historical price records to execute backtest."}
                
            dates = [p.date for p in prices]
            close_prices = [p.close for p in prices]
            
            # Run recommendations day-by-day
            signals = []
            trade_log = []
            
            # Align indices so we lookback correctly
            start_idx = len(prices) - lookback_days
            if start_idx < 10:
                start_idx = 10
                
            position = 0 # 0 = cash, 1 = long
            entry_price = 0.0
            entry_date = None
            
            portfolio_value = 10000.0
            initial_val = portfolio_value
            portfolio_history = []
            
            for idx in range(start_idx, len(prices)):
                current_date = dates[idx]
                current_price = close_prices[idx]
                
                # Compute recommendation for this historical day
                rec = self.generate_recommendation(ticker, date_val=current_date)
                action = rec["action"]
                
                # Simulate trading logic
                if action == "BUY" and position == 0:
                    position = 1
                    entry_price = current_price
                    entry_date = current_date
                    trade_log.append({
                        "date": current_date.strftime("%Y-%m-%d"),
                        "action": "BUY",
                        "price": round(current_price, 2),
                        "shares": round(portfolio_value / current_price, 4),
                        "return": 0.0
                    })
                elif action == "SELL" and position == 1:
                    position = 0
                    trade_return = (current_price - entry_price) / entry_price
                    portfolio_value *= (1.0 + trade_return)
                    trade_log.append({
                        "date": current_date.strftime("%Y-%m-%d"),
                        "action": "SELL",
                        "price": round(current_price, 2),
                        "shares": 0.0,
                        "return": round(trade_return, 4)
                    })
                    
                # Track portfolio value
                current_val = portfolio_value
                if position == 1:
                    current_val = (portfolio_value / entry_price) * current_price
                portfolio_history.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "value": current_val,
                    "price": current_price
                })
                
            if not portfolio_history:
                return {"error": "Backtest produced empty results."}
                
            df_port = pd.DataFrame(portfolio_history)
            df_port['returns'] = df_port['value'].pct_change().fillna(0.0)
            
            # Metrics
            total_return = (df_port['value'].iloc[-1] - initial_val) / initial_val
            
            # Annualized Sharpe (assuming daily returns)
            daily_std = df_port['returns'].std()
            sharpe = (df_port['returns'].mean() / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0
            
            # Max Drawdown
            df_port['peak'] = df_port['value'].cummax()
            df_port['dd'] = (df_port['value'] - df_port['peak']) / df_port['peak']
            max_dd = df_port['dd'].min()
            
            # Win Rate
            sell_trades = [t['return'] for t in trade_log if t['action'] == "SELL"]
            win_rate = np.mean([1 for r in sell_trades if r > 0]) if sell_trades else 0.0
            
            # Benchmark Buy & Hold return
            bh_return = (close_prices[-1] - close_prices[start_idx]) / close_prices[start_idx]
            
            return {
                "ticker": ticker,
                "lookback_days": lookback_days,
                "cumulative_return": round(total_return, 4),
                "buy_and_hold_return": round(bh_return, 4),
                "sharpe_ratio": round(sharpe, 2),
                "max_drawdown": round(max_dd, 4),
                "win_rate": round(win_rate, 2),
                "trade_history": trade_log,
                "portfolio_value_history": portfolio_history
            }
            
        except Exception as e:
            logger.error(f"Error running strategy backtest: {e}")
            return {"error": f"Error running backtest: {str(e)}"}
        finally:
            session.close()
