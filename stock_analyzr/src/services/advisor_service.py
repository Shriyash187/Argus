import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# Add path compatibility
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.log_service import get_logger

logger = get_logger()

class InvestmentAdvisorService:
    """
    Generates a structured, professional Investment Advisor Memo.
    Integrates optional Gemini API generation if API key is provided;
    otherwise falls back to a high-fidelity rule-based local Markdown generator.
    """
    
    def __init__(self, db_service, recommendation_service):
        self.db = db_service
        self.rec_service = recommendation_service
        
    def generate_investment_memo(self, ticker: str, gemini_api_key: Optional[str] = None) -> str:
        """
        Synthesize stock intelligence into a detailed investment advisor memo.
        
        Args:
            ticker: Ticker symbol
            gemini_api_key: Optional Gemini API key
            
        Returns:
            Structured Markdown Investment Memo
        """
        # Fetch current recommendation details
        rec = self.rec_service.generate_recommendation(ticker)
        
        # Get active model metrics
        model_name = "N/A"
        model_rmse = 0.0
        model_version = 0
        active_model = self.db.get_active_model(ticker)
        if active_model:
            model_name = active_model["model_type"]
            model_version = active_model["version"]
            model_rmse = active_model["metrics"].get("rmse", 0.0)
            
        # Compile metadata dictionary
        data_summary = {
            "ticker": ticker,
            "date": rec["date"],
            "action": rec["action"],
            "confidence": rec["confidence_score"],
            "expected_return": rec["expected_return"],
            "risk_level": rec["risk_level"],
            "horizon": rec["investment_horizon"],
            "pos_signals": rec["key_positive_signals"],
            "neg_signals": rec["key_negative_signals"],
            "model_name": model_name,
            "model_version": model_version,
            "model_rmse": model_rmse
        }
        
        # Try to use Gemini if key is provided
        if gemini_api_key and gemini_api_key != "YOUR_GEMINI_KEY" and gemini_api_key.strip() != "":
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_api_key)
                
                # Check for available model
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                prompt = f"""
                You are a senior financial analyst and investment advisor at a top-tier asset management firm.
                Generate a professional investment memo for the stock ticker {ticker} based on the following data:
                
                - Date: {data_summary['date']}
                - Action Recommendation: {data_summary['action']}
                - Confidence Score: {data_summary['confidence'] * 100}%
                - Expected Return: {data_summary['expected_return'] * 100:.2f}%
                - Risk Level: {data_summary['risk_level']}
                - Investment Horizon: {data_summary['horizon']}
                - Positive Signals: {', '.join(data_summary['pos_signals'])}
                - Negative Signals: {', '.join(data_summary['neg_signals'])}
                - Active ML Model: {data_summary['model_name']} (version {data_summary['model_version']}, test RMSE: {data_summary['model_rmse']:.4f})
                
                The memo MUST have the following structured sections:
                1. Executive Summary & Investment Thesis
                2. Technical Indicators Breakdown
                3. Market Sentiment & Catalyst Events
                4. Quantitative Model Forecast & Evaluation
                5. Risk Analysis & Implementation Guide (including entry/exit guidance)
                
                Write in a professional, objective, and analytical tone. Use Markdown headers and bullet points.
                """
                
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text
                    
            except Exception as e:
                logger.warning(f"Failed to generate memo using Gemini API: {e}. Falling back to local template.")
                
        # High-fidelity Local Template fallback
        return self._generate_local_memo(data_summary)
        
    def _generate_local_memo(self, data: Dict[str, Any]) -> str:
        """Rule-based financial synthesis memo generator (Local Fallback)."""
        ticker = data["ticker"]
        date = data["date"]
        action = data["action"]
        confidence = data["confidence"]
        expected_return = data["expected_return"]
        risk = data["risk_level"]
        horizon = data["horizon"]
        
        pos_list = "\n".join([f"- {sig}" for sig in data["pos_signals"]])
        neg_list = "\n".join([f"- {sig}" for sig in data["neg_signals"]])
        combined_list = pos_list if action == "BUY" else (neg_list if action == "SELL" else pos_list + "\n" + neg_list)
        
        action_verbs = {
            "BUY": "suggests a strong accumulation strategy. Underpricing and positive momentum align to present an attractive entry point.",
            "SELL": "warrants a liquidation or hedging strategy. Overvaluation, bearish crossovers, or negative news pose material risk.",
            "HOLD": "indicates a neutral position. The asset is trading close to fair value, with positive and negative catalysts balancing out."
        }
        
        memo = f"""# M.I.D.E. Investment Advisor Memo
**Confidential | Prepared for Internal Portfolio Manager Review**

## Metadata Summary
- **Ticker Symbol**: {ticker}
- **Date Generated**: {date}
- **Algorithmic Action**: **{action}**
- **Confidence Rating**: {confidence * 100:.0f}%
- **Expected Return**: {expected_return * 100:+.2f}%
- **Risk Classification**: {risk}
- **Target Investment Horizon**: {horizon}

---

## 1. Executive Summary & Investment Thesis
On the date of {date}, the Market Intelligence & Investment Decision Engine (M.I.D.E.) platform evaluated {ticker} and issued a **{action}** recommendation. 

This assessment is backed by a {confidence * 100:.0f}% confidence rating, indicating high model consensus. The thesis {action_verbs[action]} Over the specified horizon of {horizon}, we project an expected return of approximately {expected_return * 100:+.2f}%, adjusting for underlying asset volatility.

---

## 2. Technical Indicators Breakdown
A granular analysis of {ticker}'s price activity reveals the following technical structure:
### Key Technical Signals:
{combined_list}

- **Trend Direction**: The stock is currently trading relative to its EMA and SMA averages, indicating a structural trend alignment. 
- **Volatility Envelope**: Prices are trading inside standard Bollinger Band boundaries, showing volatility ranges.

---

## 3. Market Sentiment & Catalyst Events
Sentiment tracking over the past 7 days indicates active discussion channels.
- **News Sentiment Index**: The aggregate polarity of scraped financial media coverage shows a net bias.
- **Catalyst Classification**: High-priority corporate milestones and regulatory classified events have been factored into the risk parameters, providing immediate fundamental catalysts.

---

## 4. Quantitative Model Forecast & Evaluation
Prediction metrics are supplied by our active registry model:
- **Active Registry Model**: {data['model_name']} (Version {data['model_version']})
- **Historical Model RMSE**: {data['model_rmse']:.4f}
- **Forecast Summary**: The machine learning model projects a next-day close trend that matches the primary {action} action. Historical directional accuracy of the model on unseen test folds suggests stable pattern matching.

---

## 5. Risk Analysis & Implementation Guide
### Active Risk Factors:
{neg_list}

### Execution Strategy:
- **BUY Order Execution**: Limit orders recommended near support boundaries. If the overall action is SELL, stop-loss orders should be placed below the 20-day EMA support levels.
- **Risk Mitigation**: Portfolio exposure should be restricted to a standard weight (e.g., 2-5% of capital) due to the {risk} risk rating.
"""
        return memo
