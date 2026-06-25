import urllib.request
import urllib.parse
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from services.log_service import get_logger

logger = get_logger()

class TickerSearchService:
    """Resolves arbitrary company names/queries to validated Yahoo Finance tickers and manages the local database cache."""
    def __init__(self, db_service):
        self.db = db_service

    def search_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Query public Yahoo Finance Search API to find matching ticker symbols.
        
        Args:
            query: User's search query (e.g. "Apple", "Reliance")
            
        Returns:
            List of matching records with symbol, name, exchange, etc.
        """
        logger.info(f"Initiating stock search query for: '{query}'")
        
        # URL encode query
        encoded_query = urllib.parse.quote(query)
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={encoded_query}&quotesCount=10&newsCount=0"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36'
        }
        
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                
            results = []
            quotes = data.get("quotes", [])
            for q in quotes:
                symbol = q.get("symbol")
                name = q.get("shortname", q.get("longname", "Unknown Company"))
                exchange = q.get("exchange", "Unknown")
                country = q.get("country", "Unknown")
                quote_type = q.get("quoteType", "EQUITY")
                
                # We focus mostly on equities (stocks)
                if quote_type == "EQUITY" or not quote_type:
                    # Clean/normalize exchange name
                    exch_upper = exchange.upper()
                    detected_exch = "Unknown"
                    if "NMS" in exch_upper or "NAS" in exch_upper or "NGM" in exch_upper:
                        detected_exch = "NASDAQ"
                    elif "NYQ" in exch_upper or "NYS" in exch_upper:
                        detected_exch = "NYSE"
                    elif "NSI" in exch_upper or "NSE" in exch_upper:
                        detected_exch = "NSE"
                    elif "BSE" in exch_upper or "BOM" in exch_upper:
                        detected_exch = "BSE"
                    else:
                        detected_exch = exchange
                        
                    results.append({
                        "ticker": symbol,
                        "name": name,
                        "exchange": detected_exch,
                        "country": country,
                        "currency": q.get("currency", "USD"),
                        "sector": q.get("sector", "N/A"),
                        "industry": q.get("industry", "N/A")
                    })
                    
            logger.info(f"Search API returned {len(results)} matches for query '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Failed to query Yahoo Search API: {e}")
            return []

    def resolve_ticker(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Attempts to resolve a company name query to a single validated ticker.
        Checks cache first. If cache missed, searches public search API and caches result.
        
        Args:
            query: The stock search query
            
        Returns:
            Resolved metadata dict or None
        """
        query_cleaned = query.strip()
        if not query_cleaned:
            return None
            
        # 1. If it looks exactly like an existing ticker, check cache first
        cached = self.db.get_cached_company(query_cleaned.upper())
        if cached:
            # Check if cache is stale (older than 7 days)
            age = datetime.utcnow() - cached.last_verified
            if age < timedelta(days=7):
                logger.info(f"Cache hit for ticker: {query_cleaned.upper()}")
                return {
                    "ticker": cached.ticker,
                    "name": cached.company_name,
                    "exchange": cached.exchange,
                    "country": cached.country,
                    "currency": cached.currency,
                    "sector": cached.sector,
                    "industry": cached.industry
                }
            else:
                logger.info(f"Cache entry for {query_cleaned.upper()} is stale. Refreshing...")

        # 2. Search Yahoo Search API
        matches = self.search_query(query_cleaned)
        if not matches:
            # If no matches, check if we can directly validate the ticker using yfinance format
            # e.g. manual entry validation
            if len(query_cleaned) <= 10 and query_cleaned.isalpha():
                logger.info(f"No direct search matches, attempting validation for manual entry: {query_cleaned.upper()}")
                currency = None
                try:
                    import yfinance as yf
                    ticker_obj = yf.Ticker(query_cleaned.upper())
                    if hasattr(ticker_obj, 'fast_info'):
                        currency = ticker_obj.fast_info.get('currency')
                    if not currency:
                        currency = ticker_obj.info.get('currency')
                    if currency:
                        currency = currency.upper()
                except Exception:
                    pass
                return {
                    "ticker": query_cleaned.upper(),
                    "name": query_cleaned.upper(),
                    "exchange": "Manual",
                    "country": "Unknown",
                    "currency": currency or "USD",
                    "sector": "N/A",
                    "industry": "N/A"
                }
            return None
            
        # Select best match (first match that is equity)
        best_match = matches[0]
        
        # Dynamically fetch correct currency from yfinance to avoid defaulting to search API's fallback
        try:
            import yfinance as yf
            ticker_obj = yf.Ticker(best_match["ticker"])
            currency = None
            if hasattr(ticker_obj, 'fast_info'):
                currency = ticker_obj.fast_info.get('currency')
            if not currency:
                currency = ticker_obj.info.get('currency')
            if currency:
                best_match["currency"] = currency.upper()
        except Exception:
            pass
            
        # 3. Cache the resolved company
        self.db.cache_company(best_match["ticker"], best_match)
        
        return best_match
