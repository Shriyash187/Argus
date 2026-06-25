import re
import urllib.request
import urllib.parse
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from services.log_service import get_logger

logger = get_logger()

# Lexicons for basic financial sentiment estimation
POSITIVE_WORDS = {
    'beat', 'exceed', 'surge', 'growth', 'growth', 'gain', 'profit', 'rise', 'rally',
    'bullish', 'dividend', 'launch', 'unveils', 'record', 'highest', 'upgrade', 'buy',
    'positive', 'outperform', 'success', 'expand', 'win', 'acquisition', 'merger'
}

NEGATIVE_WORDS = {
    'miss', 'drop', 'decline', 'loss', 'fall', 'slump', 'bearish', 'layoffs', 'cut',
    'crash', 'investigation', 'lawsuit', 'fine', 'fraud', 'arrest', 'resignation',
    'death', 'regulatory', 'negative', 'underperform', 'failure', 'debt', 'risk', 'warning'
}

# Event keyword definitions for classification
EVENT_PATTERNS = {
    "CEO Appointment": [r"\bnew ceo\b", r"appoints? ceo", r"chief executive officer\b.*appoint"],
    "CEO Resignation": [r"ceo resigns?", r"ceo step.*down", r"resignation of ceo"],
    "CEO Death": [r"ceo dies\b", r"death of ceo", r"ceo passed away"],
    "CEO Arrest": [r"ceo arrested", r"arrest of ceo"],
    "Fraud": [r"fraud\b", r"scam\b", r"insider trading", r"embezzlement", r"sec charges"],
    "Acquisition": [r"acquires?\b", r"acquisition\b", r"bought out\b", r"takeover\b"],
    "Merger": [r"merger\b", r"merges?\b", r"amalgamation\b"],
    "Layoffs": [r"layoffs?\b", r"cuts? jobs?\b", r"downsizing\b", r"lay off\b"],
    "Dividend": [r"dividends?\b", r"payout\b", r"declare.*dividend"],
    "Stock Split": [r"stock split\b", r"share split\b"],
    "Patent": [r"patents?\b", r"grants? patent"],
    "Regulatory Action": [r"fined\b", r"regulatory action", r"investigated by sec", r"sanctioned?\b"],
    "Product Launch": [r"unveils?\b", r"launches?\b", r"introduces?\b", r"new product\b"],
    "Quarterly Earnings": [r"quarterly earnings", r"earnings report", r"reports q[1-4]", r"earnings beat", r"earnings miss"]
}

class NewsService:
    """Collects news, calculates sentiment scores, classifies corporate events, and saves results to the database."""
    def __init__(self, db_service, yahoo_service, news_api_key: Optional[str] = None):
        self.db = db_service
        self.yahoo = yahoo_service
        self.news_api_key = news_api_key

    def fetch_and_process_news(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Orchestrate news retrieval, run sentiment analysis/event classification, and store results.
        
        Args:
            ticker: Ticker symbol to query news for
            
        Returns:
            List of parsed and analyzed news articles
        """
        logger.info(f"Initiating news fetching pipeline for ticker: {ticker}")
        
        # 1. Fetch news from primary source (Yahoo Finance)
        articles = self.yahoo.fetch_ticker_news(ticker)
        
        # 2. If NewsAPI key is provided, fetch additional news and merge
        if self.news_api_key and self.news_api_key != "YOUR_KEY_HERE":
            logger.info("NewsAPI key detected. Querying NewsAPI for additional coverage...")
            newsapi_articles = self._fetch_newsapi(ticker)
            # Simple merge: combine and check for title duplicates
            existing_headlines = {a['headline'].lower() for a in articles}
            for art in newsapi_articles:
                if art['headline'].lower() not in existing_headlines:
                    articles.append(art)
        
        if not articles:
            logger.warning(f"No news articles recovered from any source for {ticker}")
            return []

        # 3. Analyze each article
        analyzed_articles = []
        detected_events = []
        
        for art in articles:
            text = f"{art['headline']} {art['summary']}"
            
            # Sentiment estimation
            score, confidence = self.analyze_sentiment(text)
            art['sentiment_score'] = score
            art['sentiment_confidence'] = confidence
            
            analyzed_articles.append(art)
            
            # Event detection
            event = self.classify_event(ticker, art['headline'], art['summary'], art.get('published_date', datetime.utcnow()))
            if event:
                detected_events.append(event)

        # 4. Save to Database
        self.db.save_news_articles(ticker, analyzed_articles)
        if detected_events:
            self.db.save_events(ticker, detected_events)
            
        return analyzed_articles

    def analyze_sentiment(self, text: str) -> Tuple[float, float]:
        """
        Lexicon-based financial sentiment analyzer.
        Calculates a score between -1.0 (extremely negative) and 1.0 (extremely positive).
        
        Args:
            text: Headline + Summary text
            
        Returns:
            Tuple of (sentiment_score, confidence_score)
        """
        words = re.findall(r'\b[a-z]+\b', text.lower())
        pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0, 1.0  # Neutral, high confidence
            
        score = (pos_count - neg_count) / total
        
        # Confidence calculation based on volume of sentiment indicators
        confidence = min(1.0, 0.5 + (total * 0.1))
        
        return round(score, 2), round(confidence, 2)

    def classify_event(self, ticker: str, headline: str, summary: str, pub_date: datetime) -> Optional[Dict[str, Any]]:
        """
        Classifies corporate events using regex keyword patterns and assigns confidence and severity.
        
        Args:
            ticker: Ticker symbol
            headline: Article headline
            summary: Article summary
            pub_date: Publish date
            
        Returns:
            Event dictionary or None
        """
        text = f"{headline} {summary}".lower()
        
        for event_type, patterns in EVENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    # Estimate sentiment specifically for the event context
                    score, conf = self.analyze_sentiment(text)
                    sentiment = "Neutral"
                    if score > 0.15:
                        sentiment = "Positive"
                    elif score < -0.15:
                        sentiment = "Negative"
                        
                    # Base severity on event type
                    severity = 0.2
                    if event_type in ["CEO Death", "CEO Arrest", "Fraud"]:
                        severity = 0.95
                    elif event_type in ["CEO Resignation", "Regulatory Action", "Layoffs"]:
                        severity = 0.75
                    elif event_type in ["Acquisition", "Merger", "Quarterly Earnings"]:
                        severity = 0.6
                    elif event_type in ["CEO Appointment", "Product Launch", "Dividend"]:
                        severity = 0.4
                        
                    logger.info(f"Event detected: {event_type} (Severity: {severity}) for ticker {ticker}")
                    
                    return {
                        "ticker": ticker,
                        "event_type": event_type,
                        "sentiment": sentiment,
                        "confidence": conf,
                        "severity": severity,
                        "description": headline,
                        "date": pub_date
                    }
        return None

    def _fetch_newsapi(self, ticker: str) -> List[Dict[str, Any]]:
        """Fetch news from NewsAPI (fallback)."""
        url = f"https://newsapi.org/v2/everything?q={urllib.parse.quote(ticker)}&sortBy=publishedAt&apiKey={self.news_api_key}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                
            articles = []
            for item in data.get("articles", [])[:15]:
                # Parse NewsAPI date format: YYYY-MM-DDTHH:MM:SSZ
                published_str = item.get("publishedAt", "")
                try:
                    pub_date = datetime.strptime(published_str[:19], "%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    pub_date = datetime.utcnow()
                    
                articles.append({
                    "headline": item.get("title", ""),
                    "summary": item.get("description", ""),
                    "source": item.get("source", {}).get("name", "NewsAPI"),
                    "url": item.get("url", ""),
                    "published_date": pub_date,
                    "ticker": ticker
                })
            return articles
        except Exception as e:
            logger.error(f"Error querying NewsAPI: {e}")
            return []
