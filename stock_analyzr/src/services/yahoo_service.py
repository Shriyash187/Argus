import yfinance as yf
import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime
from services.log_service import get_logger

logger = get_logger()

class YahooService:
    """Handles interaction with Yahoo Finance API for downloading prices and news."""
    def __init__(self):
        pass

    def fetch_price_history(self, symbol: str, start: Optional[str] = None, 
                            end: Optional[str] = None, period: str = "1y", 
                            interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical stock data from Yahoo Finance.
        
        Args:
            symbol: Ticker symbol (e.g. 'AAPL')
            start: Start date YYYY-MM-DD
            end: End date YYYY-MM-DD
            period: E.g. '1y', '5y', 'max'
            interval: E.g. '1d', '1wk'
            
        Returns:
            Pandas DataFrame with standard Date, Open, High, Low, Close, Volume columns
        """
        start_time = datetime.now()
        logger.info(f"Downloading historical data from Yahoo Finance for: {symbol}")
        
        try:
            df = yf.download(symbol, start=start, end=end, period=period, 
                            interval=interval, progress=False)
            
            if df.empty:
                logger.warning(f"Yahoo Finance returned empty dataframe for {symbol}")
                return pd.DataFrame()
            
            # Flatten multi-level columns if they exist (yfinance 0.2.x MultiIndex format)
            if isinstance(df.columns, pd.MultiIndex):
                if len(df.columns.levels[1]) <= 1 or symbol in df.columns.levels[1]:
                    df.columns = [col[0] for col in df.columns]
                else:
                    df.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in df.columns]
            
            # Standardize index and columns
            df = df.rename_axis('Date').reset_index()
            
            # Validate required columns
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                logger.error(f"Downloaded dataframe for {symbol} is missing columns: {missing}")
                return pd.DataFrame()
                
            # Clean data types
            df['Date'] = pd.to_datetime(df['Date'])
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Drop rows with critical null fields (Open, Close)
            df = df.dropna(subset=['Open', 'Close'])
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Successfully downloaded {len(df)} price history rows for {symbol} in {elapsed:.2f}s")
            return df[required_cols]
            
        except Exception as e:
            logger.error(f"Exception during Yahoo Finance download for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_ticker_news(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Fetch news articles for a ticker symbol using Yahoo Finance.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            List of parsed article dictionaries
        """
        logger.info(f"Fetching news for {symbol} from Yahoo Finance...")
        try:
            ticker_obj = yf.Ticker(symbol)
            raw_news = ticker_obj.news
            if not raw_news:
                logger.info(f"No news returned for ticker {symbol} from Yahoo Finance.")
                return []
                
            parsed_articles = []
            for item in raw_news:
                content = item.get("content", {})
                
                title = content.get("title", item.get("title", "No Title"))
                summary = content.get("summary", item.get("summary", content.get("description", item.get("description", ""))))
                if not summary:
                    summary = title
                
                publisher = item.get("publisher", "Yahoo Finance")
                if isinstance(publisher, dict):
                    publisher = publisher.get("displayName", "Yahoo Finance")
                elif "provider" in item and isinstance(item["provider"], dict):
                    publisher = item["provider"].get("displayName", "Yahoo Finance")
                
                link = ""
                if "canonicalUrl" in content and isinstance(content["canonicalUrl"], dict):
                    link = content["canonicalUrl"].get("url", "")
                if not link:
                    link = item.get("link", "")
                
                pub_time = datetime.utcnow()
                if "pubDate" in content:
                    try:
                        pub_time = datetime.strptime(content["pubDate"][:19], "%Y-%m-%dT%H:%M:%S")
                    except ValueError:
                        pass
                elif 'providerPublishTime' in item:
                    pub_time = datetime.utcfromtimestamp(item['providerPublishTime'])
                
                parsed_articles.append({
                    "headline": title,
                    "summary": summary,
                    "source": publisher,
                    "url": link,
                    "published_date": pub_time,
                    "ticker": symbol
                })
                
            logger.info(f"Successfully retrieved {len(parsed_articles)} news articles for {symbol}")
            return parsed_articles
        except Exception as e:
            logger.error(f"Error fetching news from Yahoo Finance for {symbol}: {e}")
            return []
