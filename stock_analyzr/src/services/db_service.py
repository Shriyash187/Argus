import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Date, ForeignKey, Text, select
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from services.log_service import get_logger, set_db_service

logger = get_logger()

# SQLite database file path
DB_DIR = "data"
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "mide.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

Base = declarative_base()

class Company(Base):
    """Metadata about watched/loaded companies."""
    __tablename__ = 'companies'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    exchange = Column(String)
    country = Column(String)
    currency = Column(String)
    sector = Column(String)
    industry = Column(String)
    
    prices = relationship("PriceHistory", back_populates="company", cascade="all, delete-orphan")

class CompanyCache(Base):
    """Cache of ticker validation and search queries to avoid yfinance rate limits."""
    __tablename__ = 'companies_cache'
    
    ticker = Column(String, primary_key=True, index=True)
    company_name = Column(String, nullable=False)
    exchange = Column(String)
    country = Column(String)
    currency = Column(String)
    sector = Column(String)
    industry = Column(String)
    last_verified = Column(DateTime, default=datetime.utcnow)

class PriceHistory(Base):
    """Historical stock prices (OHLCV)."""
    __tablename__ = 'price_history'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, ForeignKey('companies.ticker', ondelete="CASCADE"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    company = relationship("Company", back_populates="prices")

class TechnicalFeature(Base):
    """Calculated technical features (moving averages, oscillators)."""
    __tablename__ = 'technical_features'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    daily_return = Column(Float)
    log_return = Column(Float)
    moving_avg_20 = Column(Float)
    ema_20 = Column(Float)
    rsi = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_hist = Column(Float)
    bollinger_upper = Column(Float)
    bollinger_lower = Column(Float)
    volatility = Column(Float)
    volume_change = Column(Float)

class NewsArticle(Base):
    """Fetched financial news articles with sentiment classifications."""
    __tablename__ = 'news_articles'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, nullable=False, index=True)
    headline = Column(String, nullable=False)
    summary = Column(Text)
    source = Column(String)
    url = Column(String)
    published_date = Column(DateTime, nullable=False, index=True)
    sentiment_score = Column(Float)  # Continuous sentiment value between -1.0 and 1.0
    sentiment_confidence = Column(Float)  # Classification confidence

class Event(Base):
    """Detected corporate events from news intelligence classification."""
    __tablename__ = 'events'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, nullable=False, index=True)
    event_type = Column(String, nullable=False, index=True)  # Layoffs, Earnings, Regulatory Action, etc.
    sentiment = Column(String, nullable=False)  # Positive, Negative, Neutral
    confidence = Column(Float)
    severity = Column(Float)  # 0.0 (low) to 1.0 (critical impact)
    description = Column(Text)
    date = Column(DateTime, nullable=False, index=True)

class FeatureStore(Base):
    """Consolidated features (technical + sentiment) formatted for future ML pipeline ingestion."""
    __tablename__ = 'feature_store'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    features_json = Column(Text)  # Store serialized feature dictionary

class SystemLog(Base):
    """In-database logs for real-time visualization in Settings/Admin dashboard."""
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    level = Column(String, nullable=False)
    module = Column(String, nullable=False)
    message = Column(Text, nullable=False)


class DatabaseService:
    """Handles connection pooling, schema initialization, and operations on the SQLite database."""
    def __init__(self, database_url: str = DATABASE_URL):
        self.engine = create_engine(database_url, connect_args={"check_same_thread": False})
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.initialize_db()
        # Wire this service to the log handler
        set_db_service(self)
        logger.info("DatabaseService initialized and database-logging handler registered.")

    def initialize_db(self):
        """Creates all database tables if they do not exist."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables initialized successfully.")

    def get_session(self):
        """Yields a database session context."""
        session = self.SessionLocal()
        try:
            return session
        except Exception as e:
            session.close()
            raise e

    def log_to_db(self, level: str, module: str, message: str):
        """Direct insertion for log records. Bypasses session lifecycle to prevent circular dependencies."""
        session = self.SessionLocal()
        try:
            log = SystemLog(
                level=level,
                module=module,
                message=message,
                timestamp=datetime.utcnow()
            )
            session.add(log)
            session.commit()
        except Exception:
            session.rollback()
        finally:
            session.close()

    def get_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch latest system logs from the database."""
        session = self.SessionLocal()
        try:
            logs = session.query(SystemLog).order_by(SystemLog.timestamp.desc()).limit(limit).all()
            return [
                {
                    "timestamp": log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "level": log.level,
                    "module": log.module,
                    "message": log.message
                }
                for log in logs
            ]
        finally:
            session.close()

    def clear_logs(self):
        """Deletes all system logs in the database."""
        session = self.SessionLocal()
        try:
            session.query(SystemLog).delete()
            session.commit()
            logger.info("System logs cleared in database.")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to clear system logs: {e}")
        finally:
            session.close()

    # --- Companies Cache Operations ---
    def get_cached_company(self, ticker: str) -> Optional[CompanyCache]:
        """Retrieve company info from cache."""
        session = self.SessionLocal()
        try:
            return session.query(CompanyCache).filter(CompanyCache.ticker == ticker).first()
        finally:
            session.close()

    def cache_company(self, ticker: str, info: Dict[str, Any]):
        """Save company info to cache."""
        session = self.SessionLocal()
        try:
            cached = session.query(CompanyCache).filter(CompanyCache.ticker == ticker).first()
            if not cached:
                cached = CompanyCache(ticker=ticker)
                session.add(cached)
            
            cached.company_name = info.get("name", "Unknown")
            cached.exchange = info.get("exchange")
            cached.country = info.get("country")
            cached.currency = info.get("currency")
            cached.sector = info.get("sector")
            cached.industry = info.get("industry")
            cached.last_verified = datetime.utcnow()
            
            session.commit()
            logger.info(f"Cached metadata for ticker {ticker}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error caching company: {e}")
        finally:
            session.close()

    # --- Company and Price History Operations ---
    def get_company(self, ticker: str) -> Optional[Company]:
        """Get company details."""
        session = self.SessionLocal()
        try:
            return session.query(Company).filter(Company.ticker == ticker).first()
        finally:
            session.close()

    def add_company(self, ticker: str, info: Dict[str, Any]) -> Company:
        """Create or update a company record."""
        session = self.SessionLocal()
        try:
            company = session.query(Company).filter(Company.ticker == ticker).first()
            if not company:
                company = Company(ticker=ticker)
                session.add(company)
            
            company.name = info.get("name", "Unknown")
            company.exchange = info.get("exchange")
            company.country = info.get("country")
            company.currency = info.get("currency")
            company.sector = info.get("sector")
            company.industry = info.get("industry")
            
            session.commit()
            session.refresh(company)
            logger.info(f"Saved company profile {ticker} to database.")
            return company
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding company: {e}")
            raise e
        finally:
            session.close()

    def list_companies(self) -> List[Company]:
        """Get all watched companies."""
        session = self.SessionLocal()
        try:
            return session.query(Company).all()
        finally:
            session.close()

    def save_price_history(self, ticker: str, df: Any):
        """Save a pandas DataFrame containing OHLCV history into price_history table."""
        session = self.SessionLocal()
        try:
            # First, check if company exists in DB. If not, add it from cache or basic info.
            company = session.query(Company).filter(Company.ticker == ticker).first()
            if not company:
                cached = self.get_cached_company(ticker)
                currency = cached.currency if cached else None
                
                # Dynamically retrieve currency if not present in cache
                if not currency:
                    try:
                        import yfinance as yf
                        ticker_obj = yf.Ticker(ticker)
                        if hasattr(ticker_obj, 'fast_info'):
                            currency = ticker_obj.fast_info.get('currency')
                        if not currency:
                            currency = ticker_obj.info.get('currency')
                        if currency:
                            currency = currency.upper()
                    except Exception:
                        pass
                
                info = {
                    "name": cached.company_name if cached else ticker,
                    "exchange": cached.exchange if cached else None,
                    "country": cached.country if cached else None,
                    "currency": currency,
                    "sector": cached.sector if cached else None,
                    "industry": cached.industry if cached else None,
                }
                self.add_company(ticker, info)
            
            # Clear existing prices for this ticker to avoid duplicates
            session.query(PriceHistory).filter(PriceHistory.ticker == ticker).delete()
            
            # Bulk save new price records
            records = []
            for idx, row in df.iterrows():
                # Resolve date (index or column)
                date_val = idx
                if 'Date' in df.columns:
                    date_val = row['Date']
                if isinstance(date_val, str):
                    date_val = datetime.strptime(date_val, "%Y-%m-%d").date()
                elif hasattr(date_val, 'date'):
                    date_val = date_val.date()
                
                records.append(PriceHistory(
                    ticker=ticker,
                    date=date_val,
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=float(row['Volume'])
                ))
            
            session.bulk_save_objects(records)
            session.commit()
            logger.info(f"Saved {len(records)} price history records for {ticker}.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving price history: {e}")
            raise e
        finally:
            session.close()

    def get_price_history(self, ticker: str) -> List[PriceHistory]:
        """Retrieve price history from database for a ticker."""
        session = self.SessionLocal()
        try:
            return session.query(PriceHistory).filter(PriceHistory.ticker == ticker).order_by(PriceHistory.date.asc()).all()
        finally:
            session.close()

    # --- Technical Features Operations ---
    def save_technical_features(self, ticker: str, df: Any):
        """Save a pandas DataFrame with technical indicators."""
        session = self.SessionLocal()
        try:
            session.query(TechnicalFeature).filter(TechnicalFeature.ticker == ticker).delete()
            
            records = []
            for idx, row in df.iterrows():
                date_val = idx
                if 'Date' in df.columns:
                    date_val = row['Date']
                if isinstance(date_val, str):
                    date_val = datetime.strptime(date_val, "%Y-%m-%d").date()
                elif hasattr(date_val, 'date'):
                    date_val = date_val.date()
                
                records.append(TechnicalFeature(
                    ticker=ticker,
                    date=date_val,
                    daily_return=float(row['Daily_Return']) if 'Daily_Return' in row and not pd_is_nan(row['Daily_Return']) else None,
                    log_return=float(row['Log_Return']) if 'Log_Return' in row and not pd_is_nan(row['Log_Return']) else None,
                    moving_avg_20=float(row['Moving_Avg_20']) if 'Moving_Avg_20' in row and not pd_is_nan(row['Moving_Avg_20']) else None,
                    ema_20=float(row['EMA_20']) if 'EMA_20' in row and not pd_is_nan(row['EMA_20']) else None,
                    rsi=float(row['RSI']) if 'RSI' in row and not pd_is_nan(row['RSI']) else None,
                    macd=float(row['MACD']) if 'MACD' in row and not pd_is_nan(row['MACD']) else None,
                    macd_signal=float(row['MACD_Signal']) if 'MACD_Signal' in row and not pd_is_nan(row['MACD_Signal']) else None,
                    macd_hist=float(row['MACD_Hist']) if 'MACD_Hist' in row and not pd_is_nan(row['MACD_Hist']) else None,
                    bollinger_upper=float(row['Bollinger_Upper']) if 'Bollinger_Upper' in row and not pd_is_nan(row['Bollinger_Upper']) else None,
                    bollinger_lower=float(row['Bollinger_Lower']) if 'Bollinger_Lower' in row and not pd_is_nan(row['Bollinger_Lower']) else None,
                    volatility=float(row['Volatility']) if 'Volatility' in row and not pd_is_nan(row['Volatility']) else None,
                    volume_change=float(row['Volume_Change']) if 'Volume_Change' in row and not pd_is_nan(row['Volume_Change']) else None
                ))
            
            session.bulk_save_objects(records)
            session.commit()
            logger.info(f"Saved {len(records)} technical feature records for {ticker}.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving technical features: {e}")
        finally:
            session.close()

    def get_technical_features(self, ticker: str) -> List[TechnicalFeature]:
        """Fetch technical features for a ticker."""
        session = self.SessionLocal()
        try:
            return session.query(TechnicalFeature).filter(TechnicalFeature.ticker == ticker).order_by(TechnicalFeature.date.asc()).all()
        finally:
            session.close()

    # --- News and Events Operations ---
    def save_news_articles(self, ticker: str, articles: List[Dict[str, Any]]):
        """Save news articles to DB."""
        session = self.SessionLocal()
        try:
            records = []
            for art in articles:
                # Avoid exact headline duplicates for same ticker
                existing = session.query(NewsArticle).filter(
                    NewsArticle.ticker == ticker,
                    NewsArticle.headline == art['headline']
                ).first()
                if existing:
                    continue
                
                records.append(NewsArticle(
                    ticker=ticker,
                    headline=art['headline'],
                    summary=art.get('summary'),
                    source=art.get('source'),
                    url=art.get('url'),
                    published_date=art.get('published_date', datetime.utcnow()),
                    sentiment_score=art.get('sentiment_score', 0.0),
                    sentiment_confidence=art.get('sentiment_confidence', 1.0)
                ))
            
            session.bulk_save_objects(records)
            session.commit()
            logger.info(f"Saved {len(records)} news articles for {ticker}.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving news articles: {e}")
        finally:
            session.close()

    def get_news_articles(self, ticker: str, limit: int = 50) -> List[NewsArticle]:
        """Get latest news for a ticker."""
        session = self.SessionLocal()
        try:
            return session.query(NewsArticle).filter(NewsArticle.ticker == ticker).order_by(NewsArticle.published_date.desc()).limit(limit).all()
        finally:
            session.close()

    def save_events(self, ticker: str, events: List[Dict[str, Any]]):
        """Save corporate events to DB."""
        session = self.SessionLocal()
        try:
            records = []
            for ev in events:
                # Avoid duplicate events
                existing = session.query(Event).filter(
                    Event.ticker == ticker,
                    Event.event_type == ev['event_type'],
                    Event.date == ev['date']
                ).first()
                if existing:
                    continue
                
                records.append(Event(
                    ticker=ticker,
                    event_type=ev['event_type'],
                    sentiment=ev.get('sentiment', 'Neutral'),
                    confidence=ev.get('confidence', 1.0),
                    severity=ev.get('severity', 0.1),
                    description=ev.get('description'),
                    date=ev.get('date', datetime.utcnow())
                ))
            
            session.bulk_save_objects(records)
            session.commit()
            logger.info(f"Saved {len(records)} events for {ticker}.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving events: {e}")
        finally:
            session.close()

    def get_events(self, ticker: str, limit: int = 50) -> List[Event]:
        """Get classified events for a ticker."""
        session = self.SessionLocal()
        try:
            return session.query(Event).filter(Event.ticker == ticker).order_by(Event.date.desc()).limit(limit).all()
        finally:
            session.close()

def pd_is_nan(val) -> bool:
    """Helper to detect pandas nan/numpy nan without import issues."""
    try:
        import math
        return math.isnan(val)
    except Exception:
        return val != val
