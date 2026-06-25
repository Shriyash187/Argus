import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
from services.log_service import get_logger

logger = get_logger()

class FeatureService:
    """Calculates, normalizes, and stores technical indicators and consolidated features in the database."""
    def __init__(self, db_service):
        self.db = db_service

    def generate_technical_features(self, ticker: str, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes all technical features from price history and persists them.
        
        Args:
            ticker: Stock symbol
            price_df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicator columns
        """
        if price_df.empty or len(price_df) < 26:
            logger.warning(f"Insufficient historical data ({len(price_df)} rows) to construct features for {ticker}.")
            return pd.DataFrame()
            
        logger.info(f"Computing technical indicators for {ticker}...")
        df = price_df.copy().sort_values('Date').reset_index(drop=True)
        
        # 1. Daily and Log Returns
        df['Daily_Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. Moving Average and EMA (20 days)
        df['Moving_Avg_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # 3. Relative Strength Index (RSI - 14 days)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 4. MACD (Moving Average Convergence Divergence)
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # 5. Bollinger Bands (20 days, 2 stddev)
        std_20 = df['Close'].rolling(window=20).std()
        df['Bollinger_Upper'] = df['Moving_Avg_20'] + (std_20 * 2)
        df['Bollinger_Lower'] = df['Moving_Avg_20'] - (std_20 * 2)
        
        # 6. Volatility (rolling standard deviation of daily return)
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
        
        # 7. Volume Change
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Fill first rows where NaN exists due to rolling windows with sensible defaults
        df = df.bfill()
        
        # Save to database
        self.db.save_technical_features(ticker, df)
        
        return df

    def update_feature_store(self, ticker: str, tech_df: pd.DataFrame):
        """
        Consolidates technical indicators and sentiment averages for each date 
        and stores them as a JSON payload inside the feature_store table.
        
        Args:
            ticker: Stock symbol
            tech_df: DataFrame with technical indicator columns
        """
        logger.info(f"Consolidating feature store records for {ticker}...")
        session = self.db.SessionLocal()
        
        try:
            from services.db_service import FeatureStore, NewsArticle
            from sqlalchemy import func
            
            # Delete old feature store entries for this ticker using ORM
            session.query(FeatureStore).filter(FeatureStore.ticker == ticker).delete()
            
            # Fetch latest news sentiment aggregated by date using ORM
            news_res = session.query(
                func.date(NewsArticle.published_date).label('pub_date'),
                func.avg(NewsArticle.sentiment_score).label('avg_sentiment'),
                func.count(NewsArticle.id).label('article_count')
            ).filter(NewsArticle.ticker == ticker).group_by(func.date(NewsArticle.published_date)).all()
            
            sentiment_map = {row[0]: (row[1], row[2]) for row in news_res}
            
            # Bulk prepare FeatureStore objects
            records = []
            
            for idx, row in tech_df.iterrows():
                date_val = row['Date']
                if hasattr(date_val, 'date'):
                    date_val = date_val.date()
                elif isinstance(date_val, str):
                    date_val = datetime.strptime(date_val, "%Y-%m-%d").date()
                    
                date_str = date_val.strftime("%Y-%m-%d")
                
                # Fetch sentiment indicators for this date
                avg_sent, count = sentiment_map.get(date_str, (0.0, 0))
                
                feature_payload = {
                    "close": float(row['Close']),
                    "daily_return": float(row['Daily_Return']),
                    "log_return": float(row['Log_Return']),
                    "moving_avg_20": float(row['Moving_Avg_20']),
                    "ema_20": float(row['EMA_20']),
                    "rsi": float(row['RSI']),
                    "macd": float(row['MACD']),
                    "macd_signal": float(row['MACD_Signal']),
                    "macd_hist": float(row['MACD_Hist']),
                    "bollinger_upper": float(row['Bollinger_Upper']),
                    "bollinger_lower": float(row['Bollinger_Lower']),
                    "volatility": float(row['Volatility']),
                    "volume_change": float(row['Volume_Change']),
                    "sentiment_score": float(avg_sent),
                    "sentiment_count": int(count)
                }
                
                records.append(FeatureStore(
                    ticker=ticker,
                    date=date_val,
                    features_json=json.dumps(feature_payload)
                ))
                
            session.bulk_save_objects(records)
            session.commit()
            logger.info(f"Successfully compiled {len(records)} feature store items for {ticker}.")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error compiling feature store: {e}")
        finally:
            session.close()
