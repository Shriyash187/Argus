import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stock_analyzr", "src"))
from services.db_service import DatabaseService

def verify_db():
    print("Verifying database schemas and table row counts...")
    db = DatabaseService()
    session = db.SessionLocal()
    
    try:
        from services.db_service import (
            Company, CompanyCache, PriceHistory, TechnicalFeature, 
            NewsArticle, Event, FeatureStore, PortfolioState, 
            Holding, Transaction, ModelRecord
        )
        
        tables = [
            Company, CompanyCache, PriceHistory, TechnicalFeature, 
            NewsArticle, Event, FeatureStore, PortfolioState, 
            Holding, Transaction, ModelRecord
        ]
        
        for table in tables:
            count = session.query(table).count()
            print(f"  Table '{table.__tablename__}': {count} rows")
            
        print("DATABASE SANITY CHECK PASSED.")
    except Exception as e:
        print(f"DATABASE SANITY CHECK FAILED: {e}")
        sys.exit(1)
    finally:
        session.close()

if __name__ == "__main__":
    verify_db()
