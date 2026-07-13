"""
ARGUS Intelligent Market Analytics Platform Streamlit Dashboard
Modernized multi-page production interface with data-service layers.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add current folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.db_service import (
    DatabaseService, Company, CompanyCache, PriceHistory, 
    TechnicalFeature, NewsArticle, Event, FeatureStore, SystemLog
)
from services.ticker_service import TickerSearchService
from services.yahoo_service import YahooService
from services.news_service import NewsService
from services.feature_service import FeatureService
from services.log_service import get_logger
from preprocess import DataPreprocessor
from model import StockLSTMModel
from models.model_selector import ModelSelector
from services.recommendation_service import RecommendationService
from services.advisor_service import InvestmentAdvisorService

logger = get_logger()

# ----------------- SESSION STATE & SETUP -----------------
st.set_page_config(
    page_title="ARGUS Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Refined Dark Grayscale theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif !important;
        background-color: #111827 !important;
        color: #F3F4F6 !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #1B2430 !important;
        border-right: 1px solid #374151 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: #F3F4F6 !important;
    }
    
    .main-header {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #F3F4F6 !important;
        margin-bottom: 0.25rem !important;
        margin-top: 0 !important;
        letter-spacing: -0.02em !important;
    }
    .sub-header {
        font-size: 0.85rem !important;
        color: #9CA3AF !important;
        margin-bottom: 1.5rem !important;
    }
    .card {
        background-color: #202938 !important;
        padding: 1rem !important;
        border-radius: 6px !important;
        border: 1px solid #374151 !important;
        margin-bottom: 0.75rem !important;
        box-shadow: none !important;
    }
    .metric-title {
        font-size: 0.75rem !important;
        color: #9CA3AF !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        margin-bottom: 0.25rem !important;
    }
    .metric-value {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #F3F4F6 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .sentiment-positive {
        color: #22C55E !important;
        font-weight: 600 !important;
    }
    .sentiment-negative {
        color: #EF4444 !important;
        font-weight: 600 !important;
    }
    .sentiment-neutral {
        color: #9CA3AF !important;
        font-weight: 600 !important;
    }
    .badge {
        padding: 0.15rem 0.4rem !important;
        border-radius: 3px !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        display: inline-block !important;
    }
    .badge-critical {
        background-color: rgba(239, 68, 68, 0.1) !important;
        color: #EF4444 !important;
        border: 1px solid rgba(239, 68, 68, 0.2) !important;
    }
    .badge-moderate {
        background-color: rgba(245, 158, 11, 0.1) !important;
        color: #F59E0B !important;
        border: 1px solid rgba(245, 158, 11, 0.2) !important;
    }
    .badge-low {
        background-color: rgba(59, 130, 246, 0.1) !important;
        color: #3B82F6 !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
    }
    
    /* Streamlit button style overrides */
    div.stButton > button {
        background-color: #202938 !important;
        color: #F3F4F6 !important;
        border: 1px solid #374151 !important;
        border-radius: 6px !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 0.4rem 1rem !important;
        transition: all 0.1s ease !important;
    }
    div.stButton > button:hover {
        border-color: #9CA3AF !important;
        color: #F3F4F6 !important;
        background-color: #252F3F !important;
    }
    div.stButton > button:active {
        background-color: #1B2430 !important;
        border-color: #3B82F6 !important;
    }
    
    /* Form inputs styling */
    div.stTextInput > div > div > input, div.stSelectbox > div > div > div {
        background-color: #202938 !important;
        color: #F3F4F6 !important;
        border: 1px solid #374151 !important;
        border-radius: 6px !important;
        font-size: 0.875rem !important;
    }
    div.stTextInput > div > div > input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 1px #3B82F6 !important;
    }

    /* Custom Tables styling inside Streamlit markdown */
    div[data-testid="stTable"] table {
        background-color: #202938 !important;
        color: #F3F4F6 !important;
        border: 1px solid #374151 !important;
        border-collapse: collapse !important;
        font-size: 0.875rem !important;
        width: 100% !important;
        border-radius: 6px !important;
    }
    div[data-testid="stTable"] th {
        background-color: #293445 !important;
        color: #F3F4F6 !important;
        border: 1px solid #374151 !important;
        padding: 8px 12px !important;
        font-weight: 600 !important;
        text-align: left !important;
    }
    div[data-testid="stTable"] td {
        border: 1px solid #374151 !important;
        padding: 8px 12px !important;
    }
    div[data-testid="stTable"] tr:nth-child(even) {
        background-color: #252F3F !important;
    }
    div[data-testid="stTable"] tr:hover {
        background-color: #293445 !important;
    }

    /* Streamlit interactive dataframes custom wrapping */
    [data-testid="stDataFrame"] {
        border: 1px solid #374151 !important;
        background-color: #202938 !important;
    }

    /* Sidebar Radio elements customization */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label p {
        color: #9CA3AF !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover p {
        color: #F3F4F6 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Core Services inside streamlit cache or session
if 'db' not in st.session_state:
    st.session_state.db = DatabaseService()
    st.session_state.yahoo = YahooService()
    st.session_state.ticker_search = TickerSearchService(st.session_state.db)
    st.session_state.news = NewsService(st.session_state.db, st.session_state.yahoo)
    st.session_state.features = FeatureService(st.session_state.db)
    st.session_state.recommendation = RecommendationService(st.session_state.db)
    st.session_state.advisor = InvestmentAdvisorService(st.session_state.db, st.session_state.recommendation)
    st.session_state.model_selector = ModelSelector(st.session_state.db)

db = st.session_state.db
yahoo = st.session_state.yahoo
ticker_search = st.session_state.ticker_search
news_service = st.session_state.news
feature_service = st.session_state.features
rec_service = st.session_state.recommendation
advisor_service = st.session_state.advisor
model_selector_service = st.session_state.model_selector

# Ensure default watched stocks are added if empty
watched_companies = db.list_companies()
if not watched_companies:
    logger.info("Initializing default watched stocks...")
    defaults = {
        "AAPL": {"name": "Apple Inc.", "exchange": "NASDAQ", "country": "USA", "currency": "USD", "sector": "Technology", "industry": "Consumer Electronics"},
        "TSLA": {"name": "Tesla Inc.", "exchange": "NASDAQ", "country": "USA", "currency": "USD", "sector": "Consumer Cyclical", "industry": "Auto Manufacturers"},
        "NVDA": {"name": "NVIDIA Corporation", "exchange": "NASDAQ", "country": "USA", "currency": "USD", "sector": "Technology", "industry": "Semiconductors"}
    }
    for ticker, info in defaults.items():
        db.cache_company(ticker, info)
        db.add_company(ticker, info)
    watched_companies = db.list_companies()

# ----------------- SIDEBAR PAGE SELECTOR -----------------
st.sidebar.markdown("<div style='font-size: 1.15rem; font-weight: 700; color: #f0f6fc; margin-bottom: 0.25rem; letter-spacing: -0.02em;'>ARGUS Terminal</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='color: #8b949e; font-size: 0.65rem; margin-bottom: 1.25rem; text-transform: uppercase; letter-spacing: 0.05em;'>Intelligent Market Analytics Platform</div>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard", 
        "Stock Search", 
        "Stock Overview", 
        "ML Prediction", 
        "Model Registry & Compare", 
        "News Intelligence", 
        "Event Intelligence", 
        "Portfolio & Paper Trading",
        "Recommendation & Backtest",
        "AI Advisor Memo",
        "System Settings"
    ]
)

# Initialize credential session state variables from environment variables if not present
if 'alpha_key' not in st.session_state:
    st.session_state.alpha_key = os.getenv("ALPHA_VANTAGE_API_KEY", "GQ1C7TJRZ4ANOZM9")
if 'newsapi_key' not in st.session_state:
    st.session_state.newsapi_key = os.getenv("NEWS_API_KEY", "")
if 'gemini_key' not in st.session_state:
    st.session_state.gemini_key = os.getenv("GEMINI_API_KEY", "")

# Sync updated keys back to services
st.session_state.news.news_api_key = st.session_state.newsapi_key
st.session_state.features.db = db

# Compact System Status Dashboard Panel in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #F3F4F6; margin-bottom: 0.75rem;'>System Status</div>", unsafe_allow_html=True)

# Yahoo status
st.sidebar.markdown("<div style='font-size: 0.8rem; color: #9CA3AF; margin-bottom: 0.4rem; display: flex; align-items: center;'><span style='color: #22C55E; margin-right: 0.5rem; font-size: 0.9rem;'>●</span> Yahoo Finance Connected</div>", unsafe_allow_html=True)

# Alpha Vantage status
if st.session_state.alpha_key and st.session_state.alpha_key != "YOUR_KEY_HERE":
    st.sidebar.markdown("<div style='font-size: 0.8rem; color: #9CA3AF; margin-bottom: 0.4rem; display: flex; align-items: center;'><span style='color: #22C55E; margin-right: 0.5rem; font-size: 0.9rem;'>●</span> Alpha Vantage Connected</div>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<div style='font-size: 0.8rem; color: #9CA3AF; margin-bottom: 0.4rem; display: flex; align-items: center;'><span style='color: #6B7280; margin-right: 0.5rem; font-size: 0.9rem;'>●</span> Alpha Vantage Not Configured</div>", unsafe_allow_html=True)

# NewsAPI status
if st.session_state.newsapi_key:
    st.sidebar.markdown("<div style='font-size: 0.8rem; color: #9CA3AF; margin-bottom: 0.4rem; display: flex; align-items: center;'><span style='color: #22C55E; margin-right: 0.5rem; font-size: 0.9rem;'>●</span> NewsAPI Connected</div>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<div style='font-size: 0.8rem; color: #9CA3AF; margin-bottom: 0.4rem; display: flex; align-items: center;'><span style='color: #6B7280; margin-right: 0.5rem; font-size: 0.9rem;'>●</span> NewsAPI Not Configured</div>", unsafe_allow_html=True)

# Gemini AI status
if st.session_state.gemini_key:
    st.sidebar.markdown("<div style='font-size: 0.8rem; color: #9CA3AF; margin-bottom: 0.4rem; display: flex; align-items: center;'><span style='color: #22C55E; margin-right: 0.5rem; font-size: 0.9rem;'>●</span> Gemini AI Connected</div>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<div style='font-size: 0.8rem; color: #9CA3AF; margin-bottom: 0.4rem; display: flex; align-items: center;'><span style='color: #6B7280; margin-right: 0.5rem; font-size: 0.9rem;'>●</span> Gemini AI Not Configured</div>", unsafe_allow_html=True)

# Helper functions to fetch and generate indicators
def get_currency_symbol(ticker: str) -> str:
    """Detect currency symbol or ISO code dynamically from company data, cache, or provider."""
    company = db.get_company(ticker)
    currency_code = None
    if company and company.currency:
        currency_code = company.currency.upper()
    else:
        cached = db.get_cached_company(ticker)
        if cached and cached.currency:
            currency_code = cached.currency.upper()
            
    # Try fetching dynamically if not present in DB
    if not currency_code:
        try:
            import yfinance as yf
            ticker_obj = yf.Ticker(ticker)
            if hasattr(ticker_obj, 'fast_info'):
                currency_code = ticker_obj.fast_info.get('currency')
            if not currency_code:
                currency_code = ticker_obj.info.get('currency')
            if currency_code:
                currency_code = currency_code.upper()
                # Update database & cache
                session = db.SessionLocal()
                try:
                    comp = session.query(Company).filter(Company.ticker == ticker).first()
                    if comp:
                        comp.currency = currency_code
                    cached_comp = session.query(CompanyCache).filter(CompanyCache.ticker == ticker).first()
                    if cached_comp:
                        cached_comp.currency = currency_code
                    session.commit()
                except Exception:
                    session.rollback()
                finally:
                    session.close()
        except Exception:
            pass
            
    if not currency_code:
        # Heuristics based on popular exchange suffixes
        upper_ticker = ticker.upper()
        if upper_ticker.endswith('.NS') or upper_ticker.endswith('.BO'):
            currency_code = 'INR'
        elif upper_ticker.endswith('.L'):
            currency_code = 'GBP'
        elif upper_ticker.endswith('.TO') or upper_ticker.endswith('.V'):
            currency_code = 'CAD'
        elif upper_ticker.endswith('.AX'):
            currency_code = 'AUD'
        elif any(upper_ticker.endswith(sfx) for sfx in ['.DE', '.PA', '.MI', '.MC']):
            currency_code = 'EUR'
        else:
            return ""  # Unavailable, return empty so we do not assume USD
            
    # Map ISO currency codes to symbol
    CURRENCY_SYMBOLS = {
        'USD': '$', 'EUR': '€', 'GBP': '£', 'INR': '₹', 'JPY': '¥',
        'AUD': 'A$', 'CAD': 'C$', 'CHF': 'CHF', 'CNY': '¥', 'HKD': 'HK$',
        'NZD': 'NZ$', 'SEK': 'kr', 'KRW': '₩', 'SGD': 'S$', 'NOK': 'kr',
        'MXN': '$', 'RUB': '₽', 'ZAR': 'R', 'TRY': '₺', 'BRL': 'R$',
        'TWD': 'NT$', 'DKK': 'kr', 'PLN': 'zł', 'THB': '฿', 'IDR': 'Rp',
        'HUF': 'Ft', 'CZK': 'Kč', 'ILS': '₪', 'CLP': '$', 'PHP': '₱',
        'AED': 'AED', 'COP': '$', 'SAR': 'SR', 'MYR': 'RM', 'RON': 'lei'
    }
    return CURRENCY_SYMBOLS.get(currency_code, f"{currency_code} ")

def load_stock_data(ticker: str) -> pd.DataFrame:
    """Fetch history, calculate technical features, update feature store, and return price DataFrame."""
    # 1. Download price history
    price_df = yahoo.fetch_price_history(ticker, period="2y")
    if price_df.empty:
        return pd.DataFrame()
        
    # Save raw prices to DB
    db.save_price_history(ticker, price_df)
    
    # 2. Generate technical features and save
    tech_df = feature_service.generate_technical_features(ticker, price_df)
    
    # 3. Fetch latest news and compute sentiment (saves to DB)
    news_service.fetch_and_process_news(ticker)
    
    # 4. Consolidate into Feature Store
    feature_service.update_feature_store(ticker, tech_df)
    
    return tech_df

# ----------------- PAGE 1: DASHBOARD -----------------
if page == "Dashboard":
    st.markdown('<h1 class="main-header">ARGUS Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">System status, watchlists, and algorithmic rankings.</p>', unsafe_allow_html=True)
    
    # 1. Display Overview Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Watched Companies</div>
            <div class="metric-value">{len(watched_companies)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        session = db.SessionLocal()
        logs_count = session.query(SystemLog).count()
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">System Logs</div>
            <div class="metric-value">{logs_count}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        news_count = session.query(NewsArticle).count()
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Articles Analyzed</div>
            <div class="metric-value">{news_count}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        events_count = session.query(Event).count()
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Events Classified</div>
            <div class="metric-value">{events_count}</div>
        </div>
        """, unsafe_allow_html=True)
    session.close()
    
    # 2. Watchlist overview
    st.subheader("Watched Stock Overview")
    watched_tickers = [c.ticker for c in watched_companies]
    
    if watched_tickers:
        overview_data = []
        for ticker in watched_tickers:
            prices = db.get_price_history(ticker)
            if prices:
                latest = prices[-1]
                prev = prices[-2] if len(prices) > 1 else latest
                change = ((latest.close - prev.close) / prev.close) * 100
                overview_data.append({
                    "Ticker": ticker,
                    "Price": f"{get_currency_symbol(ticker)}{latest.close:.2f}",
                    "Change": f"{change:+.2f}%",
                    "Volume": f"{latest.volume:,.0f}"
                })
        if overview_data:
            st.dataframe(pd.DataFrame(overview_data), use_container_width=True)
        else:
            st.info("No pricing records stored. Go to 'Stock Overview' or 'Stock Search' to fetch data.")
            
    # 3. Algorithmic Stock Ranking (Priority Queue Demonstration)
    st.subheader("Priority Queue Performance Ranking")
    if watched_tickers:
        from fetch_data import DataFetcher
        fetcher = DataFetcher()
        for ticker in watched_tickers:
            prices = db.get_price_history(ticker)
            if len(prices) >= 30:
                yahoo_data = pd.DataFrame([{
                    "Close": p.close,
                    "Volume": p.volume
                } for p in prices])
                yahoo_data['Symbol'] = ticker
                fetcher.data_cache[ticker] = yahoo_data
                
        rankings = fetcher.rank_stocks_by_performance(watched_tickers, lookback_days=30)
        
        if rankings:
            st.markdown("Stocks ranked by performance over last **30 days**:")
            rank_df = pd.DataFrame(rankings, columns=["Ticker", "Return (30-day)"])
            rank_df["Return (30-day)"] = rank_df["Return (30-day)"].map(lambda x: f"{x*100:+.2f}%")
            st.dataframe(rank_df, use_container_width=True)
        else:
            st.info("Insufficient historical prices to compute 30-day returns. Load more stock price history first.")
            
# ----------------- PAGE 2: STOCK SEARCH -----------------
elif page == "Stock Search":
    st.markdown('<h1 class="main-header">Universal Stock Search</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Search, validate, and add assets to your watched database.</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        search_term = st.text_input("Enter Company Name or Ticker Symbol", placeholder="e.g. Reliance, Tesla, Apple, Infosys, Ola Electric")
        search_clicked = st.button("Search Assets")
        
        if search_clicked and search_term:
            with st.spinner("Searching and validating ticker..."):
                resolved = ticker_search.resolve_ticker(search_term)
                
                if resolved:
                    st.success(f"Successfully resolved ticker: **{resolved['ticker']}**")
                    st.session_state.resolved_ticker_data = resolved
                else:
                    st.error("Could not resolve company name. Please try another term or use manual entry.")
        
        # Display resolved ticker card
        if 'resolved_ticker_data' in st.session_state:
            res = st.session_state.resolved_ticker_data
            st.markdown(f"""
            <div class="card">
                <div style="font-size: 1.1rem; font-weight: 600; color: #f0f6fc; margin-bottom: 0.75rem;">{res['name']} ({res['ticker']})</div>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; font-size: 0.85rem; color: #8b949e;">
                    <div>Exchange: <span style="color: #f0f6fc;">{res['exchange']}</span></div>
                    <div>Country: <span style="color: #f0f6fc;">{res['country']}</span></div>
                    <div>Currency: <span style="color: #f0f6fc;">{res['currency']}</span></div>
                    <div>Sector: <span style="color: #f0f6fc;">{res['sector']}</span></div>
                    <div style="grid-column: span 2;">Industry: <span style="color: #f0f6fc;">{res['industry']}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Action button to add to watchlist
            if st.button("+ Add to Watchlist"):
                db.add_company(res['ticker'], res)
                st.success(f"Added {res['name']} ({res['ticker']}) to database watch list!")
                st.rerun()

    with col2:
        st.subheader("Manual Fallback Entry")
        manual_ticker = st.text_input("Enter exact ticker symbol directly", placeholder="AAPL or RELIANCE.NS").upper()
        manual_name = st.text_input("Company Name fallback", placeholder="e.g. Apple Inc.")
        manual_exch = st.selectbox("Exchange", ["NASDAQ", "NYSE", "NSE", "BSE", "Other"])
        
        if st.button("Add Ticker Manually"):
            if manual_ticker:
                # Query yfinance to get actual currency for manual entry
                detected_currency = None
                try:
                    import yfinance as yf
                    ticker_obj = yf.Ticker(manual_ticker)
                    if hasattr(ticker_obj, 'fast_info'):
                        detected_currency = ticker_obj.fast_info.get('currency')
                    if not detected_currency:
                        detected_currency = ticker_obj.info.get('currency')
                except Exception:
                    pass
                if not detected_currency:
                    detected_currency = "USD" if manual_exch in ["NASDAQ", "NYSE"] else "INR"
                else:
                    detected_currency = detected_currency.upper()
                
                info = {
                    "name": manual_name or manual_ticker,
                    "exchange": manual_exch,
                    "country": "Unknown",
                    "currency": detected_currency,
                    "sector": "N/A",
                    "industry": "N/A"
                }
                db.cache_company(manual_ticker, info)
                db.add_company(manual_ticker, info)
                st.success(f"Manually registered ticker: {manual_ticker}")
                st.rerun()

# ----------------- PAGE 3: STOCK OVERVIEW -----------------
elif page == "Stock Overview":
    st.markdown('<h1 class="main-header">Technical Overview</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Detailed charts, indicators, and support/resistance boundaries.</p>', unsafe_allow_html=True)
    
    ticker_list = [c.ticker for c in watched_companies]
    selected_ticker = st.selectbox("Select Asset to Load", ticker_list)
    
    if selected_ticker:
        with st.spinner(f"Ingesting pricing & generating indicators for {selected_ticker}..."):
            df = load_stock_data(selected_ticker)
            
        if not df.empty:
            # Stats Cards
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
            
            col1, col2, col3 = st.columns(3)
            sym = get_currency_symbol(selected_ticker)
            with col1:
                st.metric("Latest Close", f"{sym}{latest['Close']:.2f}", f"{change:+.2f}%")
            with col2:
                st.metric("RSI (14)", f"{latest['RSI']:.2f}" if 'RSI' in df.columns else "N/A")
            with col3:
                st.metric("EMA (20)", f"{sym}{latest['EMA_20']:.2f}" if 'EMA_20' in df.columns else "N/A")
                
            # Create Plotly Chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.75, 0.25])
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                name="OHLC",
                increasing_line_color='#22C55E', decreasing_line_color='#EF4444',
                increasing_fillcolor='#22C55E', decreasing_fillcolor='#EF4444'
            ), row=1, col=1)
            
            # EMA
            if 'EMA_20' in df.columns:
                fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_20'], name='EMA 20', line=dict(color='#3B82F6', width=1.5)), row=1, col=1)
            
            # Bollinger Bands
            if 'Bollinger_Upper' in df.columns:
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Bollinger_Upper'], name='BB Upper', line=dict(color='#6B7280', width=1, dash='dot')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Bollinger_Lower'], name='BB Lower', line=dict(color='#6B7280', width=1, dash='dot')), row=1, col=1)
                
            # Volume styled with positive/negative color mapping
            vol_colors = ['rgba(34, 197, 94, 0.4)' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'rgba(239, 68, 68, 0.4)' for i in range(len(df))]
            fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name="Volume", marker_color=vol_colors), row=2, col=1)
            
            # Support/Resistance levels styled as clean horizontal dash lines
            from ds_helpers import detect_support_resistance_levels
            supports, resistances = detect_support_resistance_levels(df['Close'].tolist())
            levels = [float(l) for l in (supports + resistances) if pd.notna(l)]
            for idx, lvl in enumerate(levels[:4]):
                is_support = idx % 2 == 0
                fig.add_hline(y=lvl, line_dash="dash", line_color="#22C55E" if is_support else "#EF4444", 
                             line_width=1, annotation_text=f"S{idx//2+1}" if is_support else f"R{idx//2+1}", 
                             annotation_position="top right", row=1, col=1)
                
            fig.update_layout(
                plot_bgcolor='#111827',
                paper_bgcolor='#111827',
                font=dict(family='Inter, sans-serif', color='#9CA3AF', size=11),
                xaxis=dict(showgrid=True, gridcolor='#374151', linecolor='#374151', rangeslider_visible=False),
                yaxis=dict(showgrid=True, gridcolor='#374151', linecolor='#374151', side='right'),
                xaxis2=dict(showgrid=True, gridcolor='#374151', linecolor='#374151'),
                yaxis2=dict(showgrid=True, gridcolor='#374151', linecolor='#374151', side='right'),
                margin=dict(l=10, r=10, t=10, b=10),
                height=550,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Failed to load price history data for {selected_ticker}.")

# ----------------- PAGE 4: ML PREDICTION -----------------
elif page == "ML Prediction":
    st.markdown('<h1 class="main-header">Predictive Modeling</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Train the LSTM model and validate predictions using walk-forward cross-validation.</p>', unsafe_allow_html=True)
    
    ticker_list = [c.ticker for c in watched_companies]
    selected_ticker = st.selectbox("Select Asset to Train On", ticker_list)
    
    if selected_ticker:
        # Load features
        prices = db.get_price_history(selected_ticker)
        if not prices:
            st.info("Pricing data must be downloaded first. Navigate to Stock Overview page.")
        else:
            df = pd.DataFrame([{
                "Date": p.date,
                "Open": p.open,
                "High": p.high,
                "Low": p.low,
                "Close": p.close,
                "Volume": p.volume
            } for p in prices])
            
            st.success(f"Loaded {len(df)} price history records.")
            
            col1, col2 = st.columns(2)
            with col1:
                epochs = st.slider("Training Epochs", 5, 50, 15)
                batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
                seq_len = st.slider("Sequence Lookback (days)", 10, 60, 30)
                
                train_clicked = st.button("Train LSTM Model")
                
            with col2:
                st.subheader("Model Overview")
                st.write("Constructs a 2-layer LSTM recurrent neural network with Dropout regularizers, trained using early stopping callbacks.")
                
            if train_clicked:
                # Run complete preprocessing, model building, and training pipeline
                with st.spinner("Preparing features and sequences..."):
                    preprocessor = DataPreprocessor(scaler_type='minmax')
                    processed_data = preprocessor.prepare_full_pipeline(
                        df, 
                        sequence_length=seq_len,
                        train_ratio=0.7,
                        val_ratio=0.15,
                        use_queue=True,
                        target_col='Close'
                    )
                    
                with st.spinner("Training LSTM model on CPU..."):
                    model = StockLSTMModel(input_shape=processed_data['X_train'].shape[1:])
                    model.build_model()
                    history = model.train(
                        processed_data['X_train'], processed_data['y_train'],
                        X_val=processed_data['X_val'], y_val=processed_data['y_val'],
                        epochs=epochs, batch_size=batch_size, verbose=0
                    )
                    st.success("Model trained successfully!")
                    
                # Evaluate and Plot
                test_pred = model.predict(processed_data['X_test'])
                test_pred_orig = preprocessor.inverse_transform_targets(test_pred)
                y_test_orig = processed_data['y_test_original']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=y_test_orig, name="Actual Price", line=dict(color="#3B82F6", width=1.5)))
                fig.add_trace(go.Scatter(y=test_pred_orig.flatten(), name="LSTM Pred", line=dict(color="#EF4444", width=1.5, dash="dash")))
                fig.update_layout(
                    plot_bgcolor='#111827',
                    paper_bgcolor='#111827',
                    font=dict(family='Inter, sans-serif', color='#9CA3AF', size=11),
                    xaxis=dict(showgrid=True, gridcolor='#374151', linecolor='#374151'),
                    yaxis=dict(showgrid=True, gridcolor='#374151', linecolor='#374151'),
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=350,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Backtesting validation
                st.subheader("Walk-Forward Validation")
                with st.spinner("Running 8-Window Backtest..."):
                    from backtest import WalkForwardBacktester
                    backtester = WalkForwardBacktester(model, preprocessor)
                    
                    # Convert pricing df format for backtester
                    backtest_results = backtester.walk_forward_validation(
                        df, train_window=252, test_window=21, sequence_length=seq_len
                    )
                    
                    if backtest_results:
                        metrics = backtest_results['overall_metrics']
                        col_a, col_b, col_c, col_d = st.columns(4)
                        col_a.metric("RMSE", f"{metrics['rmse']:.4f}")
                        col_b.metric("MAPE", f"{metrics['mape']*100:.2f}%")
                        col_c.metric("Directional Accuracy", f"{metrics['directional_accuracy']*100:.1f}%")
                        col_d.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0.0):.2f}")
                        
                        # Graph backtest predictions
                        fig_bt = go.Figure()
                        fig_bt.add_trace(go.Scatter(y=backtest_results['actuals'], name="Actual Price", line=dict(color="#3B82F6", width=1.5)))
                        fig_bt.add_trace(go.Scatter(y=backtest_results['predictions'], name="Backtest Pred", line=dict(color="#22C55E", width=1.5, dash="dot")))
                        fig_bt.update_layout(
                            plot_bgcolor='#111827',
                            paper_bgcolor='#111827',
                            font=dict(family='Inter, sans-serif', color='#9CA3AF', size=11),
                            xaxis=dict(showgrid=True, gridcolor='#374151', linecolor='#374151'),
                            yaxis=dict(showgrid=True, gridcolor='#374151', linecolor='#374151'),
                            margin=dict(l=10, r=10, t=10, b=10),
                            height=350,
                            showlegend=True
                        )
                        st.plotly_chart(fig_bt, use_container_width=True)
                    else:
                        st.error("Failed to calculate backtesting metrics. Make sure you have enough historical periods.")

# ----------------- PAGE 5: NEWS INTELLIGENCE -----------------
elif page == "News Intelligence":
    st.markdown('<h1 class="main-header">News Sentiment Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Financial news scraping and semantic sentiment indexing.</p>', unsafe_allow_html=True)
    
    ticker_list = [c.ticker for c in watched_companies]
    selected_ticker = st.selectbox("Select Ticker for News coverage", ticker_list)
    
    if selected_ticker:
        with st.spinner("Querying active coverage..."):
            articles = db.get_news_articles(selected_ticker)
            
        if not articles:
            st.info("No cached news articles. Click 'Stock Overview' or query search first to collect news.")
        else:
            # Compute distribution of sentiment
            sentiments = [a.sentiment_score for a in articles]
            pos = sum(1 for s in sentiments if s > 0.15)
            neg = sum(1 for s in sentiments if s < -0.15)
            neu = len(sentiments) - pos - neg
            
            fig = px.pie(
                values=[pos, neg, neu], 
                names=["Positive", "Negative", "Neutral"],
                color_discrete_sequence=["#22C55E", "#EF4444", "#374151"],
                hole=0.4
            )
            fig.update_layout(
                plot_bgcolor='#111827',
                paper_bgcolor='#111827',
                font=dict(family='Inter, sans-serif', color='#9CA3AF', size=11),
                margin=dict(l=10, r=10, t=10, b=10),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display articles list
            st.subheader("Recent Articles Feed")
            for art in articles:
                pub_date = art.published_date.strftime('%Y-%m-%d %H:%M')
                badge_class = "sentiment-neutral"
                badge_lbl = "Neutral"
                if art.sentiment_score > 0.15:
                    badge_class = "sentiment-positive"
                    badge_lbl = f"Positive ({art.sentiment_score:+.2f})"
                elif art.sentiment_score < -0.15:
                    badge_class = "sentiment-negative"
                    badge_lbl = f"Negative ({art.sentiment_score:.2f})"
                    
                st.markdown(f"""
                <div class="card">
                    <div style="font-size: 1.05rem; font-weight: 600; margin-bottom: 0.25rem;"><a href="{art.url}" target="_blank" style="color: #58a6ff; text-decoration: none;">{art.headline}</a></div>
                    <div style="color: #8b949e; font-size: 0.8rem; margin-bottom: 0.6rem;">Source: {art.source} &middot; Date: {pub_date}</div>
                    <p style="font-size: 0.875rem; color: #c9d1d9; line-height: 1.4; margin-bottom: 0.6rem;">{art.summary}</p>
                    <div style="font-size: 0.8rem; color: #8b949e;">Sentiment polarity: <span class="{badge_class}">{badge_lbl}</span> &middot; Confidence: {int(art.sentiment_confidence*100)}%</div>
                </div>
                """, unsafe_allow_html=True)

# ----------------- PAGE 6: EVENT INTELLIGENCE -----------------
elif page == "Event Intelligence":
    st.markdown('<h1 class="main-header">Event Intelligence Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time classification of corporate milestones and regulatory warnings.</p>', unsafe_allow_html=True)
    
    ticker_list = [c.ticker for c in watched_companies]
    selected_ticker = st.selectbox("Select Asset Events", ticker_list)
    
    if selected_ticker:
        events = db.get_events(selected_ticker)
        
        if not events:
            st.info("No corporate events classified for this ticker yet. Generating new coverage will index them.")
        else:
            for ev in events:
                event_date = ev.date.strftime('%Y-%m-%d %H:%M:%S')
                severity_val = f"{ev.severity:.2f}"
                sev_class = "badge-low"
                if ev.severity > 0.8:
                    sev_class = "badge-critical"
                elif ev.severity > 0.5:
                    sev_class = "badge-moderate"
                    
                sent_class = "sentiment-neutral"
                if ev.sentiment == "Positive":
                    sent_class = "sentiment-positive"
                elif ev.sentiment == "Negative":
                    sent_class = "sentiment-negative"
                    
                st.markdown(f"""
                <div class="card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                        <div style="font-size: 1.05rem; font-weight: 600; color: #f0f6fc;">{ev.event_type}</div>
                        <span class="badge {sev_class}">Severity: {severity_val}</span>
                    </div>
                    <div style="color: #8b949e; font-size: 0.8rem; margin-bottom: 0.6rem;">Date Indexed: {event_date}</div>
                    <p style="font-size: 0.875rem; color: #c9d1d9; line-height: 1.4; margin-bottom: 0.6rem;">{ev.description}</p>
                    <div style="font-size: 0.8rem; color: #8b949e;">Sentiment Impact: <span class="{sent_class}">{ev.sentiment}</span> &middot; Confidence: {int(ev.confidence*100)}%</div>
                </div>
                """, unsafe_allow_html=True)

# ----------------- PAGE: MODEL REGISTRY & COMPARE -----------------
elif page == "Model Registry & Compare":
    st.markdown('<h1 class="main-header">Model Registry & Performance Comparison</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Train all supported regressors on matching features, compare test metrics, and load versioned models from the registry.</p>', unsafe_allow_html=True)
    
    ticker_list = [c.ticker for c in watched_companies]
    selected_ticker = st.selectbox("Select Stock Symbol", ticker_list)
    
    if selected_ticker:
        # Load history
        prices = db.get_price_history(selected_ticker)
        if not prices:
            st.info("Pricing data must be downloaded first. Navigate to Stock Overview page.")
        else:
            df = pd.DataFrame([{
                "Date": p.date,
                "Open": p.open,
                "High": p.high,
                "Low": p.low,
                "Close": p.close,
                "Volume": p.volume
            } for p in prices])
            
            # Show active model
            st.subheader("Active Production Model")
            active_model = db.get_active_model(selected_ticker)
            if active_model:
                metrics = active_model.get('metrics', {})
                rmse_val = f"{metrics.get('rmse', 0.0):.4f}"
                mae_val = f"{metrics.get('mae', 0.0):.4f}"
                mape_val = f"{metrics.get('mape', 0.0)*100:.2f}"
                r2_val = f"{metrics.get('r2', 0.0):.4f}"
                dir_acc_val = f"{metrics.get('directional_accuracy', 0.0)*100:.1f}"
                
                st.markdown(f"""
                <div class="card">
                    <div style="font-size: 1.15rem; font-weight: 600; color: #F3F4F6; margin-bottom: 0.5rem;">{active_model['model_type']} <span style="font-size: 0.85rem; color: #9CA3AF; font-weight: 400;">Version {active_model['version']}</span></div>
                    <div style="color: #9CA3AF; font-size: 0.8rem; margin-bottom: 0.75rem;">Registered: {active_model['created_at']} | Path: {active_model['filepath']}</div>
                    <div style="font-size: 0.85rem; font-weight: 600; color: #F3F4F6; margin-bottom: 0.4rem;">Performance Metrics (Test Set):</div>
                    <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.5rem; text-align: left;">
                        <div style="background-color: #1B2430; padding: 0.5rem; border-radius: 4px; border: 1px solid #374151;">
                            <div style="font-size: 0.7rem; color: #9CA3AF; text-transform: uppercase; margin-bottom: 0.2rem;">RMSE</div>
                            <div style="font-size: 0.95rem; font-weight: 600; color: #F3F4F6; font-family: 'JetBrains Mono', monospace;">{rmse_val}</div>
                        </div>
                        <div style="background-color: #1B2430; padding: 0.5rem; border-radius: 4px; border: 1px solid #374151;">
                            <div style="font-size: 0.7rem; color: #9CA3AF; text-transform: uppercase; margin-bottom: 0.2rem;">MAE</div>
                            <div style="font-size: 0.95rem; font-weight: 600; color: #F3F4F6; font-family: 'JetBrains Mono', monospace;">{mae_val}</div>
                        </div>
                        <div style="background-color: #1B2430; padding: 0.5rem; border-radius: 4px; border: 1px solid #374151;">
                            <div style="font-size: 0.7rem; color: #9CA3AF; text-transform: uppercase; margin-bottom: 0.2rem;">MAPE</div>
                            <div style="font-size: 0.95rem; font-weight: 600; color: #F3F4F6; font-family: 'JetBrains Mono', monospace;">{mape_val}%</div>
                        </div>
                        <div style="background-color: #1B2430; padding: 0.5rem; border-radius: 4px; border: 1px solid #374151;">
                            <div style="font-size: 0.7rem; color: #9CA3AF; text-transform: uppercase; margin-bottom: 0.2rem;">R-Squared</div>
                            <div style="font-size: 0.95rem; font-weight: 600; color: #F3F4F6; font-family: 'JetBrains Mono', monospace;">{r2_val}</div>
                        </div>
                        <div style="background-color: #1B2430; padding: 0.5rem; border-radius: 4px; border: 1px solid #374151;">
                            <div style="font-size: 0.7rem; color: #9CA3AF; text-transform: uppercase; margin-bottom: 0.2rem;">Dir. Acc.</div>
                            <div style="font-size: 0.95rem; font-weight: 600; color: #22C55E; font-family: 'JetBrains Mono', monospace;">{dir_acc_val}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No model is currently active in the registry for this ticker. Please train models to select one.")
                
            # Training Section
            st.subheader("⚙️ Train Ensemble Models")
            col1, col2 = st.columns(2)
            with col1:
                epochs = st.slider("LSTM Epochs", 5, 50, 15)
                batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
                seq_len = st.slider("Sequence Lookback (days)", 10, 60, 30)
                
            with col2:
                st.markdown("""
                **Ensemble Regressors Included:**
                - LSTM (Sequential Deep Learning)
                - Random Forest Regressor (Ensemble Bagging)
                - XGBoost Regressor (Gradient Boosting)
                - LightGBM Regressor (Optional Leaf-wise Tree Boosting)
                - Linear Regression (Baseline)
                """)
                train_clicked = st.button("🤖 Cross-Train & Evaluate Ensemble")
                
            if train_clicked:
                with st.spinner("Preparing sequence dataset..."):
                    preprocessor = DataPreprocessor(scaler_type='minmax')
                    processed_data = preprocessor.prepare_full_pipeline(
                        df, 
                        sequence_length=seq_len,
                        train_ratio=0.7,
                        val_ratio=0.15,
                        use_queue=True,
                        target_col='Close'
                    )
                    
                with st.spinner("Training models and saving the best model..."):
                    results, best_model = model_selector_service.train_and_compare(
                        selected_ticker, processed_data, epochs=epochs, batch_size=batch_size
                    )
                    
                    st.success(f"Ensemble training completed! Best model **{best_model}** registered as active.")
                    
                    # Display comparative metrics
                    res_df = pd.DataFrame(results).T
                    st.dataframe(res_df, use_container_width=True)
                    st.rerun()
                    
            # Model Registry list & switch active
            records = db.list_models(selected_ticker)
            if records:
                st.subheader("Model Registry Versions")
                registry_data = []
                for r in records:
                    registry_data.append({
                        "Version": r["version"],
                        "Model Type": r["model_type"],
                        "Active": "Yes" if r["is_active"] else "No",
                        "RMSE": f"{r['metrics'].get('rmse', 0.0):.4f}",
                        "R²": f"{r['metrics'].get('r2', 0.0):.4f}",
                        "Directional Accuracy": f"{r['metrics'].get('directional_accuracy', 0.0)*100:.1f}%",
                        "Created At": r["created_at"]
                    })
                st.dataframe(pd.DataFrame(registry_data), use_container_width=True)
                
                # Switch active model
                st.write("**Switch Active Model Version:**")
                version_options = [r["version"] for r in records]
                selected_version = st.selectbox("Select Version to Activate", version_options)
                if st.button("Activate Selected Version"):
                    if db.set_active_model(selected_ticker, selected_version):
                        st.success(f"Version {selected_version} set as active model for {selected_ticker}!")
                        st.rerun()
                    else:
                        st.error("Failed to update active model version.")
                        
            # Explainability / SHAP Plot
            if active_model:
                st.subheader("Explainability (SHAP & Coefficients)")
                
                # Load model
                active_wrapper, active_record, active_preprocessor = model_selector_service.load_best_model(selected_ticker)
                
                if active_wrapper:
                    with st.spinner("Calculating feature importances..."):
                        # Get a sample validation set for features
                        preprocessor = DataPreprocessor(scaler_type='minmax')
                        processed_data = preprocessor.prepare_full_pipeline(
                            df, sequence_length=30, use_queue=True
                        )
                        
                        importances = model_selector_service.get_feature_importances(
                            active_wrapper, processed_data['X_val'], processed_data['feature_columns']
                        )
                        
                        if importances:
                            imp_df = pd.DataFrame({
                                "Feature": list(importances.keys()),
                                "Importance": list(importances.values())
                            }).sort_values("Importance", ascending=True)
                            
                            # Draw Plot
                            fig_imp = px.bar(
                                imp_df, x="Importance", y="Feature", orientation="h"
                            )
                            fig_imp.update_layout(
                                plot_bgcolor='#111827',
                                paper_bgcolor='#111827',
                                font=dict(family='Inter, sans-serif', color='#9CA3AF', size=11),
                                xaxis=dict(showgrid=True, gridcolor='#374151', linecolor='#374151'),
                                yaxis=dict(showgrid=True, gridcolor='#374151', linecolor='#374151'),
                                margin=dict(l=10, r=10, t=10, b=10),
                                height=300
                            )
                            fig_imp.update_traces(marker_color='#3B82F6')
                            st.plotly_chart(fig_imp, use_container_width=True)

# ----------------- PAGE: PORTFOLIO & PAPER TRADING -----------------
elif page == "Portfolio & Paper Trading":
    st.markdown('<h1 class="main-header">Portfolio Tracker & Paper Trading</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Monitor holdings, manage virtual capital, and execute real-time trades with virtual balance.</p>', unsafe_allow_html=True)
    
    # Load Portfolio State & Holdings
    state = db.get_portfolio_state()
    holdings = db.get_holdings()
    
    # Call local calculation to compile metrics
    total_holdings_val = 0.0
    parsed_holdings = []
    
    for h in holdings:
        ticker = h["ticker"]
        shares = h["shares"]
        avg_price = h["avg_purchase_price"]
        
        # Fetch latest price
        latest_price = avg_price
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            if hasattr(t, 'fast_info'):
                latest_price = t.fast_info.get('last_price', avg_price)
            if not latest_price or latest_price == avg_price:
                hist = t.history(period="1d")
                if not hist.empty:
                    latest_price = hist['Close'].iloc[-1]
        except Exception:
            pass
            
        current_val = shares * latest_price
        cost_basis = shares * avg_price
        pl = current_val - cost_basis
        pl_pct = (pl / cost_basis) * 100 if cost_basis > 0 else 0.0
        
        total_holdings_val += current_val
        parsed_holdings.append({
            "Ticker": ticker,
            "Shares": shares,
            "Avg Cost": avg_price,
            "Current Price": latest_price,
            "Market Value": current_val,
            "P/L": pl,
            "P/L (%)": pl_pct
        })
        
    total_portfolio_value = state["cash"] + total_holdings_val
    total_pl = total_portfolio_value - state["initial_capital"]
    total_pl_pct = (total_pl / state["initial_capital"]) * 100 if state["initial_capital"] > 0 else 0.0
    
    total_portfolio_value_str = f"{total_portfolio_value:,.2f}"
    total_pl_pct_str = f"{total_pl_pct:+.2f}"
    cash_str = f"{state['cash']:,.2f}"
    total_holdings_val_str = f"{total_holdings_val:,.2f}"
    total_pl_str = f"{total_pl:+,.2f}"
    total_pl_pct_color = '#22C55E' if total_pl_pct >= 0 else '#EF4444'
    total_pl_color = '#22C55E' if total_pl >= 0 else '#EF4444'
    
    # Portfolio Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Total Account Value</div>
            <div class="metric-value">${total_portfolio_value_str}</div>
            <div style="font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; font-weight: 600; color: {total_pl_pct_color}; margin-top: 0.2rem;">
                {total_pl_pct_str}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Virtual Cash Balance</div>
            <div class="metric-value">${cash_str}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Holdings Valuation</div>
            <div class="metric-value">${total_holdings_val_str}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Total Profit/Loss</div>
            <div class="metric-value">${total_pl_str}</div>
            <div style="font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; font-weight: 600; color: {total_pl_color}; margin-top: 0.2rem;">
                {total_pl_pct_str}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # Holdings table
    st.subheader("Current Holdings")
    if parsed_holdings:
        df_hold = pd.DataFrame(parsed_holdings)
        # Format columns for display
        display_df = df_hold.copy()
        display_df["Avg Cost"] = display_df["Avg Cost"].map(lambda x: f"${x:,.2f}")
        display_df["Current Price"] = display_df["Current Price"].map(lambda x: f"${x:,.2f}")
        display_df["Market Value"] = display_df["Market Value"].map(lambda x: f"${x:,.2f}")
        display_df["P/L"] = display_df["P/L"].map(lambda x: f"${x:+,.2f}")
        display_df["P/L (%)"] = display_df["P/L (%)"].map(lambda x: f"{x:+.2f}%")
        st.dataframe(display_df, use_container_width=True)
        
        # Allocations Chart
        st.subheader("Portfolio Asset Allocation")
        alloc_data = [{"Asset": "Cash", "Value": state["cash"]}]
        for ph in parsed_holdings:
            alloc_data.append({"Asset": ph["Ticker"], "Value": ph["Market Value"]})
        df_alloc = pd.DataFrame(alloc_data)
        fig_pie = px.pie(df_alloc, values="Value", names="Asset", hole=0.4)
        fig_pie.update_layout(
            plot_bgcolor='#111827',
            paper_bgcolor='#111827',
            font=dict(family='Inter, sans-serif', color='#9CA3AF', size=11),
            margin=dict(l=10, r=10, t=10, b=10),
            height=300
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No holdings in portfolio yet. Execute a paper trade below.")
        
    # Trade execution panel
    st.subheader("Trade Execution Terminal (Paper Trading)")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        action = st.selectbox("Action", ["BUY", "SELL"])
    with col_b:
        ticker_list = [c.ticker for c in watched_companies]
        trade_ticker = st.selectbox("Ticker", ticker_list)
    with col_c:
        shares = st.number_input("Shares count", min_value=0.01, step=1.0, value=10.0)
    with col_d:
        # Get current price
        current_price = 0.0
        if trade_ticker:
            try:
                import yfinance as yf
                t = yf.Ticker(trade_ticker)
                if hasattr(t, 'fast_info'):
                    current_price = t.fast_info.get('last_price', 0.0)
                if not current_price:
                    hist = t.history(period="1d")
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
            except Exception:
                pass
        price = st.number_input("Execution Price (USD)", min_value=0.01, step=0.01, value=float(current_price))
        
    if st.button("Submit Order"):
        total_cost = shares * price
        if action == "BUY":
            if state["cash"] < total_cost:
                st.error(f"Insufficient virtual cash! Needed ${total_cost:.2f}, Cash: ${state['cash']:.2f}")
            else:
                db.update_portfolio_cash(-total_cost)
                db.update_holding(trade_ticker, shares, price, "BUY")
                db.record_transaction(trade_ticker, "BUY", shares, price)
                st.success(f"Order executed: BUY {shares} shares of {trade_ticker} at ${price:.2f}")
                st.rerun()
        elif action == "SELL":
            holding = db.get_holding(trade_ticker)
            if not holding or holding["shares"] < shares:
                st.error(f"Insufficient holdings! Hold: {holding['shares'] if holding else 0}, Request: {shares}")
            else:
                db.update_portfolio_cash(total_cost)
                db.update_holding(trade_ticker, shares, price, "SELL")
                db.record_transaction(trade_ticker, "SELL", shares, price)
                st.success(f"Order executed: SELL {shares} shares of {trade_ticker} at ${price:.2f}")
                st.rerun()
                
    # Transaction history
    st.subheader("Transaction Log")
    txs = db.get_transactions(limit=25)
    if txs:
        st.dataframe(pd.DataFrame(txs), use_container_width=True)
    else:
        st.info("No transaction logs recorded.")
        
    # Reset button
    st.write("---")
    st.subheader("Reset Account Settings")
    reset_capital = st.number_input("Reset capital amount", min_value=1000.0, step=1000.0, value=100000.0)
    if st.button("Factory Reset Portfolio"):
        if db.reset_portfolio(reset_capital):
            st.success("Virtual portfolio reset successfully!")
            st.rerun()

# ----------------- PAGE: RECOMMENDATION & BACKTEST -----------------
elif page == "Recommendation & Backtest":
    st.markdown('<h1 class="main-header">Recommendation Engine & Strategy Backtester</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Evaluate real-time consensus recommendation actions and backtest strategies over history.</p>', unsafe_allow_html=True)
    
    ticker_list = [c.ticker for c in watched_companies]
    selected_ticker = st.selectbox("Select Asset to Analyze", ticker_list)
    
    if selected_ticker:
        # Generate recommendation
        with st.spinner("Aggregating technicals, sentiment, corporate events and ML forecasts..."):
            rec = rec_service.generate_recommendation(selected_ticker)
            
        action_colors = {
            "BUY": "rgba(34, 197, 94, 0.05)",
            "SELL": "rgba(239, 68, 68, 0.05)",
            "HOLD": "rgba(156, 163, 175, 0.05)"
        }
        text_colors = {
            "BUY": "#22C55E",
            "SELL": "#EF4444",
            "HOLD": "#9CA3AF"
        }
        
        expected_return_str = f"{rec['expected_return']*100:+.2f}"
        action_color = text_colors[rec['action']]
        
        st.markdown(f"""
        <div class="card" style="border-left: 4px solid {action_color}; background-color: #202938; padding: 1rem 1.2rem;">
            <div style="font-size: 0.75rem; font-weight: 600; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.2rem;">Algorithmic Recommendation</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: {action_color}; margin-bottom: 0.8rem; font-family: 'Inter', sans-serif;">{rec['action']}</div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.6rem; font-size: 0.85rem; color: #9CA3AF;">
                <div>Investment Horizon: <span style="color: #F3F4F6; font-weight: 500;">{rec['investment_horizon']}</span></div>
                <div>Risk Level: <span style="color: #F3F4F6; font-weight: 500;">{rec['risk_level']}</span></div>
                <div>Consensus Confidence: <span style="color: #F3F4F6; font-weight: 500;">{int(rec['confidence_score']*100)}%</span></div>
                <div>Expected Return: <span style="color: {action_color}; font-weight: 500; font-family: 'JetBrains Mono', monospace;">{expected_return_str}%</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display signals
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Key Positive Catalysts")
            for sig in rec["key_positive_signals"]:
                st.write(f"✓ {sig}")
        with col2:
            st.subheader("Key Negative Risk Factors")
            for sig in rec["key_negative_signals"]:
                st.write(f"⚠ {sig}")
                
        # Backtest Recommendation Logic historically
        st.write("---")
        st.subheader("Strategy Backtester")
        lookback = st.slider("Historical backtest lookback (days)", 30, 365, 120)
        
        if st.button("Run Strategy Backtest"):
            with st.spinner("Simulating historical recommendation signals day-by-day..."):
                res = rec_service.run_strategy_backtest(selected_ticker, lookback)
                
            if "error" in res:
                st.error(res["error"])
            else:
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Cumulative Return", f"{res['cumulative_return']*100:+.2f}%")
                col_b.metric("Buy & Hold Return", f"{res['buy_and_hold_return']*100:+.2f}%")
                col_c.metric("Sharpe Ratio", f"{res['sharpe_ratio']:.2f}")
                col_d.metric("Win Rate", f"{res['win_rate']*100:.1f}%")
                
                # Plot Portfolio Value vs Asset Price
                df_hist = pd.DataFrame(res["portfolio_value_history"])
                fig_hist = make_subplots(specs=[[{"secondary_y": True}]])
                fig_hist.add_trace(
                    go.Scatter(x=df_hist["date"], y=df_hist["value"], name="ARGUS Strategy ($)", line=dict(color="#22C55E", width=2)),
                    secondary_y=False
                )
                fig_hist.add_trace(
                    go.Scatter(x=df_hist["date"], y=df_hist["price"], name="Asset Price ($)", line=dict(color="#3B82F6", width=1.5, dash="dash")),
                    secondary_y=True
                )
                fig_hist.update_layout(
                    plot_bgcolor='#111827',
                    paper_bgcolor='#111827',
                    font=dict(family='Inter, sans-serif', color='#9CA3AF', size=11),
                    xaxis=dict(showgrid=True, gridcolor='#374151', linecolor='#374151'),
                    yaxis=dict(showgrid=True, gridcolor='#374151', linecolor='#374151'),
                    yaxis2=dict(showgrid=True, gridcolor='#374151', linecolor='#374151', side='right'),
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Trade logs
                st.write("**Executed Transactions during Backtest:**")
                if res["trade_history"]:
                    st.dataframe(pd.DataFrame(res["trade_history"]), use_container_width=True)
                else:
                    st.info("No trades executed during this historical period (remained flat or in HOLD state).")

# ----------------- PAGE: AI ADVISOR MEMO -----------------
elif page == "AI Advisor Memo":
    st.markdown('<h1 class="main-header">AI Investment Advisor Memo</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Generate a structured, explainable financial memo summarizing technicals, news sentiment, events, and model predictions.</p>', unsafe_allow_html=True)
    
    ticker_list = [c.ticker for c in watched_companies]
    selected_ticker = st.selectbox("Select Asset Ticker", ticker_list)
    
    if selected_ticker:
        if st.button("Compile Investment Advisor Memo"):
            with st.spinner("Synthesizing ARGUS intelligence data..."):
                memo = advisor_service.generate_investment_memo(selected_ticker, gemini_api_key=st.session_state.gemini_key)
                
            st.markdown("### Generated Memo Preview")
            st.markdown(memo)
            
            # Allow download
            st.download_button(
                label="Download Memo (Markdown)",
                data=memo,
                file_name=f"{selected_ticker}_investment_memo.md",
                mime="text/markdown"
            )

# ----------------- PAGE 7: SYSTEM SETTINGS -----------------
elif page == "System Settings":
    st.markdown('<h1 class="main-header">System Settings & Diagnostic Logs</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Manage database models, credentials, caches, and system diagnostics.</p>', unsafe_allow_html=True)
    
    # 1. Database details
    st.subheader("Database Migration Status")
    session = db.SessionLocal()
    
    metadata_counts = {
        "Companies Table": session.query(Company).count(),
        "Ticker Cache Table": session.query(CompanyCache).count(),
        "Price History Table": session.query(PriceHistory).count(),
        "Technical Features Table": session.query(TechnicalFeature).count(),
        "News Articles Table": session.query(NewsArticle).count(),
        "Corporate Events Table": session.query(Event).count(),
        "Feature Store Table": session.query(FeatureStore).count(),
        "Diagnostic Logs Table": session.query(SystemLog).count()
    }
    session.close()
    
    col_x, col_y = st.columns(2)
    with col_x:
        for table, count in metadata_counts.items():
            st.write(f"**{table}:** {count} records")
            
    with col_y:
        st.subheader("Admin Operations")
        if st.button("Clear Diagnostic Logs"):
            db.clear_logs()
            st.success("System logs wiped.")
            st.rerun()
            
        if st.button("Flush Company Cache"):
            session = db.SessionLocal()
            session.query(CompanyCache).delete()
            session.commit()
            session.close()
            st.success("Stock Discovery Ticker Cache flushed successfully.")
            st.rerun()
            
    # 2. Database Logs table
    st.subheader("Real-time System Logs")
    logs = db.get_logs(limit=50)
    if logs:
        st.dataframe(pd.DataFrame(logs), use_container_width=True)
    else:
        st.info("No log events saved in database yet.")
        
    # 3. Developer & API Credentials Override Panel
    st.write("---")
    st.subheader("Developer Credentials Settings")
    with st.expander("API Configuration overrides"):
        new_alpha = st.text_input("Alpha Vantage API Key", value=st.session_state.alpha_key, type="password")
        new_news = st.text_input("NewsAPI Key", value=st.session_state.newsapi_key, type="password")
        new_gemini = st.text_input("Gemini AI API Key", value=st.session_state.gemini_key, type="password")
        if st.button("Save API Credentials"):
            st.session_state.alpha_key = new_alpha
            st.session_state.newsapi_key = new_news
            st.session_state.gemini_key = new_gemini
            st.session_state.news.news_api_key = new_news
            st.success("API Credentials saved successfully.")
            st.rerun()
