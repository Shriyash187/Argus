"""
M.I.D.E. (Market Intelligence & Investment Decision Engine) Streamlit Dashboard
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

logger = get_logger()

# ----------------- SESSION STATE & SETUP -----------------
st.set_page_config(
    page_title="M.I.D.E. Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark/Modern theme adjustments)
st.markdown("""
<style>
    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(45deg, #1f77b4, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #1e272e;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #2f3640;
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #888888;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f5f6fa;
    }
    .sentiment-positive {
        color: #2ecc71;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #e74c3c;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #f1c40f;
        font-weight: bold;
    }
    .badge {
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-critical {
        background-color: rgba(231, 76, 60, 0.2);
        color: #e74c3c;
        border: 1px solid #e74c3c;
    }
    .badge-moderate {
        background-color: rgba(241, 196, 15, 0.2);
        color: #f1c40f;
        border: 1px solid #f1c40f;
    }
    .badge-low {
        background-color: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
        border: 1px solid #2ecc71;
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

db = st.session_state.db
yahoo = st.session_state.yahoo
ticker_search = st.session_state.ticker_search
news_service = st.session_state.news
feature_service = st.session_state.features

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
st.sidebar.markdown("<h2 style='text-align: center;'>⚡ M.I.D.E.</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; color: gray; font-size: 0.85em;'>Decision Engine & Intelligence</p>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "🔍 Stock Search", "📈 Stock Overview", "🤖 ML Prediction", "📰 News Intelligence", "🔔 Event Intelligence", "⚙️ System Settings"]
)

# Manage API keys from Sidebar Settings
st.sidebar.markdown("---")
st.sidebar.subheader("🔑 Credentials")
alpha_key = st.sidebar.text_input("Alpha Vantage Key", value=os.getenv("ALPHA_VANTAGE_API_KEY", "GQ1C7TJRZ4ANOZM9"), type="password")
newsapi_key = st.sidebar.text_input("NewsAPI Key", value=os.getenv("NEWS_API_KEY", ""), type="password")

# Sync updated keys back to services
st.session_state.news.news_api_key = newsapi_key
st.session_state.features.db = db

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
if page == "🏠 Dashboard":
    st.markdown('<h1 class="main-header">🏠 Market Intelligence Dashboard</h1>', unsafe_allow_html=True)
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
    st.subheader("📋 Watched Stock Overview")
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
    st.subheader("🏆 Priority Queue Algorithmic Ranking")
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
            st.table(rank_df)
        else:
            st.info("Insufficient historical prices to compute 30-day returns. Load more stock price history first.")
            
# ----------------- PAGE 2: STOCK SEARCH -----------------
elif page == "🔍 Stock Search":
    st.markdown('<h1 class="main-header">🔍 Universal Stock Search</h1>', unsafe_allow_html=True)
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
                <h3>{res['name']} ({res['ticker']})</h3>
                <p><b>Exchange:</b> {res['exchange']}</p>
                <p><b>Country:</b> {res['country']}</p>
                <p><b>Currency:</b> {res['currency']}</p>
                <p><b>Sector:</b> {res['sector']}</p>
                <p><b>Industry:</b> {res['industry']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Action button to add to watchlist
            if st.button("➕ Add to Watchlist"):
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
elif page == "📈 Stock Overview":
    st.markdown('<h1 class="main-header">📈 Technical Overview</h1>', unsafe_allow_html=True)
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
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                name="Price"
            ), row=1, col=1)
            
            # EMA
            if 'EMA_20' in df.columns:
                fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_20'], name='EMA 20', line=dict(color='orange', width=1.5)), row=1, col=1)
            
            # Bollinger Bands
            if 'Bollinger_Upper' in df.columns:
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Bollinger_Upper'], name='Bollinger Upper', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Bollinger_Lower'], name='Bollinger Lower', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
                
            # Volume
            fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name="Volume", marker_color='lightblue'), row=2, col=1)
            
            # Support/Resistance boundary lines (Stack helpers demonstration)
            from ds_helpers import detect_support_resistance_levels
            supports, resistances = detect_support_resistance_levels(df['Close'].tolist())
            levels = [float(l) for l in (supports + resistances) if pd.notna(l)]
            for idx, lvl in enumerate(levels[:5]):
                fig.add_hline(y=lvl, line_dash="dot", line_color="green" if idx%2==0 else "red", 
                             annotation_text=f"Level {idx+1}", row=1, col=1)
                
            fig.update_layout(height=650, title=f"{selected_ticker} Technical Chart", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Failed to load price history data for {selected_ticker}.")

# ----------------- PAGE 4: ML PREDICTION -----------------
elif page == "🤖 ML Prediction":
    st.markdown('<h1 class="main-header">🤖 Predictive Modeling</h1>', unsafe_allow_html=True)
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
                
                train_clicked = st.button("🤖 Train LSTM Model")
                
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
                fig.add_trace(go.Scatter(y=y_test_orig, name="Actual Prices", line=dict(color="blue")))
                fig.add_trace(go.Scatter(y=test_pred_orig.flatten(), name="LSTM Predictions", line=dict(color="red", dash="dash")))
                fig.update_layout(title="Predictions vs Actual Price (Test Set)", height=450)
                st.plotly_chart(fig, use_container_width=True)
                
                # Backtesting validation
                st.subheader("⌛ Walk-Forward Validation")
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
                        col_b.metric("MAPE", f"{metrics['mape']:.2f}%")
                        col_c.metric("Directional Accuracy", f"{metrics['directional_accuracy']*100:.1f}%")
                        col_d.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0.0):.2f}")
                        
                        # Graph backtest predictions
                        fig_bt = go.Figure()
                        fig_bt.add_trace(go.Scatter(y=backtest_results['actuals'], name="Actual Price", line=dict(color="blue")))
                        fig_bt.add_trace(go.Scatter(y=backtest_results['predictions'], name="Backtested Predicted Price", line=dict(color="orange", dash="dot")))
                        fig_bt.update_layout(title="Walk-Forward Validation: Simulated History vs Predictions", height=450)
                        st.plotly_chart(fig_bt, use_container_width=True)
                    else:
                        st.error("Failed to calculate backtesting metrics. Make sure you have enough historical periods.")

# ----------------- PAGE 5: NEWS INTELLIGENCE -----------------
elif page == "📰 News Intelligence":
    st.markdown('<h1 class="main-header">📰 News Sentiment Intelligence</h1>', unsafe_allow_html=True)
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
                color_discrete_sequence=["#2ecc71", "#e74c3c", "#f1c40f"],
                hole=0.4
            )
            fig.update_layout(height=350, title="Sentiment Sentiment Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Display articles list
            st.subheader("📰 Recent Articles Feed")
            for art in articles:
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
                    <h4><a href="{art.url}" target="_blank">{art.headline}</a></h4>
                    <p style="color: gray; font-size: 0.85em;">Source: {art.source} | Date: {art.published_date.strftime('%Y-%m-%d %H:%M')}</p>
                    <p>{art.summary}</p>
                    <p>Sentiment polarity: <span class="{badge_class}">{badge_lbl}</span> (Confidence: {art.sentiment_confidence*100:.0f}%)</p>
                </div>
                """, unsafe_allow_html=True)

# ----------------- PAGE 6: EVENT INTELLIGENCE -----------------
elif page == "🔔 Event Intelligence":
    st.markdown('<h1 class="main-header">🔔 Event Intelligence Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time classification of corporate milestones and regulatory warnings.</p>', unsafe_allow_html=True)
    
    ticker_list = [c.ticker for c in watched_companies]
    selected_ticker = st.selectbox("Select Asset Events", ticker_list)
    
    if selected_ticker:
        events = db.get_events(selected_ticker)
        
        if not events:
            st.info("No corporate events classified for this ticker yet. Generating new coverage will index them.")
        else:
            for ev in events:
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
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3>🚨 {ev.event_type}</h3>
                        <span class="badge {sev_class}">Severity Score: {ev.severity:.2f}</span>
                    </div>
                    <p style="color: gray; font-size: 0.85em;">Date Indexed: {ev.date.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><b>Description:</b> {ev.description}</p>
                    <p>Event Sentiment: <span class="{sent_class}">{ev.sentiment}</span> (Confidence: {ev.confidence*100:.0f}%)</p>
                </div>
                """, unsafe_allow_html=True)

# ----------------- PAGE 7: SYSTEM SETTINGS -----------------
elif page == "⚙️ System Settings":
    st.markdown('<h1 class="main-header">⚙️ System Settings & Diagnostic Logs</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Manage database models, credentials, caches, and system diagnostics.</p>', unsafe_allow_html=True)
    
    # 1. Database details
    st.subheader("💾 Database Migration Status")
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
            st.write(f"📁 **{table}:** {count} records")
            
    with col_y:
        st.subheader("Admin Operations")
        if st.button("🗑️ Clear Diagnostic Logs"):
            db.clear_logs()
            st.success("System logs wiped.")
            st.rerun()
            
        if st.button("🔥 Flush Company Cache"):
            session = db.SessionLocal()
            session.query(CompanyCache).delete()
            session.commit()
            session.close()
            st.success("Stock Discovery Ticker Cache flushed successfully.")
            st.rerun()
            
    # 2. Database Logs table
    st.subheader("📜 Real-time System Logs")
    logs = db.get_logs(limit=50)
    if logs:
        st.dataframe(pd.DataFrame(logs), use_container_width=True)
    else:
        st.info("No log events saved in database yet.")
