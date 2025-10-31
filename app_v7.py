import os
import io
import contextlib
import datetime as dt
import pandas as pd
import requests
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from textblob import TextBlob
from dotenv import load_dotenv
import os
from back import main_pipeline  # Your supplied prediction function
load_dotenv() 
# Config
st.set_page_config(page_title="FinPulse New App", layout="wide")

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Navigation
def show_home():
    st.title("Welcome to FinPulse 📈")
    st.markdown("""
    ### About FinPulse
    FinPulse is an advanced financial analysis tool that combines stock market data with sentiment analysis 
    from news articles to provide deeper insights into market trends and make predictions.
    
    #### Key Features:
    * 📊 Real-time stock price visualization
    * 📰 News sentiment analysis
    * 🤖 AI-powered price predictions
    * 📈 Technical analysis integration
    * 📱 User-friendly interface
    
    #### How it Works:
    1. **Data Collection**: We gather historical stock data and relevant news articles
    2. **Sentiment Analysis**: Our AI analyzes news sentiment using FinBERT
    3. **Price Prediction**: LSTM neural networks process both price and sentiment data
    4. **Visualization**: Clear, interactive charts and metrics for informed decision making
    
    #### Get Started:
    Select 'Stock Analysis' in the navigation to begin exploring!
    """)

def show_stock_analysis():
    # Original stock analysis code will go here
    pass

# Navigation sidebar
page = st.sidebar.radio("Navigation", ["Home", "Stock Analysis"])

def analyze_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.1:
        return "🟢 Positive", polarity
    elif polarity < -0.1:
        return "🔴 Negative", polarity
    else:
        return "⚪ Neutral", polarity

@st.cache_data
def fetch_stock_data(ticker, period="4y", interval="1d"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def fetch_news(query, days=7, page_size=20):
    if not NEWS_API_KEY:
        return []
    from_date = (dt.datetime.utcnow() - dt.timedelta(days=days)).strftime("%Y-%m-%d")
    url = f"https://newsapi.org/v2/everything?q={requests.utils.quote(query)}&language=en&sortBy=publishedAt&from={from_date}&pageSize={page_size}&apiKey={NEWS_API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        return data.get("articles", [])
    except Exception:
        return []

def plot_stock(df):
    fig = go.Figure()
    
    # Add closing price line
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Close"],
        mode='lines',
        name='Closing Price',
        line=dict(color='#00b3ff', width=2)  # Bright blue
    ))
    
    # Add volume bars with cyan color
    fig.add_trace(go.Bar(
        x=df["Date"],
        y=df["Volume"],
        name='Volume',
        yaxis='y2',
        opacity=0.3,
        marker_color="#8dfff9"  # Cyan color
    ))
    
    # Update layout for dual axis with dark mode
    fig.update_layout(
        title="Stock Price & Volume",
        template="plotly_dark",
        paper_bgcolor='#1f1f1f',
        plot_bgcolor='#1f1f1f',
        yaxis=dict(
            title="Price",
            gridcolor='#333333',
            tickprefix='$'
        ),
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            gridcolor='#333333'
        ),
        xaxis=dict(
            gridcolor="#B44040"
        ),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


# Add prediction visualization function
def plot_prediction_results(stock_results, sentiment_results):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not stock_results or not sentiment_results:
        st.warning("No prediction results available to plot.")
        return
    
    # Create subplots for comparison
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            'Stock Price Prediction (Stock Data Only)',
            'Stock Price Prediction (With Sentiment Analysis)'
        ),
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5]
    )
    
    # Set dark theme colors
    template = "plotly_dark"
    bg_color = "#1f1f1f"
    text_color = "#ffffff"
    grid_color = "#835757"
    
    # Convert dates to naive datetime objects for plotting
    def convert_dates(dates):
        return pd.to_datetime(dates).dt.tz_localize(None)
    
    # Plot for stock-only predictions
    fig.add_trace(
        go.Scatter(
            x=convert_dates(stock_results['dates']),
            y=stock_results['actual_prices'],
            mode='lines',
            name='Actual Price',
            line=dict(color='#00b3ff', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=convert_dates(stock_results['test_dates']),
            y=stock_results['test_predictions'],
            mode='lines',
            name='Stock-Only Predictions',
            line=dict(color='#ff0000', width=2)
        ),
        row=1, col=1
    )
    
    # Plot for sentiment-included predictions
    fig.add_trace(
        go.Scatter(
            x=convert_dates(sentiment_results['dates']),
            y=sentiment_results['actual_prices'],
            mode='lines',
            name='Actual Price',
            line=dict(color='#00b3ff', width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=convert_dates(sentiment_results['test_dates']),
            y=sentiment_results['test_predictions'],
            mode='lines',
            name='Sentiment-Enhanced Predictions',
            line=dict(color='#ff0000', width=2)
        ),
        row=2, col=1
    )
    
    # Update layout with better styling
    fig.update_layout(
        template=template,
        height=800,
        showlegend=True,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        margin=dict(l=40, r=40, t=80, b=40),
        hovermode='x unified',
        legend=dict(
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            orientation="h"
        )
    )
    
    # Update axes with better styling
    for i in [1, 2]:
        fig.update_xaxes(
            title_text='Date',
            row=i, col=1,
            showgrid=True,
            gridcolor=grid_color,
            tickformat='%Y-%m-%d'
        )
        fig.update_yaxes(
            title_text='Stock Price ($)',
            row=i, col=1,
            showgrid=True,
            gridcolor=grid_color,
            tickprefix='$'
        )
    
    # Add model performance metrics as annotations
    for idx, (results, row) in enumerate([(stock_results, 1), (sentiment_results, 2)]):
        metrics = results.get('test_metrics', {})
        if metrics:
            metric_text = f"Test Metrics - MAE: {metrics.get('MAE', 0):.2f} | "
            metric_text += f"RMSE: {metrics.get('RMSE', 0):.2f} | "
            metric_text += f"R²: {metrics.get('R2', 0):.3f}"
            
            fig.add_annotation(
                text=metric_text,
                xref="paper", yref="paper",
                x=0.5, y=0.95 if row == 1 else 0.45,
                showarrow=False,
                font=dict(size=10, color=text_color),
                row=row, col=1
            )
    
    # Display plot with error handling
    try:
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying plot: {str(e)}")
        st.write("Plotting data:", {
            "Stock dates available": len(stock_results.get('dates', [])),
            "Stock predictions available": len(stock_results.get('test_predictions', [])),
            "Sentiment dates available": len(sentiment_results.get('dates', [])),
            "Sentiment predictions available": len(sentiment_results.get('test_predictions', []))
        })


def plot_sentiment_results(sentiment_results):
    """Plot actual vs predicted prices for the sentiment model using Plotly (dark theme)."""
    import plotly.graph_objects as go

    if not sentiment_results:
        st.warning("No sentiment results to plot.")
        return

    # Safely extract arrays
    dates = pd.to_datetime(sentiment_results.get('dates', []))
    test_dates = pd.to_datetime(sentiment_results.get('test_dates', []))
    actual = sentiment_results.get('actual_prices', [])
    preds = sentiment_results.get('test_predictions', [])

    # Build figure
    fig = go.Figure()
    try:
        fig.add_trace(go.Scatter(
            x=dates,
            y=actual,
            mode='lines',
            name='Actual Price',
            line=dict(color='#00b3ff', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=test_dates,
            y=preds,
            mode='lines+markers',
            name='Predicted (Sentiment Model)',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=4)
        ))

        # Metrics annotation
        metrics = sentiment_results.get('test_metrics', {})
        if metrics:
            metric_text = f"Test Metrics — MAE: {metrics.get('MAE', 0):.2f} | RMSE: {metrics.get('RMSE', 0):.2f} | R²: {metrics.get('R2', 0):.3f}"
            fig.add_annotation(text=metric_text, xref='paper', yref='paper', x=0.5, y=0.95, showarrow=False,
                                font=dict(color='#ffffff', size=11))

        fig.update_layout(template='plotly_dark', paper_bgcolor='#1f1f1f', plot_bgcolor='#1f1f1f', height=520,
                          margin=dict(l=40, r=40, t=80, b=40), hovermode='x unified')
        fig.update_xaxes(title_text='Date', gridcolor='#333333', tickformat='%Y-%m-%d')
        fig.update_yaxes(title_text='Price ($)', gridcolor='#333333', tickprefix='$')

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error building sentiment plot: {e}")
        st.write({
            'dates_len': len(dates), 'test_dates_len': len(test_dates), 'preds_len': len(preds)
        })


def plot_comparison_results(stock_results, sentiment_results):
    """Plot actual prices and overlay stock-only and sentiment model predictions for comparison."""
    import plotly.graph_objects as go

    if not stock_results and not sentiment_results:
        st.warning("No results to plot.")
        return

    # Extract and normalize date arrays
    dates = pd.to_datetime(stock_results.get('dates', [])) if stock_results else pd.to_datetime(sentiment_results.get('dates', []))
    stock_test_dates = pd.to_datetime(stock_results.get('test_dates', [])) if stock_results else []
    sent_test_dates = pd.to_datetime(sentiment_results.get('test_dates', [])) if sentiment_results else []

    actual = stock_results.get('actual_prices', []) if stock_results else sentiment_results.get('actual_prices', [])
    stock_preds = stock_results.get('test_predictions', []) if stock_results else []
    sent_preds = sentiment_results.get('test_predictions', []) if sentiment_results else []

    fig = go.Figure()
    try:
        # Actual full series
        fig.add_trace(go.Scatter(
            x=dates,
            y=actual,
            mode='lines',
            name='Actual Price',
            line=dict(color='#00b3ff', width=2)
        ))

        # Stock-only predictions
        if len(stock_test_dates) and len(stock_preds):
            fig.add_trace(go.Scatter(
                x=stock_test_dates,
                y=stock_preds,
                mode='lines+markers',
                name='Stock-Only Predictions',
                line=dict(color='#2ca02c', width=2),
                marker=dict(size=4)
            ))

        # Sentiment predictions
        if len(sent_test_dates) and len(sent_preds):
            fig.add_trace(go.Scatter(
                x=sent_test_dates,
                y=sent_preds,
                mode='lines+markers',
                name='Sentiment Predictions',
                line=dict(color='#ff7f0e', width=2),
                marker=dict(size=4)
            ))

        # Add metric annotations
        ann_texts = []
        if stock_results:
            sm = stock_results.get('test_metrics', {})
            ann_texts.append(f"Stock Test — MAE: {sm.get('MAE',0):.2f}, RMSE: {sm.get('RMSE',0):.2f}, R²: {sm.get('R2',0):.3f}")
        if sentiment_results:
            tm = sentiment_results.get('test_metrics', {})
            ann_texts.append(f"Sent Test — MAE: {tm.get('MAE',0):.2f}, RMSE: {tm.get('RMSE',0):.2f}, R²: {tm.get('R2',0):.3f}")

        if ann_texts:
            fig.add_annotation(text=" | ".join(ann_texts), xref='paper', yref='paper', x=0.5, y=0.95, showarrow=False,
                               font=dict(color='#ffffff', size=11))

        fig.update_layout(template='plotly_dark', paper_bgcolor='#1f1f1f', plot_bgcolor='#1f1f1f', height=560,
                          margin=dict(l=40, r=40, t=100, b=40), hovermode='x unified')
        fig.update_xaxes(title_text='Date', gridcolor='#333333', tickformat='%Y-%m-%d')
        fig.update_yaxes(title_text='Price ($)', gridcolor='#333333', tickprefix='$')

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error building comparison plot: {e}")
        st.write({
            'dates_len': len(dates), 'stock_test_dates_len': len(stock_test_dates), 'sent_test_dates_len': len(sent_test_dates)
        })

# Navigation control
if page == "Home":
    show_home()
else:
    # Stock selection (only show in Stock Analysis page)
    st.sidebar.title("Controls")
    tickers = ["^GSPC","AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "JPM", "BAC"]
    selected_ticker = st.sidebar.selectbox("Select Stock Ticker", tickers)
    
    st.title("Stock Analysis and Prediction")
    st.write(f"Selected Ticker: **{selected_ticker}**")

    # Fetch and show stock chart
    with st.spinner("Fetching stock data..."):
        stock_df = fetch_stock_data(selected_ticker)
    plot_stock(stock_df)

# Only show news and prediction sections on Stock Analysis page
if page == "Stock Analysis":
    # Fetch and show latest news
    st.header("Latest News")
    if not NEWS_API_KEY:
        st.info("💡 Set up NEWS_API_KEY in your .env file to see latest news.", icon="ℹ️")
    else:
        query_ticker = selected_ticker
        if selected_ticker == "^GSPC":
            query_ticker = "S&P 500"
        news = fetch_news(selected_ticker) 
        if not news:
            news = fetch_news(query_ticker)
        if not news:
            st.info(f"No recent news found for {selected_ticker}.")
        else:
            for art in news[:10]:
                title = art.get("title", "")
                desc = art.get("description", "")
                url_link = art.get("url", "")
                date = art.get("publishedAt", "")
                sentiment_label, polarity = analyze_sentiment(f"{title} {desc}")
                cols = st.columns([0.9, 0.1])
                with cols[0]:
                    st.markdown(f"**{title}**")
                    if desc: st.write(desc)
                    if url_link: st.markdown(f"[Read More]({url_link})")
                    if date: st.caption(date)
                with cols[1]:
                    st.markdown(f"{sentiment_label}\n{polarity:.2f}")
                st.divider()

    # Prediction area
    st.header("Prediction Pipeline")

    if st.button("Run Prediction Pipeline"):
        progress_placeholder = st.empty()
        log_area = st.empty()
        
        def update_progress(message):
            log_area.markdown(message)
            
        try:
            with st.spinner("Running prediction pipeline..."):
                results = main_pipeline(selected_ticker, progress_callback=update_progress)

            if results:
                stock_results = results.get('stock_only', {})
                sentiment_results = results.get('with_sentiment', {})

                st.subheader("Model Performance Comparison")
                # Four columns: stock train/test and sentiment train/test
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown("##### Stock-Only Training")
                    for k, v in stock_results.get('train_metrics', {}).items():
                        st.metric(k, f"{v:.4f}")

                with col2:
                    st.markdown("##### Stock-Only Testing")
                    for k, v in stock_results.get('test_metrics', {}).items():
                        st.metric(k, f"{v:.4f}")

                with col3:
                    st.markdown("##### Sentiment Training")
                    for k, v in sentiment_results.get('train_metrics', {}).items():
                        st.metric(k, f"{v:.4f}")

                with col4:
                    st.markdown("##### Sentiment Testing")
                    for k, v in sentiment_results.get('test_metrics', {}).items():
                        st.metric(k, f"{v:.4f}")

                # Plot comparison
                try:
                    plot_comparison_results(stock_results, sentiment_results)
                except Exception as e:
                    st.error(f"Unable to render comparison plot: {e}")

                # Show sample predictions in tabs
                st.subheader("Sample Predictions (Last 10 Days)")
                tab1, tab2 = st.tabs(["Stock-Only Predictions", "Sentiment-Enhanced Predictions"])
                with tab1:
                    stock_samples = stock_results.get('sample_results')
                    if stock_samples is not None:
                        # Normalize date column
                        if 'Date' in stock_samples.columns:
                            stock_samples['Date'] = pd.to_datetime(stock_samples['Date']).dt.tz_localize(None)
                        st.dataframe(stock_samples, use_container_width=True)

                with tab2:
                    sent_samples = sentiment_results.get('sample_results')
                    if sent_samples is not None:
                        if 'Date' in sent_samples.columns:
                            sent_samples['Date'] = pd.to_datetime(sent_samples['Date']).dt.tz_localize(None)
                        st.dataframe(sent_samples, use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            log_area.error("Please try again or contact support if the issue persists.")

# Footer - show on all pages
st.caption("Developed with FinPulse backend integration")

