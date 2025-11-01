import yfinance as yf 
import pandas as pd
import numpy as np
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch 
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib 
import matplotlib.pyplot as plt
from transformers import pipeline
from langdetect import detect 
from datetime import datetime, timedelta
import time
from urllib.parse import quote
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os, certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

#fetch Stock data 
def stock_data(ticker, period='4y', interval='1d', output_dir='.', write_csv=True):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    
    # Reset index for a clean DataFrame
    hist = hist.reset_index()
    
    # Save to CSV if requested
    if write_csv:
        # Construct filename
        csv_filename = f"{output_dir}/{ticker}_stock_data_{period}_{interval}.csv"
        hist.to_csv(csv_filename, index=False)
        stock_csv_path = csv_filename
        return hist, csv_filename
    else:
        return hist

def preprocess_stock_data(stock_csv_path):
    df = pd.read_csv(stock_csv_path)
    if 'Date' in df.columns:
        # Parse 'Date' as datetime and keep only date portion to avoid timezone/time issues
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.date
        df = df.sort_values('Date')
        df = df.reset_index(drop=True)
    else:
        raise ValueError("CSV must contain 'Date' column")

    cols_needed = ['Date','Close','High','Low','Open','Volume']
    df = df[[col for col in df.columns if col in cols_needed]]

    df = df.dropna(subset=['Close'])
    # Fix fillna method: use 'ffill' (forward fill) or 'bfill' (backward fill), not 'fillna' string
    df = df.fillna(method='ffill')

    return df


def load_sentiment_model():
    model_path = os.path.join(os.path.dirname(__file__), "finbert_local")
    sentiment_model = pipeline(
        "sentiment-analysis",
        model=model_path,
        tokenizer=model_path
    )
    return sentiment_model


def fetch_gdelt_news(query, start_date, end_date, max_records=100):
    """
    Fetch news articles from GDELT API for a specific query between start_date and end_date.
    Returns a DataFrame with 'title' and 'seendate'.
    """
    if query == "^GSPC":
        query = "S&P 500"
    query = quote(query)
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "JSON",
        "maxrecords": max_records,
        "sort": "DateDesc",
        "startdatetime": start_date.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_date.strftime("%Y%m%d%H%M%S")
    }

    try:
        response = requests.get(url, params=params, timeout=30,verify=False)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])
        if articles:
            return pd.DataFrame(articles)[["title", "seendate"]]
        else:
            return pd.DataFrame(columns=["title", "seendate"])
    except Exception as e:
        print(f"Error fetching data for {start_date} to {end_date}: {e}")
        return pd.DataFrame(columns=["title", "seendate"])


def fetch_gdelt_news_batch(query, start_year, end_year, total_records=1257, progress_callback=None):
    """
    Fetches news articles from GDELT API between start_year and end_year,
    batching the requests monthly to avoid API limits and accumulate total_records.
    Returns a DataFrame with all articles fetched.
    """
    def update_status(message):
        if progress_callback:
            progress_callback(message)
        print(message)
    all_articles = pd.DataFrame(columns=["title", "seendate"])

    # Convert years to integers for datetime
    start_year_int = int(start_year)
    end_year_int = int(end_year)
    total_months = (end_year_int - start_year_int + 1) * 12
    records_per_month = max(1, total_records // total_months)

    current_date = datetime(start_year_int, 1, 1)
    end_date = datetime(end_year_int, 12, 31)

    update_status(f"📚 Fetching {total_records} articles from {start_year} to {end_year}...")

    while current_date <= end_date and len(all_articles) < total_records:
        next_month = current_date.replace(day=28) + timedelta(days=4)
        month_end = next_month - timedelta(days=next_month.day)

        # Ensure we don't go beyond end_date
        month_end = min(month_end, end_date)

        if current_date > month_end:
            current_date = month_end + timedelta(days=1)
            continue

        update_status(f"📅 Fetching articles for {current_date.strftime('%Y-%m')}...")
        monthly_articles = fetch_gdelt_news(
            query,
            current_date,
            month_end,
            max_records=records_per_month
        )

        if not monthly_articles.empty:
            all_articles = pd.concat([all_articles, monthly_articles], ignore_index=True)
            update_status(f"✨ Found {len(monthly_articles)} articles for {current_date.strftime('%Y-%m')}, total: {len(all_articles)}")

        current_date = month_end + timedelta(days=1)

        # Add delay to avoid rate limiting
        time.sleep(1)

    if len(all_articles) > total_records:
        all_articles = all_articles.head(total_records)

    return all_articles

#predict sentiment and store using FinBERT
def predict_and_store_news_sentiment(
    news_csv_path,
    output_csv="news_with_sentiment.csv",
    model_path=os.path.join(os.path.dirname(__file__), "finbert_local")
):
    """
    Loads news from CSV, runs FinBERT sentiment analysis, saves to new CSV.
    Returns the DataFrame with sentiment columns.
    """
    # Load news data
    df = pd.read_csv(news_csv_path)
    if "title" not in df.columns:
        raise ValueError("Input CSV must have a 'title' column.")

    # Load FinBERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def get_sentiment(text):
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            label_id = torch.argmax(probs).item()
            score = probs[0][label_id].item()
            label = model.config.id2label[label_id]
            return label, score
        except Exception as e:
            print(f"Error processing text: {e}")
            return None, None

    sentiments = []
    scores = []

    print("Processing sentiment analysis for news articles...")
    for title in tqdm(df['title'], desc="Analyzing sentiment"):
        sentiment, score = get_sentiment(title)
        sentiments.append(sentiment)
        scores.append(score)

    df['sentiment'] = sentiments
    df['sentiment_score'] = scores

    # Save to CSV for next pipeline step
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to '{output_csv}'")
    return df
# Example usage:
# predict_and_store_news_sentiment('tesla_news_2020_2024.csv', 'news_with_sentiment.csv')

#Merging the sentiment score with stock data
def merge_stock_with_sentiment(
    stock_csv_path,
    sentiment_csv_path,
    output_csv_path="tsla_matched_with_sentiment.csv",
    stock_date_col="Date",
    sentiment_date_col="seendate"
):
    """
    Merges stock data and news sentiment data by date.
    - stock_csv_path: path to stock data CSV (must have a date column)
    - sentiment_csv_path: path to sentiment CSV (must have a date column and 'sentiment_score')
    - output_csv_path: where to save the merged result
    - stock_date_col: name of date column in stock data
    - sentiment_date_col: name of date column in sentiment data
    """
    # Load data
    stock_df = pd.read_csv(stock_csv_path)
    sentiment_df = pd.read_csv(sentiment_csv_path)

    # Convert date columns to datetime
    stock_df[stock_date_col] = pd.to_datetime(stock_df[stock_date_col], errors='coerce', utc=True)  # Ensure datetime conversion
    sentiment_df[sentiment_date_col] = pd.to_datetime(sentiment_df[sentiment_date_col], errors='coerce', utc=True)

    # Drop rows with invalid dates
    stock_df = stock_df.dropna(subset=[stock_date_col])
    sentiment_df = sentiment_df.dropna(subset=[sentiment_date_col])

    # Aggregate sentiment by date (mean if multiple news per day)
    daily_sentiment = sentiment_df.groupby(sentiment_df[sentiment_date_col].dt.date)['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['date', 'sentiment_score']

    # Add a 'date' column to stock data for merging
    stock_df['date'] = stock_df[stock_date_col].dt.date

    # Merge on date
    merged_df = pd.merge(stock_df, daily_sentiment, on='date', how='left')

    # Fill missing sentiment scores with 0 (neutral)
    merged_df['sentiment_score'] = merged_df['sentiment_score'].fillna(method='ffill')
    # If still NaN at the start (no earlier news), set to 0
    merged_df['sentiment_score'] = merged_df['sentiment_score'].fillna(0.0)

    # Save to CSV
    merged_df.to_csv(output_csv_path, index=False)
    print(f"Merged dataset saved to '{output_csv_path}'")

    return merged_df
# Example usage:
# merged_df = merge_stock_with_sentiment('tsla_stock_data.csv', 'news_with_sentiment.csv')

# No stock-only prediction in this phase
    
    # Inverse transform
    def inverse_transform_predictions(predictions, original_data):
        dummy_array = np.zeros((len(predictions), original_data.shape[1]))
        dummy_array[:, 0] = predictions.flatten()
        return scaler.inverse_transform(dummy_array)[:, 0]

    train_predict_inv = inverse_transform_predictions(train_predict, train_data)
    test_predict_inv = inverse_transform_predictions(test_predict, test_data)
    y_train_inv = inverse_transform_predictions(y_train.reshape(-1, 1), train_data)
    y_test_inv = inverse_transform_predictions(y_test.reshape(-1, 1), test_data)

    # Calculate metrics
    def calculate_metrics(actual, predicted):
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        r2 = r2_score(actual, predicted)
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2_Score': r2
        }

    train_metrics = calculate_metrics(y_train_inv, train_predict_inv)
    test_metrics = calculate_metrics(y_test_inv, test_predict_inv)

    # Prepare dates for plotting
    train_dates = df["Date"][time_step:time_step + len(train_predict_inv)]
    test_dates = df["Date"][train_size + time_step:train_size + time_step + len(test_predict_inv)]
    
    # Create sample predictions DataFrame
    sample_predictions = pd.DataFrame({
        'Date': test_dates[-10:],
        'Actual': y_test_inv[-10:],
        'Predicted': test_predict_inv[-10:],
        'Error': np.abs(y_test_inv[-10:] - test_predict_inv[-10:])
    })

    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'dates': df["Date"].tolist(),
        'actual_prices': df["Close"].tolist(),
        'train_dates': train_dates.tolist(),
        'train_predictions': train_predict_inv.tolist(),
        'test_dates': test_dates.tolist(),
        'test_predictions': test_predict_inv.tolist(),
        'history': history.history,
        'sample_results': sample_predictions
    }

def lstm_predict_with_sentiment(
    merged_csv_path,
    time_step=60,
    epochs=100,
    batch_size=32,
    plot=True
):
    from lstm_functions import lstm_predict_with_sentiment as lstm_sentiment
    return lstm_sentiment(merged_csv_path, time_step=time_step, epochs=epochs, batch_size=batch_size, plot=plot)

    # Sequence creation
    def create_sequences(dataset, time_step=60):
        X, y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step), :])
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_data, time_step)
    X_test, y_test = create_sequences(test_data, time_step)

    # Build LSTM model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        verbose=1)

    # Predict
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transform predictions
    def inverse_transform_predictions(predictions, original_data):
        dummy_array = np.zeros((len(predictions), original_data.shape[1]))
        dummy_array[:, 0] = predictions.flatten()
        return scaler.inverse_transform(dummy_array)[:, 0]

    train_predict_inv = inverse_transform_predictions(train_predict, train_data)
    test_predict_inv = inverse_transform_predictions(test_predict, test_data)
    y_train_inv = inverse_transform_predictions(y_train.reshape(-1, 1), train_data)
    y_test_inv = inverse_transform_predictions(y_test.reshape(-1, 1), test_data)

    # Metrics
    def calculate_metrics(actual, predicted):
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        r2 = r2_score(actual, predicted)
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2_Score': r2
        }

    train_metrics = calculate_metrics(y_train_inv, train_predict_inv)
    test_metrics = calculate_metrics(y_test_inv, test_predict_inv)

    # Prepare data for plotting
    train_dates = df["Date"][time_step:time_step + len(train_predict_inv)]
    test_dates = df["Date"][train_size + time_step:train_size + time_step + len(test_predict_inv)]
    
    # Add plotting data to results
    results = {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'sample_results': pd.DataFrame({
            'Date': test_dates[:10],
            'Actual': y_test_inv[:10],
            'Predicted': test_predict_inv[:10],
            'Error': np.abs(y_test_inv[:10] - test_predict_inv[:10])
        }).round(2),
        'dates': df['Date'].values,
        'actual_prices': df['Close'].values,
        'test_dates': df['Date'].values[train_size + time_step:],
        'test_predictions': test_predict_inv.flatten(),
        'test_metrics': test_metrics,
        'dates': df["Date"].tolist(),
        'actual_prices': df["Close"].tolist(),
        'train_dates': train_dates.tolist(),
        'train_predictions': train_predict_inv.tolist(),
        'test_dates': test_dates.tolist(),
        'test_predictions': test_predict_inv.tolist(),
        'history': history.history
    }
    
    # Plotting
    if plot:
    # Plot training and validation loss
        plt.figure(figsize=(14, 6))
        plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
        plt.title('Model Loss Over Epochs', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Plot actual vs predicted stock prices
        plt.figure(figsize=(16, 8))
        plt.plot(df["Date"], df["Close"], label="Actual Price", color="blue", alpha=0.7, linewidth=2)
        plt.plot(train_dates, train_predict_inv, label="Train Predictions", color="green", linewidth=2)
        plt.plot(test_dates, test_predict_inv, label="Test Predictions", color="red", linewidth=2)
        plt.title("Stock Price Prediction with LSTM + Sentiment", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price", fontsize=12)
        plt.legend(fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Scatter plot for actual vs predicted prices
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test_inv, test_predict_inv, alpha=0.6, color='purple', edgecolor='k', s=50)
        plt.plot([y_test_inv.min(), y_test_inv.max()], [y_test_inv.min(), y_test_inv.max()], 'r--', lw=2, label='Perfect Fit')
        plt.title('Actual vs Predicted Prices (Test Set)', fontsize=16)
        plt.xlabel('Actual Prices', fontsize=12)
        plt.ylabel('Predicted Prices', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Sample predictions
    sample_results = pd.DataFrame({
        'Date': test_dates[:10],
        'Actual': y_test_inv[:10],
        'Predicted': test_predict_inv[:10],
        'Error': np.abs(y_test_inv[:10] - test_predict_inv[:10])
    })

    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "sample_results": sample_results.round(2),
        "dates": df['Date'].values,
        "actual_prices": data.flatten(),
        "test_dates": df['Date'].values[train_size + time_step:],
        "test_predictions": test_predict_inv.flatten()
    }

def lstm_predict_stock_only(stock_csv, time_step=60, epochs=100, batch_size=32, plot=True):
    """Wrapper in back.py to run stock-only LSTM from lstm_functions."""
    from lstm_functions import lstm_predict_stock_only as _stock_fn
    return _stock_fn(stock_csv, time_step=time_step, epochs=epochs, batch_size=batch_size, plot=plot)


def main_pipeline(ticker, period='4y', interval='1d',
                  news_query=None, start_year=None, end_year=None,
                  total_news_records=1200,
                  finbert_path='./finbert_local',
                  progress_callback=None):
    """Main pipeline for stock prediction with sentiment analysis"""
    def update_status(message):
        if progress_callback:
            progress_callback(message)
        print(message)

    # 1. Fetch and preprocess stock data
    update_status("📈 Fetching stock data...")
    stock_df, stock_csv = stock_data(ticker, period=period, interval=interval)
    update_status("🔄 Preprocessing stock data...")
    stock_df = preprocess_stock_data(stock_csv)

    # 2. Determine news query and date range
    update_status("🎯 Setting up news query parameters...")
    if news_query is None:
        news_query = ticker
    if start_year is None or end_year is None:
        # Use stock data date range
        min_date = pd.to_datetime(stock_df['Date']).min()
        max_date = pd.to_datetime(stock_df['Date']).max()
        start_year = min_date.year
        end_year = max_date.year

    # 3. Fetch news in batch and save
    update_status("📰 Fetching news articles...")
    news_df = fetch_gdelt_news_batch(news_query, start_year, end_year, 
                                   total_records=total_news_records,
                                   progress_callback=progress_callback)
    news_csv = f'{ticker}_news_{start_year}_{end_year}.csv'
    news_df.to_csv(news_csv, index=False)
    update_status(f"💾 News data saved to {news_csv}")

    # 4. Run sentiment analysis and save
    update_status("🤖 Running sentiment analysis on news articles...")
    sentiment_csv = f'{ticker}_news_with_sentiment.csv'
    predict_and_store_news_sentiment(news_csv, output_csv=sentiment_csv, model_path=finbert_path)
    update_status("💾 Sentiment analysis results saved")

    # 5. Merge stock and sentiment data
    merged_csv = f'{ticker}_matched_with_sentiment.csv'
    merge_stock_with_sentiment(stock_csv, sentiment_csv, output_csv_path=merged_csv)

    # 6. Run predictions with both methods
    # import functions from lstm_functions
    update_status("🤖 Running stock-only predictions...")
    from lstm_functions import lstm_predict_stock_only, lstm_predict_with_sentiment

    stock_only_results = lstm_predict_stock_only(stock_csv)

    # Then run sentiment-enhanced predictions
    update_status("🤖 Running sentiment-enhanced predictions...")
    sentiment_results = lstm_predict_with_sentiment(merged_csv)

    update_status("✅ Prediction pipeline complete!")

    final_results = {
        'stock_only': stock_only_results,
        'with_sentiment': sentiment_results
    }

    return final_results

# Example usage:
# main_pipeline('TSLA', period='4y', interval='1d')







