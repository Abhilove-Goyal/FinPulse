import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def prepare_sequences(data, time_step):
    """Helper function to create sequences"""
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """Helper function to build LSTM model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def calculate_metrics(actual, predicted):
    """Helper function to calculate metrics"""
    return {
        'MAE': mean_absolute_error(actual, predicted),
        'MSE': mean_squared_error(actual, predicted),
        'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
        'R2': r2_score(actual, predicted)
    }

def lstm_predict_stock_only(stock_csv, time_step=60, epochs=100, batch_size=32):
    """LSTM prediction using only stock data"""
    # Load data
    df = pd.read_csv(stock_csv)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract and scale close prices
    close_prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(close_prices)
    
    # Create sequences
    X, y = prepare_sequences(scaled_prices, time_step)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build and train model
    model = build_lstm_model((time_step, 1))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Make predictions
    train_predict = model.predict(X_train, verbose=0)
    test_predict = model.predict(X_test, verbose=0)
    
    # Inverse transform predictions
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train_inv.flatten(), train_predict.flatten())
    test_metrics = calculate_metrics(y_test_inv.flatten(), test_predict.flatten())
    
    # Prepare return data
    test_dates = df['Date'].iloc[train_size + time_step:].values
    
    # Create sample predictions dataframe
    sample_predictions = pd.DataFrame({
        'Date': test_dates[:10],
        'Actual': y_test_inv[:10, 0],
        'Predicted': test_predict[:10, 0],
        'Error': np.abs(y_test_inv[:10, 0] - test_predict[:10, 0])
    })
    
    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'sample_results': sample_predictions,
        'dates': df['Date'].values,
        'actual_prices': df['Close'].values,
        'test_dates': test_dates,
        'test_predictions': test_predict.flatten()
    }

def lstm_predict_with_sentiment(merged_csv_path, time_step=60, epochs=100, batch_size=32):
    """LSTM prediction using both stock and sentiment data"""
    # Load data
    df = pd.read_csv(merged_csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract features and scale them separately
    close_prices = df['Close'].values.reshape(-1, 1)
    sentiment_scores = df['sentiment_score'].values.reshape(-1, 1)
    
    # Scale features
    price_scaler = MinMaxScaler()
    sentiment_scaler = MinMaxScaler()
    
    scaled_prices = price_scaler.fit_transform(close_prices)
    scaled_sentiment = sentiment_scaler.fit_transform(sentiment_scores)
    
    # Combine scaled features
    features = np.hstack((scaled_prices, scaled_sentiment))
    
    # Create sequences
    X, y = [], []
    for i in range(len(features) - time_step):
        X.append(features[i:(i + time_step), :])
        y.append(scaled_prices[i + time_step, 0])
    X, y = np.array(X), np.array(y)
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build and train model
    model = build_lstm_model((time_step, 2))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Make predictions
    train_predict = model.predict(X_train, verbose=0)
    test_predict = model.predict(X_test, verbose=0)
    
    # Inverse transform predictions (use price_scaler since we're predicting prices)
    train_predict = price_scaler.inverse_transform(train_predict)
    test_predict = price_scaler.inverse_transform(test_predict)
    y_train_inv = price_scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_inv = price_scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train_inv.flatten(), train_predict.flatten())
    test_metrics = calculate_metrics(y_test_inv.flatten(), test_predict.flatten())
    
    # Prepare return data
    test_dates = df['Date'].iloc[train_size + time_step:].values
    
    # Create sample predictions dataframe
    sample_predictions = pd.DataFrame({
        'Date': test_dates[:10],
        'Actual': y_test_inv[:10, 0],
        'Predicted': test_predict[:10, 0],
        'Error': np.abs(y_test_inv[:10, 0] - test_predict[:10, 0])
    })
    
    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'sample_results': sample_predictions,
        'dates': df['Date'].values,
        'actual_prices': df['Close'].values,
        'test_dates': test_dates,
        'test_predictions': test_predict.flatten()
    }