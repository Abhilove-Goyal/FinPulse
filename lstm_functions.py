import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def lstm_predict_stock_only(stock_csv, time_step=60, epochs=100, batch_size=32, plot=True):
    """LSTM prediction using only stock data (returns same structure as sentiment function)."""
    df = pd.read_csv(stock_csv)
    # Convert to datetime without timezone
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert(None)
    else:
        # If no Date column, assume index or error
        raise ValueError("stock CSV must contain a 'Date' column")

    df = df.sort_values('Date')

    # Prepare price data
    price_data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_price = scaler.fit_transform(price_data)

    # Create sequences
    X, y = [], []
    for i in range(len(scaled_price) - time_step):
        X.append(scaled_price[i:(i + time_step), 0])
        y.append(scaled_price[i + time_step, 0])

    X = np.array(X)
    y = np.array(y)

    # Reshape for LSTM [samples, time_steps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Train/test split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=(X_test, y_test),
                       verbose=0)

    # Predictions
    train_predict = model.predict(X_train, verbose=0)
    test_predict = model.predict(X_test, verbose=0)

    # Inverse transform
    train_predict_inv = scaler.inverse_transform(train_predict.reshape(-1, 1))
    test_predict_inv = scaler.inverse_transform(test_predict.reshape(-1, 1))
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Metrics
    def calculate_metrics(actual, predicted):
        mae = mean_absolute_error(actual.flatten(), predicted.flatten())
        mse = mean_squared_error(actual.flatten(), predicted.flatten())
        rmse = np.sqrt(mse)
        r2 = r2_score(actual.flatten(), predicted.flatten())
        mape = np.mean(np.abs((actual.flatten() - predicted.flatten()) / actual.flatten())) * 100
        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

    train_metrics = calculate_metrics(y_train_inv, train_predict_inv)
    test_metrics = calculate_metrics(y_test_inv, test_predict_inv)

    # Dates for plotting/sample
    train_dates = df['Date'].values[time_step:time_step + len(train_predict_inv)]
    test_dates = df['Date'].values[train_size + time_step:train_size + time_step + len(test_predict_inv)]

    sample_predictions = pd.DataFrame({
        'Date': test_dates[:10],
        'Actual': y_test_inv[:10, 0],
        'Predicted': test_predict_inv[:10, 0],
        'Error': np.abs(y_test_inv[:10, 0] - test_predict_inv[:10, 0])
    }).round(2)

    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'sample_results': sample_predictions,
        'dates': df['Date'].values,
        'actual_prices': df['Close'].values,
        'test_dates': test_dates,
        'test_predictions': test_predict_inv.flatten(),
        'history': history.history
    }

def lstm_predict_with_sentiment(merged_csv_path, time_step=60, epochs=100, batch_size=32, plot=True):
    """LSTM prediction using stock data and sentiment"""
    # Load and prepare data
    df = pd.read_csv(merged_csv_path)
    # Convert to datetime without timezone
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert(None)
    df = df.sort_values("Date")
    
    # Extract features (Close price and sentiment score)
    price_data = df['Close'].values.reshape(-1, 1)
    sentiment_data = df['sentiment_score'].values.reshape(-1, 1)
    
    # Scale each feature separately
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    sentiment_scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaled_price = price_scaler.fit_transform(price_data)
    scaled_sentiment = sentiment_scaler.fit_transform(sentiment_data)
    
    # Combine scaled features
    scaled_data = np.hstack((scaled_price, scaled_sentiment))
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - time_step):
        X.append(scaled_data[i:(i + time_step), :])  # Include both features
        y.append(scaled_price[i + time_step, 0])     # Predict only the price
    
    # Convert to numpy arrays
    X = np.array(X)  # Shape: (samples, time_step, features=2)
    y = np.array(y)
    
    # Split data into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build LSTM model with same architecture as stock-only model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 2)),  # input_shape[2] = 2 for [price, sentiment]
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train
    history = model.fit(X_train, y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=(X_test, y_test),
                       verbose=0)

    # Make predictions
    train_predict = model.predict(X_train, verbose=0)
    test_predict = model.predict(X_test, verbose=0)
    
    # Predictions are only for price, so we can use price_scaler
    train_predict_inv = price_scaler.inverse_transform(train_predict.reshape(-1, 1))
    test_predict_inv = price_scaler.inverse_transform(test_predict.reshape(-1, 1))
    
    # Inverse transform actual values (they're already just prices)
    y_train_inv = price_scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_inv = price_scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    def calculate_metrics(actual, predicted):
        mae = mean_absolute_error(actual.flatten(), predicted.flatten())
        mse = mean_squared_error(actual.flatten(), predicted.flatten())
        rmse = np.sqrt(mse)
        r2 = r2_score(actual.flatten(), predicted.flatten())
        mape = np.mean(np.abs((actual.flatten() - predicted.flatten()) / actual.flatten())) * 100
        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

    train_metrics = calculate_metrics(y_train_inv, train_predict_inv)
    test_metrics = calculate_metrics(y_test_inv, test_predict_inv)

    # Get dates for sample predictions
    test_dates = df['Date'].values[train_size + time_step:]
    
    # Create sample predictions DataFrame
    sample_predictions = pd.DataFrame({
        'Date': test_dates[:10],
        'Actual': y_test_inv[:10, 0],
        'Predicted': test_predict_inv[:10, 0],
        'Error': np.abs(y_test_inv[:10, 0] - test_predict_inv[:10, 0])
    }).round(2)

    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'sample_results': sample_predictions,
        'dates': df['Date'].values,
        'actual_prices': df['Close'].values,
        'test_dates': test_dates,
        'test_predictions': test_predict_inv.flatten()
    }