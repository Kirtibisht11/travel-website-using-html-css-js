import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

# Fetch stock data using yfinance
def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError("No data fetched. Check the ticker symbol and date range.")
        return data
    except Exception as e:
        raise ValueError(f"Error fetching stock data: {str(e)}")

# Preprocess data for LSTM
def preprocess_lstm_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    if len(scaled_data) <= 60:
        raise ValueError("Insufficient data. Ensure the dataset has more than 60 entries.")

    X_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    if X_train.size == 0 or y_train.size == 0:
        raise ValueError("Error in data preprocessing. Training data is empty.")

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train, scaler

# Build LSTM Model
def build_lstm_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)  # Predict the next closing price
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Predict future prices with LSTM
def predict_with_lstm(model, scaler, data, days):
    last_60_days = data['Close'].values[-60:].reshape(-1, 1)
    scaled_last_60_days = scaler.transform(last_60_days)

    input_data = scaled_last_60_days
    future_prices = []

    for _ in range(days):
        X_test = np.reshape(input_data, (1, input_data.shape[0], 1))
        predicted_price = model.predict(X_test, verbose=0)
        future_prices.append(predicted_price[0, 0])
        input_data = np.append(input_data, predicted_price)[-60:].reshape(-1, 1)

    return scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))

# Predict future prices with ARIMA
def predict_with_arima(data, days):
    try:
        model = ARIMA(data['Close'], order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=days)
        return forecast
    except Exception as e:
        raise ValueError(f"Error in ARIMA prediction: {str(e)}")

# Flask app setup
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Handle form inputs
            ticker = request.form['ticker'].upper()
            start_date = request.form['start_date']
            end_date = request.form['end_date']
            prediction_days = int(request.form['prediction_days'])
            model_choice = request.form['model_choice']

            # Fetch data
            stock_data = fetch_stock_data(ticker, start_date, end_date)

            if model_choice == 'lstm':
                # Prepare LSTM data and model
                X_train, y_train, scaler = preprocess_lstm_data(stock_data['Close'])
                lstm_model = build_lstm_model()
                lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
                predictions = predict_with_lstm(lstm_model, scaler, stock_data, prediction_days)
            elif model_choice == 'arima':
                predictions = predict_with_arima(stock_data, prediction_days)
            else:
                raise ValueError("Invalid model choice. Select 'lstm' or 'arima'.")

            return render_template('index.html', prediction_result=predictions)
        except Exception as e:
            return render_template('index.html', error_message=str(e))
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
