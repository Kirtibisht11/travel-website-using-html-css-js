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

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to preprocess data for LSTM
def preprocess_lstm_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    X_train = []
    y_train = []

    if len(scaled_data) <= 60:
        raise ValueError("Not enough data points for preprocessing. Ensure the dataset contains more than 60 entries.")

    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    if X_train.size == 0 or y_train.size == 0:
        raise ValueError("Empty training data after preprocessing. Check the input data.")

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    return X_train, y_train, scaler

# LSTM Model
def build_lstm_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Predict the next closing price
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# LSTM Prediction
def predict_with_lstm(model, scaler, data, days):
    last_60_days = data['Close'].values[-60:].reshape(-1, 1)
    scaled_last_60_days = scaler.transform(last_60_days)
    
    future_prices = []
    input_data = scaled_last_60_days
    
    for _ in range(days):
        X_test = np.reshape(input_data, (1, input_data.shape[0], 1))
        predicted_price = model.predict(X_test)
        future_prices.append(predicted_price[0, 0])
        input_data = np.append(input_data, predicted_price)[1:].reshape(-1, 1)
    
    return scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))

# ARIMA Prediction
def predict_with_arima(data, days):
    model = ARIMA(data['Close'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=days)
    return forecast

# Flask app setup
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    # Predefined list of stock tickers
    tickers = ["AAPL", "GOOGL", "AMZN", "MSFT", "TSLA", "NFLX", "FB", "NVDA", "INTC", "IBM"]
    future_dates, future_prices, plot_path = [], [], None

    if request.method == 'POST':
        ticker = request.form['ticker']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        prediction_days = int(request.form['prediction_days'])
        model_choice = request.form['model_choice']
        
        # Fetch historical data
        data = fetch_stock_data(ticker, start_date, end_date)
        
        # Check if data is empty or has less than 60 data points
        if data.empty or len(data) < 60:
            return render_template('index.html', tickers=tickers, error="Not enough data: Ensure the dataset contains more than 60 entries.")
        
        # Prediction based on user choice
        if model_choice == "LSTM":
            X_train, y_train, scaler = preprocess_lstm_data(data)
            model = build_lstm_model()
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
            future_prices = predict_with_lstm(model, scaler, data, prediction_days)
            future_dates = [data.index[-1] + timedelta(days=i+1) for i in range(prediction_days)]
        elif model_choice == "ARIMA":
            future_prices = predict_with_arima(data, prediction_days)
            future_dates = [data.index[-1] + timedelta(days=i+1) for i in range(prediction_days)]
        else:
            return render_template('index.html', tickers=tickers, error="Invalid model choice. Please choose either LSTM or ARIMA.")
        
        # Plotting predictions
        plt.figure(figsize=(12, 6))
        plt.plot(data['Close'], label="Historical Prices")
        plt.plot(future_dates, future_prices, label="Predicted Prices", linestyle='--')
        plt.title(f"{ticker} Stock Price Prediction ({model_choice})")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.legend()
        
        # Save the plot as an image to display in HTML
        plot_path = 'static/stock_prediction.png'
        plt.savefig(plot_path)
        plt.close()
        
    return render_template('index.html', tickers=tickers, future_dates=future_dates, future_prices=future_prices, plot_path=plot_path)

if __name__ == "__main__":
    app.run(debug=True)
