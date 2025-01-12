import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to preprocess data for LSTM
def preprocess_lstm_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    X_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
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

# Main program
def main():
    # User Input
    print("Welcome to Stock Price Prediction Tool")
    ticker = input("Enter Stock Ticker (e.g., AAPL): ")
    start_date = input("Enter Start Date (YYYY-MM-DD): ")
    end_date = input("Enter End Date (YYYY-MM-DD): ")
    prediction_days = int(input("Enter Number of Days to Predict: "))
    model_choice = input("Choose Prediction Model (LSTM/ARIMA): ").strip().upper()
    
    # Fetch historical data
data = fetch_stock_data(ticker, start_date, end_date)

# Check if data is empty
if data.empty:
    print("No historical data available for the given date range.")
else:
    # Get the last date of the historical data
    last_date = data.index[-1]
    print(f"The last available date in the historical data is: {last_date.date()}")
# Fetch stock data
    print(f"Fetching data for {ticker}...")
    data = fetch_stock_data(ticker, start_date, end_date)
    
    if data.empty:
        print("No data found for the given ticker and date range. Exiting...")
                    
    print("Historical Data Preview:")
    print(data.head())
    
    # Prediction based on user choice
    if model_choice == "LSTM":
        print("Using LSTM model...")
        X_train, y_train, scaler = preprocess_lstm_data(data)
        model = build_lstm_model()
        print("Training the LSTM model. This might take a few minutes...")
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        
        print(f"Predicting next {prediction_days} days...")
        future_prices = predict_with_lstm(model, scaler, data, prediction_days)
        future_dates = [data.index[-1] + timedelta(days=i+1) for i in range(prediction_days)]
    elif model_choice == "ARIMA":
        print("Using ARIMA model...")
        print(f"Predicting next {prediction_days} days...")
        future_prices = predict_with_arima(data, prediction_days)
        future_dates = [data.index[-1] + timedelta(days=i+1) for i in range(prediction_days)]
    else:
        print("Invalid model choice. Please restart and choose either LSTM or ARIMA.")
       
    
    # Visualize Predictions
    print("Plotting the predictions...")
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label="Historical Prices")
    plt.plot(future_dates, future_prices, label="Predicted Prices", linestyle='--')
    plt.title(f"{ticker} Stock Price Prediction ({model_choice})")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
