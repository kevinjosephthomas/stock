# Import necessary libraries
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

# Step 1: Fetch stock data for a given ticker
def get_stock_data(ticker, period='1d', interval='1m'):
    stock_data = yf.download(ticker, period=period, interval=interval)
    return stock_data[['Close']]  # Only use 'Close' price for prediction

# Step 2: Preprocess the data for the LSTM model
def preprocess_data(stock_data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data)

    X, Y = [], []
    for i in range(len(scaled_data) - time_step - 1):
        X.append(scaled_data[i:(i + time_step), 0])
        Y.append(scaled_data[i + time_step, 0])

    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM
    return X, Y, scaler

# Step 3: Build the LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))  # Single output for stock price prediction
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 4: Train the model
def train_model(model, X_train, Y_train, epochs=50, batch_size=64):
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

# Step 5: Predict the next 5 minutes of stock prices
def predict_next_5_minutes(model, last_60_min_data, scaler):
    predicted_5min_prices = []
    
    # Reshape input to 3D if required by the model
    last_60_min_data = np.reshape(last_60_min_data, (1, last_60_min_data.shape[0], 1))

    for _ in range(5):
        predicted_price = model.predict(last_60_min_data)
        predicted_5min_prices.append(predicted_price[0, 0])
        
        # Reshape predicted_price to match the input shape
        predicted_price_reshaped = np.reshape(predicted_price, (1, 1, 1))
        
        # Append the predicted price, shift the last_60_min_data
        last_60_min_data = np.append(last_60_min_data[:, 1:, :], predicted_price_reshaped, axis=1)

    # Convert predicted prices back to original scale
    predicted_5min_prices = scaler.inverse_transform(np.array(predicted_5min_prices).reshape(-1, 1))

    return predicted_5min_prices

# Step 6: Visualization
def visualize_predictions(predicted_prices, ticker):
    plt.figure(figsize=(10, 5))
    plt.plot(predicted_prices, color='blue', label=f'Predicted Next 5 Min Prices for {ticker}')
    plt.title(f'Stock Price Prediction for {ticker}')
    plt.xlabel('Minutes')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# Step 7: Full process function
def predict_stock_for_ticker(ticker):
    # Fetch stock data
    stock_data = get_stock_data(ticker)

    # Preprocess data
    X_train, Y_train, scaler = preprocess_data(stock_data)

    # Create and train the LSTM model
    model = create_lstm_model((X_train.shape[1], 1))
    train_model(model, X_train, Y_train)

    # Predict the next 5 minutes using the last 60 minutes of data
    last_60_min_data = stock_data[-60:].values
    last_60_min_data_scaled = scaler.transform(last_60_min_data)

    predicted_5min_prices = predict_next_5_minutes(model, last_60_min_data_scaled, scaler)
    return predicted_5min_prices

# Step 8: Main Function to Run Streamlit App
def main():
    st.title("Stock Price Prediction")
    
    ticker = st.text_input("Enter stock ticker (e.g., 'AAPL' for Apple):")  # User provides the stock ticker
    if st.button("Predict"):
        if ticker:
            predicted_prices = predict_stock_for_ticker(ticker)
            st.write(f"Predicted stock prices for the next 5 minutes for {ticker}:")
            for i, price in enumerate(predicted_prices, 1):
                st.write(f"Minute {i}: {price[0]:.2f}")  # Change here to access the scalar value

            # Visualize the predicted stock prices
            st.subheader(f'Predicted Next 5 Min Prices for {ticker}')
            visualize_predictions(predicted_prices, ticker)
            st.pyplot(plt)  # Show the plot in Streamlit
        else:
            st.error("Please enter a valid stock ticker.")

if __name__ == "__main__":
    main()
