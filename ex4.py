import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Email Notification Function
def send_email_notification(stock_name, price, recipient_email):
    try:
        sender_email = "sgshivapalaksha@gmail.com"
        sender_password = "dxum jchz qrgk jecd"  # Ensure this is protected in actual deployment
        smtp_server = "smtp.gmail.com"
        smtp_port = 587

        # Set up the email content
        message = MIMEMultipart()
        message['From'] = 'sgshivapalaksha@gmail.com'
        message['To'] = recipient_email
        message['Subject'] = f"Stock Price Movement Alert for {stock_name}"
        body = f"The stock {stock_name} has moved. The current price is: {price}"
        message.attach(MIMEText(body, 'plain'))

        # Send the email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(message)
        server.quit()

        st.success(f"Price movement alert sent to {recipient_email}!")
    except Exception as e:
        st.error(f"Failed to send email notification. Error: {e}")

# Preprocess the stock data
def preprocess_data(stock_data, seq_length=60):
    stock_data = stock_data[['Open', 'High', 'Low', 'Close']].copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_data_scaled = scaler.fit_transform(stock_data)
    
    def create_sequences(data, seq_length):
        x, y = [], []
        for i in range(len(data) - seq_length):
            x.append(data[i:i + seq_length])
            y.append(data[i + seq_length, 3])  # Predict the 'Close' price
        return np.array(x), np.array(y)
    
    x_data, y_data = create_sequences(stock_data_scaled, seq_length)
    return x_data, y_data, scaler

# Build LSTM Model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train model
def train_model(model, x_train, y_train, epochs=10, batch_size=32):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

# Make predictions and inverse the scaling
def predict_and_inverse_transform(model, x_test, scaler):
    predictions = model.predict(x_test)
    predictions_rescaled = scaler.inverse_transform(np.concatenate((np.zeros((predictions.shape[0], 3)), predictions), axis=1))[:, 3]
    return predictions_rescaled

# Generate Buy/Sell signals using Moving Averages
def buy_sell_signals(data):
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    buy_signal = []
    sell_signal = []
    for i in range(len(data)):
        if data['SMA50'][i] > data['SMA200'][i]:  # Buy signal
            buy_signal.append(data['Close'][i])
            sell_signal.append(np.nan)
        elif data['SMA50'][i] < data['SMA200'][i]:  # Sell signal
            buy_signal.append(np.nan)
            sell_signal.append(data['Close'][i])
        else:
            buy_signal.append(np.nan)
            sell_signal.append(np.nan)
    return buy_signal, sell_signal

# Perform EDA
def perform_eda(stock_data):
    st.subheader("Exploratory Data Analysis (EDA)")
    
    st.write("## Summary Statistics")
    st.write(stock_data.describe())

    st.write("## Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(stock_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(plt)

    st.write("## Distribution of Stock Prices")
    plt.figure(figsize=(10, 6))
    for col in ['Open', 'High', 'Low', 'Close']:
        sns.histplot(stock_data[col], kde=True, label=col)
    plt.legend()
    st.pyplot(plt)

# Main App Logic
st.title("Stock Prediction App")

# Navigation options
pages = ["Home", "Predict Stock Prices", "Exploratory Data Analysis (EDA)", "Feedback"]
choice = st.radio("Navigation", pages, horizontal=True)

if choice == "Home":
    st.title("Home Page")
    st.write("Welcome to the Stock Prediction App!")
    

    st.subheader("Introduction")
    st.write("""
        This application allows you to analyze stock data using various deep learning models such as LSTM. 
        You can visualize stock trends with candlestick charts and generate buy/sell signals based on moving averages.
    """)
    st.write("Use the navigation menu to start predicting stock prices or view additional features!")

elif choice == "Predict Stock Prices":
    st.title("Stock Price Prediction App with Candlestick Analysis")

    # User input for stock ticker
    ticker = st.text_input("Enter stock ticker:", value="AAPL")  # Take input from the user

    # Time period selection
    time_period = st.selectbox("Choose Time Period", ["1d", "1h", "5d", "1mo", "6mo", "1y"])

    # Email recipient input
    recipient_email = st.text_input("Enter email for price alerts:", placeholder="your_email@example.com")

    # Fetch stock data using yfinance based on selected time period
    if st.button("Predict"):
        if time_period == "1d":
            stock_data = yf.download(ticker, period="1d", interval="1m")
        elif time_period == "1h":
            stock_data = yf.download(ticker, period="1d", interval="1h")
        elif time_period == "5d":
            stock_data = yf.download(ticker, period="5d", interval="30m")
        elif time_period == "1mo":
            stock_data = yf.download(ticker, period="1mo", interval="1d")
        elif time_period == "6mo":
            stock_data = yf.download(ticker, period="6mo", interval="1d")
        elif time_period == "1y":
            stock_data = yf.download(ticker, period="1y", interval="1d")

        if stock_data.empty:
            st.warning("No stock data found. Please try a different stock ticker.")
        else:
            # Preprocess the stock data
            x_data, y_data, scaler = preprocess_data(stock_data)

            # Split the data into training and testing sets
            train_size = int(len(x_data) * 0.8)
            x_train, y_train = x_data[:train_size], y_data[:train_size]
            x_test, y_test = x_data[train_size:], y_data[train_size:]

            # Build and train the LSTM model
            model = build_lstm_model(input_shape=(x_train.shape[1], x_train.shape[2]))
            model = train_model(model, x_train, y_train, epochs=10, batch_size=32)

            # Make predictions
            predicted_prices = predict_and_inverse_transform(model, x_test, scaler)

            # Add predictions to the stock data for comparison
            stock_data['Predicted Close'] = np.nan
            stock_data.iloc[-len(predicted_prices):, stock_data.columns.get_loc('Predicted Close')] = predicted_prices

            # Plot actual and predicted closing prices
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Actual Close'))
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Predicted Close'], mode='lines', name='Predicted Close'))
            fig.update_layout(title=f"Actual vs Predicted Closing Prices for {ticker}", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig)

            # Generate Buy/Sell signals and plot them
            stock_data['Buy'], stock_data['Sell'] = buy_sell_signals(stock_data)
            fig = go.Figure(data=[go.Candlestick(x=stock_data.index, open=stock_data['Open'], high=stock_data['High'], low=stock_data['Low'], close=stock_data['Close'], name='Candlestick')])
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Buy'], mode='markers', name='Buy Signal', marker=dict(color='green', symbol='triangle-up')))
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Sell'], mode='markers', name='Sell Signal', marker=dict(color='red', symbol='triangle-down')))
            fig.update_layout(title=f"Candlestick Chart with Buy/Sell Signals for {ticker}", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig)

            # Send price movement email alert if email is provided
            if recipient_email:
                current_price = stock_data['Close'].iloc[-1]
                send_email_notification(ticker, current_price, recipient_email)

elif choice == "Exploratory Data Analysis (EDA)":
    st.title("Exploratory Data Analysis (EDA)")
    ticker = st.text_input("Enter stock ticker for EDA:", value="AAPL")
    
    if st.button("Perform EDA"):
        stock_data = yf.download(ticker, period="1y")
        if stock_data.empty:
            st.warning("No stock data found. Please try a different stock ticker.")
        else:
            perform_eda(stock_data)

elif choice == "Feedback":
    st.title("Feedback")
    feedback = st.text_area("We value your feedback!", placeholder="Enter your feedback here...")
    if st.button("Submit"):
        st.success("Thank you for your feedback!")
