import streamlit as st
import pandas as pd
import numpy as np
import requests

from alpha_vantage.timeseries import TimeSeries
from sklearn.model_selection import train_test_split
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import plotly.graph_objects as go
import plotly.express as px

# Your Alpha Vantage API Key
# This key is now correctly placed in your app for data fetching
API_KEY = "AZKWKA2YRWA88QX9"
ts = TimeSeries(key=API_KEY, output_format='pandas')

st.set_page_config(layout="wide", page_title="Stock Price Forecasting App")
st.title("📈 Dynamic Stock Price Forecasting App")
st.markdown("Predict future stock prices using various time series models on any stock ticker.")

### Sidebar for user input ###
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input('Stock Ticker', 'GOOGL').upper()
start_date_str = st.sidebar.text_input("Start Date (YYYY-MM-DD)", "2020-01-01")

### Data Loading and Preprocessing ###

# Function to load data using Alpha Vantage
@st.cache_data
def load_data(ticker):
    try:
        # Fetch daily adjusted stock data
        data, meta_data = ts.get_daily_adjusted(symbol=ticker, outputsize='full')
        data = data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. adjusted close': 'Adj Close',
            '6. volume': 'Volume'
        })
        data.index = pd.to_datetime(data.index)
        # Sort data by date
        data = data.sort_index()
        return data

    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return None

# Load the data
data = load_data(ticker)

if data is not None and not data.empty:
    st.subheader(f'Historical Data for {ticker}')
    st.write(data)

    # Plot raw data
    st.subheader("Price Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig.layout.update(
        xaxis_rangeslider_visible=True,
        title_text="Stock Price History",
        xaxis_title="Date",
        yaxis_title="Price (USD)"
    )
    st.plotly_chart(fig)

    # Convert to Prophet-friendly format
    prophet_df = data.reset_index()[['index', 'Close']]
    prophet_df.columns = ['ds', 'y']

    # Prophet Model
    st.subheader("Prophet Model Forecasting")
    m = Prophet(daily_seasonality=True)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)

    # Plot Prophet forecast
    prophet_fig = m.plot(forecast)
    st.write(prophet_fig)

    # LSTM Model
    st.subheader("LSTM Model Forecasting")

    # Data preprocessing for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Create training data
    train_data = scaled_data[:int(len(scaled_data) * 0.8)]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create testing data
    test_data = scaled_data[len(train_data) - 60:, :]
    x_test = []
    y_test = scaled_data[len(train_data):, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Make predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Plot predictions
    train = data[:len(train_data)]
    valid = data[len(train_data):]
    valid['Predictions'] = predictions

    lstm_fig = go.Figure()
    lstm_fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Training Data'))
    lstm_fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Validation Data'))
    lstm_fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Predictions'))
    lstm_fig.layout.update(
        title_text="LSTM Model Predictions",
        xaxis_title="Date",
        yaxis_title="Price (USD)"
    )
    st.plotly_chart(lstm_fig)

    # Display error metrics
    rmse = np.sqrt(mean_squared_error(valid['Close'], valid['Predictions']))
    st.markdown(f"**LSTM Model RMSE:** {rmse:.2f}")

