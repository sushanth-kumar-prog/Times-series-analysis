# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import backoff
import json

# --- Page Config ---
st.set_page_config(
    page_title="Stock Forecasting App",
    page_icon="📈",
    layout="wide",
)

# --- Functions for Data Fetching and Forecasting ---

# Function to safely fetch stock data with retries for network issues
@st.cache_data(show_spinner=False)
@backoff.on_exception(
    backoff.expo,
    (yf.YFInvalidTickerError, ConnectionError, TimeoutError),
    max_tries=3,
)
def get_stock_data(ticker):
    """
    Fetches historical stock data for a given ticker.
    """
    try:
        data = yf.download(ticker, start="2010-01-01", progress=False)
        if data.empty:
            st.error(f"Could not find a stock with the ticker: {ticker}. Please check the ticker symbol.")
            return None
        return data
    except yf.YFInvalidTickerError:
        st.error(f"Invalid ticker symbol: {ticker}. Please enter a valid stock ticker.")
        return None
    except Exception as e:
        st.error(f"An error occurred while fetching data: {e}. Please try again.")
        return None

def plot_stock_data(data, title):
    """
    Plots the closing price of the stock.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Stock Close Price'))
    fig.update_layout(title_text=title, xaxis_rangeslider_visible=True, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


def plot_prophet_forecast(model, future, forecast):
    """
    Plots the Prophet forecast.
    """
    fig1 = plot_plotly(model, forecast)
    fig1.update_layout(title="Prophet Forecast Plot", template="plotly_dark")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Prophet Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2, use_container_width=True)


def plot_lstm_forecast(data, forecast, title):
    """
    Plots the LSTM forecast.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data, name="Actual Price", mode="lines"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast['Prediction'], name="Predicted Price", mode="lines"))
    fig.update_layout(title_text=title, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


# --- UI and Main Logic ---

st.title("Stock Forecasting App")
st.write("This app predicts stock prices using various machine learning models.")

# Sidebar for user inputs
st.sidebar.header("User Input")
stock_ticker = st.sidebar.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL").upper()
n_days = st.sidebar.slider("Number of days to forecast", 30, 365, 90)

# Main content
st.info("Please wait while the app loads data and trains the models...")

# Fetch stock data
df = get_stock_data(stock_ticker)

if df is not None and not df.empty:
    st.success("Data fetched successfully!")

    # --- Data Processing and Display ---
    st.subheader(f"Historical Data for {stock_ticker}")
    st.write(df.tail())
    plot_stock_data(df, f"Historical Closing Price for {stock_ticker}")

    # --- Model Selection and Forecasting ---
    st.header("Stock Price Forecasting")
    model_choice = st.selectbox(
        "Select a forecasting model",
        ("Prophet", "LSTM", "Seasonal Decomposition"),
    )

    if model_choice == "Prophet":
        st.subheader("Forecasting with Prophet")
        data_prophet = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        model_prophet = Prophet()
        model_prophet.fit(data_prophet)

        future_prophet = model_prophet.make_future_dataframe(periods=n_days)
        forecast_prophet = model_prophet.predict(future_prophet)

        st.subheader(f"Prophet Forecast for the next {n_days} days")
        st.write(forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        plot_prophet_forecast(model_prophet, future_prophet, forecast_prophet)

        st.subheader("Model Evaluation")
        y_true = data_prophet['y']
        y_pred = forecast_prophet['yhat'].iloc[:-n_days]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    elif model_choice == "LSTM":
        st.subheader("Forecasting with LSTM")

        # Prepare data for LSTM
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

        training_data_len = int(len(scaled_data) * 0.8)
        train_data = scaled_data[0:training_data_len, :]

        # Create training data set
        x_train, y_train = [], []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i - 60 : i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Build the LSTM model
        model_lstm = Sequential()
        model_lstm.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model_lstm.add(LSTM(50, return_sequences=False))
        model_lstm.add(Dense(25))
        model_lstm.add(Dense(1))

        model_lstm.compile(optimizer='adam', loss='mean_squared_error')
        model_lstm.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)

        # Make predictions for the future
        future_data = scaled_data[-60:].reshape(1, 60, 1)
        future_predictions = []

        for _ in range(n_days):
            prediction = model_lstm.predict(future_data, verbose=0)
            future_predictions.append(prediction[0, 0])
            future_data = np.append(future_data[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

        # Inverse transform the predictions
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # Create a dataframe for the forecast
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=n_days, freq='D')
        forecast_lstm = pd.DataFrame(future_predictions, index=future_dates, columns=['Prediction'])

        st.subheader(f"LSTM Forecast for the next {n_days} days")
        st.write(forecast_lstm.head())
        plot_lstm_forecast(df['Close'], forecast_lstm, f"LSTM Forecast for {stock_ticker}")
        
    elif model_choice == "Seasonal Decomposition":
        st.subheader("Seasonal Decomposition Analysis")
        st.write("This model decomposes the stock price into its trend, seasonal, and residual components.")

        try:
            decomposition = seasonal_decompose(df['Close'], model='multiplicative', period=365)
            fig_sd = decomposition.plot()
            st.pyplot(fig_sd)
        except Exception as e:
            st.warning(f"Could not perform seasonal decomposition. This can happen with limited data. Error: {e}")

