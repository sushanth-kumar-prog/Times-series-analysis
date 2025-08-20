# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import backoff  # <-- ADD THIS IMPORT
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf
from datetime import date, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

st.set_page_config(layout="wide", page_title="Stock Price Forecasting App")

st.title('📈 Dynamic Stock Price Forecasting App')
st.markdown("Predict future stock prices using various time series models on any stock ticker.")

# --- Sidebar for user input ---
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input('Stock Ticker', 'GOOGL').upper()

# Set default date range
today = date.today()
end_date = st.sidebar.date_input('End Date', today)
start_date = st.sidebar.date_input('Start Date', end_date - timedelta(days=365*5))

# --- Data Loading ---
if st.sidebar.button('Reload Data'):
    st.cache_data.clear()

# --- THIS IS THE FINAL FIX ATTEMPT ---
@st.cache_data
@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
def load_data(ticker, start, end):
    """
    Loads stock data from Yahoo Finance using the Ticker method with retries.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        # The most robust way to download, with a timeout.
        data = ticker_obj.history(start=start, end=end, timeout=10)
        
        if data.empty:
            # Check if the ticker is valid at all
            if not ticker_obj.info:
                 st.error(f"Invalid ticker symbol: {ticker}")
                 return pd.DataFrame()
            st.error(f"No data found for ticker: {ticker} in the selected date range.")
            return pd.DataFrame()

        return data[['Close']].copy()
        
    except Exception as e:
        st.error(f"An error occurred while fetching data: {e}")
        return pd.DataFrame()

# --- END OF FIX ATTEMPT ---

df = load_data(ticker, start_date, end_date)

if df.empty:
    st.stop()

# --- The rest of your code remains exactly the same ---
st.subheader(f'Raw Data for {ticker} (Last 5 Rows)')
st.write(df.tail())

# --- Plot Raw Data with Plotly ---
st.subheader(f'{ticker} Closing Price Over Time')
fig_raw = px.line(df, x=df.index, y='Close', title=f'{ticker} Historical Closing Price',
                  labels={'Date': 'Date', 'Close': 'Close Price (USD)'})
fig_raw.update_layout(xaxis_rangeslider_visible=True, template='plotly_dark')
st.plotly_chart(fig_raw, use_container_width=True)

# --- Model Selection and Forecasting Parameters ---
st.sidebar.header("Forecasting Settings")
model_choice = st.sidebar.selectbox(
    'Choose a forecasting model:',
    ('ARIMA', 'Prophet', 'LSTM')
)
forecast_steps = st.sidebar.slider('Select number of days to forecast', 30, 365, 90)

# --- Forecasting Button ---
if st.sidebar.button('Generate Forecast'):
    time_step = 60
    if len(df) < time_step + 1 and model_choice == 'LSTM':
        st.error(f"Not enough historical data to train the LSTM model. It requires at least {time_step + 1} data points. Your dataset has {len(df)}.")
        st.stop()

    train_size = int(len(df) * 0.8)
    train_data, test_data = df[0:train_size], df[train_size:len(df)]

    st.subheader(f'{model_choice} Forecast')
    accuracy_metric = {}

    with st.spinner(f'Generating {model_choice} forecast...'):
        try:
            if model_choice == 'ARIMA':
                model_test = ARIMA(train_data['Close'], order=(5, 1, 0))
                model_test_fit = model_test.fit()
                test_predictions = model_test_fit.forecast(steps=len(test_data))
                
                mse = mean_squared_error(test_data['Close'], test_predictions)
                rmse = np.sqrt(mse)
                accuracy_metric['ARIMA'] = {'MSE': mse, 'RMSE': rmse}

                model_full = ARIMA(df['Close'], order=(5, 1, 0))
                model_full_fit = model_full.fit()
                future_predictions = model_full_fit.forecast(steps=forecast_steps)
                
                last_historical_date = df.index[-1]
                forecast_index = pd.date_range(start=last_historical_date + pd.Timedelta(days=1), periods=forecast_steps, freq='B')
                forecast_df = pd.DataFrame({'Forecast': future_predictions}, index=forecast_index)
                
                fig_arima = go.Figure()
                fig_arima.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Historical Data', line=dict(color='royalblue')))
                fig_arima.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='ARIMA Forecast', line=dict(color='red', dash='dash')))
                fig_arima.update_layout(title=f'ARIMA {ticker} Price Forecast', xaxis_title='Date', yaxis_title='Close Price (USD)', template='plotly_dark')
                st.plotly_chart(fig_arima, use_container_width=True)
                
            elif model_choice == 'Prophet':
                prophet_df_train = train_data.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
                prophet_model_train = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True, interval_width=0.95)
                prophet_model_train.fit(prophet_df_train)

                future_test = prophet_model_train.make_future_dataframe(periods=len(test_data), freq='B')
                forecast_test = prophet_model_train.predict(future_test)
                test_predictions = forecast_test['yhat'].values[-len(test_data):]
                
                mse = mean_squared_error(test_data['Close'], test_predictions)
                rmse = np.sqrt(mse)
                accuracy_metric['Prophet'] = {'MSE': mse, 'RMSE': rmse}

                prophet_df_full = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
                prophet_model_full = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True, interval_width=0.95)
                prophet_model_full.fit(prophet_df_full)
                future_forecast = prophet_model_full.make_future_dataframe(periods=forecast_steps, freq='B')
                forecast_future = prophet_model_full.predict(future_forecast)
                
                fig_prophet = go.Figure()
                fig_prophet.add_trace(go.Scatter(x=forecast_future['ds'], y=forecast_future['yhat_lower'], mode='lines', name='Lower Bound', line=dict(width=0), showlegend=False))
                fig_prophet.add_trace(go.Scatter(x=forecast_future['ds'], y=forecast_future['yhat_upper'], mode='lines', name='Upper Bound', fill='tonexty', fillcolor='rgba(152, 251, 152, 0.4)', line=dict(width=0)))
                fig_prophet.add_trace(go.Scatter(x=prophet_df_full['ds'], y=prophet_df_full['y'], mode='lines', name='Historical Data', line=dict(color='royalblue')))
                fig_prophet.add_trace(go.Scatter(x=forecast_future['ds'], y=forecast_future['yhat'], mode='lines', name='Prophet Forecast', line=dict(color='green', dash='dash')))
                fig_prophet.update_layout(title=f'Prophet {ticker} Price Forecast', xaxis_title='Date', yaxis_title='Close Price (USD)', template='plotly_dark')
                st.plotly_chart(fig_prophet, use_container_width=True)

                with st.expander("See Prophet Forecast Components"):
                    fig_prophet_components = prophet_model_full.plot_components(forecast_future)
                    st.pyplot(fig_prophet_components)

            elif model_choice == 'LSTM':
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

                def create_dataset(dataset, time_step=1):
                    dataX, dataY = [], []
                    for i in range(len(dataset) - time_step):
                        a = dataset[i:(i + time_step), 0]
                        dataX.append(a)
                        dataY.append(dataset[i + time_step, 0])
                    return np.array(dataX), np.array(dataY)
                
                train_data_scaled = scaled_data[0:train_size]
                test_data_scaled = scaled_data[train_size - time_step:]

                X_train, y_train = create_dataset(train_data_scaled, time_step)
                X_test, y_test = create_dataset(test_data_scaled, time_step)
                
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                
                model_lstm = Sequential()
                model_lstm.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
                model_lstm.add(LSTM(50, return_sequences=False))
                model_lstm.add(Dense(25))
                model_lstm.add(Dense(1))
                model_lstm.compile(optimizer='adam', loss='mean_squared_error')
                model_lstm.fit(X_train, y_train, batch_size=64, epochs=10, verbose=0)
                
                test_predictions_scaled = model_lstm.predict(X_test)
                test_predictions_lstm = scaler.inverse_transform(test_predictions_scaled)
                y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

                mse = mean_squared_error(y_test_actual, test_predictions_lstm)
                rmse = np.sqrt(mse)
                accuracy_metric['LSTM'] = {'MSE': mse, 'RMSE': rmse}

                current_input_sequence_scaled = list(scaled_data[-time_step:].flatten())
                predictions_scaled = []
                for i in range(forecast_steps):
                    x_input = np.array(current_input_sequence_scaled).reshape(1, time_step, 1)
                    yhat_scaled = model_lstm.predict(x_input, verbose=0)[0, 0]
                    predictions_scaled.append(yhat_scaled)
                    current_input_sequence_scaled.append(yhat_scaled)
                    current_input_sequence_scaled = current_input_sequence_scaled[1:]
                
                predictions_lstm = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))
                last_historical_date = df.index[-1]
                forecast_index = pd.date_range(start=last_historical_date + pd.Timedelta(days=1), periods=len(predictions_lstm), freq='B')
                forecast_df = pd.DataFrame({'Forecast': predictions_lstm.flatten()}, index=forecast_index)

                fig_lstm = go.Figure()
                fig_lstm.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Historical Data', line=dict(color='royalblue')))
                fig_lstm.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='LSTM Forecast', line=dict(color='purple', dash='dash')))
                fig_lstm.update_layout(title=f'LSTM {ticker} Price Forecast', xaxis_title='Date', yaxis_title='Close Price (USD)', template='plotly_dark')
                st.plotly_chart(fig_lstm, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during forecasting: {e}")
            import traceback
            st.text(traceback.format_exc())

    st.markdown("---")
    st.subheader(f'Model Accuracy on Test Data (Last 20% of the {ticker} dataset)')
    st.markdown("Metrics are calculated by training the model on 80% of the data and predicting on the remaining 20%.")
    
    if model_choice in accuracy_metric:
        st.markdown(f"**{model_choice} Metrics:**")
        st.write(f"Mean Squared Error (MSE): **{accuracy_metric[model_choice]['MSE']:.2f}**")
        st.write(f"Root Mean Squared Error (RMSE): **${accuracy_metric[model_choice]['RMSE']:.2f}**")
    else:
        st.warning("Accuracy metrics could not be calculated for this model.")

st.sidebar.success('App created by a fellow data enthusiast!')