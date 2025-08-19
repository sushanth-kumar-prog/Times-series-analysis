# app.py
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
import yfinance as yf

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

st.set_page_config(layout="wide", page_title="Stock Price Forecasting App")

st.title('📈 Stock Price Forecasting App')
st.markdown("This app fetches live stock data and performs time series forecasting using ARIMA, Prophet, and LSTM.")

# --- Stock Data Selection ---
st.sidebar.header("Stock Selection")
ticker_symbol = st.sidebar.text_input('Enter Stock Ticker (e.g., AAPL, GOOG, MSFT)', 'GOOG')

today = date.today()
start_date = st.sidebar.date_input('Start date', date(2020, 1, 1))
end_date = st.sidebar.date_input('End date', today)

@st.cache_data
def get_data(ticker, start, end):
    """Fetches stock data from Yahoo Finance."""
    data = yf.download(ticker, start=start, end=end, progress=False)
    return data

data = get_data(ticker_symbol, start_date, end_date)

if data is not None and not data.empty:
    # --- Plot Raw Data with Plotly ---
    st.subheader(f'Raw Data for {ticker_symbol} (Last 5 Rows)')
    st.write(data.tail())

    st.subheader(f'{ticker_symbol} Closing Price Over Time')
    fig_raw = px.line(data, x=data.index, y='Close', title=f'{ticker_symbol} Historical Closing Price',
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
        # Check for sufficient data for LSTM's time_step
        time_step = 60
        if model_choice == 'LSTM' and len(data) < time_step + 1:
            st.error(f"Not enough historical data to train the LSTM model. It requires at least {time_step + 1} data points. Your dataset has {len(data)}.")
            st.stop()
        
        # Split data into training and testing sets (80/20 split)
        train_size = int(len(data) * 0.8)
        train_data, test_data = data[0:train_size], data[train_size:len(data)]

        st.subheader(f'{model_choice} Forecast')
        accuracy_metric = {}

        with st.spinner(f'Generating {model_choice} forecast...'):
            if model_choice == 'ARIMA':
                history = [x for x in train_data['Close']]
                test_predictions = []
                for t in range(len(test_data)):
                    model = ARIMA(history, order=(5, 1, 0))
                    model_fit = model.fit()
                    output = model_fit.forecast()
                    yhat = output[0]
                    test_predictions.append(yhat)
                    history.append(test_data['Close'].iloc[t])
                
                test_predictions = pd.Series(test_predictions, index=test_data.index)
                mse = mean_squared_error(test_data['Close'], test_predictions)
                rmse = np.sqrt(mse)
                accuracy_metric['ARIMA'] = {'MSE': mse, 'RMSE': rmse}

                future_history = [x for x in data['Close']]
                future_predictions = []
                for t in range(forecast_steps):
                    model = ARIMA(future_history, order=(5,1,0))
                    model_fit = model.fit()
                    output = model_fit.forecast()
                    yhat = output[0]
                    future_predictions.append(yhat)
                    future_history.append(yhat)
                
                last_historical_date = data.index[-1]
                forecast_index = pd.date_range(start=last_historical_date + pd.Timedela(days=1), periods=len(future_predictions), freq='B')
                forecast_df = pd.DataFrame({'Forecast': future_predictions}, index=forecast_index)

                fig_arima = go.Figure()
                fig_arima.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Data', line=dict(color='royalblue')))
                fig_arima.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='ARIMA Forecast', line=dict(color='red', dash='dash')))
                fig_arima.update_layout(title=f'ARIMA {ticker_symbol} Price Forecast', xaxis_title='Date', yaxis_title='Close Price (USD)', template='plotly_dark')
                st.plotly_chart(fig_arima, use_container_width=True)
                
            elif model_choice == 'Prophet':
                prophet_df_train = train_data.reset_index().rename(columns={'index': 'ds', 'Close': 'y'})
                prophet_model_train = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True, interval_width=0.95)
                prophet_model_train.fit(prophet_df_train)

                future_test = prophet_model_train.make_future_dataframe(periods=len(test_data), freq='B', include_history=False)
                forecast_test = prophet_model_train.predict(future_test)
                test_predictions = forecast_test['yhat'].values
                
                mse = mean_squared_error(test_data['Close'], test_predictions)
                rmse = np.sqrt(mse)
                accuracy_metric['Prophet'] = {'MSE': mse, 'RMSE': rmse}

                prophet_df_full = data.reset_index().rename(columns={'index': 'ds', 'Close': 'y'})
                prophet_model_full = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True, interval_width=0.95)
                prophet_model_full.fit(prophet_df_full)
                future_forecast = prophet_model_full.make_future_dataframe(periods=forecast_steps, freq='B')
                forecast_future = prophet_model_full.predict(future_forecast)
                
                fig_prophet = go.Figure()
                fig_prophet.add_trace(go.Scatter(x=forecast_future['ds'], y=forecast_future['yhat_lower'], mode='lines', name='Lower Bound', line=dict(width=0), showlegend=False))
                fig_prophet.add_trace(go.Scatter(x=forecast_future['ds'], y=forecast_future['yhat_upper'], mode='lines', name='Upper Bound', fill='tonexty', fillcolor='rgba(152, 251, 152, 0.4)', line=dict(width=0)))
                fig_prophet.add_trace(go.Scatter(x=prophet_df_full['ds'], y=prophet_df_full['y'], mode='lines', name='Historical Data', line=dict(color='royalblue')))
                fig_prophet.add_trace(go.Scatter(x=forecast_future['ds'], y=forecast_future['yhat'], mode='lines', name='Prophet Forecast', line=dict(color='green', dash='dash')))
                fig_prophet.update_layout(title=f'Prophet {ticker_symbol} Price Forecast', xaxis_title='Date', yaxis_title='Close Price (USD)', template='plotly_dark')
                st.plotly_chart(fig_prophet, use_container_width=True)

                with st.expander("See Prophet Forecast Components"):
                    fig_prophet_components = prophet_model_full.plot_components(forecast_future)
                    st.pyplot(fig_prophet_components)

            elif model_choice == 'LSTM':
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data_train = scaler.fit_transform(train_data[['Close']])
                
                def create_dataset(dataset, time_step=1):
                    dataX, dataY = [], []
                    for i in range(len(dataset) - time_step):
                        a = dataset[i:(i + time_step), 0]
                        dataX.append(a)
                        dataY.append(dataset[i + time_step, 0])
                    return np.array(dataX), np.array(dataY)
                
                X_train, y_train = create_dataset(scaled_data_train, time_step)
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                
                model_lstm = Sequential()
                model_lstm.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
                model_lstm.add(LSTM(50, return_sequences=False))
                model_lstm.add(Dense(25))
                model_lstm.add(Dense(1))
                model_lstm.compile(optimizer='adam', loss='mean_squared_error')
                model_lstm.fit(X_train, y_train, batch_size=64, epochs=10, verbose=0)
                
                # Prepare test data for accuracy evaluation
                full_dataset_scaled = scaler.transform(data[['Close']])
                test_data_scaled = full_dataset_scaled[train_size - time_step:]
                X_test, y_test = create_dataset(test_data_scaled, time_step)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                test_predictions_scaled = model_lstm.predict(X_test, verbose=0)
                test_predictions_lstm = scaler.inverse_transform(test_predictions_scaled)

                mse = mean_squared_error(test_data['Close'], test_predictions_lstm.flatten())
                rmse = np.sqrt(mse)
                accuracy_metric['LSTM'] = {'MSE': mse, 'RMSE': rmse}

                # Prepare full dataset for final forecast
                scaled_data_full = scaler.fit_transform(data[['Close']])
                X_full, y_full = create_dataset(scaled_data_full, time_step)
                X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], 1)
                
                model_lstm_forecast = Sequential()
                model_lstm_forecast.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
                model_lstm_forecast.add(LSTM(50, return_sequences=False))
                model_lstm_forecast.add(Dense(25))
                model_lstm_forecast.add(Dense(1))
                model_lstm_forecast.compile(optimizer='adam', loss='mean_squared_error')
                model_lstm_forecast.fit(X_full, y_full, batch_size=64, epochs=10, verbose=0)

                current_input_sequence = list(scaled_data_full[-time_step:].flatten())
                predictions_scaled = []
                for i in range(forecast_steps):
                    x_input = np.array(current_input_sequence).reshape(1, time_step, 1)
                    yhat_scaled = float(model_lstm_forecast.predict(x_input, verbose=0)[0, 0])
                    predictions_scaled.append(yhat_scaled)
                    current_input_sequence.append(yhat_scaled)
                    current_input_sequence = current_input_sequence[1:]
                
                predictions_lstm = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))
                last_historical_date = data.index[-1]
                forecast_index = pd.date_range(start=last_historical_date + pd.Timedelta(days=1), periods=len(predictions_lstm), freq='B')
                forecast_df = pd.DataFrame({'Forecast': predictions_lstm.flatten()}, index=forecast_index)

                fig_lstm = go.Figure()
                fig_lstm.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Data', line=dict(color='royalblue')))
                fig_lstm.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='LSTM Forecast', line=dict(color='purple', dash='dash')))
                fig_lstm.update_layout(title=f'LSTM {ticker_symbol} Price Forecast', xaxis_title='Date', yaxis_title='Close Price (USD)', template='plotly_dark')
                st.plotly_chart(fig_lstm, use_container_width=True)

        st.markdown("---")
        st.subheader(f'Model Accuracy on Test Data (Last 20% of the {ticker_symbol} dataset)')
        st.markdown("Metrics are calculated by training the model on 80% of the data and predicting on the remaining 20%.")
        
        if model_choice in accuracy_metric:
            st.markdown(f"**{model_choice} Metrics:**")
            st.write(f"Mean Squared Error (MSE): **{accuracy_metric[model_choice]['MSE']:.2f}**")
            st.write(f"Root Mean Squared Error (RMSE): **${accuracy_metric[model_choice]['RMSE']:.2f}**")
        else:
            st.warning("Accuracy metrics could not be calculated for this model.")
