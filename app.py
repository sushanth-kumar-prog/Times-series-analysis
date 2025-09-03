# app.py
# Version 2.2 - Adjusted for Polygon.io 2-year data limit
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.metrics import mean_squared_error
from polygon import RESTClient

# Models
import pmdarima as pm
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Plotly
import plotly.graph_objects as go

# ---------------------------------
# Streamlit Page Config
# ---------------------------------
st.set_page_config(page_title="Dynamic Stock Price Forecasting", layout="wide")
st.title("📈 Enhanced Stock Price Forecasting App")
st.markdown("Predict future stock prices using Auto-ARIMA, Prophet, and LSTM with proper validation.")

# ---------------------------------
# Sidebar Inputs
# ---------------------------------
st.sidebar.header("⚙️ Input Parameters")
ticker = st.sidebar.text_input("Stock Ticker", "GOOGL").upper()

# Your API Key from the previous step
POLYGON_API_KEY = "nd8GPrvHgo_5bXBZniizp3alkYdFGbOX"

# ✅ FIX 1: Change the default start date to be within the 2-year limit
end_date = st.sidebar.date_input("End Date", date.today())
start_date = st.sidebar.date_input("Start Date", end_date - timedelta(days=365*1)) # Changed from 5 years to 1 year

# ✅ FIX 2: Add an informative message about the data limit
st.sidebar.info("Note: The free data source (Polygon.io) provides access to a maximum of 2 years of historical daily data.")

train_ratio = st.sidebar.slider("Train/Test Split Ratio", 0.6, 0.95, 0.8, 0.05)
forecast_days = st.sidebar.number_input("Days to Forecast into Future", min_value=7, max_value=365, value=30, step=7)

st.sidebar.markdown("---")
st.sidebar.header("🤖 Model Selection")
run_arima = st.sidebar.checkbox("Run Auto-ARIMA", True)
run_prophet = st.sidebar.checkbox("Run Prophet", True)
run_lstm = st.sidebar.checkbox("Run LSTM (Can be slow)", True)

# ---------------------------------
# Data Loading Function
# ---------------------------------
@st.cache_data
def load_stock_data(api_key, ticker, start, end):
    """
    Loads split-adjusted stock data from Polygon.io.
    """
    if not api_key or api_key == "YOUR_POLYGON_API_KEY":
        st.error("Polygon.io API key is not set. Please add your key to the script.")
        return None
    try:
        client = RESTClient(api_key)
        aggs = client.get_aggs(ticker=ticker, multiplier=1, timespan="day", from_=start, to=end, adjusted=True, limit=50000)
        
        if not aggs:
            raise ValueError("No data found for the given ticker and date range from Polygon.io.")
            
        df = pd.DataFrame(aggs)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.rename(columns={'close': 'Close'}, inplace=True)
        
        return df[['Close']]

    except Exception as e:
        st.error(f"Failed to load data for {ticker} from Polygon.io: {e}")
        return None

# ---------------------------------
# Main App Logic
# ---------------------------------
data = load_stock_data(POLYGON_API_KEY, ticker, str(start_date), str(end_date))

if data is not None:
    # The rest of your code remains the same and is correct
    st.subheader(f"📊 Historical Prices for {ticker} (Adjusted Close)")
    st.line_chart(data["Close"])

    split_index = int(len(data) * train_ratio)
    train_df, test_df = data.iloc[:split_index], data.iloc[split_index:]
    st.write(f"Training set: {len(train_df)} days | Test set: {len(test_df)} days")

    forecasts = {}
    metrics = {}

    if st.button("🚀 Generate Forecast"):
        if run_arima:
            with st.spinner("Fitting Auto-ARIMA model..."):
                try:
                    arima_model = pm.auto_arima(train_df["Close"], seasonal=False, stepwise=True, suppress_warnings=True, error_action="ignore")
                    test_pred, _ = arima_model.predict(n_periods=len(test_df), return_conf_int=True)
                    metrics["ARIMA"] = {"RMSE": np.sqrt(mean_squared_error(test_df["Close"], test_pred))}
                    full_arima_model = pm.auto_arima(data["Close"], seasonal=False, stepwise=True, suppress_warnings=True, error_action="ignore")
                    future_pred = full_arima_model.predict(n_periods=forecast_days)
                    forecasts["ARIMA"] = future_pred
                    st.success("ARIMA model fitted and forecasted successfully.")
                except Exception as e:
                    st.error(f"ARIMA failed: {e}")

        if run_prophet:
            with st.spinner("Fitting Prophet model..."):
                try:
                    prophet_train_df = train_df.reset_index().rename(columns={"timestamp": "ds", "Close": "y"})
                    prophet_model = Prophet(daily_seasonality=True).fit(prophet_train_df)
                    test_future = prophet_model.make_future_dataframe(periods=len(test_df), freq='B')
                    test_pred_df = prophet_model.predict(test_future)
                    test_pred = test_pred_df['yhat'][-len(test_df):]
                    metrics["Prophet"] = {"RMSE": np.sqrt(mean_squared_error(test_df["Close"], test_pred))}
                    full_prophet_df = data.reset_index().rename(columns={"timestamp": "ds", "Close": "y"})
                    full_prophet_model = Prophet(daily_seasonality=True).fit(full_prophet_df)
                    future_df = full_prophet_model.make_future_dataframe(periods=forecast_days, freq='B')
                    future_pred_df = full_prophet_model.predict(future_df)
                    forecasts["Prophet"] = future_pred_df['yhat'][-forecast_days:]
                    st.success("Prophet model fitted and forecasted successfully.")
                except Exception as e:
                    st.error(f"Prophet failed: {e}")
        
        if run_lstm:
            with st.spinner("Fitting LSTM model... (this may take a moment)"):
                try:
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    train_scaled = scaler.fit_transform(train_df)
                    
                    def create_sequences(data, window_size):
                        X, y = [], []
                        for i in range(window_size, len(data)):
                            X.append(data[i-window_size:i, 0])
                            y.append(data[i, 0])
                        return np.array(X), np.array(y)
                    
                    window_size = 60
                    if len(train_scaled) <= window_size:
                        raise ValueError(f"Not enough training data ({len(train_scaled)} points) to create a sequence of length {window_size}.")

                    X_train, y_train = create_sequences(train_scaled, window_size)
                    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

                    lstm_model = Sequential([
                        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                        LSTM(50, return_sequences=False),
                        Dense(25),
                        Dense(1)
                    ])
                    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
                    lstm_model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=0)
                    
                    inputs = data['Close'][len(data) - len(test_df) - window_size:].values.reshape(-1,1)
                    inputs = scaler.transform(inputs)
                    
                    X_test = []
                    for i in range(window_size, len(inputs)):
                        X_test.append(inputs[i-window_size:i, 0])
                    X_test = np.array(X_test)
                    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                    
                    test_pred_scaled = lstm_model.predict(X_test, verbose=0)
                    test_pred = scaler.inverse_transform(test_pred_scaled).flatten()
                    metrics["LSTM"] = {"RMSE": np.sqrt(mean_squared_error(test_df["Close"], test_pred))}
                    
                    last_sequence_full_data = data['Close'][-window_size:].values.reshape(-1, 1)
                    last_sequence_scaled = scaler.transform(last_sequence_full_data)

                    future_preds_scaled = []
                    current_sequence = last_sequence_scaled.reshape(1, window_size, 1)

                    for _ in range(forecast_days):
                        pred = lstm_model.predict(current_sequence, verbose=0)[0,0]
                        future_preds_scaled.append(pred)
                        new_item = np.array([[pred]])
                        current_sequence = np.append(current_sequence[:, 1:, :], new_item.reshape(1,1,1), axis=1)

                    future_pred = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1,1)).flatten()
                    forecasts["LSTM"] = future_pred
                    st.success("LSTM model fitted and forecasted successfully.")
                except Exception as e:
                    st.error(f"LSTM failed: {e}")

        if forecasts:
            st.subheader("📈 Forecast vs Actuals")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train_df.index, y=train_df['Close'], mode='lines', name='Training Data', line=dict(color='royalblue')))
            fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Close'], mode='lines', name='Actual Test Data', line=dict(color='orange')))
            
            future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=forecast_days, freq='B')
            
            colors = {"ARIMA": "red", "Prophet": "limegreen", "LSTM": "magenta"}
            for model_name, pred in forecasts.items():
                fig.add_trace(go.Scatter(x=future_dates, y=pred, mode='lines', name=f'{model_name} Forecast', line=dict(color=colors[model_name], dash='dash')))
            
            fig.update_layout(title=f'{ticker} Price Forecast', xaxis_title='Date', yaxis_title='Price (USD)', template='plotly_dark', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

        if metrics:
            st.subheader("📊 Model Performance on Test Set (RMSE)")
            metrics_df = pd.DataFrame(metrics).T
            st.dataframe(metrics_df)