from prophet import Prophet
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset (assuming 'stock_data.csv' is already generated and preprocessed)
data = pd.read_csv('stock_data.csv', parse_dates=['Date'], index_col='Date')
df = data[['Close']].copy()

# Split data into training and testing sets (80/20 split)
train_size = int(len(df) * 0.8)
train_data, test_data = df[0:train_size], df[train_size:len(df)]

# --- Prophet Model ---
# Prophet requires the dataframe to have specific column names: 'ds' for date and 'y' for value
prophet_train_df = train_data.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})

# Initialize Prophet model. daily_seasonality=True is often useful for stock data.
model_prophet = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
model_prophet.fit(prophet_train_df)

# Create a dataframe for future predictions.
# We make future predictions for the entire test set duration.
future = model_prophet.make_future_dataframe(periods=len(test_data), freq='B') # 'B' for business day frequency
forecast_prophet = model_prophet.predict(future)

# Evaluate the model on the same 30-day subset of the test data for consistent comparison
forecast_steps = min(30, len(test_data))
test_subset = test_data['Close'][:forecast_steps]

# Extract predictions corresponding to the test_subset dates
# We need to align the forecast with the test_subset's dates.
# Prophet's forecast 'ds' column contains all dates, so we can filter.
predictions_prophet_df = forecast_prophet[forecast_prophet['ds'].isin(test_subset.index)]['yhat']
predictions_prophet = predictions_prophet_df.values[:forecast_steps] # Take only the first 'forecast_steps'

# Ensure predictions_prophet has the same length as test_subset
if len(predictions_prophet) != len(test_subset):
    print(f"Warning: Prophet predictions length ({len(predictions_prophet)}) does not match test subset length ({len(test_subset)}). Adjusting for RMSE calculation.")
    # Pad or truncate predictions to match test_subset length
    if len(predictions_prophet) < len(test_subset):
        predictions_prophet = np.pad(predictions_prophet, (0, len(test_subset) - len(predictions_prophet)), 'constant', constant_values=np.nan)
    else:
        predictions_prophet = predictions_prophet[:len(test_subset)]

# Evaluate the model
if not any(np.isnan(predictions_prophet)):
    rmse_prophet = np.sqrt(mean_squared_error(test_subset, predictions_prophet))
    print(f'Prophet Test RMSE (first {forecast_steps} days): {rmse_prophet:.2f}')
else:
    print("Prophet predictions could not be fully generated for RMSE calculation.")
    rmse_prophet = np.nan

# Plot the forecast using Prophet's built-in plotting function
fig1 = model_prophet.plot(forecast_prophet)
plt.title('Prophet Stock Price Forecast (Full Forecast)')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.show()

# Optionally, plot the specific test subset forecast for comparison with other models
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Close'], color='blue', label='Historical Data')
if predictions_prophet_df is not None and not predictions_prophet_df.empty:
    plt.plot(test_subset.index, predictions_prophet, color='purple', linestyle='--', label='Prophet Forecast (Test Subset)')
plt.title('Prophet Stock Price Forecast (Test Subset Comparison)')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
