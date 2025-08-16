from statsmodels.tsa.arima.model import ARIMA
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

# --- ARIMA Model ---
# Initialize history with training data for rolling forecast
history = [x for x in train_data['Close']]
predictions_arima = []

# We will forecast for a specific number of steps in the test set
# For demonstration, we'll forecast the first 30 days of the test set.
forecast_steps = min(30, len(test_data)) # Ensure we don't try to forecast more than available test data
test_subset = test_data['Close'][:forecast_steps]

print(f"Attempting to forecast {forecast_steps} steps using ARIMA.")

# Perform rolling forecast: train the model, make one prediction, and then add the actual observation to history
for t in range(len(test_subset)):
    try:
        # ARIMA model with (p,d,q) = (5,1,0) as a common starting point
        # This means 5 AR terms, 1 differencing, and 0 MA terms.
        model_arima = ARIMA(history, order=(5,1,0))
        model_arima_fit = model_arima.fit()
        
        # Forecast the next step
        output = model_arima_fit.forecast()
        yhat = output[0]
        predictions_arima.append(yhat)
        
        # Get the actual observation from the test subset
        obs = test_subset.iloc[t]
        # Add the actual observation to the history for the next iteration
        history.append(obs)
    except Exception as e:
        print(f"ARIMA forecasting error at step {t}: {e}")
        # If an error occurs, append NaN to predictions and break or continue as needed
        predictions_arima.append(np.nan)
        break # Exit the loop if an error occurs

# Evaluate the model only if predictions were made
if predictions_arima and not any(np.isnan(predictions_arima)):
    rmse_arima = np.sqrt(mean_squared_error(test_subset, predictions_arima))
    print(f'ARIMA Test RMSE (first {forecast_steps} days): {rmse_arima:.2f}')
else:
    print("ARIMA predictions could not be fully generated due to errors or insufficient data.")
    rmse_arima = np.nan # Set RMSE to NaN if predictions failed

# Plot the forecast
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Close'], color='blue', label='Historical Data')

# Plot the ARIMA forecast against the actual test data subset
if predictions_arima:
    # Create a proper index for the predictions
    forecast_index = test_subset.index
    plt.plot(forecast_index, predictions_arima, color='red', linestyle='--', label='ARIMA Forecast')

plt.title('ARIMA Stock Price Forecast (First 30 Days of Test Set)')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
