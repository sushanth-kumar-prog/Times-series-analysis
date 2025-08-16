from statsmodels.tsa.statespace.sarimax import SARIMAX
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

# --- SARIMA Model ---
# Initialize history with training data for rolling forecast
history_sarima = [x for x in train_data['Close']]
predictions_sarima = []

# We will forecast for the same number of steps as ARIMA for comparison
forecast_steps = min(30, len(test_data))
test_subset = test_data['Close'][:forecast_steps]

print(f"Attempting to forecast {forecast_steps} steps using SARIMA.")

# Perform rolling forecast similar to ARIMA
for t in range(len(test_subset)):
    try:
        # SARIMA model parameters:
        # order=(p,d,q) for the non-seasonal part (e.g., (1,1,1))
        # seasonal_order=(P,D,Q,S) for the seasonal part.
        # Using S=5 for weekly seasonality, which is more memory-efficient for daily data
        # compared to S=252 for yearly seasonality in a rolling forecast.
        model_sarima = SARIMAX(history_sarima, order=(1, 1, 1), seasonal_order=(1, 1, 0, 5))
        # disp=False suppresses convergence messages during fitting
        model_sarima_fit = model_sarima.fit(disp=False)
        
        # Forecast the next step
        output = model_sarima_fit.forecast()
        yhat = output[0]
        predictions_sarima.append(yhat)
        
        # Get the actual observation from the test subset
        obs = test_subset.iloc[t]
        # Add the actual observation to the history for the next iteration
        history_sarima.append(obs)
    except Exception as e:
        print(f"SARIMA forecasting error at step {t}: {e}")
        predictions_sarima.append(np.nan) # Append NaN if an error occurs
        # If an error occurs, break the loop to prevent further issues
        break 

# Evaluate the model only if predictions were successfully made and are not NaNs
if predictions_sarima and not any(np.isnan(predictions_sarima)):
    rmse_sarima = np.sqrt(mean_squared_error(test_subset, predictions_sarima))
    print(f'SARIMA Test RMSE (first {forecast_steps} days): {rmse_sarima:.2f}')
else:
    print("SARIMA predictions could not be fully generated due to errors or insufficient data.")
    rmse_sarima = np.nan # Set RMSE to NaN if predictions failed

# Plot the forecast
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Close'], color='blue', label='Historical Data')

# Plot the SARIMA forecast against the actual test data subset
# Ensure forecast_index matches the length of predictions_sarima for plotting
if predictions_sarima and not any(np.isnan(predictions_sarima)):
    # Use the actual length of predictions_sarima to slice forecast_index
    plot_forecast_index = test_subset.index[:len(predictions_sarima)]
    plt.plot(plot_forecast_index, predictions_sarima, color='green', linestyle='--', label='SARIMA Forecast')
else:
    print("Skipping SARIMA plot due to incomplete or erroneous predictions.")

plt.title('SARIMA Stock Price Forecast (First 30 Days of Test Set)')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
