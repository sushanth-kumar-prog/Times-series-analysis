import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
from sklearn.metrics import mean_squared_error

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Load the dataset (assuming 'stock_data.csv' is already generated and preprocessed)
data = pd.read_csv('stock_data.csv', parse_dates=['Date'], index_col='Date')
df = data[['Close']].copy()

# --- LSTM Model ---
# Scale the data to be between 0 and 1, which is good practice for neural networks
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Create training and testing sets from the scaled data
train_size = int(len(scaled_data) * 0.8)
train_scaled, test_scaled = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]

# Function to create a dataset suitable for LSTM
# It creates input sequences (X) and corresponding output values (Y)
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0] # Features: previous 'time_step' values
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0]) # Target: the value at 'time_step' + 1
    return np.array(dataX), np.array(dataY)

# Define the time step (look-back period) for LSTM
time_step = 60 # Using the past 60 days to predict the next day's price

X_train, y_train = create_dataset(train_scaled, time_step)
X_test, y_test = create_dataset(test_scaled, time_step)

# Reshape input to be [samples, time steps, features]
# LSTM expects a 3D input: (number_of_samples, number_of_timesteps, number_of_features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model_lstm = Sequential()
# First LSTM layer with 50 units, returning sequences for the next LSTM layer
model_lstm.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
# Second LSTM layer with 50 units, not returning sequences (output for the Dense layer)
model_lstm.add(LSTM(50, return_sequences=False))
# A Dense layer with 25 units
model_lstm.add(Dense(25))
# Output Dense layer with 1 unit (for predicting the single next price)
model_lstm.add(Dense(1))

# Compile the model using Adam optimizer and mean squared error loss
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

print("Training LSTM model...")
# Train the model
# batch_size: number of samples per gradient update
# epochs: number of times the entire dataset is passed forward and backward through the neural network
model_lstm.fit(X_train, y_train, batch_size=64, epochs=20, verbose=1)
print("LSTM model training complete.")

# Make predictions
predictions_lstm_scaled = model_lstm.predict(X_test)

# Inverse transform the predictions to get actual price values
predictions_lstm = scaler.inverse_transform(predictions_lstm_scaled)

# Inverse transform y_test to get actual price values for comparison
y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate the model using RMSE
# Note: The original RMSE calculation (np.sqrt(np.mean(predictions_lstm - y_test)**2)) was incorrect.
# It should be np.sqrt(mean_squared_error(actual_values, predicted_values)).
# Also, ensure y_test is inverse transformed for a meaningful comparison.
rmse_lstm = np.sqrt(mean_squared_error(y_test_inverse, predictions_lstm))
print(f'LSTM Test RMSE: {rmse_lstm:.2f}')

# Plot the forecast
# Prepare data for plotting
train = df[:train_size]
# The validation set (actual test data) needs to be adjusted because LSTM uses a look-back period.
# The predictions start after the initial 'time_step' + 1 days in the test_data.
valid = df[train_size:]
# Ensure 'valid' starts from the correct index where predictions begin
valid_start_index = valid.index[time_step + 1] if len(valid) > (time_step + 1) else valid.index[0]
# Explicitly create a copy to avoid SettingWithCopyWarning
valid_for_plot = valid.loc[valid_start_index:].copy()

# Add predictions to the valid DataFrame for easy plotting
if len(predictions_lstm) == len(valid_for_plot):
    valid_for_plot['Predictions'] = predictions_lstm
else:
    print(f"Warning: Length mismatch for plotting. Predictions: {len(predictions_lstm)}, Actual Valid: {len(valid_for_plot)}")
    # Adjust predictions or valid_for_plot to match for plotting
    min_len = min(len(predictions_lstm), len(valid_for_plot))
    valid_for_plot = valid_for_plot.iloc[:min_len]
    valid_for_plot['Predictions'] = predictions_lstm[:min_len]


plt.figure(figsize=(16, 8))
plt.title('LSTM Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(train['Close'], label='Train Data')
plt.plot(valid_for_plot['Close'], label='Actual Test Data')
plt.plot(valid_for_plot['Predictions'], label='LSTM Predictions')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
