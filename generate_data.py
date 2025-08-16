import pandas as pd
import numpy as np
from datetime import timedelta

# Define the number of rows (data points)
num_rows = 10000

# Generate dates
# Start date: January 1, 2000
start_date = pd.to_datetime('2000-01-01')
# Create a date range, assuming business days (Mon-Fri)
# This will generate more than 10,000 dates, so we'll slice it later
dates = pd.date_range(start=start_date, periods=num_rows * 1.5, freq='B') # Generate more to ensure 10k business days

# Initialize a base price and add some random walk
# Starting price for the stock
base_price = 100.0
# Generate daily returns with a small mean and standard deviation
# np.random.randn generates samples from a standard normal distribution (mean 0, variance 1)
# We multiply by a small factor (0.005) to simulate realistic daily price fluctuations
# Add a small positive drift (0.0001) to simulate a general upward trend over long periods
daily_returns = np.random.normal(loc=0.0001, scale=0.005, size=len(dates))
# Calculate cumulative returns
cumulative_returns = np.exp(np.cumsum(daily_returns))
# Calculate prices
prices = base_price * cumulative_returns

# Create a DataFrame
# Ensure we only take exactly num_rows
df_generated = pd.DataFrame({
    'Date': dates[:num_rows],
    'Close': prices[:num_rows]
})

# Set 'Date' as index
df_generated.set_index('Date', inplace=True)

# Save the DataFrame to a CSV file
# You can change the filename here if you used a different name in app.py
output_filename = 'large_stock_data.csv'
df_generated.to_csv(output_filename)

print(f"Successfully generated and saved {num_rows} rows of synthetic stock data to '{output_filename}'.")
print("First 5 rows of the generated data:")
print(df_generated.head())
print("\nLast 5 rows of the generated data:")
print(df_generated.tail())
