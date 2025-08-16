import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset from the CSV file
# parse_dates=['Date'] converts the 'Date' column to datetime objects
# index_col='Date' sets the 'Date' column as the DataFrame index
data = pd.read_csv('stock_data.csv', parse_dates=['Date'], index_col='Date')

# For our analysis, we will focus solely on the 'Close' price
df = data[['Close']].copy()

# Check for any missing values in the 'Close' price column
print("Missing values in 'Close' price:\n", df.isnull().sum())

# Plot the historical closing price to visually inspect for trends and patterns
plt.figure(figsize=(14, 7))
plt.title('GOOGL Stock Closing Price Over Time') # Changed title to GOOGL
plt.plot(df['Close'], color='blue', linewidth=1.5)
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()

# Decompose the time series into trend, seasonality, and residuals
# We use a multiplicative model as stock prices often exhibit multiplicative seasonality,
# meaning the seasonal fluctuations change with the level of the series.
# The period is set to 252, which is the approximate number of trading days in a year,
# to capture yearly seasonality.
try:
    # Note: With 10,000 rows, a period of 252 is fine for decomposition.
    # This might take a moment due to the larger dataset.
    decomposition = seasonal_decompose(df['Close'], model='multiplicative', period=252)

    # Plot the decomposed components
    fig = decomposition.plot()
    fig.set_size_inches(14, 10)
    fig.suptitle('Time Series Decomposition of GOOGL Stock Closing Price', y=1.02) # Changed title to GOOGL
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout for suptitle
    plt.show()
except Exception as e:
    print(f"Could not perform seasonal decomposition. This might happen if there aren't enough data points for the specified period ({252}). Error: {e}")

print("\nFirst 5 rows of the preprocessed DataFrame:")
print(df.head())
