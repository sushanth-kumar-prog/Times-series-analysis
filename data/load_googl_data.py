import pandas as pd

# Define the path to your new, larger CSV file
# Make sure this matches the filename you used in generate_data.py
file_path = 'large_stock_data.csv'

try:
    # Define the correct column names based on your generated CSV's structure
    # The generated CSV has 'Date' and 'Close' columns
    column_names = ['Date', 'Close']

    # Load the data from the CSV file
    # For the generated data, we don't need to skip rows or specify header=None
    # as it's a clean CSV with 'Date' and 'Close' as the first row.
    # parse_dates=['Date']: Converts the 'Date' column to datetime objects.
    # index_col='Date': Sets the 'Date' column as the DataFrame index.
    ticker_df = pd.read_csv(
        file_path,
        parse_dates=['Date'],
        index_col='Date'
    )

    # Save the data to 'stock_data.csv' for consistency with subsequent steps.
    # This ensures all other scripts (preprocessing, models) will use this larger data.
    ticker_df.to_csv('stock_data.csv')

    print(f"Successfully loaded {len(ticker_df)} rows of data from {file_path}.")
    print("Data saved to 'stock_data.csv' for consistent use in subsequent steps.")
    print("First 5 rows of the loaded data:")
    print(ticker_df.head())
    print("\nLast 5 rows of the loaded data:")
    print(ticker_df.tail())

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please ensure '{output_filename}' is in the same directory as this script.")
except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")
