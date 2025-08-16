import yfinance as yf
import os

os.makedirs("data/raw", exist_ok=True)

ticker = 'GOOGL'
start_date = "2015-01-01"
end_date = "2025-01-01"

print(f"Downloading data for {ticker}...")
data = yf.download(ticker, start=start_date, end=end_date)
data.to_csv(f"data/raw/{ticker}.csv")
print(f"Saved {ticker} data to data/raw/{ticker}.csv")
