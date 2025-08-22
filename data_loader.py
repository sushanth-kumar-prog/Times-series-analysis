import os
import pandas as pd
import yfinance as yf

class DataLoadError(Exception):
    pass

def load_from_csv(ticker: str) -> pd.DataFrame:
    csv_candidates = [f"{ticker}.csv", f"{ticker.upper()}.csv", "stock_data.csv", "large_stock_data.csv"]
    for f in csv_candidates:
        if os.path.exists(f):
            df = pd.read_csv(f)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"]).sort_values("Date")
                df = df.set_index("Date")
            return df
    raise DataLoadError("No local CSV found for ticker.")

def load_stock_data(ticker: str, start: str, end: str, mode="auto") -> pd.DataFrame:
    """
    mode = 'auto' → try Yahoo first, fallback to CSV
    mode = 'local' → only use CSV
    """
    if mode == "local":
        return load_from_csv(ticker)

    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            raise Exception("Empty dataframe from Yahoo")
        return df
    except Exception as e:
        print(f"Yahoo fetch failed: {e}. Falling back to CSV...")
        return load_from_csv(ticker)
