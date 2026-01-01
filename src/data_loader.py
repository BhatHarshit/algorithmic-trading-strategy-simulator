import yfinance as yf
import pandas as pd
import os
from src.config import STOCKS


DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

def download_stock(ticker, start="2019-01-01", end="2024-01-01"):
    df = yf.download(ticker, start=start, end=end)
    df["Ticker"] = ticker
    df.reset_index(inplace=True)
    return df

def download_all_stocks():
    all_data = []
    for ticker in STOCKS:
        print(f"Downloading {ticker}")
        df = download_stock(ticker)
        df.to_csv(f"{DATA_DIR}/{ticker}.csv", index=False)
        all_data.append(df)
    return pd.concat(all_data)

if __name__ == "__main__":
    data = download_all_stocks()
    print(data.head())
def load_stock_data(ticker: str) -> pd.DataFrame:
    """
    Load stock data from CSV if exists, else download it.
    Ensures numeric Close prices for strategy use.
    """
    file_path = f"{DATA_DIR}/{ticker}.csv"

    if not os.path.exists(file_path):
        df = download_stock(ticker)
        df.to_csv(file_path, index=False)
    else:
        df = pd.read_csv(file_path)

    # Standardize column names
    df.columns = [col.lower() for col in df.columns]

    # Use adjusted close if available, else close
    price_col = "adj close" if "adj close" in df.columns else "close"

    df = df[["date", price_col]].copy()
    df.rename(columns={price_col: "Close"}, inplace=True)

    # Ensure numeric
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # Drop bad rows
    df.dropna(inplace=True)

    return df
