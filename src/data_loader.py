import yfinance as yf
import pandas as pd
import os
from config import STOCKS

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
