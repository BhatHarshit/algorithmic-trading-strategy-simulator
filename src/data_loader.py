import yfinance as yf
import pandas as pd

def download_stock_data(ticker, start="2019-01-01", end="2024-01-01"):
    data = yf.download(ticker, start=start, end=end)
    data.reset_index(inplace=True)
    return data

if __name__ == "__main__":
    df = download_stock_data("AAPL")
    print(df.head())
