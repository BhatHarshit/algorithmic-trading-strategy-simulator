import pandas as pd

def run_backtest(df):
    """
    Expects df with columns:
    ['Close', 'Signal']
    """

    data = df.copy()

    # Market returns
    data["Market_Return"] = data["Close"].pct_change()

    # Strategy positions (shift to avoid lookahead bias)
    data["Position"] = data["Signal"].shift(1)

    # Strategy returns
    data["Strategy_Return"] = data["Position"] * data["Market_Return"]

    # Cumulative returns
    data["Cumulative_Market"] = (1 + data["Market_Return"]).cumprod()
    data["Cumulative_Strategy"] = (1 + data["Strategy_Return"]).cumprod()

    return data
