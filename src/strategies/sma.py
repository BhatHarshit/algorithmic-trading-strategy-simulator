# src/strategies/sma.py

import pandas as pd


def sma_strategy(
    data: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50,
    trend_window: int = 200
) -> pd.DataFrame:
    """
    SMA crossover strategy with trend regime filter.

    Trades are only allowed when price is above long-term trend SMA.

    Parameters
    ----------
    data : pd.DataFrame
        Price data with 'Close' column.
    short_window : int
        Short-term SMA window.
    long_window : int
        Long-term SMA window.
    trend_window : int
        Long-term trend filter window (default = 200).

    Returns
    -------
    pd.DataFrame
        DataFrame with SMA values and trading signals.
    """

    df = data.copy()

    # ======================
    # Moving Averages
    # ======================
    df["SMA_Short"] = df["Close"].rolling(window=short_window).mean()
    df["SMA_Long"] = df["Close"].rolling(window=long_window).mean()
    df["SMA_Trend"] = df["Close"].rolling(window=trend_window).mean()

    # ======================
    # Signal Logic
    # ======================
    df["Signal"] = 0

    # Bullish crossover ONLY in uptrend
    df.loc[
        (df["SMA_Short"] > df["SMA_Long"]) &
        (df["Close"] > df["SMA_Trend"]),
        "Signal"
    ] = 1

    # Exit condition
    df.loc[
        (df["SMA_Short"] < df["SMA_Long"]) |
        (df["Close"] < df["SMA_Trend"]),
        "Signal"
    ] = 0

    # ======================
    # Position Change
    # ======================
    df["Position"] = df["Signal"].diff()

    return df
