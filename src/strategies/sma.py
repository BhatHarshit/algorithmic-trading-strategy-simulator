# src/strategies/sma.py

import pandas as pd


def sma_strategy(
    data: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50
) -> pd.DataFrame:
    """
    Simple Moving Average (SMA) crossover strategy.

    Parameters
    ----------
    data : pd.DataFrame
        Price data with 'Close' column.
    short_window : int
        Short-term SMA window.
    long_window : int
        Long-term SMA window.

    Returns
    -------
    pd.DataFrame
        DataFrame with SMA values and trading signals.
    """

    df = data.copy()

    # Calculate SMAs
    df["SMA_Short"] = df["Close"].rolling(window=short_window).mean()
    df["SMA_Long"] = df["Close"].rolling(window=long_window).mean()

    # Initialize signal column
    df["Signal"] = 0

    # Generate signals
    df.loc[df["SMA_Short"] > df["SMA_Long"], "Signal"] = 1
    df.loc[df["SMA_Short"] < df["SMA_Long"], "Signal"] = -1

    # Position change (trade execution points)
    df["Position"] = df["Signal"].diff()

    return df
