# src/strategies/sma.py

import pandas as pd
import numpy as np


def sma_strategy(
    data: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50,
    trend_window: int = 200,
    rsi_window: int = 14
) -> pd.DataFrame:
    """
    Trend-following SMA strategy with RSI-based risk modulation
    (Resume-aligned implementation)
    """

    df = data.copy()

    # ======================
    # Moving Averages
    # ======================
    df["SMA_Short"] = df["Close"].rolling(short_window).mean()
    df["SMA_Long"] = df["Close"].rolling(long_window).mean()
    df["SMA_Trend"] = df["Close"].rolling(trend_window).mean()

    # ======================
    # RSI
    # ======================
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(rsi_window).mean()
    avg_loss = loss.rolling(rsi_window).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # ======================
    # Base SMA Signal
    # ======================
    df["Signal"] = 0.0

    trend_condition = (
        (df["SMA_Short"] > df["SMA_Long"]) &
        (df["Close"] > df["SMA_Trend"])
    )

    df.loc[trend_condition, "Signal"] = 1.0

    # ======================
    # RSI Risk Modulation (NOT ENTRY FILTER)
    # ======================
    df.loc[df["RSI"] > 70, "Signal"] *= 0.5
    df.loc[df["RSI"] < 30, "Signal"] *= 0.5

    # ======================
    # Position Change
    # ======================
    df["Position"] = df["Signal"].diff()

    return df
