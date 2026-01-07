import pandas as pd
import numpy as np


def rsi_strategy(
    data: pd.DataFrame,
    window: int = 14,
    neutral_low: int = 40,
    neutral_high: int = 60
) -> pd.DataFrame:
    """
    RSI risk modulation strategy (resume-aligned).
    Produces smooth signal instead of binary trades.
    """

    df = data.copy()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # ======================
    # Smooth RSI Signal
    # ======================
    df["Signal"] = 0.0

    # Bullish bias in lower-neutral zone
    df.loc[
        (df["RSI"] >= neutral_low) & (df["RSI"] < neutral_high),
        "Signal"
    ] = (neutral_high - df["RSI"]) / (neutral_high - neutral_low)

    # Bearish bias above neutral zone
    df.loc[
        df["RSI"] > neutral_high,
        "Signal"
    ] = - (df["RSI"] - neutral_high) / (100 - neutral_high)

    # Normalize
    df["Signal"] = df["Signal"].clip(-1, 1)

    df["Position"] = df["Signal"].diff()

    return df
