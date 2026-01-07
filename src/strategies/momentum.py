import pandas as pd
import numpy as np


def momentum_strategy(
    data: pd.DataFrame,
    window: int = 20,
    vol_window: int = 20
) -> pd.DataFrame:
    """
    Volatility-normalized momentum strategy.
    """

    df = data.copy()

    if "Close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Close' column")

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # ======================
    # Momentum
    # ======================
    raw_momentum = df["Close"].pct_change(window)

    # ======================
    # Volatility Normalization
    # ======================
    volatility = df["Close"].pct_change().rolling(vol_window).std()

    df["Signal"] = raw_momentum / volatility
    df["Signal"] = df["Signal"].clip(-2, 2)

    # Normalize to [-1, 1]
    df["Signal"] = df["Signal"] / df["Signal"].abs().rolling(50).max()

    df["Position"] = df["Signal"].diff()

    df = df.dropna().reset_index(drop=True)

    return df
