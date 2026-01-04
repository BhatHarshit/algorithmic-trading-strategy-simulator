import pandas as pd


def rsi_strategy(
    data: pd.DataFrame,
    window: int = 14,
    overbought: int = 70,
    oversold: int = 30
) -> pd.DataFrame:

    df = data.copy()
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Signal"] = 0
    df.loc[df["RSI"] < oversold, "Signal"] = 1
    df.loc[df["RSI"] > overbought, "Signal"] = -1

    df["Position"] = df["Signal"].diff()

    return df
