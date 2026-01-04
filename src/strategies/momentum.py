import pandas as pd


def momentum_strategy(
    data: pd.DataFrame,
    window: int = 10
) -> pd.DataFrame:
    """
    Momentum-based trading strategy.
    Generates momentum, trading signal, and position changes.
    """

    # Defensive copy
    df = data.copy()

    # Ensure required column exists
    if "Close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Close' column")

    # Ensure numeric Close prices
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # Calculate momentum
    df["Momentum"] = df["Close"].pct_change(periods=window)

    # Initialize Signal column
    df["Signal"] = 0

    # Generate trading signals
    df.loc[df["Momentum"] > 0, "Signal"] = 1
    df.loc[df["Momentum"] < 0, "Signal"] = -1

    # Position change (entry / exit)
    df["Position"] = df["Signal"].diff()

    # Remove NaN rows caused by pct_change & diff
    df = df.dropna().reset_index(drop=True)

    return df
