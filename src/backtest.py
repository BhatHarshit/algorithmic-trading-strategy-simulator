# src/backtest.py

import pandas as pd
import numpy as np


def run_backtest(df, target_vol=0.02, vol_window=20):
    """
    Expects df with columns:
    ['Close', 'Signal']

    Adds:
    - Volatility-scaled position sizing
    - Risk-adjusted strategy returns
    """

    data = df.copy()

    # ======================
    # Market Returns
    # ======================
    data["Market_Return"] = data["Close"].pct_change()

    # ======================
    # Volatility Estimation
    # ======================
    data["Volatility"] = data["Market_Return"].rolling(vol_window).std()

    # ======================
    # Strategy Positioning
    # ======================
    # Shift signal to avoid lookahead bias
    data["Position"] = data["Signal"].shift(1)

    # Volatility-scaled position sizing
    data["Position_Size"] = np.where(
        data["Volatility"] > 0,
        target_vol / data["Volatility"],
        0
    )

    # Cap exposure (no leverage explosion)
    data["Position_Size"] = data["Position_Size"].clip(0, 1)

    # ======================
    # Strategy Returns
    # ======================
    data["Strategy_Return"] = (
        data["Position"] *
        data["Position_Size"] *
        data["Market_Return"]
    )

    # ======================
    # Cumulative Returns
    # ======================
    data["Cumulative_Market"] = (1 + data["Market_Return"]).cumprod()
    data["Cumulative_Strategy"] = (1 + data["Strategy_Return"]).cumprod()

    return data
