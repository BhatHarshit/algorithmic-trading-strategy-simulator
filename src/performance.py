import numpy as np
import pandas as pd



def calculate_metrics(df, risk_free_rate=0.0):
    returns = df["Strategy_Return"].dropna()

    total_return = df["Cumulative_Strategy"].iloc[-1] - 1

    sharpe_ratio = (
        np.sqrt(252) * (returns.mean() - risk_free_rate) / returns.std()
        if returns.std() != 0 else 0
    )

    cumulative = df["Cumulative_Strategy"]
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    return {
        "Total Return": round(total_return, 4),
        "Sharpe Ratio": round(sharpe_ratio, 4),
        "Max Drawdown": round(max_drawdown, 4),
    }
