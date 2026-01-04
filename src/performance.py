import numpy as np
import pandas as pd



import numpy as np
import pandas as pd


# =========================
# STEP 6 — Strategy Metrics
# =========================
def calculate_strategy_metrics(df, risk_free_rate=0.0):
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
        "Total Return": round(float(total_return), 4),
        "Sharpe Ratio": round(float(sharpe_ratio), 4),
        "Max Drawdown": round(float(max_drawdown), 4),
    }


# =========================
# STEP 7 — Portfolio Metrics
# =========================
def calculate_metrics(df, column="Strategy_Return"):
    returns = df[column].dropna()

    total_return = (1 + returns).prod() - 1
    sharpe = returns.mean() / returns.std() if returns.std() != 0 else 0

    cumulative = (1 + returns).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1
    max_drawdown = drawdown.min()

    return {
        "Total Return": round(float(total_return), 4),
        "Sharpe Ratio": round(float(sharpe), 4),
        "Max Drawdown": round(float(max_drawdown), 4),
    }
