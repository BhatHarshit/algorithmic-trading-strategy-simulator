# src/performance.py

import numpy as np
import pandas as pd


TRADING_DAYS = 252


# =========================
# Strategy Metrics
# =========================
def calculate_strategy_metrics(df, risk_free_rate=0.0):
    returns = df["Strategy_Return"].dropna()

    annual_return = (1 + returns.mean()) ** TRADING_DAYS - 1
    annual_vol = returns.std() * np.sqrt(TRADING_DAYS)

    sharpe_ratio = (
        (annual_return - risk_free_rate) / annual_vol
        if annual_vol != 0 else 0
    )

    cumulative = df["Cumulative_Strategy"]
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    return {
        "Annual Return": round(float(annual_return), 4),
        "Annual Volatility": round(float(annual_vol), 4),
        "Sharpe Ratio": round(float(sharpe_ratio), 4),
        "Max Drawdown": round(float(max_drawdown), 4),
    }


# =========================
# Portfolio Metrics
# =========================


def calculate_metrics(
    df: pd.DataFrame,
    column: str = "Portfolio_Return",
    risk_free_rate: float = 0.02,
    trading_days: int = 252
) -> dict:

    returns = df[column].dropna()

    # Annualized Return
    cumulative_return = (1 + returns).prod()
    annual_return = cumulative_return ** (trading_days / len(returns)) - 1

    # Annualized Volatility
    annual_volatility = returns.std() * np.sqrt(trading_days)

    # Sharpe Ratio (Proper)
    excess_return = annual_return - risk_free_rate
    sharpe_ratio = excess_return / annual_volatility if annual_volatility != 0 else 0

    # Max Drawdown (Equity Curve)
    equity_curve = (1 + returns).cumprod()
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1
    max_drawdown = drawdown.min()

    return {
        "Annual Return": round(annual_return, 4),
        "Annual Volatility": round(annual_volatility, 4),
        "Sharpe Ratio": round(sharpe_ratio, 4),
        "Max Drawdown": round(max_drawdown, 4)
    }
