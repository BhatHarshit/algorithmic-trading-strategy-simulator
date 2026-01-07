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
def calculate_metrics(df, column="Portfolio_Return", risk_free_rate=0.0):
    returns = df[column].dropna()

    annual_return = (1 + returns.mean()) ** TRADING_DAYS - 1
    annual_vol = returns.std() * np.sqrt(TRADING_DAYS)

    sharpe_ratio = (
        (annual_return - risk_free_rate) / annual_vol
        if annual_vol != 0 else 0
    )

    cumulative = (1 + returns).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1
    max_drawdown = drawdown.min()

    return {
        "Annual Return": round(float(annual_return), 4),
        "Annual Volatility": round(float(annual_vol), 4),
        "Sharpe Ratio": round(float(sharpe_ratio), 4),
        "Max Drawdown": round(float(max_drawdown), 4),
    }
