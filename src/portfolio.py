# src/portfolio.py

import pandas as pd
import numpy as np


def run_portfolio_backtest(
    strategy_results: dict,
    vol_window: int = 20,
    target_vol: float = 0.10  # 10% annualized target volatility
) -> pd.DataFrame:
    """
    Combines multiple strategy backtest DataFrames into a portfolio
    using:
    - Inverse volatility (risk parity) weighting
    - Portfolio volatility targeting
    - Drawdown-aware risk shaping (Step A)
    """

    returns = pd.DataFrame()

    # ============================
    # Collect Strategy Returns
    # ============================
    for ticker, df in strategy_results.items():
        returns[ticker] = df["Strategy_Return"]

    returns = returns.dropna()

    # ============================
    # Asset Volatility Estimation
    # ============================
    asset_vol = returns.rolling(vol_window).std()

    # ============================
    # Inverse Volatility Weights
    # ============================
    inv_vol = 1 / asset_vol
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)

    # ============================
    # Portfolio Raw Return
    # ============================
    portfolio_return = (weights * returns).sum(axis=1)

    # =====================================================
    # Step A: Drawdown-Aware Risk Shaping (IMPROVED VERSION)
    # =====================================================
    equity = (1 + portfolio_return).cumprod()
    rolling_peak = equity.cummax()
    drawdown = (equity - rolling_peak) / rolling_peak

    # Exposure scaling based on drawdown
    exposure = pd.Series(1.0, index=drawdown.index)

    exposure[drawdown <= -0.05] = 0.75
    exposure[drawdown <= -0.10] = 0.50
    exposure[drawdown <= -0.15] = 0.30

    # Apply exposure scaling
    portfolio_return = portfolio_return * exposure

    # ============================
    # Portfolio Volatility Targeting
    # ============================
    portfolio_vol = portfolio_return.rolling(vol_window).std() * np.sqrt(252)

    scaling_factor = target_vol / portfolio_vol
    scaling_factor = scaling_factor.clip(upper=1.5)

    portfolio_return_scaled = portfolio_return * scaling_factor

    # ============================
    # Final Portfolio DataFrame
    # ============================
    portfolio = pd.DataFrame({
        "Portfolio_Return": portfolio_return_scaled
    })

    portfolio["Cumulative_Return"] = (1 + portfolio["Portfolio_Return"]).cumprod()

    return portfolio
